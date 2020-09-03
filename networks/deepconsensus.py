import multiprocessing

import sys
sys.path.insert(0, "/home/ai/Code/sfmnet")

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.unsuperpoint import brute_force_match, SiameseUnsuperPoint

def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1),
        nn.BatchNorm1d(out_channels),
        #nn.InstanceNorm1d(out_channels),
        nn.ReLU())

def batch_ids_lookup(P, ids):
    # P[B,2,N], ids[B,N]
    B,N = ids.shape
    #return P.transpose(0,1).reshape(2,-1)[:,ids.view(-1)].reshape(2,B,N).transpose(0,1)
    return torch.stack([P[b,:,ids[b]] for b in range(B)])

def batch_mask_lookup(P, mask):
    # P[B,2,N], mask[B,N]
    B,K,N = P.shape
    return [P[b,:,mask[b]] for b in range(B)] # [[2,?] * B]

class Tnet(nn.Module):

    def __init__(self, K=3):
        super().__init__()
        
        # Dimensions of data
        self.K = K

        self.blocks = nn.Sequential(
            block(K, 64),
            block(64, 128),
            block(128, 1024))
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            #nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            #nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, K*K))

    def forward(self, x):
        # [B,K,N]
        N = x.shape[-1]
        x = self.blocks(x) # [B,1024,N]
        x = F.max_pool1d(x, N).squeeze(2) # [B,1024,1] -> [B,1024]
        
        x = self.mlp(x)
        # TODO: Weird instance norm hack
        #x = self.mlp(x.unsqueeze(1)) # [B,K^2]
        #x = x.squeeze(1)
        
        identity = torch.eye(self.K, dtype=torch.double).view(-1).to(x.device)
        x += identity # batch broadcasted
        x = x.view(-1, self.K, self.K) # [B,K,K]
        return x

class PointNetBase(nn.Module):

    def __init__(self, K=3):
        super().__init__()

        self.input_t = Tnet(K)
        self.embedding_t = Tnet(64)

        self.mlp1 = nn.Sequential(
            block(K, 64),
            block(64, 64))

        self.mlp2 = nn.Sequential(
			block(64, 64),
			block(64, 128),
            block(128, 1024))

    def forward(self, x):
        # [B,K,N]
        N = x.shape[-1]
        T1 = self.input_t(x) # [B,K,K]
        x = torch.bmm(T1, x) # [B,K,N]

        x = self.mlp1(x) # [B,64,N]
        T2 = self.embedding_t(x) # [B,64,64]
        local_embedding = torch.bmm(T2, x) # [B,64,N]

        global_feature = self.mlp2(local_embedding) # [B,1024,N]
        global_feature = F.max_pool1d(global_feature, N).squeeze(2) # [B,1024,1] -> [B,1024]

        return global_feature, local_embedding, T2

class PointNetBinSeg(nn.Module):

    def __init__(self, K=3):
        super().__init__()

        self.base = PointNetBase(K)

        # Binary segmentation
        self.binseg = nn.Sequential(
            block(1024+64, 512),
            block(512, 256),
            block(256, 128),
            block(128, 128),
            nn.Conv1d(128, 1, 1),
            #nn.Sigmoid() We do this at a later stage instead
        )

    def forward(self, x):
        # x -> [B,K,N]
        N = x.shape[-1]
        global_feature, local_embedding, T = self.base(x) # [B,1024], [B,64,N], [B,64,64]
        global_expand = global_feature.unsqueeze(-1).repeat(1,1,N) # [B,1024,N]
        cat_feat = torch.cat( [ local_embedding, global_expand ], dim=1)
        binseg = self.binseg(cat_feat).squeeze(1) # [B,1,N] -> [B,N]
        return { "x": x, "binseg": binseg, "Temb": T }

class FundamentalConsensus(nn.Module):

    def __init__(self):
        super().__init__()

        self.siamese_unsuperpoint = SiameseUnsuperPoint()
        self.pointnet_binseg = PointNetBinSeg(K=4)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.siamese_unsuperpoint.eval() # lol
            for param in self.siamese_unsuperpoint.parameters():
                param.requires_grad = False
    
    def forward(self, data):
        data = { **data, **self.siamese_unsuperpoint(data) }
        
        AP = data["A"]["P"] # [B,2,N]
        AF = data["A"]["F"] # [B,256,N]
        BP = data["B"]["P"]
        BF = data["B"]["F"]
        B,_,N = AP.shape

        ids, mask = brute_force_match(AF, BF) # [B,N], [B,N]
        Ap = batch_mask_lookup(AP, mask) # [[2,n] * B]
        Bp = batch_mask_lookup(batch_ids_lookup(BP, ids), mask) # [[2,n] * B]

        # temp
        data["Ap"] = Ap
        data["Bp"] = Bp

        L = torch.min(mask.sum(dim=1))

        x = torch.stack([ torch.cat([Ap[b], Bp[b]], dim=0)[:,torch.randperm(Ap[b].shape[1],device=Ap[b].device)][:,:L] for b in range(B) ], dim=0)

        data = { **data, **self.pointnet_binseg(x) }

        """
        X = [torch.cat([Ap[b], Bp[b]], dim=0).unsqueeze(0) for b in range(B)] # [[1,K=4,n] * B]

        data = { **data,
            "pointnet": [self.pointnet_binseg(X[b]) for b in range(B)] # [{"x", "w", "Temb"} * B]
        }
        """

        return data

class ConsensusLoss():

    def __init__(self, r, d):
        self.r = r
        self.d = d

    def vandermonde_matrix(self, u, v):
        pass

    def __call__(self, data):

        Ap, Bp = data["x"][:,:self.d], data["x"][:,self.d:]

        binseg_pred = data["binseg"]
        Temb = data["Temb"]

        # Soft count inliers
        inlier_tresh = 0.5
        #inlier_prob = 1.0 - torch.sigmoid(binseg_pred) # 1 - x
        #inlier_soft = torch.mean(torch.sigmoid(5*(inlier_prob - inlier_tresh)))
        inlier_prob = torch.sigmoid(binseg_pred)
        inlier_soft = torch.mean(inlier_prob)
        data["inlier_prob"] = inlier_prob

        # Add small random values to produce high vander loss in case of too few inliers
        weights = inlier_prob# + torch.rand_like(inlier_prob) * 1e-9

        # Normalize weights, sum to 1 across all points
        #weights = weights / torch.norm(weights, dim=1, keepdim=True)
        
        # Multiply weights by vandermonde matrix
        vandermonde = self.vandermonde_matrix(Ap, Bp)
        weighted_vandermonde = torch.diag_embed(weights) @ vandermonde
        data["WM"] = weighted_vandermonde
        
        # Extract nullspace from vandermonde matrix
        U,S,VT = torch.svd(weighted_vandermonde)
        V = VT.transpose(1,2)
        basis = V[:,:,V.shape[2]-self.r:]
        data["basis"] = basis

        # Singular values to be minimized
        #singular_lam = 1e4#0.003#1e5#0.15
        singular_lam = 1.0
        s = S[:,S.shape[1]-self.r:]
        data["S"] = S
        data["s"] = s

        # PointNet regularization 
        #reg_lam = 1.0
        reg_lam = 1e-3
        I = torch.eye(Temb.shape[1],dtype=torch.double,device=Temb.device)
        reg = Temb @ Temb.transpose(1,2)
        reg_loss = reg_lam * ((reg-I)**2).sum()

        # inlier lamdba
        inlier_lam = 10.0

        # Loss terms
        inlier_loss = -inlier_lam * inlier_soft
        singular_loss = singular_lam * s.mean()

        # Total loss
        loss_cons = inlier_loss + singular_loss + reg_loss

        print(inlier_loss, singular_loss, reg_loss)
        percent = 100.0*(inlier_prob[0] > 0.5).sum()/inlier_prob.shape[1]
        print(percent, inlier_prob.shape[1])

        return loss_cons, data

def fundamental_homography_vandermonde_matrix(u, v):
    # u/v --> [B,2,N]
    B,_,N = u.shape
    """
    M = torch.stack([
        u[:,0,:], 
        u[:,1,:], 
        v[:,0,:], 
        v[:,1,:], 
        u[:,0,:] * v[:,0,:],
        u[:,0,:] * v[:,1,:],
        u[:,1,:] * v[:,0,:],
        
        u[:,1,:] * v[:,1,:],
        torch.ones(B,N,device=u.device),

        #u[:,1,:],
        #v[:,1,:],
        
        
    ], dim=1).transpose(1,2)
    """
    M = torch.stack([
        u[:,0,:] * v[:,0,:], # u_x * v_x
        u[:,0,:] * v[:,1,:], # u_x * v_y
        u[:,0,:],            # u_x
        u[:,1,:] * v[:,0,:], # u_y * v_x
        u[:,1,:] * v[:,1,:], # u_y * v_y
        u[:,1,:],            # u_y
        v[:,0,:],            # v_x
        v[:,1,:],            # v_y
        torch.ones(B,N,device=u.device), # 1
    ], dim=1).transpose(1,2)
    # Normalize
    #M = (M/torch.norm(M, dim=2, keepdim=True).expand(-1,-1,9))
    return M


class FundamentalConsensusLoss(ConsensusLoss):

    def __init__(self):
        super().__init__(r=1, d=2)

    def vandermonde_matrix(self, u, v):
        return fundamental_homography_vandermonde_matrix(u, v)

    def __call__(self, data):
        loss_cons, data = super().__call__(data)
        basis = data["basis"]
        B = basis.shape[0]
        
        F = basis.reshape(B,3,3)
        _,S,_ = torch.svd(F)
        lam_f = 1e1
        loss_f = lam_f * S[:,2].mean() # Fundamental matrix regularization
        #print(loss_f)

        loss_total = loss_cons + loss_f

        return 0.01*loss_total, data

class HomographyConsensusLoss(ConsensusLoss):

    def __init__(self):
        super().__init__(r=3, d=2)

    def vandermonde_matrix(self, u, v):
        return fundamental_homography_vandermonde_matrix(u, v)

    def __call__(self, data):
        loss_cons, data = super().__call__(data)
        
        loss_total = loss_cons

        # # # # # # # # # # # # # # # #
        # Construct homography matrix #
        # # # # # # # # # # # # # # # #
        basis = data["basis"] # nullspace vectors 9x3
        # target basis:
        #      0 x x
        #      0 x x
        #      0 x x
        #------------------
        #      x 0 x
        #      x 0 x
        #      x 0 x
        #------------------
        #      x x x
        #      x x x
        #      x x x
        A1 = basis[:,0:3,:]
        A2 = basis[:,3:6,:]

        n1 = torch.svd(A1)[2].transpose(1, 2)[:,:,-1] # nullspace of A1
        n2 = torch.svd(A2)[2].transpose(1, 2)[:,:,-1] # nullspace of A2
        
        # Change of basis using nullspace of A1 and A2
        # will introduce zeroes like in the target basis
        x1 = (basis @ n1.unsqueeze(2)).squeeze(2)
        x2 = (basis @ n2.unsqueeze(2)).squeeze(2)
        
        # Scale equations such that h3 from x1 aligns with h3 from x2
        x1s = x1 / torch.norm(x1[:,3:6], dim=1, keepdim=True)
        x2s = x2 / torch.norm(x2[:,0:3], dim=1, keepdim=True)
        
        # Correct for sign (will be -1 if different, 1 if the same)
        different_signs = torch.sign(torch.sum(x2s[:,0:3], dim=1)) * torch.sign(torch.sum(x1s[:,3:6], dim=1))
        x2s = x2s * different_signs.unsqueeze(1)

        # Assemble homography
        h11 = x2s[:,6]
        h12 = x2s[:,7]
        h13 = x2s[:,8]

        h21 = x1s[:,6]
        h22 = x1s[:,7]
        h23 = x1s[:,8]

        h31 = -x1s[:,3]
        h32 = -x1s[:,4]
        h33 = -x1s[:,5]

        h1 = torch.stack((h11, h12, h13), dim=1)
        h2 = torch.stack((h21, h22, h23), dim=1)
        h3 = torch.stack((h31, h32, h33), dim=1)
        HH = torch.stack((h1, h2, h3), dim=1)
        H = HH / torch.norm(HH, dim=(1,2), keepdim=True)

        data["H_pred"] = H

        #return 0.01*loss_total, data
        return 1.0*loss_total, data



def main():
    data = { "x": torch.randn(4,4,150) }

    pointNetModel = PointNetBinSeg(K=data["x"].shape[1])
    pointNetModel.train()

    loss_fn = FundamentalConsensusLoss()

    data = pointNetModel.forward(data)
    loss, debug = loss_fn(data)
    
    print(data["w"].shape, data["Temb"].shape)
    print("bye!")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    main()