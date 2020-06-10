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
            nn.Sigmoid()
        )

    def forward(self, x):
        # x -> [B,K,N]
        N = x.shape[-1]
        global_feature, local_embedding, T = self.base(x) # [B,1024], [B,64,N], [B,64,64]
        global_expand = global_feature.unsqueeze(-1).repeat(1,1,N) # [B,1024,N]
        cat_feat = torch.cat( [ local_embedding, global_expand ], dim=1)
        w = self.binseg(cat_feat).squeeze(1) # [B,1,N] -> [B,N]
        return { "x": x, "w": w, "Temb": T }

class FundamentalConsensus(nn.Module):

    def __init__(self):
        super().__init__()

        self.siamese_unsuperpoint = SiameseUnsuperPoint()
        self.pointnet_binseg = PointNetBinSeg(K=4)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            print("F model train mode!!!! #####################3")
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
        w = data["w"]
        Temb = data["Temb"]
        
        W = torch.diag_embed(w)
        M = self.vandermonde_matrix(Ap, Bp)
        SVD = W@M
        U,S,VT = torch.svd(SVD)
        P = VT.transpose(1,2)[:,:,VT.shape[1]-self.r:]
        #P = VT[:,:,VT.shape[1]-self.r:]
        data["P"] = P

        s = S[:,S.shape[1]-self.r:]

        I = torch.eye(Temb.shape[1],dtype=torch.double,device=Temb.device)
        reg = Temb @ Temb.transpose(1,2)
        reg_loss = ((reg-I)**2).sum() # PointNet regularization

        lam = 1e5#1e5#0.15
        lam_r = 1e-1
        wl = -w.mean()
        sl = lam * s.mean()
        rl = lam_r * reg_loss
        loss_cons = wl + sl + rl # + -0.01*data["mask"].sum() nono
        print(wl, sl, rl)

        percent = 100.0*(w[0] > 0.9).sum()/w.shape[1]
        print(percent, w.shape[1])

        return loss_cons, data

class FundamentalConsensusLoss(ConsensusLoss):

    def __init__(self):
        super().__init__(r=1, d=2)

    def vandermonde_matrix(self, u, v):
        # u/v --> [B,2,N]
        B,_,N = u.shape
        M = torch.stack([
            u[:,0,:], 
            u[:,1,:], 
            v[:,0,:], 
            v[:,1,:], 
            u[:,0,:] * v[:,0,:],
            u[:,0,:] * v[:,1,:],
            u[:,1,:] * v[:,0,:],
            u[:,1,:],
            v[:,1,:],
            #torch.ones(B,N,device=u.device)
        ], dim=1).transpose(1,2)
        return M

    def __call__(self, data):
        loss_cons, data = super().__call__(data)
        P = data["P"]
        B = P.shape[0]
        #F = P.reshape(B,3,3)
        #_,S,_ = torch.svd(F)

        # Fundamental matrix regularization
        #lam_f = 1e2
        #fl = lam_f * S[:,2].mean()
        #loss_f =  lam_f * fl
        #print(fl)

        loss_total = loss_cons# + loss_f

        return loss_total, data

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