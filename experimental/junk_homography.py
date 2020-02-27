
def sample_homography(
    shape,
    shift=0,
    patch_ratio=1.0,
    perspective=True,
    perspective_amplitude_x=0.1,
    perspective_amplitude_y=0.1,
    scaling=True,
    n_scales=5,
    scaling_amplitude=0.1,
    translation=True,
    translation_overflow=0.0,
    rotation=True,
    max_angle=np.pi/2,
    n_angles=25,
    allow_artifacts=False,
):

    shape = np.array(shape)

    output_corners = np.stack([
        [0., 0.], 
        [0., 1.], 
        [1., 1.], 
        [1., 0.]], axis=0)

    H, W = shape
    patch_ratio = H / W

    margin = (1 - patch_ratio) / 2

    input_corners = margin + np.array([
        [0., 0.], 
        [0., patch_ratio], 
        [patch_ratio, patch_ratio],
        [patch_ratio, 0.]])

    std_trunc = 2

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        
        perspective_displacement = truncnorm(-std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)
        h_displacement_left      = truncnorm(-std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        h_displacement_right     = truncnorm(-std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        
        input_corners += np.array([
            [h_displacement_left,   perspective_displacement],
            [h_displacement_left,  -perspective_displacement],
            [h_displacement_right,  perspective_displacement],
            [h_displacement_right, -perspective_displacement],
        ]).squeeze()

    if scaling:
        scales = truncnorm(-std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(input_corners, axis=0, keepdims=True)
        scaled = (input_corners - center)[np.newaxis,:,:] * scales[:,np.newaxis,np.newaxis] + center
        if allow_artifacts:
            # All scales are valid except scale=1
            valid = np.arange(n_scales)
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[randint(valid.shape[0], size=1)].squeeze().astype(int)
        input_corners = scaled[idx,:,:]

    if translation:
        t_min = np.min(input_corners, axis=0)
        t_max = np.min(1 - input_corners, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        input_corners += np.array([
            uniform(-t_min[0], t_max[0], 1),
            uniform(-t_min[1], t_max[1], 1)]).T

    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)
        center = np.mean(input_corners, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack(
            [np.cos(angles), -np.sin(angles), 
             np.sin(angles),  np.cos(angles)], axis=1
            ), [-1, 2,2])
        rotated = np.matmul((input_corners - center)[np.newaxis,:,:], rot_mat) + center
        if allow_artifacts:
            # All rotations are valid except rotation=0
            valid = np.arange(n_angles)
        else:
            valid = (rotated >= 0) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[randint(valid.shape[0], size=1)].squeeze().astype(int)
        input_corners = rotated[idx,:,:]
    
    shape = shape[::-1] # [h,w] -> [w,h]
    output_corners *= shape[np.newaxis,:]
    input_corners *= shape[np.newaxis,:]

    homography = cv2.getPerspectiveTransform(
        np.float32(output_corners + shift),
        np.float32(input_corners  + shift))

    return torch.tensor(homography, dtype=torch.float32)

def sample_homographies(shape, **args):
    return torch.stack([ sample_homography(shape=shape[-2:], **args) for _ in range(shape[0]) ])
