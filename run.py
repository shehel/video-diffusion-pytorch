import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 2, 2),
)
im_size = 64
diffusion = GaussianDiffusion(
    model,
    image_size = im_size,
    num_frames = 6,
    timesteps = 100,   # number of steps
    loss_type = 'l2'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    '../NeurIPS2021-traffic4cast/data/raw/',                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = 4,
    train_lr = 2e-4,
    save_and_sample_every = 1000,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,
    im_size = 64# turn on mixed precision
)

trainer.load(18)
trainer.train()
