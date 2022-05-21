import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

from clearml import Task

task = Task.init(project_name="t4c_gen", task_name="Train diffusion")
logger = task.get_logger()
args = {
    'im_size': 128,
    'batch_size': 4,
    'train_lr': 4,
    'save_sample_every': 1000,
    'train_steps': 700000,
    'num_workers': 2,
    'data': '../NeurIPS2021-traffic4cast/data/raw/',
    'num_channels': 8,
    'num_frames': 6,
    'timesteps': 1000,
    'loss_type': 'l2',
    'amp': False,
    'load_model': None,
    'dim': 64,
    'dim_mults': (1,2,2,2),
    'cond': True
    }

task.connect(args)
print ('Arguments: {}'.format(args))

model = Unet3D(
    dim = args['dim'],
    dim_mults = args['dim_mults'],
)
diffusion = GaussianDiffusion(
    model,
    image_size = args['im_size'],
    num_frames = args['num_frames'],
    timesteps = args['timesteps'],   # number of steps
    loss_type = args['loss_type']    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    args['data'],                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = args['batch_size'],
    train_lr = args['train_lr'],
    save_and_sample_every = args['save_sample_every'],
    train_num_steps = args['train_steps'],         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = args['amp'],
    im_size = args['im_size'],
    cond = args['cond']
)

if args['load_model'] is not None:
    trainer.load(args['load_model'])

trainer.train(logger=logger)
