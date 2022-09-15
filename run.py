import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

from clearml import Task

task = Task.init(project_name="t4c_gen", task_name="Test video diffusion from load")
logger = task.get_logger()
args = {
    'im_size': 128,
    'batch_size': 1,
    'train_lr': 1e-4,
    'save_sample_every': 1,
    'train_steps': 700000,
    'num_workers': 0,
    'data': '../NeurIPS2021-traffic4cast/data/raw/',
    'channels': 1,
    'num_frames': 6,
    'timesteps': 100,
    'loss_type': 'l2',
    'amp': True,
    'load_model': 35,
    'dim': 64,
    'dim_mults': (1,2,4,8),
    'cond': True,
    'grad_accum': 2,
    't_start': 6,
    't_end': 12,
    'ch_start': 1,
    'ch_end': 2,
    'in_frames': None,
    'out_frames': None,
    'file_filter': None
    }

task.connect(args)
print ('Arguments: {}'.format(args))

model = Unet3D(
    dim = args['dim'],
    dim_mults = args['dim_mults'],
    ch_start = args['ch_start'],
    ch_end = args['ch_end'],
    t_start = args['t_start'],
    t_end = args['t_end'],
    channels = args['channels']

)

model = torch.nn.DataParallel(model)
diffusion = GaussianDiffusion(
    model,
    image_size = args['im_size'],
    num_frames = args['num_frames'],
    timesteps = args['timesteps'],   # number of steps
    loss_type = args['loss_type'],    # L1 or L2
    channels = args['channels'],
    ).cuda()

trainer = Trainer(
    diffusion,
    args['data'],                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = args['batch_size'],
    train_lr = args['train_lr'],
    save_and_sample_every = args['save_sample_every'],
    train_num_steps = args['train_steps'],         # total training steps
    gradient_accumulate_every = args['grad_accum'],    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = args['amp'],
    im_size = args['im_size'],
    cond = args['cond'],
    in_frames = args['in_frames'],
    out_frames = args['out_frames'],
    ch_start = args['ch_start'],
    ch_end = args['ch_end'],
    file_filter = args['file_filter']
)

if args['load_model'] is not None:
    trainer.load(args['load_model'])

#trainer.train(logger=logger)
trainer.infer(logger=logger, milestone=args['load_model'])
