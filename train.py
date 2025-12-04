import sys
sys.path.append('/home/honsen/honsen/VO_SLAM/DROID-SLAM/droid_slam')

import cv2
import numpy as np
from collections import OrderedDict
import wandb
import torch
from torch.utils.data import DataLoader
from droid_slam.data_readers.factory import dataset_factory
from lietorch import SO3, SE3, Sim3
from droid_slam.geom import losses
from droid_slam.geom.graph_utils import build_frame_graph
# network
from droid_slam.droid_net import DroidNet
from droid_slam.logger import Logger
# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'

def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()


def report_loss(sumrot, sumtr, sumfe, sumpx, wandb_run, total_steps):

    train_loss_dict = {}

    train_loss_dict["train/f_error"] = sumfe
    train_loss_dict["train/1px"] = sumpx
    train_loss_dict["train/rot_error"] = sumrot
    train_loss_dict["train/tr_error"] = sumtr
    train_loss_dict['train/step'] = total_steps

    wandb_run.log(train_loss_dict)

def train(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    # setup_ddp(gpu, args)
    rng = np.random.default_rng(12345)

    N = args.n_frames

    model = DroidNet()
    model.cuda()

    model.train()

    droid_dict = torch.load("/home/honsen/honsen/VO_SLAM/DROID-SLAM/ckpt1/checkpoint_name_005000.pth")#/home/hyc/hyc_work/DVO/droid.pth

    # state_dict = OrderedDict([
    #    (k.replace("module.", ""), v) for (k, v) in droid_dict.items()])
    # state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
    # state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
    # state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
    # state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
    
    # for k in list(state_dict.keys()):
    #    if 'corr' in k:
    #        del state_dict[k]

    model.load_state_dict(droid_dict)
    model.cuda()
    model.train()
    model.fnet.requires_grad_(False)
    model.cnet.requires_grad_(False)
    # fetch dataloader
    db = dataset_factory(['tartan'], datapath=args.datapath, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=args.world_size, rank=gpu)

    train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=2)

    # fetch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False)
    # torch.autograd.set_detect_anomaly(True)

    for i in range(3000):
        optimizer.zero_grad()
        optimizer.step()
        scheduler.step()
    #
    # wandb_run = wandb.init(project="xxxx",
    #                        entity="xxxx",
    #                        group="xxxx",
    #                        name="xxxx",
    #                        )

    logger = Logger(args.name, scheduler)
    should_keep_training = True
    total_steps = 5000
    sumloss = 0
    sumrot = 0
    sumtr = 0
    sumfe = 0
    sumpx = 0
    sum1 = 0
    sum = 0
    while should_keep_training:
        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()

            images, poses, disps, intrinsics = [x.to('cuda') for x in item]

            # convert poses w2c -> c2w
            Ps = SE3(poses).inv()
            Gs = SE3.IdentityLike(Ps)

            # randomize frame graph
            if np.random.rand() < 0.5:#
                graph = build_frame_graph(poses, disps, intrinsics, num=args.edges)
            
            else:
                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]

            # fix first to camera poses
            Gs.data[:,0] = Ps.data[:,0].clone()
            Gs.data[:,1:] = Ps.data[:,[1]].clone()
            disp0 = torch.ones_like(disps[:,:,3::8,3::8])

            # perform random restarts
            r = 0
            while r < args.restart_prob:
                r = rng.random()
                
                intrinsics0 = intrinsics / 8.0
                poses_est, disps_est, residuals = model(Gs, images, disp0, intrinsics0,
                                                        graph, num_steps=args.iters, fixedp=2)

                geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False)
                res_loss, res_metrics = losses.residual_loss(residuals)
                flo_loss, flo_metrics = losses.flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph)

                loss = args.w1 * geo_loss + args.w2 * res_loss + args.w3 * flo_loss
                sumloss += loss
                sum += 1
                print(loss)
                loss.backward()

                Gs = poses_est[-1].detach()
                disp0 = disps_est[-1][:,:,3::8,3::8].detach()

            metrics = {}
            metrics.update(geo_metrics)
            metrics.update(res_metrics)
            metrics.update(flo_metrics)

            sumrot+=metrics['rot_error']
            sumtr+=metrics['tr_error']
            sumfe+=metrics['f_error']
            sumpx+=metrics['1px']
            sum1+=1
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_steps += 1
            if total_steps>0 and total_steps%100==0:
                # report_loss(sumrot/sum1, sumtr/sum1, sumfe/sum1, sumpx/sum1, wandb_run, total_steps)
                print(sumloss/sum)
                sum=0
                sum1=0
                sumrot =0
                sumtr =0
                sumfe =0
                sumpx =0
                sumloss=0

            if gpu == 0:
                logger.push(metrics)

            if total_steps % 1000 == 0 and gpu == 0:
                PATH = '/home/honsen/honsen/VO_SLAM/DROID-SLAM/ckpt1/%s_%06d.pth' % (args.name, total_steps)

                torch.save(model.state_dict(), PATH)

            if total_steps >= args.steps:
                should_keep_training = False
                break

    dist.destroy_process_group()
                

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='checkpoint_name', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    parser.add_argument('--datapath', default='/home/hyc/hyc_work/slam_Data/dataset', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=1)

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--steps', type=int, default=160000)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.01)
    parser.add_argument('--w3', type=float, default=0.05)
    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)

    args = parser.parse_args()

    args.world_size = args.gpus
    print(args)

    import os
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    args = parser.parse_args()
    args.world_size = args.gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    mp.spawn(train, nprocs=args.gpus, args=(args,))

