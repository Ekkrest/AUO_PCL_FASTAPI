import argparse
from argparse import Namespace
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss
import yaml
import zipfile
import json


import pcl.builder_adjust
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pcl.loader
import pcl.builder

import Res2netFFM  # USE FEATURE FUSION MODULE TO RESNET50
 
from customDataset import CustomDataset
from customDataset import SingleDataset

from train_pcl_api import _critical_section_updata_weight_list, _critical_section_weight_id

weight_dir = "./weights"
weight_info = "./weights/info.json"
weight_type = ('h5', 'ckpt', 'pth', 'pt', '.tar', '.pth.tar')
job_dir = "./jobs"
job_info = "./jobs/job.json"

# 將字典轉換為 Namespace
def dict_to_namespace(d):
    return Namespace(**d)

def main(j_id):
    # 讀取 YAML 配置文件
    with open('./config/config.yaml', 'r') as file:
        args = yaml.safe_load(file)
        args = dict_to_namespace(args)
        print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        print(args.world_size )

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    args.num_cluster = args.num_cluster.split(',')
    args.j_id = j_id
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, j_id=j_id)


def main_worker(gpu, ngpus_per_node, args, j_id=None):

    args.gpu = gpu
    job_id = args.j_id
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master    
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model Res2netFFM adding CCL")
    model = pcl.builder_adjust.MoCo(
        Res2netFFM.MocoV2_PCL, 
        args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.mlp,
        negative_samples_path= args.negative_samples_path)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        print(args.gpu)
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.Resize(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        
    # center-crop augmentation 
    eval_augmentation = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
        ])    

    defect_path = 'defect_img'
    mask_path = 'defect_mask'
    defect_data_path = os.path.join(args.data_path, defect_path)
    mask_data_path = os.path.join(args.data_path, mask_path)
       
    # train_dataset 與 eval_dataset
    train_dataset = CustomDataset(folder_path1= defect_data_path, 
                                  folder_path2= mask_data_path,
                                  transform=pcl.loader.TwoCropsTransform(transforms.Compose(augmentation), transforms.Compose(augmentation)))

    eval_dataset = SingleDataset(folder_path1= defect_data_path, 
                                  folder_path2= mask_data_path,
                                  transform=eval_augmentation) 
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)
    
    for epoch in range(args.start_epoch, args.epochs):
        
        cluster_result = None
        if epoch>=args.warmup_epoch:
            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args)         
            
            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
            for num_cluster in args.num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(len(eval_dataset),dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),args.low_dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda()) 

            if args.gpu == 0:
                features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
                features = features.numpy()
                cluster_result = run_kmeans(features,args)  #run kmeans clustering on master node
                # save the clustering result
                # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))  
                
            dist.barrier()  
            # broadcast clustering result
            for k, data_list in cluster_result.items():
                for data_tensor in data_list:                
                    dist.broadcast(data_tensor, 0, async_op=False)     
    
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, cluster_result)

        if (epoch%10 == 9) and (not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)) and epoch != args.epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch), j_id=job_id)
        elif (epoch%10 == 9) and (not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)) and epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=True, filename='best.pth.tar', j_id=job_id)

    # for watchdog
    status = dict()
    status['epoch'] = epoch
    status['status'] = "Finished"
    status['idle'] = True
    status['completed'] = True
    with open('./status.json', 'w') as f:
        json.dump(status, f)        



def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    infoNCE_losses = AverageMeter('InfoNCE Loss', ':.4e')
    protoNCE_losses = AverageMeter('ProtoNCE Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')   
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, infoNCE_losses, protoNCE_losses, losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
                
        # compute output
        output, target, output_proto, target_proto = model(im_q=images[0], im_k=images[1], cluster_result=cluster_result, index=index)
        
        # InfoNCE loss
        infoNCE_loss = criterion(output, target)  
        
        # ProtoNCE loss
        protoNCE_loss = 0
        if output_proto is not None:
            for proto_out,proto_target in zip(output_proto, target_proto):
                protoNCE_loss += criterion(proto_out, proto_target)  
                accp = accuracy(proto_out, proto_target)[0] 
                acc_proto.update(accp[0], images[0].size(0))
                
            # average loss across all sets of prototypes
            protoNCE_loss /= len(args.num_cluster) 
            protoNCE_losses.update(protoNCE_loss.item(), images[0].size(0))
            #loss += loss_proto   

        total_loss = infoNCE_loss + protoNCE_loss    

        infoNCE_losses.update(infoNCE_loss.item(), images[0].size(0))
        
        losses.update(total_loss.item(), images[0].size(0))
        acc = accuracy(output, target)[0] 
        acc_inst.update(acc[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),args.low_dim).cuda()
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images,is_eval=True) 
            features[index] = feat
    dist.barrier()        
    dist.all_reduce(features, op=dist.ReduceOp.SUM)     
    return features.cpu()

def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 5000
        clus.min_points_per_centroid = 1000

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu    
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        # concentration estimation (phi)        
        density = np.zeros(k)
        cluster_instance_number = []
        for i,dist in enumerate(Dcluster):
            cluster_instance_number.append(len(dist))
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
        print(cluster_instance_number)  
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = args.temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', j_id=None):
    # 保存 state 到 .pth.tar 文件
    # torch.save(state, filename)
    try:
        if j_id is None:
            raise ValueError("j_id is None")  

        if is_best == False: 
            weight_list_id, w = _critical_section_weight_id()
            w["name"] = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]
            w["info"] = f'training checkpoint'
            w_dir_path = os.path.join(weight_dir, str(w["weight_id"]))
            if not os.path.exists(w_dir_path):
                os.mkdir(w_dir_path)
            w["file_path"] = w_dir_path

            # 将保存的文件路径修改为新创建的目录下
            save_path = os.path.join(w_dir_path, filename)          
            # 保存 checkpoint 文件到新的路径
            torch.save(state, save_path)
            print(f"Checkpoint saved as {save_path}")

            for root, dirs, files in os.walk(w_dir_path):
                for f in files:
                    print(f)
                    if f.endswith(weight_type):
                        w["file_name"] = f

            _critical_section_updata_weight_list(weight_list_id, w)
            print('update checkpoint successful')
        elif is_best == True:
            filename = 'best.pth.tar'
            j_dir_path = os.path.join(job_dir, str(j_id))
            if not os.path.exists(j_dir_path):
                os.mkdir(j_dir_path)
            print('create j_path', j_dir_path)
            save_path = os.path.join(j_dir_path, filename)
            torch.save(state, save_path)
            print("Records save successful.")

            modelcfg_path = os.path.join("./config", "config.yaml")
            shutil.copy2(modelcfg_path, j_dir_path)
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")    


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Main PCL Training Script")
    parser.add_argument('--j_id', type=int, required=True, help="Job ID for this training session")
    temp = parser.parse_args()

    j_id = temp.j_id
    print('j_id: ', j_id)

    # for watchdog
    status = dict()
    status['status'] = "Training"
    status['idle'] = False
    status['completed'] = False
    with open('./status.json', 'w') as f:
        json.dump(status, f)

    main(j_id)
