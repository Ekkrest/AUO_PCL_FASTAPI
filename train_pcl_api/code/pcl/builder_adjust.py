import torch
import torch.nn as nn
from random import sample
from torchvision import transforms
import os
from PIL import Image

#import Res2netFFM_2layer
import Res2netFFM

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, r=16384, m=0.999, T=0.1, mlp=False, negative_samples_path=None):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T
        self.negative_samples_path = negative_samples_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(model=Res2netFFM.Res2Net(block=Res2netFFM.Bottle2neck, layers=[3, 4, 6, 3], num_classes=2048), head=Res2netFFM.ContrastiveHead(2048))
        self.encoder_k = base_encoder(model=Res2netFFM.Res2Net(block=Res2netFFM.Bottle2neck, layers=[3, 4, 6, 3], num_classes=2048), head=Res2netFFM.ContrastiveHead(2048))

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.encoder.fc.weight.shape[1]
            self.encoder_q.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.encoder.fc)
            self.encoder_k.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.encoder.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.negative_samples = self._load_initial_negative_samples().to(self.device)
        self._initialize_queue()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _load_initial_negative_samples(self):
        # Load initial 2048 negative samples from the specified folder.
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        origin_path = os.path.join(self.negative_samples_path, 'non_defect_img')
        position_path = os.path.join(self.negative_samples_path, 'non_defect_mask')

        file_names = sorted(os.listdir(origin_path))[:256]
        negative_samples = []
        for file_name in file_names:
            origin_file_path = os.path.join(origin_path, file_name)
            position_file_path = os.path.join(position_path, file_name)

            origin_image = Image.open(origin_file_path).convert('RGB')
            position_image = Image.open(position_file_path).convert('RGB')

            origin_image = transform(origin_image)
            position_image = transform(position_image)

            combined_image = torch.cat((origin_image, position_image), dim=0)
            negative_samples.append(combined_image)

        self.current_index = 256
        print('256 initial images loaded.')
        negative_samples = torch.stack(negative_samples)
        return negative_samples   

    @torch.no_grad()
    def _load_next_batch(self, batch_size):
        # Load next batch of negative samples from the specified folder.
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        origin_path = os.path.join(self.negative_samples_path, 'non_defect_img')
        position_path = os.path.join(self.negative_samples_path, 'non_defect_mask')

        negative_samples = []
        while len(negative_samples) < batch_size:
            end_index = self.current_index + batch_size - len(negative_samples)
            file_names = sorted(os.listdir(origin_path))[self.current_index:end_index]
            if len(file_names) == 0:  # 如果没有文件了，重新开始
                self.current_index = 0
                continue

            for file_name in file_names:
                origin_file_path = os.path.join(origin_path, file_name)
                position_file_path = os.path.join(position_path, file_name)

                origin_image = Image.open(origin_file_path).convert('RGB')
                position_image = Image.open(position_file_path).convert('RGB')

                origin_image = transform(origin_image)
                position_image = transform(position_image)

                combined_image = torch.cat((origin_image, position_image), dim=0)
                negative_samples.append(combined_image)

            self.current_index += len(file_names)
            if self.current_index >= len(os.listdir(origin_path)):
                self.current_index = 0  # 重新开始

        #print(f'Next {batch_size} images loaded.')
        negative_samples = torch.stack(negative_samples)

        return negative_samples     

    @torch.no_grad()
    def _initialize_queue(self):
        print("Initializing queue...")
        self.encoder_k = self.encoder_k.to(self.device)
        k = self.encoder_k(self.negative_samples.to(self.device))
        k = nn.functional.normalize(k, dim=1)
        self.queue[:, :k.size(0)] = k.t()
        print(f"Queue initialized with shape {self.queue.shape}.")


    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        #print("Dequeue and enqueue...")
        batch_size = 32

        # Load next batch of negative samples
        k_batch = self._load_next_batch(batch_size).to(self.device)
        k = self.encoder_k(k_batch)
        k = nn.functional.normalize(k, dim=1)

        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0


        if k.size(0) != batch_size:
            print(f"Error: Expected k.size(0) to be {batch_size}, but got {k.size(0)}")
            return

        # Dequeue and enqueue
        self.queue[:, -batch_size:] = k.t()  # Add new samples to the front
        self.queue = torch.cat((k.t(), self.queue[:, :-batch_size]), dim=1)  # Remove old samples from the end

        ptr = (ptr + batch_size) % self.r
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k=None, is_eval=False, cluster_result=None, index=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        
        if is_eval:
            k = self.encoder_k(im_q)  
            k = nn.functional.normalize(k, dim=1)            
            return k
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        
        # dequeue and enqueue
        self._dequeue_and_enqueue()
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # prototypical contrast
        if cluster_result is not None:  
            proto_labels = []
            proto_logits = []
            for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
                # get positive prototypes
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]    
                
                # sample negative prototypes
                #all_proto_id = [i for i in range(im2cluster.max()+1)]       
                #neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
                #print(f'all_proto_id: {all_proto_id}')
                #print(f'pos_proto_id: {pos_proto_id}')
                #print(f'neg_proto_id: {neg_proto_id}')
                #neg_proto_id = list(neg_proto_id)  # 将集合转换为列表
                #neg_proto_id = sample(neg_proto_id,self.r) #sample r negative prototypes 
                #neg_prototypes = prototypes[neg_proto_id]    

                #proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
                
                # compute prototypical, q與cluster center logits 內積
                all_sim = torch.mm(q,prototypes.t())
                # 挑选出每个 q 对应的聚类中心的内积值
                pos_logits = all_sim[torch.arange(all_sim.size(0)), pos_proto_id]

                # 创建一个新的 logits_proto 张量
                logits_proto = torch.zeros_like(all_sim)
                logits_proto[:, 0] = pos_logits  # 将挑选出的内积值移动到 index=0 的位置

                # 填充剩余的 logits_proto
                mask = torch.ones_like(all_sim).scatter_(1, pos_proto_id.unsqueeze(1), 0)
                neg_logits = all_sim[mask.bool()].view(all_sim.size(0), -1)
                logits_proto[:, 1:] = neg_logits
                
                # targets for prototype assignment
                labels_proto = torch.zeros(q.size(0)).long().cuda()
                
                # scaling temperatures for the selected prototypes
                temp_proto = density
                all_sim /= temp_proto
                
                proto_labels.append(labels_proto)
                proto_logits.append(all_sim)
            return logits, labels, proto_logits, proto_labels
        else:
            return logits, labels, None, None


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
