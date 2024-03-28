from mkdt.utils import get_dataset, get_network, get_daparam, TensorDataset, epoch, ParamDiffAug
from torch.utils.data import Subset
from tqdm import tqdm
import argparse
import copy
import os
import pickle 
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    args.dsa = True if args.dsa == 'True' else False
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    print(args.run_id)
    
    channel, im_size, _, dst_train, _, args.zca_trans = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args, set_seed=False)
    
    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["cifar10", "cifar100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir += f"_{args.run_id}"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    ''' organize the real dataset '''
    images_all = []
    
    print("Processing Trainset")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))

    images_all = torch.cat(images_all, dim=0).to("cpu")

    result_dir = "target_rep"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    labels_all = torch.load(args.train_labels_path, map_location="cpu").to("cpu")
    print("train label shape", labels_all.shape)
    
    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    
    # Subset using Distill idx if specified 
    if args.distill_idx is not None:
        with open(args.distill_idx, "rb") as f:
            distill_idx = pickle.load(f)
            dst_train = Subset(dst_train, indices=distill_idx)

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=4, pin_memory=True)

    print("Dataset creation complete")
    
    criterion = None
    if args.criterion == "mse":
        criterion = nn.MSELoss().to(args.device)
    elif args.criterion == "ce":
        criterion = nn.CrossEntropyLoss().to(args.device)
    trajectories = []

    ''' set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # for whole-dataset training
    print('DC augmentation parameters: \n', args.dc_aug_param)

    for it in range(0, args.num_experts):

        # Sample model 
        teacher_net = get_network(args.model, channel, labels_all.shape[1], im_size).to(args.device) # get a random model

        if it == 0:
            print(teacher_net)
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=args.lr_teacher, momentum=args.mom, weight_decay=args.l2)  # optimizer_img for synthetic data
        teacher_optim.zero_grad()

        timestamps = []
        
        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        # Train on real data to get trajectory
        for e in range(args.train_epochs):

            train_loss = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, args=args, aug=True if args.enable_aug else False)

            print(f"Itr: {it} \tEpoch: {e} \tTrain Loss: {train_loss}")

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        trajectories.append(timestamps)
        
        print("Saving trajectory checkpoint")
        n = 0
        while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
            n += 1
        print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
        torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
        trajectories = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=1.0, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='False', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='/data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers_figure2', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=1)
    
    # new params
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--train_labels_path', type=str)
    parser.add_argument('--criterion', type=str, default="mse")
    parser.add_argument('--device', type=int, default=0, help='gpu number')
    parser.add_argument('--distill_idx', type=str, default=None, help='path to indices to distill')
    parser.add_argument('--enable_aug', action='store_true')
    
    args = parser.parse_args()
    main(args)


