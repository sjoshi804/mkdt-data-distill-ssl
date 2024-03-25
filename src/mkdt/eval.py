from mkdt.full_finetune import ft_run
from mkdt.linear_evaluation import le_run
from mkdt.pretrain import pretrain_run
from mkdt.utils import get_dataset, get_network, SoftLabelDataset, ParamDiffAug
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid
import argparse
import numpy as np
import os
import pickle
import random
import torch
import wandb

def aggregate_results(results):
    aggregated_results = {}
    for dataset in results[0].keys():
        dataset_results = np.array([result[dataset] for result in results])
        mean = np.mean(dataset_results, axis=0)
        std = np.std(dataset_results, axis=0)
        aggregated_results[dataset] = {'mean': mean.tolist(), 'std': std.tolist()}
    return aggregated_results

def main(args):
    target_datasets = ["tiny", "cifar100", "CIFAR10", "aircraft", "cub2011", "dogs", "flowers"]

    # Wandb init
    wandb.init(
        project="data_distillation_evaluation",
        config=args
    )

    if args.result_dir is not None:
        path = os.path.basename(args.result_dir.replace('/', '_')) 
    else:
        path = f"train_{args.train_dataset}_{os.path.basename(args.subset_path) if args.subset_path is not None else None}_{args.num_subset_within}"

    print(f"Evaluate for {path}")

    final_res = {}

    for td in target_datasets:
        args.test_dataset = td
    
        res_list = []
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)

        # default augment
        args.dsa_param = ParamDiffAug()
        args.dsa_strategy = 'color_crop_cutout_flip_scale_rotate' 

        # seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # data
        _, train_img_size,_, dst_train_original, _, args.zca_trans = get_dataset(args.train_dataset, args.data_path, args.batch_size, args.subset, args=args)     
        args.channel, im_size, num_classes, dst_train, dst_test, _ = get_dataset(args.test_dataset, args.data_path, args.batch_size, args.subset, subset_size=args.subset_frac, args=args)
        # Used for test (le and ft)
        dl_tr = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size, shuffle=False if args.test_algorithm == "linear_evaluation" else True, num_workers=4, pin_memory=True)
        dl_te = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        print("Test training data: ", len(dst_train))

        labels_all = torch.load(args.label_path, map_location="cpu")
        
        args.train_img_shape = train_img_size
        args.test_img_shape = im_size
        args.num_classes = num_classes

        if args.result_dir is None:
            if args.subset_path is not None:
                if args.num_subset_within is not None:
                    with open(args.subset_path, "rb") as f:
                        subset_idx = pickle.load(f)[0:args.num_subset_within]
                else:
                    with open(args.subset_path, "rb") as f:
                        subset_idx = pickle.load(f)
            
            else:
                subset_idx = None
                
            dst_train_initial = SoftLabelDataset(dst_train_original, labels_all, subset_idx)
            print(f"num_images: {len(dst_train_initial)}")
            dl_syn = torch.utils.data.DataLoader(dst_train_initial, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
        else:
            if args.use_krrst:
                x_syn = torch.load(f"{args.result_dir}/x_syn_final.pt") 
            else:
                x_syn = torch.load(f"{args.result_dir}/images_{args.distilled_steps}.pt")
                # x_syn = torch.load(f"{args.result_dir}")
            try:
                syn_lr = torch.load(f"{args.result_dir}/synlr_{args.distilled_steps}.pt") 
                args.pre_lr = syn_lr 
            except:
                print("No syn lr found, fall back to original pre_lr")
            try:
                if args.use_krrst:
                    y_syn = torch.load(f"{args.result_dir}/y_syn_final.pt")
                else:
                    y_syn = torch.load(f"{args.result_dir}/labels_{args.distilled_steps}.pt")
            except:
                # target model 
                from mkdt.teacher_repr.bt_model import BTModel
                teacher_model = BTModel(feature_dim=1024, dataset=args.train_dataset)
                teacher_model = teacher_model.to(device)        
                teacher_model.load_state_dict(torch.load(f"/home/jennyni/krrst_orig/saved_bt_models/barlow_twins_resnet18_{args.train_dataset}.pth", map_location="cpu"), strict=False)
    

                teacher_model.eval()
                with torch.no_grad():
                    x_syn = x_syn.to(device)
                    y_syn = teacher_model(x_syn)
                    y_syn = y_syn.detach().clone()
            
            assert x_syn.requires_grad == False and y_syn.requires_grad == False
            assert x_syn.grad is None and y_syn.grad is None
            assert  y_syn.shape[-1] == labels_all.shape[1]

            x_syn, y_syn = x_syn.detach(), y_syn.detach()
            x_syn, y_syn = x_syn.to(device), y_syn.to(device)

            dst_train_final = TensorDataset(x_syn, y_syn)
            print(f"num_images: {len(dst_train_final)}")
            dl_syn = DataLoader(dst_train_final, batch_size=args.batch_size, shuffle=True, num_workers=0)

            # visualize
            grid = make_grid(x_syn.clone(), nrow=100)
            wandb.log({"final images": wandb.Image(grid.detach().cpu())})


        args.num_target_features = labels_all.shape[1]
        print("teacher representations dim:", args.num_target_features)
        print("pretraining lr:", args.pre_lr)

        ckpt_dir = "ckpt_dir"
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # check random encoder perf
        random_model =get_network(args.train_model, args.channel, args.num_target_features, args.train_img_shape, fix_net=True).to(device)
        if args.test_algorithm == "linear_evaluation":
            rd_loss, rd_acc = le_run(args, device, random_model, dl_tr, dl_te)
        else:
            rd_loss, rd_acc = ft_run(args, device, random_model, dl_tr, dl_te)
        print("rd_loss", rd_loss)
        print("rd_acc", rd_acc)
        wandb.log({f"{td}_random_loss": rd_loss, f"{td}_random_accuracy": rd_acc})
        del random_model
        res_list.append(rd_acc)

        # final performance
        ckpt_file_final = f"{ckpt_dir}/{path}_pre_epoch_{args.pre_epoch}_{args.seed}_{args.train_dataset}_{args.train_model}_{args.distilled_steps}_{args.pre_lr}_final_full.pt"
        if os.path.exists(ckpt_file_final):
            print(f"find {ckpt_file_final}")
            final_model = get_network(args.train_model, args.channel, args.num_target_features, args.train_img_shape, fix_net=True).to(device)
            final_model.load_state_dict(torch.load(ckpt_file_final, map_location=device))
        else:
            final_model = pretrain_run(args, device, dl_syn)
            torch.save(final_model.state_dict(), ckpt_file_final)
        if args.test_algorithm == "linear_evaluation":
            final_loss, final_acc = le_run(args, device, final_model, dl_tr, dl_te)
        else:
            final_loss, final_acc = ft_run(args, device, final_model, dl_tr, dl_te)
        del final_model
        print("final_loss", final_loss)
        print("final_acc", final_acc)
        wandb.log({f"{td}_final_loss": final_loss, f"{td}_subset_final_accuracy": final_acc})
        res_list.append(final_acc)
    
        final_res[td] = res_list

    print(final_res)

    wandb.log(final_res)

    wandb.finish(quiet=True)

    return final_res
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=0)

    # data
    parser.add_argument('--data_path', type=str, default="/data")
    parser.add_argument('--train_dataset', type=str, default="cifar100")
    parser.add_argument('--test_dataset', type=str, default="cifar100")
    parser.add_argument('--num_workers', type=int, default=0)

    # label
    parser.add_argument('--label_path', type=str)

    # subset
    parser.add_argument('--subset_path', type=str, default=None)
    parser.add_argument('--num_subset_within', type=int, default=None)

    # original
    parser.add_argument('--use_flip', action="store_true")
    parser.add_argument('--use_diff_aug', action="store_true")
    parser.add_argument('--zca', action="store_true")
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')

    
    # hparams for model
    parser.add_argument('--train_model', type=str, default="ConvNet")


    # hparms for pretrain
    parser.add_argument('--pre_opt', type=str, default="sgd") 
    parser.add_argument('--pre_epoch', type=int, default=20) 
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pre_lr', type=float, default=0.1) # 0.174
    parser.add_argument('--pre_wd', type=float, default=1e-4)
    parser.add_argument('--pre_sch', action="store_true")

    # hparms for test
    parser.add_argument('--test_algorithm', type=str, choices=['linear_evaluation', 'full_finetune'], default="linear_evaluation")
    parser.add_argument('--test_opt', type=str, default="sgd")
    parser.add_argument('--test_epoch', type=int, default=50) 
    parser.add_argument('--test_lr', type=float, default=0.05)
    parser.add_argument('--test_wd', type=float, default=0.0)
    parser.add_argument('--subset_frac', type=float, default=None)

    # gpus
    parser.add_argument('--gpu_id', type=int, default=0)

    # for distilled set directory
    parser.add_argument('--result_dir', type=str, default=None)

    parser.add_argument('--distilled_steps', type=int, default=1000) 
    parser.add_argument('--use_krrst', action="store_true")

    # args = parser.parse_args()

    seeds = [0, 1, 2]
    results = []
    
    args = parser.parse_args()
    for seed in seeds:
        args.seed = seed
        results.append(main(args))

    final_aggregated_results = aggregate_results(results)
    print(final_aggregated_results)

    # wandb.log(final_aggregated_results)
    wandb.finish(quiet=True)