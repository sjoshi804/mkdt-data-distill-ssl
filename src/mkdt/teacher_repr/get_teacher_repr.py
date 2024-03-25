from mkdt.teacher_repr.bt_model import BTModel
from mkdt.utils import get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import torch
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    _, _, _, dst_train, _, _ = get_dataset(args.dataset, args.data_path, args.batch_size, args=args, set_seed=False)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    teacher_model = BTModel(feature_dim=1024, dataset=args.dataset, arch=args.arch)
    teacher_model = teacher_model.to(args.device)  
    teacher_model.load_state_dict(torch.load(args.model_path, map_location="cpu"), strict=False)
    teacher_model.eval()
    
    train_repr = []
    trainloader = DataLoader(dataset=dst_train, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for X, _ in tqdm(trainloader, desc="Encoding"):
            X = X.to(args.device)
            image_repr = teacher_model(X)
            train_repr.append(image_repr)       
    train_repr = torch.cat(train_repr, dim=0)
    torch.save(train_repr.detach().cpu(), os.path.join(args.save_dir, f"{args.dataset}_{args.run_prefix}_teacher_rep_train.pt"))
    
    print("train repr shape", train_repr.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teacher Representations')
    parser.add_argument('--dataset', type=str, help='dataset name', choices=["cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument('--save_dir', type=str, help="directory to save representations", default="/home/sjoshi/mkdt-data-distill-ssl/src/mkdt/teacher_repr/saved/")
    parser.add_argument('--data_path', type=str, help='path to dataset parent folder', default="/data")
    parser.add_argument('--batch_size', type=int, help='batch size', default=128)
    parser.add_argument('--arch', type=str, help='architecture of model', default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument('--model_path', type=str, help='path to local copy of model provided by mixbt repository')
    parser.add_argument('--run_prefix', type=str, default="test", help="prefix for .pt file storing training data representations")
    parser.add_argument('--device', type=int, default=0, help='gpu number')
    
    args = parser.parse_args()
    main(args)



