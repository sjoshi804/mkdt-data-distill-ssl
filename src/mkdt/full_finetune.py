from tqdm import tqdm
from mkdt.utils import get_network
import torch
import torch.nn as nn
import torch.nn.functional as F


import random
import numpy as np

def ft_run(
    args, device, 
    init_model, 
    dl_tr, dl_te,
):  

    # set seed first 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    zca = args.zca_trans is not None

    """FULL FINETUNE"""

    # model
    model = get_network(args.train_model, args.channel, args.num_classes, args.test_img_shape, fix_net=True).to(device)
    if hasattr(init_model, "classifier"):
        del init_model.classifier
    model.load_state_dict(init_model.state_dict(), strict=False)
    backbone_param = [ p for n, p in model.named_parameters() if not "classifier" in n ]
    fc_param = [ p for n, p in model.named_parameters() if "classifier" in n ]

    # opt
    if args.test_opt == "sgd":
        opt = torch.optim.SGD([
            {"params": backbone_param, "lr": args.encoder_lr, "momentum": 0.9, "weight_decay": args.test_wd},
            {"params": fc_param, "lr": args.test_lr, "momentum": 0.9, "weight_decay": args.test_wd}
        ])
    elif args.test_opt == "adam":
        opt = torch.optim.AdamW([
            {"params": backbone_param, "lr": args.encoder_lr, "weight_decay": args.test_wd},
            {"params": fc_param, "lr": args.test_lr, "weight_decay": args.test_wd}
        ])
    else:
        raise NotImplementedError
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.test_epoch)

    print("Full Finetune")
    print("encoder_lr:", args.encoder_lr)
    print("classifer_lr:", args.test_lr)
    model.train()
    pbar = tqdm(range(args.test_epoch), desc="finetuning")
    for _ in pbar:
        epoch_loss = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            if zca:
                x = args.zca_trans(x)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sch.step()
        pbar.set_postfix_str(f"loss: {epoch_loss / len(dl_tr)}")
    
    model.eval()
    with torch.no_grad():        
        meta_loss, meta_acc, denominator = 0., 0., 0.
        for x, y in dl_te:
            x, y = x.to(device), y.to(device)
            if zca:
                x = args.zca_trans(x)
            l = model(x)
            meta_loss += F.cross_entropy(l, y, reduction="sum")
            meta_acc += torch.eq(l.argmax(dim=-1), y).float().sum()
            denominator += x.shape[0]
        meta_loss /= denominator; meta_acc /= (denominator/100.)

    del model

    return meta_loss.item(), meta_acc.item()