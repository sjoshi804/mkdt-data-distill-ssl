from tqdm import trange
from utils import get_network
import numpy as np
import random
import torch
import torch.nn as nn


def pretrain_run(
    args, device, 
    dl_syn,
):  

    # set seed first 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    model = get_network(args.train_model, args.channel, args.num_target_features, args.train_img_shape).to(device)
    if args.pre_opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.pre_lr, momentum=0.9, weight_decay=args.pre_wd)
    elif args.pre_opt == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.pre_lr, weight_decay=args.pre_wd)
    else:
        raise NotImplementedError

    criterion = nn.MSELoss()

   
    print("Pretrain")
    model.train()

    pbar = trange(args.pre_epoch, desc="pre-training")
    for epoch in pbar:
        loss_avg, num_exp = 0, 0
        for datum in dl_syn:
            img = datum[0].float().to(device)
            lab = datum[1].float().to(device)
            #lab = datum[1].to(device)
            #indices = indices.to(device)
            
            n_b = img.shape[0]

            output = model(img)
            
            loss = criterion(output, lab)
                    
            loss_avg += loss.item()*n_b
            num_exp += n_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_avg /= num_exp
        pbar.set_postfix_str(f"loss: {loss_avg}")
        # wandb.log({"Loss": loss.detach().cpu(),
        #            "Iteration": epoch})
        print(loss_avg)
    

    return model