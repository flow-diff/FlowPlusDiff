from flows import AugmentedSequential 
import torch
from torch.amp import GradScaler
from data_factory.data_loader import get_loader_segment   
from tqdm import tqdm
import numpy as np
from torch.distributions import Normal
from evaluation.metrics import get_metrics
import json
import sys
import time
import csv
import os, random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cwd = os.getcwd()
print("Current directory:", cwd)

counter=0
train=False 

if len(sys.argv) > 1:
    datasetname = sys.argv[1].upper()
else:
    datasetname = ""
seedlist=[1000,1001,1002,1003,1004]

for z in range(5):
    set_seed(seedlist[z])
    seed = seedlist[z]
    data_path = cwd+"/dataset/"+datasetname
    with open(cwd+"/config.json", "r") as f:
        configs = json.load(f)
    params = configs[datasetname]
    batch_size = params["batch_size"]
    win_size = params["win_size"]
    d_model = params["d_model"]
    stride = params["stride"]
    n_flow_layers = params["n_flow_layers"]
    nhead = params["nhead"]
    flowattlayer = params["flowattlayer"]
    ratio = params["ratio"]
    diff_layers = params["diff_layers"]
    betamin = params["betamin"]
    betamax = params["betamax"]
    n_times = params["n_times"]
    infer_samples = params["infer_samples"]
    U_d = params["U_d"]
    diffst = params["diffst"]
    diffusion_conditioning = params["diffusion_conditioning"]
    only_transformer = params["only_transformer"]
    removed_binary = params["removed_binary"]
    score_type = params["score_types"]
    LMO=params["LMO"]

    savename=str(datasetname)+f"_seed_{seed}"+"_i"+"_d_"+str(d_model)+"_nfl_"+str(n_flow_layers)+"_ndl_"+str(diff_layers)+"_p_"+str(ratio)+"_ntimes_"+str(n_times)+"_bmin_"+str(betamin)+"_bmax_"+str(betamax)+"_nh_"+str(nhead)+"_fat_"+str(flowattlayer)+"_st_"+str(stride)+"_ud_"+str(U_d)+"_cond_"+str(diffusion_conditioning)+"_ot_"+str(only_transformer)+"_lmo_"+str(LMO)
    train_loader,val_loader,test_loader,Fea = get_loader_segment(data_path, batch_size=batch_size, win_size=win_size, step=stride,
                                                mode='train',
                                                dataset=datasetname,nratio=0.0,removed_binary=removed_binary)
    if LMO: 
        Fea = Fea*2

    model_cfg = dict(
        d_model=d_model,
        n_flow_layers=n_flow_layers,
        nhead=nhead,
        flowattlayer=flowattlayer,
        window=win_size,
        ratio=ratio,
        Fea=Fea,
    )
    diff_cfg = dict(
        diff_layers=diff_layers,
        betamin=betamin,
        betamax=betamax,
        n_times=n_times,
        infer_samples=infer_samples,
        U_d=U_d,
        only_transformer=only_transformer,
        diffst=diffst,
        diffusion_conditioning=diffusion_conditioning
    )
    model = AugmentedSequential(
        **model_cfg,
        **diff_cfg,
        device=device
    ).to(device)

    # Print model parameters
    denoiser_params = sum(p.numel() for p in model.denoiser.parameters() if p.requires_grad)
    transformer_params = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
    flow_params = sum(p.numel() for p in model.modeles.parameters() if p.requires_grad)
    log_var_params = model.log_var.numel()

    total_params = (
        denoiser_params
        + transformer_params
        + flow_params
        + log_var_params
    )

    print(f"Denoiser:    {denoiser_params:,}")
    print(f"Transformer: {transformer_params:,}")
    print(f"Flow:        {flow_params:,}")
    print(f"log_var:     {log_var_params:,}")
    print("-" * 30)
    print(f"Total:       {total_params:,}")
    opt_flow = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler("cuda")
    epochs = 10

    best_val_loss = float('inf')
    patience = 3

    if train:
        epoch_times = []
        for epoch in range(epochs):
            epoch_start_time = time.time()
            model.train()
            running_loss_flow = 0.0
            epoch_flow_loss = []
            epoch_flow_loss_val=[]

    
            for i, (batch, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                batch = batch.to(device)
                if LMO:
                    batchmean = batch.mean(dim=1, keepdim=True)
                    batch = batch - batchmean
                    batchmean = batchmean.repeat(1, win_size, 1)
                    batch = torch.cat((batch, batchmean), dim=-1)  # Concatenate along the feature dimension


                opt_flow.zero_grad()
                with torch.amp.autocast('cuda'):
                    z_0, logdet_tot = model.flow(batch)
                    diff_z_0, sigma = model.diffusion(z_0, batch)
                    dist = Normal(diff_z_0, sigma)
                    log_prob = dist.log_prob(z_0).sum(-1)
                    loss_flow = -(logdet_tot.squeeze(-1).float() + log_prob).mean()

                scaler.scale(loss_flow).backward(retain_graph=True)
                scaler.step(opt_flow)
                scaler.update()

                epoch_flow_loss.append(loss_flow.item())
                running_loss_flow += loss_flow.item()

            avg_loss_flow = running_loss_flow / len(train_loader)

            model.eval()
            val_loss_flow = 0.0


            with torch.inference_mode():
                for batch, _ in tqdm(val_loader):
                    batch = batch.to(device)
                    if LMO:
                        batchmean = batch.mean(dim=1, keepdim=True)
                        batch = batch - batchmean
                        batchmean = batchmean.repeat(1, win_size, 1)
                        batch = torch.cat((batch, batchmean), dim=-1)  # Concatenate along the feature dimension
                    z_0, logdet_tot = model.flow(batch)
                    diff_z_0, sigma = model.diffusion(z_0, batch)
                    dist = Normal(diff_z_0, sigma)
                    log_prob = dist.log_prob(z_0).sum(-1)
                    loss_flow_val = -(logdet_tot.squeeze(-1) + log_prob).mean()
                    val_loss_flow += loss_flow_val.item()
                    epoch_flow_loss_val.append(loss_flow_val.item())

            val_loss_flow /= len(val_loader)
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            if val_loss_flow < best_val_loss:
                best_val_loss = val_loss_flow
                os.makedirs("main_pth", exist_ok=True)
                torch.save(model.state_dict(), os.path.join("main_pth", savename+".pth"))
                print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_loss_flow:.4f}, val_loss={val_loss_flow:.4f}, time={epoch_time:.2f}s - Saved")
                counter = 0  # reset patience counter
            else:
                counter += 1
                print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_loss_flow:.4f}, val_loss={val_loss_flow:.4f}, time={epoch_time:.2f}s - No improvement ({counter}/{patience})")

            if counter >= patience:
                print("Early stopping triggered.")
                break
        
        # Print average epoch time
        avg_epoch_time = np.mean(epoch_times)
        print(f"\nAverage time per epoch: {avg_epoch_time:.2f}s")
        print(f"Total training time: {sum(epoch_times):.2f}s\n")

    model.load_state_dict(torch.load(os.path.join(cwd, "main_pth", savename+".pth"), weights_only=True))
    print("Model loaded successfully.")
    model.eval() 
    all_y_prob, test_labels,pred,true,predstd = [], [],[],[],[]
    z_flow_all, mu_all = [], []
    total_time = 0.0
    total_samples = 0

    with torch.inference_mode():
        for i, (x, labels) in enumerate(tqdm(test_loader)):
            B, L, Fea = x.shape
            total_samples += B

            x = x.to(device)
            true.append(x.clone().detach().cpu())
                        # Start timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            if LMO:
                xmean = x.mean(dim=1, keepdim=True)
                x = x - xmean
                xmean = xmean.repeat(1, win_size, 1)
                x=torch.cat((x, xmean), dim=-1)  # Concatenate along the feature dimension

            predout, feature_reconstd, y_prob, z_flow, mu = model.infer(
                x, score_type=score_type
            )

            # End timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()

            total_time += (end - start)
            
            feature_reconstd = feature_reconstd[:, :, :Fea]

            if LMO:
                predout = predout[:, :, :Fea] + xmean.to(device)
            avg_time=(end - start)
            print(f"Batch {i+1}: Inference time: {avg_time * 1000:.3f} ms, Batch size: {B}")
            labels = labels.reshape(-1)
            y_prob = y_prob.reshape(-1)

            all_y_prob.append(y_prob.detach().cpu())
            test_labels.append(labels.detach().cpu())
            pred.append(predout.detach().cpu())
            predstd.append(feature_reconstd.detach().cpu())
            z_flow_all.append(z_flow.detach().cpu())
            mu_all.append(mu.detach().cpu())

        avg_time_per_sample = total_time / total_samples
        throughput = total_samples / total_time
        print(f"Average inference time: {avg_time_per_sample * 1000:.3f} ms/sample")
        print(f"Throughput: {throughput:.2f} samples/s")
        all_y_prob = np.concatenate(all_y_prob, axis=0).reshape(-1)
        all_y_prob = np.array(all_y_prob)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        predstd = np.concatenate(predstd, axis=0).reshape(-1,Fea)
        predstd = np.array(predstd)
        true = np.concatenate(true, axis=0).reshape(-1,Fea)
        true = np.array(true)
        pred = np.concatenate(pred, axis=0).reshape(-1,Fea)
        pred = np.array(pred)
        z_flow_all = np.concatenate(z_flow_all, axis=0).reshape(-1, z_flow_all[0].shape[-1])
        z_flow_all = np.array(z_flow_all)
        mu_all = np.concatenate(mu_all, axis=0).reshape(-1, mu_all[0].shape[-1])
        mu_all = np.array(mu_all)
        error=(pred-true)**2
        error=error.mean(-1)
        weightstart = all_y_prob.shape[0]
        induced_values= np.zeros_like(all_y_prob) # Initialize output tensor
        windowroll=L
        full=all_y_prob.shape[0]


        if score_type=="l2_recon":
            all_scores=error
        elif score_type=="wo_avg":
            all_scores=all_y_prob
        else:
            for l in range(0,full):
                l_min = max(l - windowroll, 0)   # Ensure valid lower index
                l_max = min(l + windowroll, full) # Ensure valid upper index (exclusive)
                induced_values[l] = np.mean(all_y_prob[l_min:l_max])
            all_scores=induced_values
        gt = test_labels.astype(int)
    
        get_metrics_dict = get_metrics(all_scores, gt, pred=None, slidingWindow=windowroll)
        print(get_metrics_dict)

        out_dir = os.path.join(cwd, "main_pth")
        os.makedirs(out_dir, exist_ok=True)
        csv_file = os.path.join(out_dir, str(seed)+"metrics_by_dataset.csv")

        metrics_sorted = sorted(get_metrics_dict.keys())
        row = [datasetname]
        for metric in metrics_sorted:
            value = get_metrics_dict.get(metric, "")
            if isinstance(value, float):
                value = round(value, 3)
            row.append(value)

        write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
        with open(csv_file, "a", newline="") as wf:
            writer = csv.writer(wf)
            if write_header:
                writer.writerow(["Dataset"] + metrics_sorted)
            writer.writerow(row)
        print(f"Appended metrics to {csv_file}")


