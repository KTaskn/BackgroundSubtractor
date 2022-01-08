import torch
from tqdm import tqdm
from dataset import DataSet
import argparse
from models import WrapperI3D, WrapperResNet
import pandas as pd

N_BATCHES = 5
N_WORKERS = 18

# Number of frames per feature
F = 16

def extract(dataset, net, n_batches=N_BATCHES, n_workers=N_WORKERS, cuda=False):
    net = net.cuda() if cuda else net
    net.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=n_batches,
        shuffle=False,
        num_workers=n_workers)
    
    outputs, outputs_label = [], []
    with torch.no_grad():
        with tqdm(total=len(loader), unit="batch") as pbar:
            for batches, b_labels in loader:
                # B x Cls x F x C x H x W if video
                # B x Cls x 1 x C x H x W if image
                batches = batches.cuda() if cuda else batches
                predicts = net(batches)
                predicts = predicts.cpu() if cuda else predicts
                outputs.append(predicts)
                outputs_label.append(b_labels)
                pbar.update(1)
    return torch.cat(outputs), torch.cat(outputs_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pathlist", help="Path to the dataset", type=str)
    parser.add_argument("output_path", help="Path to the output file", type=str)
    parser.add_argument("--video", action='store_true', help="Whether the dataset is video or image")
    parser.add_argument("--gpu", action='store_true', help="Use GPU")
    parser.add_argument("--resize", help="resize Image", type=float, default=1.0)
    parser.add_argument("--F", help="trajectories frame", type=int, default=F)
    
    args = parser.parse_args()
    print(f"pathlist: {args.pathlist}")
    print(f"output_path: {args.output_path}")
    print(f"video: {args.video}")
    print(f"gpu: {args.gpu}")
    print(f"resize: {args.resize}")
    print(f"F: {args.F}")
    
    # You can change the model here
    net = WrapperI3D() if args.video else WrapperResNet()
    
    # Get the image path and label from a input file
    with open(args.pathlist) as f:
        grp_path_and_label = [
            row.split(" ")
            for row in f.read().split("\n")
            if row
        ]
        df = pd.DataFrame({
            "grp": [int(grp) for grp, _, _ in grp_path_and_label],
            "path": [path for _, path, _ in grp_path_and_label],
            "label": [int(label) for _, _, label in grp_path_and_label],
        })
    
    outputs, labels = [], []
    for idx, df_grp in df.groupby('grp'):
        print(f"{idx}:")
        ds = DataSet(
            df_grp["path"].tolist(),
            df_grp["label"].tolist(),
            resize=args.resize,
            is_video=args.video,
            F=args.F,
            n_batches=N_BATCHES,
            n_workers=N_WORKERS)
        o, l = extract(ds, net, cuda=args.gpu)
        outputs.append(o)
        labels.append(l)
    
    outputs = torch.cat(outputs)
    labels = torch.cat(labels)
        
    print(f"features_size: {outputs.size()}")
    print(f"labels_size: {labels.size()}")
    
    torch.save({
        "features": outputs,
        "labels": labels
    }, args.output_path)