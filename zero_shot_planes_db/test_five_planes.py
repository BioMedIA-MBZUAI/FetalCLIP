import os
import json

import torch
import numpy as np
import pandas as pd
import open_clip
import json

from PIL import Image

from tqdm import tqdm
from torchmetrics import Accuracy, F1Score

# Constants and Configuration
DIR_IMAGES = "<path_to_fetalplanesdb_data>/Images"
PATH_CSV = "<path_to_fetalplanesdb_data>/FETAL_PLANES_DB_data.csv"
PATH_FETALCLIP_WEIGHT = "../FetalCLIP_weights.pt"
PATH_FETALCLIP_CONFIG = "../FetalCLIP_config.json"
PATH_TEXT_PROMPTS = "test_five_planes_prompts.json"
BATCH_SIZE = 16
NUM_WORKERS = 4

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load model configuration
with open(PATH_FETALCLIP_CONFIG, "r") as file:
    config_fetalclip = json.load(file)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

class DatasetFetalPlanesDB(torch.utils.data.Dataset):
    def __init__(
            self, dir_images, path_csv, preprocess, split='all',
            exclude_planes=['Other'],
        ):
        self.root = dir_images
        self.preprocess = preprocess
        
        df = pd.read_csv(path_csv, sep=';')
        if split == 'train':
            df = df[df['Train '] == 1]
        elif split == 'test':
            df = df[df['Train '] == 0]
        
        for plane in exclude_planes:
            df = df[df['Plane'] != plane]
        
        self.data = []
        for _, row in df.iterrows():
            self.data.append({
                'img': os.path.join(self.root, f"{row['Image_name']}.png"),
                'plane': row['Plane'],
                'pid': row['Patient_num'],
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = Image.open(self.data[index]['img'])
        img = make_image_square_with_zero_padding(img)

        img = self.preprocess(img)
        plane = self.data[index]['plane']
        
        return img, plane, self.data[index]['img']

def make_image_square_with_zero_padding(image):
    width, height = image.size

    # Determine the size of the square
    max_side = max(width, height)

    # Create a new square image with black padding (0 for black in RGB or L modes)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Create a new square image with black padding (0 for black in RGB or L modes)
    if image.mode == "RGB":
        padding_color = (0, 0, 0)  # Black for RGB images
    elif image.mode == "L":
        padding_color = 0  # Black for grayscale images

    # Create a new square image
    new_image = Image.new(image.mode, (max_side, max_side), padding_color)

    # Calculate padding
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2

    # Paste the original image in the center of the new square image
    new_image.paste(image, (padding_left, padding_top))

    return new_image

def main(checkpoint):
    plane_db_to_planename = {
        'Fetal brain': 'brain',
        'Fetal abdomen': 'abdomen',
        'Fetal thorax': 'heart',
        'Fetal femur': 'femur',
        'Maternal cervix': 'cervix',
    }

    with open(PATH_TEXT_PROMPTS, 'r') as json_file:
        text_prompts = json.load(json_file)

    planename_to_index = {key: i for i, key in enumerate(text_prompts.keys())}
    index_to_planename = {val: key for key, val in planename_to_index.items()}

    model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=checkpoint)
    tokenizer = open_clip.get_tokenizer("FetalCLIP")

    ds = DatasetFetalPlanesDB(DIR_IMAGES, PATH_CSV, preprocess, split='all', exclude_planes=['Other'])
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model.eval()

    model.to(device)

    list_text_features = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for plane, prompts in tqdm(text_prompts.items()):
            prompts = tokenizer(prompts).to(device)
            text_features = model.encode_text(prompts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            text_features = text_features.mean(dim=0).unsqueeze(0)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            list_text_features.append(text_features)

    list_paths = []
    list_gt = []
    list_probs = []
    for imgs, planes, paths in tqdm(dl):
        with torch.no_grad(), torch.cuda.amp.autocast():
            imgs = imgs.to(device)
            image_features = model.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            list_text_logits = []
            for text_features in list_text_features:
                text_logits = (100.0 * image_features @ text_features.T).mean(dim=-1)[:,None]
                list_text_logits.append(text_logits)
            text_probs = torch.cat(list_text_logits, dim=1).softmax(dim=-1)
            
            list_paths.extend(paths)
            list_gt.extend(planes)
            list_probs.append(text_probs.cpu())

    probs = torch.cat(list_probs, dim=0).detach().cpu()
    preds = torch.argmax(probs, dim=1)
    
    list_pred = [index_to_planename[pred.item()] for pred in preds]

    targets = [planename_to_index[plane_db_to_planename[x]] for x in list_gt]
    targets = torch.tensor(targets)

    acc = Accuracy(task="multiclass", num_classes=len(planename_to_index), top_k=1, average='none')(probs, targets)
    f1 = F1Score(task="multiclass", num_classes=len(planename_to_index), top_k=1, average='none')(probs, targets)

    acc_top2 = Accuracy(task="multiclass", num_classes=len(planename_to_index), top_k=2, average='none')(probs, targets)
    f1_top2 = F1Score(task="multiclass", num_classes=len(planename_to_index), top_k=2, average='none')(probs, targets)

    acc_top3 = Accuracy(task="multiclass", num_classes=len(planename_to_index), top_k=3, average='none')(probs, targets)
    f1_top3 = F1Score(task="multiclass", num_classes=len(planename_to_index), top_k=3, average='none')(probs, targets)

    list_targets = sorted(targets.unique().tolist())
    acc = retrieve_based_on_indexes(acc, list_targets)
    f1 = retrieve_based_on_indexes(f1, list_targets)
    acc_top2 = retrieve_based_on_indexes(acc_top2, list_targets)
    f1_top2 = retrieve_based_on_indexes(f1_top2, list_targets)
    acc_top3 = retrieve_based_on_indexes(acc_top3, list_targets)
    f1_top3 = retrieve_based_on_indexes(f1_top3, list_targets)
    list_classes = [index_to_planename[x] for x in list_targets]

    print('Classes')
    print(list_classes)
    print('')
    print(f'acc: {np.mean(acc):.4f} | {[f"{x:.4f}" for x in acc]}')
    print(f'f1 : {np.mean(f1):.4f} | {[f"{x:.4f}" for x in f1]}')
    print('')
    print(f'acc_top2: {np.mean(acc_top2):.4f} | {[f"{x:.4f}" for x in acc_top2]}')
    print('')
    print(f'acc_top3: {np.mean(acc_top3):.4f} | {[f"{x:.4f}" for x in acc_top3]}')

    data = {
            'plane_db_to_planename': plane_db_to_planename,
            'planename_to_index': planename_to_index,
            'targets': targets.detach().cpu(),
            'probs': probs.detach().cpu(),
            'paths': list_paths,
            'list_classes': list_classes,
            'acc': acc,
            'f1': f1,
            'acc_top2': acc_top2,
            'acc_top3': acc_top3,
        }
    list_data = [{
        'model': "FetalCLIP",
        'f1': np.mean(data['f1']),
        'acc': np.mean(data['acc']),
        'acc_top2': np.mean(data['acc_top2']),
        'acc_top3': np.mean(data['acc_top3']),
        **{
            f'f1_{key}': val for key, val in zip(data['list_classes'], data['f1'])
        },
        **{
            f'acc_{key}': val for key, val in zip(data['list_classes'], data['acc'])
        },
        **{
            f'acc_top2_{key}': val for key, val in zip(data['list_classes'], data['acc_top2'])
        },
        **{
            f'acc_top3_{key}': val for key, val in zip(data['list_classes'], data['acc_top3'])
        },
    }]

    df = pd.DataFrame(list_data)
    df.to_csv('test_five_planes_results.csv', index=False)

    with open("test_five_planes_prediction.json", "w") as f:
        json.dump(
            {"prediction": list_pred, "label": [plane_db_to_planename[x] for x in list_gt]},
            f, indent=2
        )

def retrieve_based_on_indexes(vals, list_indexes):
    return [x.detach().item() for i, x in enumerate(vals) if i in list_indexes]

if __name__ == '__main__':
    main(PATH_FETALCLIP_WEIGHT)