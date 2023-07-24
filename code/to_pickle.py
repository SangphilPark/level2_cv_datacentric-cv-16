import pickle
from dataset import CustomSceneTextDataset
from east_dataset import EASTDataset
from tqdm import tqdm
from utils import split_folder, get_gt_bboxes, get_pred_bboxes

for size in range(1280, 2304 + 1, 256):
    print(size)
    root_dir, train_images, valid_images = split_folder(seed=1110, test_size=0.2, data_dir='/opt/ml/input/data/medical')
    
    train_dataset = CustomSceneTextDataset(
        root_dir,
        train_images,
        split='train',
        image_size=size,
        crop_size=1024,
        ignore_tags=['masked', 'excluded-region', 'maintable', 'stamp']
    )

    train_data = EASTDataset(train_dataset)
    
    for i in tqdm(range(len(train_data))):
        g = train_data.__getitem__(i)
        with open(file=f"/opt/ml/input/data/multi_scaled/{i}_{size}.pkl", mode="wb") as f: ## 저장경로
            pickle.dump(g, f)