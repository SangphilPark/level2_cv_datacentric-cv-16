import os
import os.path as osp
import shutil
from detect import *
from tqdm import tqdm
import cv2
import json
from sklearn.model_selection import train_test_split
import pickle


def get_gt_bboxes(root_dir, ufo_dir, valid_images) : 
    
    '''
    ground truth bboxes dictionary를 구하는 함수 입니다.
    '''   
    
    gt_bboxes = dict()
    ufo_file_root = root_dir + ufo_dir
    
    with open(ufo_file_root, 'r') as f:
        ufo_file = json.load(f)
            
    ufo_file_images = ufo_file['images']
    for valid_image in tqdm(valid_images) :
        gt_bboxes[valid_image] = []
        for idx in ufo_file_images[valid_image]['words'].keys() :
            gt_bboxes[valid_image].append(ufo_file_images[valid_image]['words'][idx]['points'])
    return gt_bboxes        


def get_pred_bboxes(model, data_dir, valid_images, input_size, batch_size, split='valid') : 
    
    '''
    모델을 통해 예측된 bboxes dictionary를 구하는 함수 입니다.
    '''   
    
    image_fnames, by_sample_bboxes = [], []

    images = []
    for valid_image in tqdm(valid_images) :
        image_fpath = osp.join(data_dir,'img/{}/{}'.format(split, valid_image))
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))
        
    pred_bboxes = dict()   
    for idx in range(len(image_fnames)) :
        image_fname = image_fnames[idx]
        sample_bboxes = by_sample_bboxes[idx]
        pred_bboxes[image_fname] = sample_bboxes
    
    return pred_bboxes
    
def split_folder(seed, test_size, data_dir) :
    
    '''
    train, valid set 폴더를 생성하는 함수입니다.
    해당 함수에서 문제가 발생할 경우, code 파일 내부에 생성된 pkls 폴더과
    data 폴더 내부에 생성된 split 폴더을 제거하고 다시 실행해주세요!
    '''
    
    train_data_dir = data_dir + '/img/train'
    images = os.listdir(train_data_dir)
    images = sorted(images)
    new_root_path = '/opt/ml/input/data/split'
    pkl_path = '/opt/ml/input/code/pkls'
    
    if os.path.isdir(new_root_path) : 
        # split 폴더가 존재하면 데이터가 이미 나뉘었다고 간주
        with open(pkl_path+'/train_valid_set.pkl', 'rb') as f :
            train_valid_set = pickle.load(f)
            
        return new_root_path, train_valid_set['train'], train_valid_set['valid']
    
    else :
        new_img_path = new_root_path + '/img'
        new_ufo_path = new_root_path + '/ufo'
        os.makedirs(new_img_path)
        os.mkdir(new_ufo_path)
        
        new_train_data_path = new_img_path + '/train'
        new_valid_data_path = new_img_path + '/valid'
        os.mkdir(new_train_data_path)
        os.mkdir(new_valid_data_path)
        
        shutil.copy(data_dir+'/ufo/train.json',new_ufo_path+'/train.json')
        shutil.copy(data_dir+'/ufo/train.json',new_ufo_path+'/valid.json')
        
        train_images, valid_images = train_test_split(images, test_size=test_size, random_state=seed)
        for train_image in train_images :
            shutil.copy(train_data_dir+'/'+train_image, new_train_data_path+'/'+train_image)
        for valid_image in valid_images :
            shutil.copy(train_data_dir+'/'+valid_image, new_valid_data_path+'/'+valid_image)
        
        os.mkdir(pkl_path)
        train_valid_set = {'train':train_images, 'valid':valid_images}
        with open(pkl_path+"/train_valid_set.pkl", 'wb') as f:
            pickle.dump(train_valid_set,f)
            
        return new_root_path, train_images, valid_images
        