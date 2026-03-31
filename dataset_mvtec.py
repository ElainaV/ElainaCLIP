import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        
        #原mvtec
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']

        '''
        #pipeAndMVTEC
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood','pipe']
        '''
        
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    # ------------------- 修改开始 -------------------
    # 1) 将 obj_list 中的类名统一为小写（避免 'Class1' vs 'class1' 不一致）
    obj_list = [k.lower() for k in obj_list]

    # 2) 将映射表的键也转换为小写（确保 class_name_map_class_id['class1'] 有对应的 id）
    class_name_map_class_id = {k.lower(): v for k, v in class_name_map_class_id.items()}
    # ------------------- 修改结束 -------------------

    return obj_list, class_name_map_class_id


class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        
        # ------------------- 修复在进行test时遇到的报错 不同数据集train，test大小写不一致 -------------------
        if mode not in meta_info:
            alt_mode = mode.capitalize()
            if alt_mode in meta_info:
                mode = alt_mode
            else:
                raise KeyError(f"meta.json 中找不到 {mode} 或 {alt_mode} 键，请检查 meta.json 结构")
        # ------------------------------------------------------------------------------------
        
        meta_info = meta_info[mode]

        # ------------------- 通用修复：统一 meta.json 的类名为小写，避免 KeyError -------------------
        meta_info_lower = {}
        for k, v in meta_info.items():
            meta_info_lower[k.lower()] = v
        meta_info = meta_info_lower
        # ---------------------------------------------------------------------------------------

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask

        # ------------------- 修复说明 -------------------
        # 这里 cls_name 可能是小写 'class1'，而映射表现在已支持小写，不需额外修改
        # -----------------------------------------------
        #return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
        #        'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}
        
        # ------------------- 修复 KeyError: 'Woven_001' -------------------
        cls_name_lower = cls_name.lower()
        return {
            'img': img,
            'img_mask': img_mask,
            'cls_name': cls_name,  # 保留原始大小写
            'anomaly': anomaly,
            'img_path': os.path.join(self.root, img_path),
            "cls_id": self.class_name_map_class_id[cls_name_lower]  # 用小写索引映射表
        }
        # ---------------------------------------------------------------

