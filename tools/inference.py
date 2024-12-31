##############################################################
# % Author: Castle
# % Date:14/01/2023
###############################################################
import argparse
import os
import numpy as np
import cv2
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc
from datasets.io import IO
from datasets.data_transforms import Compose
import json
from models.Transformer_utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_config', 
        help = 'yaml config file')
    parser.add_argument(
        'model_checkpoint', 
        help = 'pretrained weight')
    parser.add_argument('--pc_root', type=str, default='', help='Pc root')
    parser.add_argument('--pc', type=str, default='', help='Pc file')   
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='whether to save img of complete point cloud') 
    parser.add_argument(
        '--out_pc_root',
        type=str,
        default='',
        help='root of the output pc file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config is not None
    assert args.model_checkpoint is not None
    assert (args.pc != '') or (args.pc_root != '')

    return args

def inference_single(model, pc_path, args, config, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path
    # read single point cloud
    pc_ndarray = IO.get(pc_file).astype(np.float32)
    # transform it according to the model 
    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # normalize it to fit the model on ShapeNet-55/34
        centroid = np.mean(pc_ndarray, axis=0)
        pc_ndarray = pc_ndarray - centroid
        m = np.max(np.sqrt(np.sum(pc_ndarray**2, axis=1)))
        pc_ndarray = pc_ndarray / m

    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 256  #2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    
    pc_ndarray_normalized = transform({'input': pc_ndarray})

    # Save the processed point cloud before inference
    # if args.out_pc_root != '':
    #     target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
    #     os.makedirs(target_path, exist_ok=True)
    #     np.save(os.path.join(target_path, 'processed.npy'), pc_ndarray_normalized['input'].numpy())

    # inference
    ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    coarse_points = ret[0].squeeze(0).detach().cpu().numpy()

    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # denormalize it to adapt for the original input
        dense_points = dense_points * m
        dense_points = dense_points + centroid

    if args.out_pc_root != '':
        target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
        os.makedirs(target_path, exist_ok=True)

        np.save(os.path.join(target_path, 'fine.npy'), dense_points)
        np.save(os.path.join(target_path, 'coarse.npy'), coarse_points)
        if args.save_vis_img:
            input_img = misc.get_ptcloud_img(pc_ndarray_normalized['input'].numpy())
            dense_img = misc.get_ptcloud_img(dense_points)
            cv2.imwrite(os.path.join(target_path, 'input.jpg'), input_img)
            cv2.imwrite(os.path.join(target_path, 'fine.jpg'), dense_img)
    
    return


def inference_ShapeNet(model, pc_path, args, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path

    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 256  #2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])

    with open(os.path.join(pc_file, "ShapeNet_Car_Seq.json")) as f:
        dataset_categories = json.loads(f.read())

    samples = dataset_categories["test"]
    for s in samples:
        gt_path = os.path.join(pc_file, "test", "complete", f'{s}.pcd')
        gt = IO.get(gt_path).astype(np.float32)
        for j in range(15):
            partial_path = os.path.join(pc_file, "test", "partial", s, f'{j:02}')
            print(partial_path)
            # partial_path = os.path.dirname(partial_path)
            # print(partial_path)

            # partial_dir = os.path.dirname(partial_path)
            # List all files in the directory to count available partial files
            available_files = sorted([f for f in os.listdir(partial_path) if f.endswith('.pcd')])
            view_count = len(available_files)
            # print(view_count)

            for i in range(view_count):
                partial = IO.get(os.path.join(partial_path, f'{i:03}.pcd')).astype(np.float32)
                partial_data = {'input': partial}
                partial_data = transform(partial_data)
                partial_data = partial_data['input']
                ret = model(partial_data.unsqueeze(0).to(args.device.lower()))

                dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
                # coarse_points = ret[0].squeeze(0).detach().cpu().numpy()

                if args.out_pc_root != '':
                    target_path = os.path.join(args.out_pc_root, s, f'{j:02}')
                    os.makedirs(target_path, exist_ok=True)

                    # np.save(os.path.join(target_path, f'{i:03}_input.npy'), partial)
                    # np.save(os.path.join(target_path, f'{i:03}_input256.npy'), partial_data.numpy())
                    np.save(os.path.join(target_path, f'{i:03}_fine.npy'), dense_points)
                    # np.save(os.path.join(target_path, f'{i:03}_coarse.npy'), coarse_points)
                    # np.save(os.path.join(target_path, 'gt.npy'), gt)

    return

inference_list = ["10555502fa7b3027283ffcfc40c29975",
"12097984d9c51437b84d944e8a1952a5",
"202648a87dd6ad2573e10a7135e947fe",
"272791fdabf46b2d5921daf0138cfe67",
"324434f8eea2839bf63ee8a34069b7c5"]


def inference_ShapeNet_m(model, pc_path, args, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path

    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 256  #2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])

    with open(os.path.join(pc_file, "ShapeNet_Car_Seq.json")) as f:
        dataset_categories = json.loads(f.read())

    samples = dataset_categories["test"]
    for s in samples:
        # if not s in inference_list:
        #     continue
        # else:
        gt_path = os.path.join(pc_file, "test", "complete", f'{s}.pcd')
        gt = IO.get(gt_path).astype(np.float32)
        for j in range(15):
            partial_path = os.path.join(pc_file, "test", "partial", s, f'{j:02}')
            print(partial_path)
            # partial_path = os.path.dirname(partial_path)
            # print(partial_path)

            # partial_dir = os.path.dirname(partial_path)
            # List all files in the directory to count available partial files
            available_files = sorted([f for f in os.listdir(partial_path) if f.endswith('.pcd')])
            view_count = len(available_files)
            # print(view_count)

            window_size = 9
            half_window = window_size // 2

            for i in range(half_window + 1, view_count - half_window - 1):
                partials_data = []
                # Loop through the window size to gather partials from (i - half_window) to (i + half_window)
                for offset in range(-half_window, half_window + 1):
                    idx = i + offset
                    # print(idx)

                    # Ensure the index is within valid bounds
                    if 0 <= idx < len(available_files):
                        partial = IO.get(os.path.join(partial_path, f'{idx:03}.pcd')).astype(np.float32)
                        partial_data = {'input': partial}
                        partial_data = transform(partial_data)
                        partial_data = partial_data['input']
                        # Append the concatenated result to the list
                        partials_data.append(partial_data.unsqueeze(0))

                        # ret = model(partials_data.to(args.device.lower()))
                # print(len(partials_data))
                concatenated_tensor = torch.cat(partials_data, dim=1).float()  #(B, 256*9, 3)
                # print(concatenated_tensor.shape)

                # Gather coor and x
                # fps_idx = pointnet2_utils.furthest_point_sample(concatenated_tensor, 256)  # (B, N)
                # updated_tensor = pointnet2_utils.gather_operation(concatenated_tensor.transpose(1, 2).contiguous().float(),
                #                                                 fps_idx).transpose(1, 2).contiguous()  # [B, N, 3]
                indices = torch.randint(0, 2304, (1, 256), device = concatenated_tensor.device)
                updated_tensor = torch.stack([
                    concatenated_tensor[b, idx, :] for b, idx in enumerate(indices)])
                # cuda_window_partials = [window_partial.cuda() for window_partial in window_partials]
                # print(fps_idx.shape)
                # print(updated_tensor.shape)
                # Forward pass
                ret = model(updated_tensor.cuda())
                # cuda_partials = [partial.to(args.device.lower()) for partial in partials_data]
                # ret = model(cuda_partials)

                dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
                # coarse_points = ret[0].squeeze(0).detach().cpu().numpy()
                # updated_coor = ret[-2].squeeze(0).detach().cpu().numpy()
                # coor = ret[-1].squeeze(0).detach().cpu().numpy()
                #
                # inputs = [partial.squeeze(0).numpy() for partial in partials_data]
                # input = np.vstack(inputs)

                if args.out_pc_root != '':
                    target_path = os.path.join(args.out_pc_root, s, f'{j:02}')
                    os.makedirs(target_path, exist_ok=True)

                    np.save(os.path.join(target_path, f'{i:03}_fine.npy'), dense_points)
                    # np.save(os.path.join(target_path, f'{i:03}_coarse.npy'), coarse_points)
                    # np.save(os.path.join(target_path, f'{i:03}_ucoor.npy'), updated_coor)
                    # np.save(os.path.join(target_path, f'{i:03}_coor.npy'), coor)
                    # np.save(os.path.join(target_path, f'{i:03}_input.npy'), input)
                    # np.save(os.path.join(target_path, 'gt.npy'), gt)

    return





def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    # if args.pc_root != '':
    #     pc_file_list = os.listdir(args.pc_root)
    #     for pc_file in pc_file_list:
    #         inference_single(base_model, pc_file, args, config, root=args.pc_root)
    # else:
    #     inference_single(base_model, args.pc, args, config)
    inference_ShapeNet_m(base_model, args.pc_root, args)

if __name__ == '__main__':
    main()