from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor
from multi_person_tracker import MPT_WD
from natsort import natsorted
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import functools
import subprocess
import argparse
import asyncio
import signal
import pickle
import glob
import os

pymaf_path = "/home/wd-vujos/Work/PyMAF/"
darkpose_path = "/home/wd-vujos/Work/darkpose/"
kps_manager_path = "/home/wd-vujos/Work/2d_kps_manager/"
optimizer_path = "/home/wd-vujos/Work/person_optimization/"
sdg_path = "/home/wd-vujos/Work/synthetic-data-generator-blender/"

def sub_call(command, cwd, should_exit=True):
    '''
        Wrapper around popen, so we could interupt process by KeyboardInterrupt.
    '''
    p = subprocess.Popen(command, shell=True, cwd=cwd)
    try:
        (output, err) = p.communicate()
        p_status = p.wait()
    except KeyboardInterrupt:
        p.kill()
        p_status = p.wait()
        if should_exit:
            exit()
    return p_status

def set_in_thread(num_of_threads):
    _executor = ThreadPoolExecutor(num_of_threads)
    async def in_thread(func):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, func)
    return in_thread

def run_mpt(image_folder, min_frame_num=None, force=False):
    if(force or not os.path.isfile(os.path.join(image_folder, 'tracking_results.pickle'))):
        mot = MPT_WD(output_format='dict', square=False, detection_threshold=0.5, tracker_off=False)
        tracking_results_path = os.path.join(image_folder, "tracking_results.pickle")
        tracking_results = mot(image_folder, batch_size=1, min_frame_num=min_frame_num)
        pickle.dump(tracking_results, open(tracking_results_path, "wb"))

async def run_async_mpt(num_of_threads, glob_query, min_frame_num=None):
    in_thread = set_in_thread(num_of_threads)
    img_folders = glob.glob(glob_query)
    tasks = []
    for img_folder in img_folders:
        tasks.append(in_thread(functools.partial(run_mpt, img_folder, min_frame_num)))
    await asyncio.gather(*tasks)

def export_as_ds(output, opt_path, kps_path, images, prefix_to_remove=None, seq_size=None, gender=None):
    tracking_results = np.load(os.path.join(images, 'tracking_results.pickle'), allow_pickle=True)
    imgnames_, parts_, centers_, scales_, pose_, shape_ = [], [], [], [], [], []
    image_paths = glob.glob(os.path.join(images, "*.png"))
    image_paths.extend(glob.glob(os.path.join(images, "*.jpg")))
    image_paths = np.sort(image_paths)

    if prefix_to_remove:
        image_paths = np.array([img.replace(prefix_to_remove, '') for img in image_paths])

    kps_order_dp = [16,14,12,11,13,15,10,8,6,5,7,9,0,1,2,3,4]
    kps_indices = [0,1,2,3,4,5,6,7,8,9,10,11,19,20,21,22,23]
    for opt_npz in np.sort(glob.glob(os.path.join(opt_path, "*_dataset_params_optimization.npz"))):
        p_id = os.path.basename(opt_npz).split('_')[0]

        opt = np.load(opt_npz)
        kps = np.load(os.path.join(kps_path, '1_'+p_id + ".npy"))

        good_indices = np.delete(np.arange(kps.shape[0]), np.where(np.sum(kps[..., 2],axis=1) == 0)[0]) # From all possible indices, delete those where sum of confidence is equal to 0
        
        if good_indices.shape[0] < 32:
            continue

        if seq_size:
            good_seq_images = []
            holes = np.nonzero(np.diff(good_indices)-1)
            holes = np.insert(holes, 0, 0) # Insert 0 at the begining
            holes = np.insert(holes, len(holes), len(good_indices))
            for i in range(len(holes)-1):
                valid_elems = holes[i+1] - holes[i]
                for j in range(int(np.floor(valid_elems/seq_size))-1):
                    if j < 0:
                        print(a)
                    good_seq_images.append(good_indices[holes[i]+seq_size*j:holes[i]+seq_size*(j+1)])
                    shape = good_seq_images[-1].shape[0]
            if len(good_seq_images) == 0:
                continue
            good_indices = np.concatenate(good_seq_images, axis=0)

        p_tracking_res = tracking_results[int(p_id)]
        
        bboxes = p_tracking_res['bbox'][good_indices]
        bboxes[..., 2:] = np.max(bboxes[..., 2:], axis=-1).reshape(*bboxes.shape[:-1], 1)
        
        centers_.append([bboxes[..., 0], bboxes[..., 1]])
        scales_.append(bboxes[..., 2]/200)
        
        shape_.append(opt['smpl_betas'][good_indices])
        pose_.append(opt['smpl_pose_3d'][good_indices])
        all_kps = np.zeros((kps.shape[0], 24, 3))

        all_kps[:, kps_indices] = kps[:, kps_order_dp]
        parts_.append(all_kps[good_indices])

        imgnames_.append(image_paths[p_tracking_res['frames']][good_indices])

    if len(imgnames_):
        imgnames_ = np.concatenate(imgnames_, axis=0)
        to_save = {
            "imgname":imgnames_,
            "center":np.concatenate(centers_, axis=1).T,
            "scale":np.concatenate(scales_, axis=0).ravel(),
            "part":np.concatenate(parts_, axis=0),
            "pose":np.concatenate(pose_, axis=0),
            "shape":np.concatenate(shape_, axis=0),
            "has_smpl":np.ones(imgnames_.shape[0])
        }

        if gender: to_save['gender'] = np.full(imgnames_.shape[0], gender[0])

        np.savez(output, **to_save)
        return True
    return False

def merge_npz(output, npz_files):
    data_all = [np.load(fname) for fname in npz_files]
    merged_data = defaultdict(list)
    for data in data_all:
        [merged_data[k].append(v) for k, v in data.items()]
    for k in merged_data.keys():
        merged_data[k] = np.concatenate(merged_data[k])
    np.savez(output, **merged_data)

def main(args):
    if args.cfg_file:
        # TODO
        pass

    # args.output = '/home/wd-vujos/Desktop/h36m_300/h36m_test/'

    # all_datasets = []
    # for img_folder in tqdm(glob.glob('/home/wd-vujos/Desktop/h36m_300/images/')):
    args.output = '/mnt/nas2/datasets/pseudo_ds/videos_from_yt/'
    prefix_to_remove = '/mnt/nas2/datasets/videos_from_yt/'

    all_datasets = []
    args.image_folder = '/mnt/nas2/datasets/videos_from_yt/*/'

    # female_folders = ['Franzi_studio', 'Nadia_outdoor', 'Natalia_outdoor', 'Simeng_moving_cam', 'Simeng_outdoor', 'Simeng_studio']
    for image_folder in tqdm(glob.glob(args.image_folder)):
        print(image_folder)
        num_of_threads = args.num_of_threads

        pymaf_checkpoint = args.pymaf_checkpoint if os.path.isabs(args.pymaf_checkpoint) else os.path.abspath(args.pymaf_checkpoint)
        output_folder = args.output

        # gender = 'female' if actual_image_name in female_folders else 'male'
        gender = 'male'

        if not os.path.isabs(output_folder):
            output_folder = os.path.abspath(output_folder)

        os.makedirs(output_folder, exist_ok = True)
        kps_output = os.path.join(output_folder, 'output.npz')

        if image_folder.endswith('/'): image_folder = image_folder[:-1]
        image_name = os.path.basename(image_folder)

        kps_output = os.path.join(output_folder, 'kps')
        pymaf_output = os.path.join(output_folder, 'pymaf')

        # args.focal_length = 8.5
        # args.aperture = 7.5

        # colors = ['1,0,0,1', '0,1,0,1', '0,0,1,1']
        # ds_names = [image_name, 'only_h36m', 'h36m_dp_head']
        # for color, seq_name in zip(colors, ds_names):
        kps_final = os.path.join(output_folder, 'kps_final', image_name) # SPECIFIC CASE
        # kps_final = os.path.join(output_folder, 'kps_final', seq_name)
        opt_output = os.path.join(output_folder, 'opt', image_name)

        # run_mpt(image_folder, args.min_frame_num)
        # sub_call(f"python3 test.py --input='{image_folder}' --output '{kps_output}' --num_of_threads {num_of_threads}", os.path.dirname(darkpose_path))
        # sub_call(f"python3 demo_wd.py --skip_existing --output_folder='{pymaf_output}' --checkpoint='{pymaf_checkpoint}' --image_folder '{image_folder}' --no_render", os.path.dirname(pymaf_path))
        # sub_call(f"python3 main.py --img_folders='{image_folder}' --results_3d '{os.path.join(pymaf_output, image_name)}' --kps_results '{kps_output}' --output '{kps_final}'", os.path.dirname(kps_manager_path))
        sub_call(f"python3 demo_single_cam.py --gender {gender} --opt_params smpl_pose_3d smpl_betas --img_folders='{image_folder}' --aperture {args.focal_length} --focal_length {args.aperture} --results_3d '{os.path.join(pymaf_output, image_name)}' --results_2d '{kps_final}' --output '{opt_output}'", os.path.dirname(optimizer_path))
        # for opt_npz in glob.glob(os.path.join(opt_output, "*_blender_params_optimization.npz")):
        #     p_id = os.path.basename(opt_npz).split('_')[0]
        #     blender_output = os.path.join(output_folder, 'blender', image_name, p_id + '.blend')
        #     # if seq_name == image_name:
        #     sub_call(f"blender --background --python output_to.py -- --aperture {args.focal_length} --focal_length {args.aperture} --fbx_file smpl_{gender} --cam_type PERSP --name {p_id}_orig --color '0.8,0.8,0.8,1' --background_image='{image_folder}' --npz_file '{os.path.join(pymaf_output, image_name, p_id + '_blender_params.npz')}' --blender_file_path '{blender_output}'", os.path.dirname(sdg_path))
        #         # sub_call(f"blender {blender_output} --background --python output_to.py -- --keep_scene --fbx_file smpl_male --cam_type PERSP --name {p_id}_gt --color '0.2,0.2,0.2,1' --background_image='{image_folder}' --npz_file /home/wd-vujos/Desktop/h36m_300/h36m_test/{p_id}_h36m_300.npz --blender_file_path '{blender_output}'", os.path.dirname(sdg_path))
        #     sub_call(f"blender {blender_output} --background --python output_to.py -- --keep_scene --fbx_file smpl_{gender} --cam_type PERSP --name {p_id}_optim --color '1,0,0,1' --background_image='{image_folder}' --npz_file '{os.path.join(opt_output, p_id + '_blender_params_optimization.npz')}' --blender_file_path '{blender_output}'", os.path.dirname(sdg_path))
        #     os.remove(blender_output + "1")
            # export_as_ds(os.path.join(output_folder, 'ds_test', image_name), opt_output, kps_final, image_folder, prefix_to_remove)
    #     ds_path = os.path.join(output_folder, 'datasets', 'all', image_name)
    #     os.makedirs(os.path.join(output_folder, 'datasets', 'all'), exist_ok=True)
    #     succ = export_as_ds(ds_path, opt_output, kps_final, image_folder, prefix_to_remove, gender=None)
    #     if succ:
    #         all_datasets.append(ds_path + ".npz")
    # merge_npz(os.path.join(args.output, 'datasets', 'datasets_all.npz'), all_datasets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_file', type=str,
                        help='Path to cfg file with all info')
    parser.add_argument('--image_folder', type=str,
                        help='Path to images')
    parser.add_argument('--output', type=str, default='output',
                        help='Output path')
    parser.add_argument('--pymaf_checkpoint', type=str,
                        help='PyMaf checkpoint', default='data/model_best.pt')
    parser.add_argument('--num_of_threads', type=int, default=8,
                        help='Number of threads to use')
    parser.add_argument('--min_frame_num', type=int, default=32,
                        help='Min number of frames person need to be on')
    parser.add_argument('--focal_length', type=float, default=25,
                        help='Focal length to use')
    parser.add_argument('--aperture', type=float, default=25,
                        help='Aperture to use')
    args = parser.parse_args()

    main(args)
