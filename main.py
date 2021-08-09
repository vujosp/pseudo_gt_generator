from concurrent.futures import ThreadPoolExecutor
from multi_person_tracker import MPT_WD
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

def run_mpt(image_folder):
    model_path = os.path.join(os.path.expanduser("~"), '.torch/models/yolov5x6.pt')  # it automatically downloads yolov5s model to given path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if(not os.path.isfile(os.path.join(image_folder, 'tracking_results.pickle'))):
        mot = MPT_WD(model_path=model_path, output_format='dict', square=False, detection_threshold=0.5, tracker_off=False)
        tracking_results_path = os.path.join(image_folder, "tracking_results.pickle")
        tracking_results = mot(image_folder, batch_size=1)
        pickle.dump(tracking_results, open(tracking_results_path, "wb"))

async def run_async_mpt(num_of_threads, glob_query):
    in_thread = set_in_thread(num_of_threads)
    img_folders = glob.glob(glob_query)
    tasks = []
    for img_folder in img_folders:
        tasks.append(in_thread(functools.partial(run_mpt, img_folder)))
    await asyncio.gather(*tasks)

def export_as_ds(output, opt_path, kps_path, images):
    tracking_results = np.load(os.path.join(images, 'tracking_results.pickle'), allow_pickle=True)
    images_, bbox, center = [], [], []
    image_paths = glob.glob(images + "*.png")
    image_paths.extend(glob.glob(images + ".jpg"))
    for opt_npz in glob.glob(os.path.join(opt_output, "*")):
        p_id = os.path.basename(opt_npz).split('_')[0]
        kps = np.load(os.path.join(kps_path, '1_'+p_id + ".npz"))
        p_tracking_res = tracking_results[p_id]

        images_.append(image_paths['frames'])


def main(args):
    if args.cfg_file:
        # TODO
        pass

    args.output = '/home/wd-vujos/Desktop/monoperfcap/'

    for img_folder in glob.glob('/mnt/nas2/datasets/3dpose_datasets/monoperfcap/*/')[2:]: # We skip first for now
        actual_image_name = os.path.basename(img_folder[:-1])
        args.image_folder = os.path.join(img_folder, actual_image_name, 'images')

        num_of_threads = args.num_of_threads or 8

        pymaf_checkpoint = args.pymaf_checkpoint  or "/mnt/nas/models/PyMAF/logs/pymaf_res50_mix/pymaf_res50_mix_as_lp3_mlp256-128-64-5_Jun07-09-14-53-KNj/checkpoints/model_best.pt"
        image_folder = args.image_folder or "/home/wd-vujos/Downloads/test_seq/"
        output_folder = os.path.join(args.output, actual_image_name)

        if not os.path.isabs(output_folder):
            output_folder = os.path.abspath(output_folder)

        os.makedirs(output_folder, exist_ok = True)
        kps_output = os.path.join(output_folder, 'output.npz')

        if image_folder.endswith('/'): image_folder = image_folder[:-1]
        image_name = os.path.basename(image_folder)

        kps_output = os.path.join(output_folder, 'kps')
        pymaf_output = os.path.join(output_folder, 'pymaf')
        kps_final = os.path.join(output_folder, 'kps_final', image_name)
        opt_output = os.path.join(output_folder, 'opt', image_name)

        # export_as_ds('', opt_output, kps_final, args.image_folder)

        # mot = MPT_WD(model_path=model_path, output_format='dict', square=False, detection_threshold=0.5, tracker_off=args.tracker_off)
        # sub_call(f"python3 test.py --input='{image_folder}' --output '{kps_output}' --num_of_threads {num_of_threads}", os.path.dirname(darkpose_path))
        sub_call(f"python3 demo_wd.py --output_folder='{pymaf_output}' --checkpoint='{pymaf_checkpoint}' --image_folder '{image_folder}' --no_render", os.path.dirname(pymaf_path))
        # sub_call(f"python3 main.py --img_folders='{image_folder}' --kps_results '{kps_output}' --output '{kps_final}'", os.path.dirname(kps_manager_path))
        # sub_call(f"python3 demo_single_cam.py --img_folders='{image_folder}' --results_3d '{os.path.join(pymaf_output, image_name)}' --results_2d '{kps_final}' --output '{opt_output}'", os.path.dirname(optimizer_path))

        # for opt_npz in glob.glob(os.path.join(opt_output, "*")):
        #     p_id = os.path.basename(opt_npz).split('_')[0]
        #     blender_output = os.path.join(output_folder, 'blender', p_id + '.blend')
        #     sub_call(f"blender --background --python output_to.py -- --keep_scene --fbx_file male --cam_type PERSP --name {p_id}_optim --background_image='{image_folder}' --npz_file '{os.path.join(opt_output, p_id + '_blender_params_optimization.npz')}' --blender_file_path '{blender_output}'", os.path.dirname(sdg_path))
        #     sub_call(f"blender {blender_output} --background --python output_to.py -- --keep_scene --aperture 25 --focal_length 25 --fbx_file male --cam_type PERSP --name {p_id}_orig --background_image='{image_folder}' --npz_file '{os.path.join(pymaf_output, image_name, p_id + '_blender_params.npz')}' --blender_file_path '{blender_output}'", os.path.dirname(sdg_path))
        #     os.remove(blender_output + "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_file', type=str,
                        help='Path to cfg file with all info')
    parser.add_argument('--image_folder', type=str,
                        help='Path to images')
    parser.add_argument('--output', type=str, default='output',
                        help='Output path')
    parser.add_argument('--pymaf_checkpoint', type=str,
                        help='PyMaf checkpoint')
    parser.add_argument('--num_of_threads', type=int, default=8,
                        help='Number of threads to use')

    args = parser.parse_args()

    main(args)
