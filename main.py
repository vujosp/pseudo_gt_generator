import subprocess
import argparse
import signal
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

def main(args):
    if args.cfg_file:
        # TODO
        pass

    num_of_threads = args.num_of_threads or 8

    pymaf_checkpoint = args.pymaf_checkpoint  or "/mnt/nas/models/PyMAF/logs/pymaf_res50_mix/pymaf_res50_mix_as_lp3_mlp256-128-64-5_Jun07-09-14-53-KNj/checkpoints/model_best.pt"
    image_folder = args.image_folder or "/mnt/nas2/tmp/test_mini_sample/"
    output_folder = args.output_path or '/home/wd-vujos/Desktop/test_pl/'
    
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
    sub_call(f"python3 test.py --input={image_folder} --output {kps_output} --num_of_threads {num_of_threads}", os.path.dirname(darkpose_path))
    sub_call(f"python3 demo_wd.py --output_folder={pymaf_output} --checkpoint={pymaf_checkpoint} --image_folder {image_folder}", os.path.dirname(pymaf_path))
    sub_call(f"python3 main.py --img_folders={image_folder} --kps_results {kps_output} --output {kps_final}", os.path.dirname(kps_manager_path))
    sub_call(f"python3 demo_single_cam.py --img_folders={image_folder} --results_3d {os.path.join(pymaf_output, image_name)} --results_2d {kps_final} --output {opt_output}", os.path.dirname(optimizer_path))

    pids = []
    for opt_npz in glob.glob(os.path.join(opt_output, "*")):
        p_id = os.path.basename(opt_npz).split('_')[0]
        blender_output = os.path.join(output_folder, 'blender', p_id + '.blend')
        sub_call(f"blender --background --python output_to.py -- --keep_scene --fbx_file male --cam_type PERSP --name {p_id}_optim --background_image={image_folder} --npz_file {os.path.join(opt_output, p_id + '_blender_params_optimization.npz')} --blender_file_path {blender_output}", os.path.dirname(sdg_path))
        sub_call(f"blender {blender_output} --background --python output_to.py -- --keep_scene --aperture 25 --focal_length 25 --fbx_file male --cam_type PERSP --name {p_id}_orig --background_image={image_folder} --npz_file {os.path.join(pymaf_output, image_name, p_id + '_blender_params.npz')} --blender_file_path {blender_output}", os.path.dirname(sdg_path))
        os.remove(blender_output + "1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_file', type=str,
                        help='Path to cfg file with all info')
    parser.add_argument('--image_folder', type=str,
                        help='Path to images')
    parser.add_argument('--output_path', type=str, default='output',
                        help='Output path')
    parser.add_argument('--pymaf_checkpoint', type=str,
                        help='PyMaf checkpoint')
    parser.add_argument('--num_of_threads', type=int, default=8,
                        help='Number of threads to use')

    args = parser.parse_args()

    main(args)
