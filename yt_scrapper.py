from string import ascii_uppercase
from random import choices
from pytube import YouTube
import subprocess
import argparse
import shutil
import time
import os

transnetv2_path = '/home/wd-vujos/Work/TransNetV2/inference/'

def extract_files(scenes, path, threshold=32):
    cut_prefix = os.path.basename(path) + "_cut_"
    with open(scenes) as f:
        for i, line in enumerate(f):
            start_frame, end_frame = line.split()
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            num_of_frames = end_frame - start_frame + 1
            if num_of_frames > threshold:
                extract_path = os.path.join(path, cut_prefix + str(i))
                os.makedirs(extract_path, exist_ok = True)
                for f in range(num_of_frames):
                    shutil.move(os.path.join(path, 'tmp', str(f+start_frame).zfill(6) + ".jpg"), os.path.join(extract_path, str(f).zfill(6) + ".jpg"))
        shutil.rmtree(os.path.join(path, 'tmp'))


def main(args):
    if args.youtube_links.endswith('.txt'):
        youtube_links = open(args.youtube_links).readlines()
    elif args.youtube_links.startswith('https://www.youtube.com') or args.youtube_links.startswith('www.youtube.com'):
        youtube_links  = [args.youtube_links]
    else:
        raise ValueError("Youtube links not supported, please pass txt file containing youtube links or youtube link itself")

    for i, yt_link in enumerate(youtube_links):
        myVideo = YouTube(yt_link)
        streams = myVideo.streams.filter(only_video=True)
        stream = streams.filter(resolution='1080p').first()
        if stream is None: stream = streams.filter(resolution='702p').first()
        if stream is None:
            print("Didn't found resolution that sattisfy our criteraia, skipping video")
            print(yt_link)
            print(streams.first().default_filename)
        else:
            vidname = os.path.join(args.output, stream.default_filename)
            print("Downloading video: ", yt_link)
            stream.download(args.output)

            if not args.keep_name:
                vidname_ = os.path.join(args.output, ''.join(choices(ascii_uppercase, k=10))) + '.' + vidname.split('.')[-1]
                os.rename(vidname, vidname_)
                vidname = vidname_
            
            folder_path = os.path.splitext(vidname)[0]

            print("Extracting video: ")
            print(folder_path)
            
            os.makedirs(os.path.join(folder_path, 'tmp'))
            command = ['ffmpeg',
                    '-i', f'"{vidname}"',
                    '-f', 'image2',
                    '-v', 'error',
                    '-start_number', '0',
                    f'"{folder_path}/tmp/%06d.jpg"']

            subprocess.Popen(f'{" ".join(command)}', shell=True).wait()
            
            print("Separating scenes: ")
            subprocess.Popen(f"conda run -n transnet python3 transnetv2.py '{vidname}'", shell=True, cwd=os.path.dirname(transnetv2_path)).wait()
            print("Separating scenes into subfolders: ")
            extract_files(vidname + ".scenes.txt", folder_path)

            if not args.keep_video:
                os.remove(vidname)
            os.remove(vidname + ".scenes.txt")
            os.remove(vidname + ".predictions.txt")

            time.sleep(5) # Just add so we don't spam youtube, dunno

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--youtube_links', type=str,
                        help='Path to txt with youtube links or youtube link', required=True)
    parser.add_argument('--output', type=str, default='output',
                        help='Output path', required=True)
    parser.add_argument('--keep_video', action="store_true",
                        help="If passed, video file will be saved")
    parser.add_argument('--keep_name', action="store_true",
                        help="If passed, video will keep original name, othervise it will be randomly generated")
    args = parser.parse_args()

    main(args)