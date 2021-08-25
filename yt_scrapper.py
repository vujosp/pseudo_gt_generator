from pytube import YouTube
import subprocess
import time
import os

download_path = '/mnt/nas2/datasets/videos_from_yt/'
with open('/home/wd-vujos/Desktop/video.txt') as f:
    for i, yt_link in enumerate(f):
        if i < 10:
            continue
        myVideo = YouTube(yt_link)
        streams = myVideo.streams.filter(only_video=True)
        stream = streams.filter(resolution='1080p').first()
        if stream is None: stream = streams.filter(resolution='702p').first()
        if stream is None:
            print("Didn't found resolution that sattisfy our criteraia, skipping video")
            print(yt_link)
            print(streams.first().default_filename)
        else:
            downloaded_file = os.path.join(download_path, stream.default_filename)
            print("Skipping ", downloaded_file)
            if os.path.exists(downloaded_file):
                print("Skipping ", downloaded_file) 
                print("Video already downloaded!")
            else:
                print("Downloading: ", yt_link)
                stream.download(download_path)
            
            folder_path = os.path.splitext(downloaded_file)[0]
            if not os.path.exists(folder_path):
                print("Extracting video")
                print(folder_path)
                os.makedirs(folder_path)
                command = ['ffmpeg',
                        '-i', f'"{downloaded_file}"',
                        '-f', 'image2',
                        '-v', 'error',
                        '-start_number', '0',
                        f'"{folder_path}/%06d.jpg"']
            
                print('\n\n')
                print(f'{" ".join(command)}')
                print('\n\n')
                subprocess.Popen(f'{" ".join(command)}', shell=True)
            else:
                print("Video already extracted")
                print(folder_path)
            
            time.sleep(5) # Just add so we don't spam youtube, dunno