import subprocess
import glob
import os


def ffmpeg_video_creation(img_dir_path, outpath):
    duration = 0.05
    filenames = list()
    os.chdir(img_dir_path)
    for file in glob.glob("*.jpg"):
        filenames.append(file)

    with open("ffmpeg_input.txt", "wb") as outfile:
        for filename in filenames:
            outfile.write(f"file '{img_dir_path}/{filename}'\n".encode())
            outfile.write(f"duration {duration}\n".encode())

    command_line = f"ffmpeg -r 29.97 -f concat -safe 0 -i ffmpeg_input.txt -c:v libx264 -pix_fmt yuv420p {outpath}\\out.mp4"
    print(command_line)

    pipe = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE).stdout
    output = pipe.read().decode()
    pipe.close()

def ffmpeg_img_slicing_from_video(inputvideo, outputpath):
    command_line = f"ffmpeg -i {inputvideo}"
    print(command_line)
    pipe = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE).stdout
    output = pipe.read().decode()
    pipe.close()

    command_line = f"ffmpeg -i {inputvideo} {outputpath}/%05d.jpg"
    print(command_line)
    pipe = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE).stdout
    output = pipe.read().decode()
    pipe.close()

if __name__ == "__main__":
    #inputvideo = "C:/Users/mum21730/Desktop/5_Safe/Bilder/petpa/camera_top/1.mp4"
    outputpath_imgs = "C:/Users/mum21730/Desktop/5_Safe/Bilder/petpa/camera_top/result"
    outputpath_vid = "C:/Users/mum21730/Desktop/5_Safe/Bilder/petpa/camera_top/"

    #ffmpeg_img_slicing_from_video(inputvideo, outputpath_imgs)

    ffmpeg_video_creation(outputpath_imgs, outputpath_vid)