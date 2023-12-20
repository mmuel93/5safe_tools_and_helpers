import cv2 as cv
from cv2 import WARP_INVERSE_MAP
import numpy as np
from matplotlib import pyplot as plt
import os
import subprocess
import glob
from LightGlue.lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from LightGlue.lightglue.utils import load_image, rbd



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

    command_line = f"ffmpeg -r 29.97 -f concat -safe 0 -i ffmpeg_input.txt -c:v libx264 -pix_fmt yuv420p {outpath}\\out_stab.mp4"
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

def create_sift_feature_matches_for_corresponding_images(img1, img2):

    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    return pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c, ch = img1.shape
    #img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    #img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
    
def show_images(img1, img2):
    plt.subplot(121),plt.imshow(img1)
    plt.subplot(122),plt.imshow(img2)
    plt.show()

def draw_features_to_images(img, feature_list):
    for feature in feature_list:
        img = cv.circle(img, (int(feature[0]), int(feature[1])), 5, (0, 255, 0), thickness=-1)

def get_all_files_in_dir_with_extension(dir, extension):
    filelist = []
    for file in os.listdir(dir):
        if file.endswith(extension):
            filelist.append(os.path.join(dir, file))
    return filelist

def imgstab_ai(filelist, outputpath_imgs_stab):
    # SuperPoint+LightGlue
    #extractor = SuperPoint(max_num_keypoints=2048, resize="2048").eval()  # load the extractor
    #matcher = LightGlue(features='superpoint').eval()  # load the matcher

    # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
    extractor = DISK(max_num_keypoints=2048).eval()  # load the extractor, set resize to 2048
    matcher = LightGlue(features='disk').eval() # load the matcher

    image0 = load_image(filelist[0])
    img1 = cv.imread(filelist[0])
    count = 0
    for file in filelist[1:]:
        img2 = cv.imread(file)
        print("Processing: %05d of %05d" % ((count + 1), (len(filelist) - 1)))

        # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
        image1 = load_image(file)

        # extract local features
        feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
        feats1 = extractor.extract(image1)

        # match the features
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

        points0 = points0.detach().numpy()
        points1 = points1.detach().numpy()

        H = cv.findHomography(points0, points1, cv.RANSAC, 5.0)
        H_inv  = np.linalg.inv(H[0])
        result = cv.warpPerspective(img2, H_inv, (img1.shape[1], img1.shape[0]), cv.WARP_INVERSE_MAP)

 
        cv.imwrite(outputpath_imgs_stab + os.path.split(file)[-1], result)
        count += 1
    return 1

def imgstab_cv2(filelist, outputpath_imgs_stab):
    count = 0
    for file in filelist[1:]:
        img2 = cv.imread(file)
        print("Processing: %05d of %05d" % ((count + 1), (len(filelist) - 1)))

        #img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
        #img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

        features_img1, features_img2 = create_sift_feature_matches_for_corresponding_images(img1, img2)
        features_img1 = np.int32(features_img1)
        features_img2 = np.int32(features_img2)

        H = cv.findHomography(features_img1, features_img2, cv.RANSAC, 5.0)
        H_inv  = np.linalg.inv(H[0])
        result = cv.warpPerspective(img2, H_inv, (img1.shape[1], img1.shape[0]), WARP_INVERSE_MAP)
        cv.imwrite(outputpath_imgs_stab + os.path.split(file)[-1], result)
        count += 1
    return 1

if __name__ == '__main__':
    basepath = "C:/Users/mum21730/Desktop/5_Safe/Bilder/VUP_231108/topview_videos/"
    vidname = "drone_ped"

    inputpath_video = basepath + vidname + ".MP4"

    if not os.path.exists(basepath + vidname + "_stab"):
        os.makedirs(basepath + vidname + "_stab")
    if not os.path.exists(basepath + vidname + "_stab/out"):
        os.makedirs(basepath + vidname + "_stab/out")
    if not os.path.exists(basepath + vidname + "_stab/result"):
        os.makedirs(basepath + vidname + "_stab/result")

    inputpath_imgs = basepath + vidname + "_stab" + "/out/"
    outputpath_imgs_stab = basepath + vidname + "_stab" + "/result/"
    outputpath_video_stab = basepath + vidname + "_stab"

    
    ffmpeg_img_slicing_from_video(inputpath_video, inputpath_imgs)

    filelist = get_all_files_in_dir_with_extension(inputpath_imgs, ".jpg")

    # Read and Write first Image to Outputdirectory. This image is the Keyframe.
    img1 = cv.imread(filelist[0])
    cv.imwrite(outputpath_imgs_stab + os.path.split(filelist[0])[-1], img1)
    
    #status_imgstab = imgstab_cv2(filelist, outputpath_imgs_stab)

    status_imgstab = imgstab_ai(filelist, outputpath_imgs_stab)
    
    if status_imgstab:
        ffmpeg_video_creation(outputpath_imgs_stab, outputpath_video_stab)
    else:
        raise Exception("Video Creation was not successfully because Imgstab finished with an Error. Start Debugging at Imgstab")
    
    



    