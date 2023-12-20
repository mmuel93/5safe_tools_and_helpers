import os
import sys
import json
import subprocess



def exctract_angle_from_imagmetata_dji_mini_3(input_file):
    exe="C:/Users/mum21730/Downloads/exiftool-12.49/exiftool(-a).exe"
    # process is a variable
    process=subprocess.Popen([exe,input_file],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    angle = None
    altitude = None
    # For the output
    for output in process.stdout:
        #print(output.strip())
        if "Gimbal Pitch Degree" in output:
            labelstring = output.strip()
            angle = output.strip().split(":")[1].strip()
        if "Relative Altitude" in output:
            labelstring = output.strip()
            altitude = output.strip().split(":")[1].strip()
    if (angle is not None and altitude is not None):
        return angle, altitude
        
    return None, None

def read_all_files_in_directory(path_to_dir):
    filenamelist = os.listdir(path_to_dir)
    for file in os.listdir(path_to_dir):
        if file.endswith(".jpg" or ".JPG"):
            filenamelist.append(os.path.join("/mydir", file))
    if len(filenamelist) > 0:
        return filenamelist
    else:
        raise Exception("No .jpg Files found in given Directory! Check Input Directory")
    
def save_dict_to_json(dict, filename, outdir):
    json_object = json.dumps(dict, indent=4)
    
    filename, _ = os.path.splitext(filename)


    with open(outdir + "/%s.json" %filename, "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    input_dir="C:/Users/mum21730/Desktop/Paper_IMG/Proposed_for_Comp"
    filelist = read_all_files_in_directory(input_dir)
    for filename in filelist:
        angle, altitude = exctract_angle_from_imagmetata_dji_mini_3(input_dir + "/" + filename)
        if not angle:
            raise Exception("Warning! Could not extract Angle from Imagemetadata for %s! Delete or check File and try again." % filename)
        if not altitude:
            raise Exception("Warning! Could not extract Altitude from Imagemetadata for %s! Delete or check File and try again." % filename)
        labeldict = {"Angle": float(angle), 
                     "Relative Altitude:": float(altitude)}
        save_dict_to_json(labeldict, filename, input_dir)
        print(filename, angle, altitude)