import json
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('input', type=str, default='transform_cv2.json', 
        help='path to input json file')
parser.add_argument('--output', type=str, default='traj.txt',
        help='path to output txt file, default is traj.txt')

args = parser.parse_args()

with open(f"{args.input}", 'rb') as json_file:
    json_dict = json.load(json_file)

frames = json_dict['frames']


poses = []
for frame in frames:
    matrix = frame["transform_matrix"]
    #matrix = matrix.flatten()
    poses.append(str(matrix).replace("[","").replace("]","").replace(",",""))

with open(args.output, 'w') as f:
    f.write(os.linesep.join(poses))

parser.exit(0, message= f'Done saved in {args.output}')