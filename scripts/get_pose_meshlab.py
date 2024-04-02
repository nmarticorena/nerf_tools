import json
import spatialmath as sm
import numpy as np
import pyperclip

template = '<!DOCTYPE ViewState><project> <VCGCamera PixelSizeMm="0.0369161 0.0369161" TranslationVector="{}" FocalMm="13.49" CameraType="0" LensDistortion="0 0" ViewportPx="1920 1440" BinaryData="0" CenterPx="966 718" RotationMatrix="{}"/> <ViewSettings TrackScale="0.695179" NearPlane="0.003109" FarPlane="5.79931"/></project>'


def get_pose_meshlab(filename, index):
    with open(filename) as f:
        data = json.load(f)
    T = np.array(data["frames"][index]["transform_matrix"])
    T = sm.SE3(T, check=False)
    T = T * sm.SE3.Rx(-90, 'deg')

    R = T.R
    R_2 = np.c_[R, np.array([0,0,0])]
    R3 = np.r_[R_2, np.array([[0,0,0,1]])]
    rot = R3.flatten().tolist()
    rot = str(rot).replace('[','').replace(']','').replace(',','')
    t = T.A[:,3]

    t = str(t).replace('[','').replace(']','').replace(',','')
    
    print(data["frames"][index])
    print(template.format(t, rot))
    pyperclip.copy(template.format(t, rot))

get_pose_meshlab("transforms.json", 2)
