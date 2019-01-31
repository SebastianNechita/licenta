from scipy.io import loadmat
from collections import defaultdict
import pims
from matplotlib import animation
from tqdm import tqdm_notebook
import cv2 as cv
from copy import deepcopy
import matplotlib.pyplot as plt

def addAnnotationsToImage(img, ann, config):
    img2 = deepcopy(img)
    for personAnn in ann:
        pos = personAnn['pos']
        color = config.possibleLabels[personAnn['lbl']]
        pos = [int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])]
        img2 = cv.rectangle(img2, (pos[0],pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), color, 1)
    return img2

def getAnnotatedImage(video, frameNumber, config, annotations = None):
    image = video[frameNumber]
    if annotations is not None:
        ann = getAnnotationsForFrame(frameNumber, annotations)
        image = addAnnotationsToImage(image, ann, config)
    return image

def printImage(video, frameNumber, config, annotations = None):
    image = getAnnotatedImage(video, frameNumber, config, annotations)
    fig, ax = plt.subplots(1, figsize=(20, 10))
    ax.imshow(image, animated=True)
    plt.show()

def saveAnnotatedVideo(video, videoName, config, annotations = None):
    fig, ax = plt.subplots(1, figsize=(20, 10))
    frameList = []
    for frameNumber in tqdm_notebook(range(len(video))):
        img = getAnnotatedImage(video, frameNumber, config, annotations)
        frameList.append([ax.imshow(img, animated = True)])

    ani = animation.ArtistAnimation(fig, frameList, interval=50, blit=True, repeat_delay=10)
    print("Saving animation...")
    ani.save(videoName)
    print("Animation saved!")

def getAnnotationsForFrame(frameNumber, anns):
    temp = anns['frames'][frameNumber]
    rez = []
    for ann in temp:
        rez.append({'pos': ann['pos'], 'lbl': ann['lbl']})
    return rez

def getAllLabels(annotations):
    allLabels = set()
    for frame in range(annotations['nFrame']):
        ann = getAnnotationsForFrame(frame, annotations)
        thisFrameLabels = [ann[i]['lbl'] for i in range(len(ann))]
        allLabels = allLabels.union(set(thisFrameLabels))
    return allLabels

def read_seq(path):
    return pims.PyAVReaderIndexed(path)

# Copied from https://github.com/hizhangp/caltech-pedestrian-converter/blob/master/converter.py
def read_vbb(path):
    assert path[-3:] == 'vbb'
    vbb = loadmat(path)
    nFrame = int(vbb['A'][0][0][0][0][0])
    objLists = vbb['A'][0][0][1][0]
    maxObj = int(vbb['A'][0][0][2][0][0])
    objInit = vbb['A'][0][0][3][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    objStr = vbb['A'][0][0][5][0]
    objEnd = vbb['A'][0][0][6][0]
    objHide = vbb['A'][0][0][7][0]
    altered = int(vbb['A'][0][0][8][0][0])
    log = vbb['A'][0][0][9][0]
    logLen = int(vbb['A'][0][0][10][0][0])

    data = {}
    data['nFrame'] = nFrame
    data['maxObj'] = maxObj
    data['log'] = log.tolist()
    data['logLen'] = logLen
    data['altered'] = altered
    data['frames'] = defaultdict(list)

    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0:
            for id, pos, occl, lock, posv in zip(obj['id'][0],
                                                 obj['pos'][0],
                                                 obj['occl'][0],
                                                 obj['lock'][0],
                                                 obj['posv'][0]):
                keys = obj.dtype.names
                id = int(id[0][0]) - 1  # MATLAB is 1-origin
                p = pos[0].tolist()
                pos = [p[0] - 1, p[1] - 1, p[2], p[3]]  # MATLAB is 1-origin
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()

                datum = dict(zip(keys, [id, pos, occl, lock, posv]))
                datum['lbl'] = str(objLbl[datum['id']])
                # MATLAB is 1-origin
                datum['str'] = int(objStr[datum['id']]) - 1
                # MATLAB is 1-origin
                datum['end'] = int(objEnd[datum['id']]) - 1
                datum['hide'] = int(objHide[datum['id']])
                datum['init'] = int(objInit[datum['id']])

                data['frames'][frame_id].append(datum)

    return data

