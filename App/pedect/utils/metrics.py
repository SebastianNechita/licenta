def intersectionBox(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x1, y1, x2, y2


def area(box):
    if box[0] < box[2] and box[1] < box[3]:
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    return 0


def IOU(box1, box2):
    inter = area(intersectionBox(box1, box2))
    union = area(box1) + area(box2) - inter
    iou = 1.0 * inter / union
    return iou

