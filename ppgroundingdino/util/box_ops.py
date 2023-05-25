"""
Utilities for bounding box manipulation and GIoU.
"""
import paddle


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(axis=-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return paddle.stack(x=b, axis=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(axis=-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return paddle.stack(x=b, axis=-1)


def box_xywh_to_xyxy(boxes):
    x, y, w, h = boxes.unbind(-1)
    boxes = paddle.stack([x, y, x + w, y + h], axis=-1)
    return boxes


def box_xyxy_to_xywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1  # x2 - x1
    h = y2 - y1  # y2 - y1
    boxes = paddle.stack((x1, y1, w, h), axis=-1)
    return boxes


def box_convert(boxes, in_fmt, out_fmt):
    allowed_fmts = ("xyxy", "xywh", "cxcywh")
    if in_fmt not in allowed_fmts or out_fmt not in allowed_fmts:
        raise ValueError("Unsupported Bounding Box Conversions for given in_fmt and out_fmt")

    if in_fmt == out_fmt:
        return boxes.clone()

    if in_fmt != 'xyxy' and out_fmt != 'xyxy':
        # convert to xyxy and change in_fmt xyxy
        if in_fmt == "xywh":
            boxes = box_xywh_to_xyxy(boxes)
        elif in_fmt == "cxcywh":
            boxes = box_cxcywh_to_xyxy(boxes)
        in_fmt = 'xyxy'

    if in_fmt == "xyxy":
        if out_fmt == "xywh":
            boxes = box_xyxy_to_xywh(boxes)
        elif out_fmt == "cxcywh":
            boxes = box_xyxy_to_cxcywh(boxes)
    elif out_fmt == "xyxy":
        if in_fmt == "xywh":
            boxes = box_xywh_to_xyxy(boxes)
        elif in_fmt == "cxcywh":
            boxes = box_cxcywh_to_xyxy(boxes)
    return boxes



def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = paddle.maximum(boxes1[:, (None), :2], boxes2[:, :2])
    rb = paddle.minimum(boxes1[:, (None), 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, (None)] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = paddle.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = paddle.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / (area + 1e-06)


def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = paddle.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = paddle.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2)
    lt = paddle.minimum(boxes1[:, :2], boxes2[:, :2])
    rb = paddle.maximum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(min=0)
    area = wh[:, 0] * wh[:, 1]
    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return paddle.zeros(shape=(0, 4))
    h, w = masks.shape[-2:]
    y = paddle.arange(start=0, end=h).astype('float32')
    x = paddle.arange(start=0, end=w).astype('float32')
    y, x = paddle.meshgrid(y, x)
    x_mask = masks * x.unsqueeze(axis=0)
    x_max = x_mask.flatten(start_axis=1).max(axis=-1)
    x_min = paddle.where(~masks.astype(dtype='bool'), x_mask, 100000000.0
        ).flatten(start_axis=1).min(axis=-1)
    y_mask = masks * y.unsqueeze(axis=0)
    y_max = y_mask.flatten(start_axis=1).max(axis=-1)
    y_min = paddle.where(~masks.astype(dtype='bool'), y_mask, 100000000.0
        ).flatten(start_axis=1).min(axis=-1)
    return paddle.stack(x=[x_min, y_min, x_max, y_max], axis=1)


if __name__ == '__main__':
    x = paddle.rand(shape=[5, 4])
    y = paddle.rand(shape=[3, 4])
    iou, union = box_iou(x, y)
    import ipdb
    ipdb.set_trace()
