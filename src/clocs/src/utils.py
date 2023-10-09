import torch
from torch import Tensor
from typing import Optional
from mmcv.ops import nms_rotated

def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.
    Args:
        boxes_xywhr (torch.Tensor | np.ndarray): Rotated boxes in XYWHR format.

    Returns:
        (torch.Tensor | np.ndarray): Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[..., 2] / 2
    half_h = boxes_xywhr[..., 3] / 2

    boxes[..., 0] = boxes_xywhr[..., 0] - half_w
    boxes[..., 1] = boxes_xywhr[..., 1] - half_h
    boxes[..., 2] = boxes_xywhr[..., 0] + half_w
    boxes[..., 3] = boxes_xywhr[..., 1] + half_h
    boxes[..., 4] = boxes_xywhr[..., 4]
    return boxes

def nms_bev(boxes: Tensor,
            scores: Tensor,
            thresh: float,
            pre_max_size: Optional[int] = None,
            post_max_size: Optional[int] = None) -> Tensor:
    """NMS function GPU implementation (for BEV boxes). The overlap of two
    boxes for IoU calculation is defined as the exact overlapping area of the
    two boxes. In this function, one can also set ``pre_max_size`` and
    ``post_max_size``.

    Args:
        boxes (Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (Tensor): Scores of boxes with the shape of [N].
        thresh (float): Overlap threshold of NMS.
        pre_max_size (int, optional): Max size of boxes before NMS.
            Defaults to None.
        post_max_size (int, optional): Max size of boxes after NMS.
            Defaults to None.

    Returns:
        Tensor: Indexes after NMS.
    """
    assert boxes.size(1) == 5, 'Input boxes shape should be [N, 5]'
    order = scores.sort(0, descending=True)[1]
    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = boxes[order].contiguous()
    scores = scores[order]

    # xyxyr -> back to xywhr
    # note: better skip this step before nms_bev call in the future
    boxes = torch.stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
         boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 4]),
        dim=-1)
    keep = nms_rotated(boxes, scores, thresh)[1]
    keep = order[keep]
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep
