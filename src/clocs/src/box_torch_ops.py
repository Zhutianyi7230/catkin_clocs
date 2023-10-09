#对目标检测框进行几何变换等相关操作
import numpy as np
import torch
from torch import stack as tstack

def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2**ndim, ndim)
    return corners



def rotation_3d_in_axis(points, angles, axis=0):
    '''
    Args:
        points: [N, point_size, 3]
        angles: [N]

    Returns:
        [N, point_size, 3]
    '''
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = tstack([
            tstack([rot_cos, zeros, -rot_sin]),
            tstack([zeros, ones, zeros]),
            tstack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = tstack([
            tstack([rot_cos, -rot_sin, zeros]),
            tstack([rot_sin, rot_cos, zeros]),
            tstack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = tstack([
            tstack([zeros, rot_cos, -rot_sin]),
            tstack([zeros, rot_sin, rot_cos]),
            tstack([ones, zeros, zeros])
        ])
    else:
        raise ValueError("axis should in range")

    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack(
        [tstack([rot_cos, -rot_sin]),
         tstack([rot_sin, rot_cos])])
    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def center_to_corner_box3d(centers,
                           dims,
                           angles,
                           origin=[0.5, 1.0, 0.5],
                           axis=1):
    """convert kitti locations, dimensions and angles to corners
    根据3D框的locations, dimensions and angles计算得到8个角点的坐标
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.view(-1, 1, 3)
    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners


def project_to_image(points_3d, proj_mat):
    '''把3D坐标点投影到像素坐标系的2D坐标
    Args:
        points_3d: [N, point_size, 3]
        proj_mat: [4, 4]

    Returns:
        [N, point_size, 2]
    '''
    #print("!!!!!!!!!!this information is from project_to_image:",points_3d.type(),proj_mat.type(),points_3d.shape,proj_mat.shape)
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    points_4 = torch.cat(
        [points_3d, torch.zeros(*points_shape).type_as(points_3d)], dim=-1)
    # point_2d = points_4 @ tf.transpose(proj_mat, [1, 0])
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def camera_to_lidar(points, r_rect, velo2cam):
    '''从相机坐标系到雷达坐标系的点的坐标转化
    Args:
        points (torch tensor, shape=[N, 3]): 中心点坐标.
        r_rect (torch tensor, shape=[4, 4]): 旋转修正矩阵.
        velo2cam (torch tensor, shape=[4, 4]): 雷达坐标系到相机坐标系变换矩阵.

    Returns:
        torch tensor, shape=[N, 7]
    '''
    num_points = points.shape[0]
    points = torch.cat(
        [points, torch.ones(num_points, 1).type_as(points)], dim=-1)
    lidar_points = points @ torch.inverse((r_rect @ velo2cam).t())
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    '''从雷达坐标系到相机坐标系的点的坐标转化
    Args:
        points (torch tensor, shape=[N, 3]): 中心点坐标.
        r_rect (torch tensor, shape=[4, 4]): 旋转修正矩阵.
        velo2cam (torch tensor, shape=[4, 4]): 雷达坐标系到相机坐标系变换矩阵.

    Returns:
        torch tensor, shape=[N, 7]
    '''
    num_points = points.shape[0]
    points = torch.cat(
        [points, torch.ones(num_points, 1).type_as(points)], dim=-1)
    camera_points = points @ (r_rect @ velo2cam).t()
    return camera_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    """将3D目标检测框的中心点从相机坐标系转到雷达坐标系
    Args:
        data (torch tensor, shape=[N, 7]): 相机坐标系下3D框中心点坐标，维度，航向角.
        r_rect (torch tensor, shape=[4, 4]): 旋转修正矩阵.
        velo2cam (torch tensor, shape=[4, 4]): 雷达坐标系到相机坐标系变换矩阵.

    Returns:
        torch tensor, shape=[N, 7]
    """
    xyz = data[..., 0:3]
    l, h, w = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return torch.cat([xyz_lidar, w, l, h, r], dim=-1)


def box_lidar_to_camera(data, r_rect, velo2cam):
    """将3D目标检测框的中心点从雷达坐标系转到相机坐标系
    Args:
        data (torch tensor, shape=[N, 7]): 雷达坐标系下3D框中心点坐标，维度，航向角.
        r_rect (torch tensor, shape=[4, 4]): 旋转修正矩阵.
        velo2cam (torch tensor, shape=[4, 4]): 雷达坐标系到相机坐标系变换矩阵.

    Returns:
        torch tensor, shape=[N, 7]
    """
    xyz_lidar = data[..., 0:3]
    w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return torch.cat([xyz, l, h, w, r], dim=-1)




