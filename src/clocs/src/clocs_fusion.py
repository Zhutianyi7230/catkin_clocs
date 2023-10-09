#!/data/yum_install/usr/bin/python 
# -*- coding: utf-8 -*-
import std_msgs
import rospy
from std_msgs.msg import Float32MultiArray,Int32
import torch
import numba
import numpy as np
from torch import nn
import box_torch_ops
import onnxruntime
from utils import xywhr2xyxyr,nms_bev

device = torch.device('cpu')

@numba.jit(nopython=True,parallel=True)
def build_stage2_training(boxes, query_boxes, criterion, scores_3d, scores_2d, dis_to_lidar_3d,overlaps,tensor_index):
    '''计算2D与3D目标检测框的重合度，构建表示两者关联的稀疏张量，用于输入后融合网络
    Args:
        boxes : [N, 4]. 投影到2D的3D框. [x1,y1,x2,y2] . xy1表示左上点，xy2表示右下点
        query_boxes : [K, 4]. 2D检测框 . [x1,y1,x2,y2] . xy1表示左上点，xy2表示右下点
        criterion : 面积计算类型.默认=-1
        scores_3d : [N, 1]. 3D检测框的分数
        scores_2d : [K, 1]. 2D检测框的分数
        dis_to_lidar_3d: [N, 1]. 3D检测框中心距离lidar的水平距离
        overlaps: [K*N, 4]. 构建表示2D与3D目标检测框关联的稀疏张量
        tensor_index : [K*N, 2]. 表示overlaps中某一位置张量所对应的2D和3D检测框索引
        
    '''
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    max_num = 900000
    ind = 0
    ind_max = ind

    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))

        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))

            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))

                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih
                        )
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0

                    overlaps[ind, 0] = iw * ih / ua
                    overlaps[ind, 1] = scores_3d[n, 0]
                    overlaps[ind, 2] = scores_2d[k, 0]
                    overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
                    tensor_index[ind, 0] = k
                    tensor_index[ind, 1] = n
                    ind = ind + 1

                elif k == K - 1:
                    overlaps[ind, 0] = -10
                    overlaps[ind, 1] = scores_3d[n, 0]
                    overlaps[ind, 2] = -10
                    overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
                    tensor_index[ind, 0] = k
                    tensor_index[ind, 1] = n
                    ind = ind + 1

            elif k == K - 1:
                overlaps[ind, 0] = -10
                overlaps[ind, 1] = scores_3d[n, 0]
                overlaps[ind, 2] = -10
                overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
                tensor_index[ind, 0] = k
                tensor_index[ind, 1] = n
                ind = ind + 1

    if ind > ind_max:
        ind_max = ind

    # return overlaps[:ind], tensor_index[:ind], ind
    return overlaps, tensor_index, ind

class CLOCs_Node(nn.Module):
    def __init__(self,
                 num_class=1
                 ):
        super().__init__()
        rospy.init_node('clocs_fusion', anonymous=True)
        self.num_class = num_class
        self._multiclass_nms = (self.num_class>1)
        self._nms_score_threshold = 0.2
        self._use_rotate_nms = True
        self.box_code_size = 7
        self._nms_pre_max_size = 1000
        self._nms_post_max_size = 50
        self._nms_iou_threshold = torch.tensor([0.0099],device=device)
        

        self.det2d_topic_in = rospy.get_param('~det2d_topic_in')
        self.det3d1_topic_in = rospy.get_param('~det3d1_topic_in')
        self.det3d2_topic_in = rospy.get_param('~det3d2_topic_in')
        self.rect_topic_in = rospy.get_param('~rect_topic_in')
        self.Trv2c_topic_in = rospy.get_param('~Trv2c_topic_in')
        self.P2_topic_in = rospy.get_param('~P2_topic_in')
        
        # 初始化数据存储变量
        self.det2d_data = None
        self.det3d_data1 = None
        self.det3d_data2 = None
        self.rect_data = None
        self.Trv2c_data = None
        self.P2_data = None
        

        # 订阅多个话题
        rospy.Subscriber(self.det2d_topic_in, Float32MultiArray, self.callback_topic1)
        rospy.Subscriber(self.det3d1_topic_in, Float32MultiArray, self.callback_topic2)
        rospy.Subscriber(self.det3d2_topic_in, Float32MultiArray, self.callback_topic3)
        rospy.Subscriber(self.rect_topic_in, Float32MultiArray, self.callback_topic4)
        rospy.Subscriber(self.Trv2c_topic_in, Float32MultiArray, self.callback_topic5)
        rospy.Subscriber(self.P2_topic_in, Float32MultiArray, self.callback_topic6)

        self.pub1 = rospy.Publisher('num_object', Int32, queue_size=10)
        self.pub2 = rospy.Publisher('predictions_result', Float32MultiArray, queue_size=10)
        self.sess = onnxruntime.InferenceSession("/media/zhutianyi/KESU/project/catkin_clocs/src/clocs/fusion_model_3_sim.onnx",providers=onnxruntime.get_available_providers())

    def callback_topic1(self,data):
        rospy.loginfo("Recieve det2d_topic.")
        self.det2d_data = data.data
        self.process_data()
        
    def callback_topic2(self,data):
        rospy.loginfo("Recieve det3d1_topic.")
        self.det3d_data1 = data.data
        self.process_data()
        
    def callback_topic3(self,data):
        rospy.loginfo("Recieve det3d2_topic.")
        self.det3d_data2 = data.data
        self.process_data()

    def callback_topic4(self,data):
        rospy.loginfo("Recieve rect_topic.")
        self.rect_data = data.data
        self.process_data()

    def callback_topic5(self,data):
        rospy.loginfo("Recieve Trv2c_topic.")
        self.Trv2c_data = data.data
        self.process_data()

    def callback_topic6(self,data):
        rospy.loginfo("Recieve P2_topic.")
        self.P2_data = data.data
        self.process_data()
        
    def process_data(self):
        if self.det2d_data and self.det3d_data1 and self.det3d_data2 and self.rect_data and self.Trv2c_data and self.P2_data:
            top_predictions = torch.FloatTensor(self.det2d_data).reshape([-1,6]).to(device)
            
            if self.num_class==1 :
                box_preds_car = torch.FloatTensor(self.det3d_data1).reshape([10000,7]).to(device)
                cls_preds_car = torch.FloatTensor(self.det3d_data2).reshape([10000,1]).to(device)
            else:
                box_preds = torch.FloatTensor(self.det3d_data1).reshape([20000,7]).to(device)
                cls_preds = torch.FloatTensor(self.det3d_data2).reshape([20000,3]).to(device)
                
                top_scores, top_labels = torch.max(cls_preds, dim=-1)
                num_cars = torch.where(top_labels==2)[0].shape[0]
                if num_cars>10000:
                    car_indices = torch.where(top_labels==2)[0][:10000]#因为top_scores本身就是降序的，所以前10000个也是分数最高的10000个
                else:
                    car_indices = torch.where(top_labels==2)[0][:num_cars]
                cls_preds_3class = cls_preds[car_indices]#[m,3]
                cls_preds_car = top_scores[car_indices].unsqueeze(-1)#[m,1]
                box_preds_car = box_preds[car_indices,:]#[m,7]

                cls_preds_3d_nocar = cls_preds[top_labels!=2]#torch.Size([5548, 3])
                box_preds_3d_nocar = box_preds[top_labels!=2]#torch.Size([5548, 7])

            rect = torch.FloatTensor(self.rect_data).reshape(4,4).to(device)
            Trv2c = torch.FloatTensor(self.Trv2c_data).reshape(4,4).to(device)
            P2 = torch.FloatTensor(self.P2_data).reshape(4,4).to(device)
 
            example = dict()
            example['rect'] = rect
            example['Trv2c'] = Trv2c
            example['P2'] = P2
            example['image_shape'] = torch.tensor([375, 1242], dtype=torch.int32)
            
            #clocs前处理
            predictions_dict, non_empty_iou_test_tensor, non_empty_tensor_index_tensor = \
            self.clocs(top_predictions, box_preds_car, cls_preds_car, example)
            
            #清零
            self.det2d_data = None
            self.det3d_data1 = None
            self.det3d_data2 = None
            self.rect_data = None
            self.Trv2c_data = None
            self.P2_data = None
            
            fusion_input = {
                'non_empty_iou_test_tensor':non_empty_iou_test_tensor.detach().cpu().numpy(),#torch.Size([1, 4, 1, N])
                'non_empty_tensor_index_tensor':non_empty_tensor_index_tensor.detach().cpu().numpy()#torch.Size([N, 2])
            }
            
            #获取fusion网络onnx的输入输出名
            input_name0 = self.sess.get_inputs()[0].name
            input_name1 = self.sess.get_inputs()[1].name
            input_name2 = self.sess.get_inputs()[2].name
            output_name0 = self.sess.get_outputs()[0].name
            
            fusion_input['sparse_tensor'] = np.zeros([1,200,10000]).astype(np.float32)
            fusion_input['sparse_tensor'][:,:,:] = -999.0
            ##后融合fusion onnx推理
            pred_onx = self.sess.run([output_name0], {input_name0:fusion_input['non_empty_iou_test_tensor'],input_name1:fusion_input['non_empty_tensor_index_tensor'],input_name2:fusion_input['sparse_tensor']})
            #[1,10000,1]
            if self.num_class==1:
                cls_preds = torch.from_numpy(pred_onx[0]).to(device)
                cls_preds = cls_preds.sigmoid().squeeze(0)#[10000,1]
                box_preds = box_preds_car
            else:
                if num_cars>10000:
                    pred_onx = torch.from_numpy(pred_onx[0]).to(device)
                else:
                    pred_onx = torch.from_numpy(pred_onx[0]).to(device)[:,:num_cars,:]
                pred_onx = pred_onx.sigmoid().squeeze(0)#
                cls_preds_3class[:,2] = pred_onx[:,0]

                cls_preds = torch.concat([cls_preds_3d_nocar,cls_preds_3class],dim=0)
                box_preds = torch.concat([box_preds_3d_nocar,box_preds_car],dim=0)
                
            #后处理
            postprocess_input = {
                'box_preds':box_preds.to(device),
                'cls_preds':cls_preds
            }  
            predictions_dict = self.postprocess(postprocess_input,example)
            
            data = Float32MultiArray()
            data.data.extend(predictions_dict['bbox'].flatten().tolist())
            data.data.extend(predictions_dict['box3d_camera'].flatten().tolist())
            data.data.extend(predictions_dict['box3d_lidar'].flatten().tolist())
            data.data.extend(predictions_dict['scores'].flatten().tolist())
            data.data.extend(predictions_dict['label_preds'].flatten().tolist())
            
            num_object = Int32()
            num_object.data = predictions_dict['bbox'].shape[0]
            self.pub1.publish(num_object)
            self.pub2.publish(data)
            rospy.loginfo("Publishing prediction result.")
            
        else:
            return 

    def postprocess(self,preds_dict,example):
        box_preds = preds_dict['box_preds'].float()
        cls_preds = preds_dict['cls_preds'].float()    
        rect = example['rect'].float()
        Trv2c = example['Trv2c'].float()
        P2 = example['P2'].float()     
        total_scores = cls_preds
        
        nms_func = nms_bev#mmdet3d的函数
        
        if self._multiclass_nms:#多类别的情况
            bboxes = []
            scores = []
            labels = []
            for i in range(self.num_class):
                #先进行阈值筛选
                cls_inds = total_scores[:, i] > self._nms_score_threshold
                if not cls_inds.any():
                    continue
                
                _scores = total_scores[cls_inds, i]
                box_preds_class = box_preds[cls_inds, :]
                boxes_for_nms = box_preds_class[:, [0, 1, 3, 4, 6]]
                #Convert a rotated boxes in XYWHR format to XYXYR format.
                boxes_for_nms_xyxyr = xywhr2xyxyr(boxes_for_nms)
                #Apply NMS
                selected = nms_func(
                    boxes_for_nms_xyxyr, #torch.Size([N, 5]),XYXYR format
                    _scores, #torch.Size([N])
                    thresh=self._nms_iou_threshold,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                )
                cls_label = box_preds.new_full((len(selected), ),
                                        i,
                                        dtype=torch.long)
                if box_preds_class[selected].shape[0] != 0:##
                    bboxes.append(box_preds_class[selected])
                    scores.append(_scores[selected])
                    labels.append(cls_label)
                
            ###
            if len(bboxes)!=0:
                bboxes = torch.cat(bboxes, dim=0)
                scores = torch.cat(scores, dim=0)
                labels = torch.cat(labels, dim=0)
                selected_boxes = bboxes
                selected_scores = scores
                selected_labels = labels
            else:
                selected_boxes = torch.Tensor(bboxes)
                selected_scores = torch.Tensor(scores)
                selected_labels = torch.Tensor(labels)
            
        else:##单类别的情况
            top_scores = total_scores.squeeze(-1)
            top_labels = torch.zeros(
                total_scores.shape[0],
                device=total_scores.device,
                dtype=torch.long)
            #先进行阈值筛选
            if self._nms_score_threshold > 0.0:
                thresh = torch.tensor(
                    [self._nms_score_threshold],
                    device=total_scores.device).type_as(total_scores)
                top_scores_keep = (top_scores >= thresh)
                top_scores = top_scores.masked_select(top_scores_keep)
                if top_scores.shape[0] != 0:
                    if self._nms_score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]

                    #Convert a rotated boxes in XYWHR format to XYXYR format.
                    boxes_for_nms_xyxyr = xywhr2xyxyr(boxes_for_nms)#######
                    #Apply NMS.
                    selected = nms_func(
                        boxes_for_nms_xyxyr, #torch.Size([10000, 5]),XYXYR format
                        top_scores, #torch.Size([10000])
                        thresh=self._nms_iou_threshold,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                    )
                else:
                    selected = []
        
                selected_boxes = box_preds[selected]#torch.Size([N, N])
                selected_scores = top_scores[selected]#torch.Size([N])
                selected_labels = box_preds.new_full((len(selected), ),
                                        0,
                                        dtype=torch.long)
                
        # finally generate predictions.
        if selected_boxes.shape[0] != 0:
            box_preds = selected_boxes
            scores = selected_scores
            labels = selected_labels

            final_box_preds = box_preds
            final_scores = scores
            final_labels = labels

            final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                final_box_preds, rect, Trv2c)
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            #获得3D检测框的各个角点
            box_corners = box_torch_ops.center_to_corner_box3d(
                locs, dims, angles, camera_box_origin, axis=1)
            #把角点投影到2D像素坐标系
            box_corners_in_image = box_torch_ops.project_to_image(
                box_corners, P2)
            # box_corners_in_image: [N, 8, 2]
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            # predictions
            predictions_dict = {
                "bbox": box_2d_preds.detach().cpu().numpy(),
                "box3d_camera": final_box_preds_camera.detach().cpu().numpy(),
                "box3d_lidar": final_box_preds.detach().cpu().numpy(),
                "scores": final_scores.detach().cpu().numpy(),
                "label_preds": final_labels.detach().cpu().numpy(),
            }
        else:
            predictions_dict = {
                "bbox": np.zeros([0, 4], dtype=np.float32),
                "box3d_camera": np.zeros([0, 7], dtype=np.float32),
                "box3d_lidar": np.zeros([0, 7], dtype=np.float32),
                "scores": np.zeros([0], dtype=np.float32),
                "label_preds": np.zeros([0, 4], dtype=np.float32)
            }
        return predictions_dict

    def clocs(self,top_predictions, box_preds, cls_preds, example):
        final_box_preds = box_preds.float()
        final_scores = cls_preds.float()
        rect = example['rect'].float()
        Trv2c = example['Trv2c'].float()
        P2 = example['P2'].float()
        
        #torch.Size([10000, 7])转坐标系
        final_box_preds_camera = box_torch_ops.box_lidar_to_camera(final_box_preds, rect, Trv2c)
        locs = final_box_preds_camera[:, :3]
        dims = final_box_preds_camera[:, 3:6]
        angles = final_box_preds_camera[:, 6]
        camera_box_origin = [0.5, 1.0, 0.5]
        box_corners = box_torch_ops.center_to_corner_box3d(
            locs, dims, angles, camera_box_origin, axis=1)#torch.Size([10000, 8, 3])

        box_corners_in_image = box_torch_ops.project_to_image(
            box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        img_height = example['image_shape'][0]
        img_width = example['image_shape'][1]
        minxy[:,0] = torch.clamp(minxy[:,0],min = 0,max = img_width)
        minxy[:,1] = torch.clamp(minxy[:,1],min = 0,max = img_height)
        maxxy[:,0] = torch.clamp(maxxy[:,0],min = 0,max = img_width)
        maxxy[:,1] = torch.clamp(maxxy[:,1],min = 0,max = img_height)
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)#torch.Size([10000, 4]) 3D框投影到2D的结果
        
        # predictions
        predictions_dict = {
            "bbox": box_2d_preds, # 3D预测框投影到2D的结果
            "box3d_camera": final_box_preds_camera, # 3D预测框投影到相机坐标的结果
            "box3d_lidar": final_box_preds,
            "scores": final_scores
        } 

        #math.aqrt(43.52*43.52+25.6*25.6) = 50.49
        dis_to_lidar = torch.norm(box_preds[:,:2],p=2,dim=1,keepdim=True)/50.49
        box_2d_detector = top_predictions[:,:4]#(K,4)
        box_2d_scores = top_predictions[:,4].reshape(-1,1)#(K,1)
        
        K = box_2d_detector.shape[0] # K=4
        N = box_2d_preds.shape[0] #N = 10000
        overlaps1 = np.zeros((K*N,4),dtype=box_2d_preds.detach().cpu().numpy().dtype)
        tensor_index1 = np.zeros((K*N,2),dtype=box_2d_preds.detach().cpu().numpy().dtype)
        overlaps1[:,:] = -1
        tensor_index1[:,:] = -1
        
        # 构建输入后融合网络的稀疏张量
        iou_test,tensor_index, max_num = build_stage2_training(box_2d_preds.detach().cpu().numpy(),
                                            box_2d_detector.detach().cpu().numpy(),
                                            -1,
                                            final_scores.detach().cpu().numpy(),
                                            box_2d_scores.detach().cpu().numpy(),
                                            dis_to_lidar.detach().cpu().numpy(),
                                            overlaps1,
                                            tensor_index1)
        
        iou_test_tensor = torch.FloatTensor(iou_test)  
        tensor_index_tensor = torch.tensor(tensor_index, dtype=torch.long)
        iou_test_tensor = iou_test_tensor.permute(1,0)
        iou_test_tensor = iou_test_tensor.reshape(1,4,1,-1)
        tensor_index_tensor = tensor_index_tensor.reshape(-1,2)
        
        if max_num == 0:
            non_empty_iou_test_tensor = torch.zeros(1,4,1,2)
            non_empty_iou_test_tensor[:,:,:,:] = -1
            non_empty_tensor_index_tensor = torch.zeros(2,2)
            non_empty_tensor_index_tensor[:,:] = -1
        else:
            non_empty_iou_test_tensor = iou_test_tensor[:,:,:,:max_num]
            non_empty_tensor_index_tensor = tensor_index_tensor[:max_num,:]
        
        print(non_empty_iou_test_tensor.shape)
        print(non_empty_tensor_index_tensor.shape)
        return predictions_dict, non_empty_iou_test_tensor, non_empty_tensor_index_tensor

    def run(self):
        rospy.spin()
    
    
    
if __name__=="__main__":

    node = CLOCs_Node(num_class=3)
    rospy.loginfo("Complete initialization.")
    node.run()
        
    
