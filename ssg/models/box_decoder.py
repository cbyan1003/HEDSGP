import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import TypeVar
from typing import Type
from ssg.models.confusion_memory_block import Confusion_Memory_Block
from torch.multiprocessing import Pool
from typing import Optional
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.interpolate import make_interp_spline

TBoxTensor = TypeVar("TBoxTensor", bound="BoxTensor")

def _box_shape_ok(t: Tensor) -> bool:
  if len(t.shape) < 2:
    return False
  else:
    if t.size(-2) != 2:
      return False

    return True


def _shape_error_str(tensor_name, expected_shape, actual_shape):
  return "Shape of {} has to be {} but is {}".format(tensor_name,
                             expected_shape,
                             tuple(actual_shape))


class BoxTensor(object):
  """ A wrapper to which contains single tensor which
  represents single or multiple boxes.

  Have to use composition instead of inheritance because
  it is not safe to interit from :class:`torch.Tensor` because
  creating an instance of such a class will always make it a leaf node.
  This works for :class:`torch.nn.Parameter` but won't work for a general
  box_tensor.
  """

  def __init__(self, data: Tensor) -> None:
    """
    Arguments:
      data: Tensor of shape (**, zZ, num_dims). Here, zZ=2, where
        the 0th dim is for bottom left corner and 1st dim is for
        top right corner of the box
    """

    if _box_shape_ok(data):
      self.data = data
    else:
      raise ValueError(_shape_error_str('data', '(**,2,num_dims)', data.shape))
    super().__init__()
    

  def __repr__(self):
    return 'box_tensor_wrapper(' + self.data.__repr__() + ')'

  @property
  def min_offset(self) -> Tensor:
    """Lower left coordinate as Tensor"""

    return self.data[..., 0, :]

  @property
  def max_offset(self) -> Tensor:
    """Top right coordinate as Tensor"""
    return self.data[..., 1, :]

  @property
  def center(self) -> Tensor:
    """Centre coordinate as Tensor"""

    return (self.min_offset + self.max_offset)/2

  @classmethod
  def from_zZ(cls: Type[TBoxTensor], min_offset: Tensor, max_offset: Tensor) -> TBoxTensor:
    """
    Creates a box by stacking z and Z along -2 dim.
    That is if z.shape == Z.shape == (**, num_dim),
    then the result would be box of shape (**, 2, num_dim)
    """

    if min_offset.shape != max_offset.shape:
      raise ValueError(
        "Shape of z and Z should be same but is {} and {}".format(
          min_offset.shape, max_offset.shape))
    min_offset.to('cuda')
    max_offset.to('cuda')
    box_val: Tensor = torch.stack((min_offset, max_offset), -2)

    return cls(box_val)

  @classmethod
  def from_split(cls: Type[TBoxTensor], t: Tensor,
           dim: int = -1) -> TBoxTensor:
    """Creates a BoxTensor by splitting on the dimension dim at midpoint

    Args:
      t: input
      dim: dimension to split on

    Returns:
      BoxTensor: output BoxTensor

    Raises:
      ValueError: `dim` has to be even
    """
    len_dim = t.size(dim)

    if len_dim % 2 != 0:
      raise ValueError(
        "dim has to be even to split on it but is {}".format(
          t.size(dim)))
    split_point = int(len_dim / 2)
    t.to('cuda')
    min_offset = t.index_select(dim, torch.tensor(list(range(split_point)), dtype=torch.int64, device=t.device))

    max_offset = t.index_select(dim,torch.tensor(list(range(split_point, len_dim)), dtype=torch.int64, device=t.device))

    return cls.from_zZ(min_offset, max_offset)
  
  
    
class CenterSigmoidBoxTensor(BoxTensor):
  
  @property
  def center(self) -> Tensor:
    return (self.min_offset + self.max_offset)/2

  @property
  def min_offset(self) -> Tensor:
    min_offset = self.data[..., 0, :] \
      - torch.nn.functional.softplus(self.data[..., 1, :], beta=10.)
    return torch.sigmoid(min_offset)

  @property
  def max_offset(self) -> Tensor:
    max_offset = self.data[..., 0, :] \
      + torch.nn.functional.softplus(self.data[..., 1, :], beta=10.)
    return torch.sigmoid(max_offset)
  
  @classmethod
  def gumbel_intersection_box(self, box1, box2, gumbel_beta):
    intersections_min = gumbel_beta * torch.logsumexp(
            torch.stack((box1.min_offset / gumbel_beta, box2.min_offset / gumbel_beta)),
            0
    )
    intersections_min = torch.max(
        intersections_min,
        torch.max(box1.min_offset, box2.min_offset)
    )
    intersections_max = - gumbel_beta * torch.logsumexp(
        torch.stack((-box1.max_offset / gumbel_beta, -box2.max_offset / gumbel_beta)),
        0
    )
    intersections_max = torch.min(
        intersections_max,
        torch.min(box1.max_offset, box2.max_offset)
    )
    
    return intersections_min, intersections_max
  
  def test(inputs):
      box1, box2, gumbel_beta = inputs
      intersections_min = gumbel_beta * torch.logsumexp(
              torch.stack((box1.min_offset / gumbel_beta, box2.min_offset / gumbel_beta)),
              0
      )
      intersections_min = torch.max(
          intersections_min,
          torch.max(box1.min_offset, box2.min_offset)
      )
      intersections_max = - gumbel_beta * torch.logsumexp(
          torch.stack((-box1.max_offset / gumbel_beta, -box2.max_offset / gumbel_beta)),
          0
      )
      intersections_max = torch.min(
          intersections_max,
          torch.min(box1.max_offset, box2.max_offset)
      )
      
      return (intersections_min, intersections_max)
  
  @classmethod
  def gumbel_intersection_mutli_box(self, box1s, box2s, gumbel_beta):
    '''
    for boxs classifier
    wanna compare every box1 with each box2
    box1: type class
    box2: node
    output: box1_0 <-> box2_0, box1_1 <-> box2_0, ... , box1_n <-> box2_0, 
            box1_0 <-> box2_1, box1_1 <-> box2_1, ... , box1_n <-> box2_1, 
            box1_0 <-> box2_2, ...
            box1_0 <-> box2_n, box1_1 <-> box2_n, ... , box1_n <-> box2_n, 
    '''
    
    box1s_size = box1s.data.shape[0]
    if box2s.data.ndim == 2:
      box2s_size = 1
      box2s.data = box2s.data.view(1,2,256)
    else:
      box2s_size = box2s.data.shape[0]
    
    for i in range(box2s_size):
      for j in range(box1s_size):
        intersections_min, intersections_max = self.gumbel_intersection_box(BoxTensor.from_zZ(box2s.min_offset[i],box2s.max_offset[i]), BoxTensor.from_zZ(box1s.min_offset[j],box1s.max_offset[j]), gumbel_beta)
        if i == 0 and j == 0:
          intersections_mins = intersections_min.unsqueeze(0)
          intersections_maxs = intersections_max.unsqueeze(0)
        else:
          intersections_mins = torch.cat((intersections_mins, intersections_min.unsqueeze(0)), 0)
          intersections_maxs = torch.cat((intersections_maxs, intersections_max.unsqueeze(0)), 0)
    return intersections_mins, intersections_maxs
  
  @classmethod
  def gumbel_intersection_relation_box(self, front_boxs, back_boxs, gumbel_beta, gt_node, REL_SIZE, BOX_DIM, node_dict):
    '''
    for relation classifier
    input: 
          front_boxs:  F_r1(Box_1), F_r2(Box_1), ... F_r7(Box_1),      F_r1(Box_2), F_r2(Box_2), ... F_r7(Box_2), ....   
          back_boxs:   B_r1(Box_1), B_r2(Box_1), ... B_r7(Box_1),      B_r1(Box_2), B_r2(Box_2), ... B_r7(Box_2), ....   
    compare F_r1_[] with B_r1_[], only compare same position
    '''
    boxs_size = front_boxs.data.shape[0]
    front_boxs_data = front_boxs.data.view(REL_SIZE, boxs_size // REL_SIZE, 2, BOX_DIM)
    back_boxs_data = back_boxs.data.view(REL_SIZE, boxs_size // REL_SIZE, 2, BOX_DIM)
    gt_boxs = []
    front_box_pred_cls_boxs = []
    back_boxs_pred_cls_boxs = []
    for i in range(gt_node.shape[0]):
      gt_box = gt_node[i].repeat(REL_SIZE,1).squeeze(1)
      front_box_pred_cl = node_dict[i]['node_cls_pred_id'].repeat(REL_SIZE,1).squeeze(1)
      back_boxs_pred_cl = node_dict[i]['node_cls_pred_id'].repeat(REL_SIZE,1).squeeze(1)
      gt_boxs.append(gt_box)
      front_box_pred_cls_boxs.append(front_box_pred_cl)
      back_boxs_pred_cls_boxs.append(back_boxs_pred_cl)
    gt_boxs = torch.cat(gt_boxs, dim=0)
    front_box_pred_cls_boxs = torch.cat(front_box_pred_cls_boxs, dim=0)
    back_boxs_pred_cls_boxs = torch.cat(back_boxs_pred_cls_boxs, dim=0)
    
    rel_ids = torch.tensor(list(range(REL_SIZE))).cuda()
    
    for k in rel_ids: # control relation type
      for i in range(boxs_size // REL_SIZE): # control front box
        for j in range(boxs_size // REL_SIZE): # control back box
          if i != j:
            intersections_min, intersections_max = self.gumbel_intersection_box(CenterSigmoidBoxTensor(front_boxs.data[REL_SIZE * i + k]), CenterSigmoidBoxTensor(back_boxs.data[REL_SIZE * j + k]), gumbel_beta)
            front_gt_value = gt_boxs[REL_SIZE * i]
            back_gt_value = gt_boxs[REL_SIZE * j]
            front_box_pred_cl = front_box_pred_cls_boxs[REL_SIZE * i]
            back_boxs_pred_cl = back_boxs_pred_cls_boxs[REL_SIZE * j]
            rel_id = k
            back_intersections_min_offset = back_boxs.min_offset[REL_SIZE * j + k]
            back_intersections_max_offset = back_boxs.max_offset[REL_SIZE * j + k]
          else:
            continue
          if i == 0 and j == 1 and k.item() == 0:
            intersections_mins = intersections_min.unsqueeze(0)
            intersections_maxs = intersections_max.unsqueeze(0)
            front_gt_values = front_gt_value.unsqueeze(0)
            back_gt_values = back_gt_value.unsqueeze(0)
            front_box_pred_cls = front_box_pred_cl.unsqueeze(0)
            back_boxs_pred_cls = back_boxs_pred_cl.unsqueeze(0)
            rel_ids1 = rel_id.view(1)
            back_intersections_min_offsets = back_intersections_min_offset.unsqueeze(0)
            back_intersections_max_offsets = back_intersections_max_offset.unsqueeze(0)
          else:
            intersections_mins = torch.cat((intersections_mins, intersections_min.unsqueeze(0)), 0)
            intersections_maxs = torch.cat((intersections_maxs, intersections_max.unsqueeze(0)), 0)
            front_gt_values = torch.cat((front_gt_values, front_gt_value.unsqueeze(0)), 0)
            back_gt_values = torch.cat((back_gt_values, back_gt_value.unsqueeze(0)), 0)
            front_box_pred_cls = torch.cat((front_box_pred_cls, front_box_pred_cl.unsqueeze(0)), 0)
            back_boxs_pred_cls = torch.cat((back_boxs_pred_cls, back_boxs_pred_cl.unsqueeze(0)), 0)
            rel_ids1 = torch.cat((rel_ids1, rel_id.view(1)), 0)
            back_intersections_min_offsets = torch.cat((back_intersections_min_offsets, back_intersections_min_offset.unsqueeze(0)), 0)
            back_intersections_max_offsets = torch.cat((back_intersections_max_offsets, back_intersections_max_offset.unsqueeze(0)), 0)
            
    return intersections_mins, intersections_maxs, front_gt_values, back_gt_values, rel_ids1, back_intersections_min_offsets, back_intersections_max_offsets, front_box_pred_cls, back_boxs_pred_cls
  
    for i in range(boxs_size):
      intersections_min, intersections_max = self.gumbel_intersection_box(CenterSigmoidBoxTensor(front_boxs.data[i]), CenterSigmoidBoxTensor(back_boxs.data[i]), gumbel_beta)
      if i == 0:
        intersections_mins = intersections_min.unsqueeze(0)
        intersections_maxs = intersections_max.unsqueeze(0)
      else:
        intersections_mins = torch.cat((intersections_mins, intersections_min.unsqueeze(0)), 0)
        intersections_maxs = torch.cat((intersections_maxs, intersections_max.unsqueeze(0)), 0)
    return intersections_mins, intersections_maxs
  
  @classmethod
  def log_soft_volume(self, min_offset, max_offset, euler_gamma, temp: float = 1.,scale: float = 1.,gumbel_beta: float = 0.):
    eps = torch.finfo(min_offset.dtype).tiny  # type: ignore

    if isinstance(scale, float):
      s = torch.tensor(scale)
    else:
      s = scale
    if gumbel_beta <= 0.:
      log_vol = torch.sum(torch.log(torch.nn.functional.softplus(max_offset - min_offset, beta=temp).clamp_min(eps)), dim=-1) + torch.exp(s)
      return log_vol              # need this eps to that the derivative of log does not blow
    else:
      log_vol = torch.sum(torch.log(torch.nn.functional.softplus(max_offset - min_offset - 2 * euler_gamma * gumbel_beta, beta=temp).clamp_min(eps)), dim=-1) + torch.exp(s)
      return log_vol
    
  @classmethod
  def log_soft_mutli_volume(self, min_offsets, max_offsets, euler_gamma, temp: float = 1.,scale: float = 1.,gumbel_beta: float = 0.):
    box_num = min_offsets.shape[0]
    for i in range(box_num):
      log_vol = self.log_soft_volume(min_offsets[i], max_offsets[i], euler_gamma, temp, scale, gumbel_beta)
      if i == 0:
        log_vols = log_vol.unsqueeze(0)
      else:
        log_vols = torch.cat((log_vols, log_vol.unsqueeze(0)), 0)
    return log_vols

class Type_box(nn.Module):
  def __init__(self,
               num_embeddings: int,
               embedding_dim: int):
    super(Type_box, self).__init__()
    self.num_embeddings = num_embeddings
    self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim * 2)
  def forward(self):
    type_input = self.embedding_layer(torch.arange(0,
                            self.num_embeddings,
                            dtype=torch.int64,
                            device="cuda"))
    type_box = CenterSigmoidBoxTensor.from_split(type_input)
    return type_box  

class BoxDecoder(nn.Module):
  def __init__(self, input_dim, output_dim, alpha = 0.1):
      super(BoxDecoder, self).__init__()
      self.input_dim = input_dim
      self.output_dim = output_dim
      self.fc = nn.Linear(input_dim, 2 * output_dim)
      self.activation = nn.LeakyReLU(alpha)

  def forward(self, x):
      x = self.fc(x)
      x = self.activation(x)
      
      left_bottom = x[:, :self.output_dim]
      right_top = x[:, self.output_dim:]

      box = BoxTensor.from_zZ(left_bottom, right_top)
      return box

class Box_Classifier(nn.Module):
  def __init__(self,
               num_embeddings: int,
               box_dim: int,
               inv_softplus_temp: float = 1.,
               softplus_scale: float = 1.,
               gumbel_beta: float = 1.0):
    super(Box_Classifier, self).__init__()
    self.num_embeddings = num_embeddings
    self.box_embedding_dim = box_dim
    self.box_embeddings = nn.Embedding(num_embeddings, box_dim * 2).to("cuda")
    self.inv_softplus_temp = inv_softplus_temp
    self.softplus_scale = softplus_scale
    self.gumbel_beta = gumbel_beta
    self.euler_gamma = 0.57721566490153286060
    self.fc = nn.Linear(num_embeddings, num_embeddings)
    self.dropout = nn.Dropout(p=0.1)

    # self.type_box = CenterSigmoidBoxTensor.from_split(self.type_input)
    mean = 0.0
    std_dev = 1.0
    initial_values = torch.normal(mean, std_dev, size=(self.num_embeddings, self.box_embedding_dim * 2))
    self.type_box_input = nn.Parameter(initial_values)
            
  def new_node_classifer(self, type_box, box, gumbel_beta, euler_gamma, temp, scale, gt_node):
    type_box_size = type_box.data.shape[0]
    if box.data.ndim == 2:
      box_size = 1
      box.data = box.data.view(1,2,256)
    else:
      box_size = box.data.shape[0]
    node_dict = []
    all_obj_prob_list = []
    
    for i in range(box_size):
      obj_log_prob_list = []
      for j in range(type_box_size):
        intersections_min, intersections_max = CenterSigmoidBoxTensor.gumbel_intersection_box(BoxTensor.from_zZ(box.min_offset[i], box.max_offset[i]), BoxTensor.from_zZ(type_box.min_offset[j], type_box.max_offset[j]), gumbel_beta)
        inter_vol = CenterSigmoidBoxTensor.log_soft_volume(intersections_min, intersections_max, euler_gamma, temp, scale, gumbel_beta)
        type_box_vol = CenterSigmoidBoxTensor.log_soft_volume(type_box.min_offset[j], type_box.max_offset[j], euler_gamma, temp, scale, gumbel_beta)
        obj_log_prob = inter_vol - type_box_vol
        obj_log_prob_list.append(obj_log_prob)
        
        # with torch.no_grad():
        #   value1 = list(range(256))
        #   lower_bounds1 = box.min_offset[i].cpu()
        #   upper_bounds1 = box.max_offset[i].cpu()
        #   heights1 = np.array(upper_bounds1) - np.array(lower_bounds1)
        #   bottoms1 = lower_bounds1
        #   plt.bar(value1, heights1, bottom=bottoms1, color='green', alpha=0.7)
          
        #   value2 = list(range(256))
        #   lower_bounds2 = type_box.min_offset[j].cpu()
        #   upper_bounds2 = type_box.max_offset[j].cpu()
        #   heights2 = np.array(upper_bounds2) - np.array(lower_bounds2)
        #   bottoms2 = lower_bounds2
        #   plt.bar(value2, heights2, bottom=bottoms2, width = 1.0, color='red', alpha=0.7)
        #   plt.savefig('/home/***/3DSSG/ssg/{}.png'.format(i))
        
        
      obj_prob = torch.stack(obj_log_prob_list, dim=0)
      node_dict.append({'node_cls_pred_vol': torch.max(obj_prob, 0)[0], 'node_cls_pred_id': torch.max(obj_prob, 0)[1], 'node_cls_gt': gt_node[i]})
      all_obj_prob_list.append(obj_prob)
    log_probs = torch.stack(all_obj_prob_list, dim=0)
    return log_probs, node_dict
  
  def forward(self, box, gt_node):
    type_box_input = self.type_box_input[torch.arange(0, self.num_embeddings, dtype=torch.int64, device="cuda")]
    # type_box_input = nn.Embedding(self.num_embeddings, self.box_embedding_dim * 2).to("cuda")(torch.arange(0, self.num_embeddings, dtype=torch.int64, device="cuda"))
    # type_box_input = self.box_embeddings(torch.arange(0,
    #                         self.box_embeddings.num_embeddings,
    #                         dtype=torch.int64,
    #                         device="cuda"))
    type_box = CenterSigmoidBoxTensor.from_split(type_box_input)
    log_probs, node_dict = self.new_node_classifer(type_box, box, self.gumbel_beta, self.euler_gamma, self.inv_softplus_temp, self.softplus_scale, gt_node)
    return log_probs, node_dict
    # min_point, max_point = CenterSigmoidBoxTensor.gumbel_intersection_mutli_box(type_box, box, self.gumbel_beta) # type 1, type 2 , .... , type 20 
    # intersection_box_vol1 = CenterSigmoidBoxTensor.log_soft_mutli_volume(min_point, 
    #                                                   max_point,
    #                                                   self.euler_gamma,
    #                                                   temp=self.inv_softplus_temp,
    #                                                   scale=self.softplus_scale,
    #                                                   gumbel_beta=self.gumbel_beta)
    # box_vol2 = CenterSigmoidBoxTensor.log_soft_mutli_volume(type_box.min_offset, 
    #                                      type_box.max_offset,
    #                                      self.euler_gamma,
    #                                      temp=self.inv_softplus_temp,
    #                                      scale=self.softplus_scale,
    #                                      gumbel_beta=self.gumbel_beta)
    # for i in range(box_vol2.shape[0]):
    #   if i == 0:
    #     box_vol2_expand = box_vol2[i].unsqueeze(0).repeat(box.data.shape[0])
    #   else:
    #     box_vol2_expand = torch.cat((box_vol2_expand, box_vol2[i].unsqueeze(0).repeat(box.data.shape[0])), 0)

    # log_probs = intersection_box_vol1 - box_vol2_expand
    
    # node_num = len(log_probs) / 20
    # log_probs = log_probs.view(int(node_num), 20)
    # return log_probs
  
class Box_Rel_Classifier(nn.Module):
  def __init__(self,
               rel_size = 9,
               box_dim = 256,
               ent_size = 20,
               device = 'cuda',
               gumbel_beta: float = 1.0,
               inv_softplus_temp: float = 1.,
               softplus_scale: float = 1.,
               lambda_trans = 0.3,
               lambda_scale = 0.3
               ):
    self.REL_SIZE = rel_size
    self.BOX_DIM = box_dim
    self.ENT_SIZE = ent_size
    self.device = device
    self.gumbel_beta = gumbel_beta
    self.euler_gamma = 0.57721566490153286060
    self.inv_softplus_temp = inv_softplus_temp
    self.softplus_scale = softplus_scale
    self.lambda_trans = lambda_trans
    self.lambda_scale = lambda_scale
    self.old_max = -999
    self.old_min = 999
    super(Box_Rel_Classifier, self).__init__()
    '''relation affine transformation'''# forget the dim of box
    rel_trans_for_front = torch.empty(self.REL_SIZE,self.BOX_DIM)
    rel_scale_for_front = torch.empty(self.REL_SIZE,self.BOX_DIM)
    torch.nn.init.normal_(rel_trans_for_front, mean=0, std=1e-4)
    torch.nn.init.normal_(rel_scale_for_front, mean=1, std=0.2)
    
    rel_trans_for_back = torch.empty(self.REL_SIZE,self.BOX_DIM)
    rel_scale_for_back = torch.empty(self.REL_SIZE,self.BOX_DIM)
    torch.nn.init.normal_(rel_trans_for_back, mean=0, std=1e-4)
    torch.nn.init.normal_(rel_scale_for_back, mean=1, std=0.2)       
    
    self.rel_trans_for_front, self.rel_scale_for_front = nn.Parameter(rel_trans_for_front.to(device)), nn.Parameter(
        rel_scale_for_front.to(device))
    self.rel_trans_for_back, self.rel_scale_for_back = nn.Parameter(rel_trans_for_back.to(device)), nn.Parameter(
        rel_scale_for_back.to(device))
    
  def transform_front_box(self, box, Confusion_Memory_Block):  
    relu = nn.ReLU()
    rel_ids = torch.tensor(list(range(self.REL_SIZE)))
    translation = self.rel_trans_for_front[rel_ids] #7种关系的平移
    scale = relu(self.rel_scale_for_front[rel_ids]) #7种关系的放缩
    
    translation =  Confusion_Memory_Block.mean_front_translation * self.lambda_trans + translation * (1 - self.lambda_trans)
    scale = Confusion_Memory_Block.mean_front_scale * self.lambda_scale + scale * (1 - self.lambda_scale)
    
    min_offsets = []
    max_offsets = []
    for i in range(box.data.shape[0]):
      min_offsets1 = box.min_offset[i].unsqueeze(0).repeat(self.REL_SIZE,1)
      max_offsets1 = box.max_offset[i].unsqueeze(0).repeat(self.REL_SIZE,1)
      min_offsets1 = min_offsets1 + translation
      max_offsets1 = (max_offsets1 - min_offsets1) * scale + min_offsets1
      
      min_offsets.append(min_offsets1)
      max_offsets.append(max_offsets1)
    min_offsets_cat = torch.cat(min_offsets, dim=0)
    max_offsets_cat = torch.cat(max_offsets, dim=0)
    
    front_box = BoxTensor.from_zZ(min_offsets_cat, max_offsets_cat) # 7*box num
    return front_box, translation, scale

  def transform_back_box(self, box, Confusion_Memory_Block):
    relu = nn.ReLU()
    rel_ids = torch.tensor(list(range(self.REL_SIZE)))
    translation = self.rel_trans_for_back[rel_ids] #7种关系的平移
    scale = relu(self.rel_scale_for_back[rel_ids]) #7种关系的放缩
    
    translation =  Confusion_Memory_Block.mean_back_translation * self.lambda_trans + translation * (1 - self.lambda_trans)
    scale = Confusion_Memory_Block.mean_back_scale * self.lambda_scale + scale * (1 - self.lambda_scale)
    
    min_offsets = []
    max_offsets = []
    for i in range(box.data.shape[0]):
      min_offsets1 = box.min_offset[i].unsqueeze(0).repeat(self.REL_SIZE,1)
      max_offsets1 = box.max_offset[i].unsqueeze(0).repeat(self.REL_SIZE,1)
      min_offsets1 = min_offsets1 + translation
      max_offsets1 = (max_offsets1 - min_offsets1) * scale + min_offsets1
      
      min_offsets.append(min_offsets1)
      max_offsets.append(max_offsets1)
    min_offsets_cat = torch.cat(min_offsets, dim=0)
    max_offsets_cat = torch.cat(max_offsets, dim=0)
    back_box = BoxTensor.from_zZ(min_offsets_cat, max_offsets_cat) # 7*box num
    return back_box, translation, scale
  
  def new_transform_front_box(self, box, Confusion_Memory_Block):  
    relu = nn.ReLU()
    rel_ids = torch.tensor(list(range(self.REL_SIZE)))
    translation = self.rel_trans_for_front[rel_ids] #7种关系的平移
    scale = relu(self.rel_scale_for_front[rel_ids]) #7种关系的放缩
    
    translation =  Confusion_Memory_Block.mean_front_translation * self.lambda_trans + translation * (1 - self.lambda_trans)
    scale = Confusion_Memory_Block.mean_front_scale * self.lambda_scale + scale * (1 - self.lambda_scale)

    min_offsets1 = box.min_offset.unsqueeze(0).repeat(self.REL_SIZE,1)
    max_offsets1 = box.max_offset.unsqueeze(0).repeat(self.REL_SIZE,1)
    min_offsets1 = min_offsets1 + translation
    max_offsets1 = (max_offsets1 - min_offsets1) * scale + min_offsets1
    front_box = BoxTensor.from_zZ(min_offsets1, max_offsets1) # 7*box num
    return front_box, translation, scale

  def new_transform_back_box(self, box, Confusion_Memory_Block):
    relu = nn.ReLU()
    rel_ids = torch.tensor(list(range(self.REL_SIZE)))
    translation = self.rel_trans_for_back[rel_ids] #7种关系的平移
    scale = relu(self.rel_scale_for_back[rel_ids]) #7种关系的放缩
    
    translation =  Confusion_Memory_Block.mean_back_translation * self.lambda_trans + translation * (1 - self.lambda_trans)
    scale = Confusion_Memory_Block.mean_back_scale * self.lambda_scale + scale * (1 - self.lambda_scale)
    
    min_offsets1 = box.min_offset.unsqueeze(0).repeat(self.REL_SIZE,1)
    max_offsets1 = box.max_offset.unsqueeze(0).repeat(self.REL_SIZE,1)
    min_offsets1 = min_offsets1 + translation
    max_offsets1 = (max_offsets1 - min_offsets1) * scale + min_offsets1

    back_box = BoxTensor.from_zZ(min_offsets1, max_offsets1) # 7*box num
    return back_box, translation, scale
  
  def new_relation_box(self, node_to_node, Box, Confusion_Memory_Block, gumbel_beta, euler_gamma, temp, scale):
    edge_num = node_to_node.edge_index.shape[1]
    new_dict = []
    for i in range(edge_num):
      idx0 = node_to_node.edge_index[0][i]
      idx1 = node_to_node.edge_index[1][i]
      gt_rel = node_to_node.y[i]
      front_box, front_translation, front_scale = self.new_transform_front_box(BoxTensor(Box.data[idx0]), Confusion_Memory_Block)
      back_box, back_translation, back_scale = self.new_transform_back_box(BoxTensor(Box.data[idx1]), Confusion_Memory_Block)
      rel_prob_list = []
      for j in range(self.REL_SIZE):
        intersections_min, intersections_max = CenterSigmoidBoxTensor.gumbel_intersection_box(BoxTensor.from_zZ(front_box.min_offset[j], front_box.max_offset[j]), BoxTensor.from_zZ(back_box.min_offset[j], back_box.max_offset[j]), gumbel_beta)
        inter_vol = CenterSigmoidBoxTensor.log_soft_volume(intersections_min, intersections_max, euler_gamma, temp, scale, gumbel_beta)
        back_vol =  CenterSigmoidBoxTensor.log_soft_volume(back_box.min_offset[j], back_box.max_offset[j], euler_gamma, temp, scale, gumbel_beta)
        rel_log_prob = inter_vol - back_vol
        rel_prob_list.append(rel_log_prob)
      
      max_value = max(rel_prob_list)
      min_value = min(rel_prob_list)
      max_index = rel_prob_list.index(max_value)
      min_index = rel_prob_list.index(min_value)
      
      with torch.no_grad():
        def calculate_overlap(h1,b1,h2,b2):
          print("h1:{},b1:{},h2:{},b2:{}".format(len(h1),len(b1),len(h2),len(b2)))
          x1_h = []
          x1_b = []
          x2_h = []
          x2_b = []
          key_b = []
          key_h = []
          ex1_h = []
          ex1_b = []
          ex2_h = []
          ex2_b = []
          for i in range(len(h1)):
            h11 = np.maximum(h1[i], b1[i])
            b11 = np.minimum(h1[i], b1[i])
            h22 = np.maximum(h2[i], b2[i])
            b22 = np.minimum(h2[i], b2[i])
            inter_b = np.maximum(b11, b22)
            inter_h = np.minimum(h11, h22)
            if inter_h > inter_b:
              if h11 > inter_h and inter_h > b11 and h11 > h22:
                x1_h.append(h11)
                x1_b.append(inter_h)
                key_h.append(inter_h)
                key_b.append(inter_b)
                x2_h.append(inter_b)
                x2_b.append(b11)
                ex1_h.append(0)
                ex1_b.append(0)
                ex2_h.append(0)
                ex2_b.append(0)
              elif h22 > inter_h and inter_h > b22 and h11 < h22:
                x2_h.append(h22)
                x2_b.append(inter_h)
                key_h.append(inter_h)
                key_b.append(inter_b)
                x1_h.append(inter_b)
                x1_b.append(b11)
                ex1_h.append(0)
                ex1_b.append(0)
                ex2_h.append(0)
                ex2_b.append(0)
              elif inter_h == h22 and inter_b == b22:
                x1_h.append(h11)
                x1_b.append(inter_h)
                key_h.append(inter_h)
                key_b.append(inter_b)
                x2_h.append(0)
                x2_b.append(0)
                ex1_h.append(inter_b)
                ex1_b.append(b11)
                ex2_h.append(0)
                ex2_b.append(0)
              elif inter_h == h11 and inter_b == b11:
                x1_h.append(0)
                x1_b.append(0)
                key_h.append(inter_h)
                key_b.append(inter_b)
                x2_h.append(h22)
                x2_b.append(inter_h)
                ex1_h.append(0)
                ex1_b.append(0)
                ex2_h.append(inter_b)
                ex2_b.append(b22)
              else:
                print("wrong")
                print("h1:{},b1:{},h2:{},b2:{},inter_h:{},inter_b:{}".format(h11,b11,h22,b22,inter_h,inter_b))
            elif inter_h <= inter_b:
              x1_h.append(h11)
              x1_b.append(b11)
              x2_h.append(h22)
              x2_b.append(b22)
              key_h.append(0)
              key_b.append(0)
              ex1_h.append(0)
              ex1_b.append(0)
              ex2_h.append(0)
              ex2_b.append(0)
            else:
              print("error")
          return x1_h, x1_b, x2_h, x2_b, key_h, key_b, ex1_h, ex1_b, ex2_h, ex2_b
        if max_value > self.old_max:
          self.old_max = max_value
          plt.figure(figsize=(40, 5),dpi=500)
          categories = list(range(256))
          y_num=np.arange(len(categories)* 6.5)
          lower_bounds1 = front_box.min_offset[max_index].cpu()
          upper_bounds1 = front_box.max_offset[max_index].cpu()
          heights1 = np.array(upper_bounds1) - np.array(lower_bounds1)
          bottoms1 = lower_bounds1
          
          lower_bounds2 = back_box.min_offset[max_index].cpu()
          upper_bounds2 = back_box.max_offset[max_index].cpu()
          heights2 = np.array(upper_bounds2) - np.array(lower_bounds2)
          bottoms2 = lower_bounds2
          
          plt.bar(np.arange(len(categories)) * 6.5, heights1, bottom=bottoms1, width = 5.5, color='red', alpha=0.4)
          plt.bar(np.arange(len(categories)) * 6.5, heights2, bottom=bottoms2, width = 5.5, color='green', alpha=0.4)
          # 绘制交集颜色
          # x1_h, x1_b, x2_h, x2_b, key_h, key_b, ex1_h, ex1_b, ex2_h, ex2_b = calculate_overlap(heights1, bottoms1, heights2, bottoms2)
          # print("x1_h:{},x1_b:{},x2_h:{},x2_b:{},key_h:{},key_b:{}, ex1_h:{}, ex1_b:{}, ex2_h:{}, ex2_b:{}".format(len(x1_h),len(x1_b),len(x2_h),len(x2_b),len(key_h),len(key_b),len(ex1_h),len(ex1_b),len(ex2_h),len(ex2_b)))
          # plt.bar(np.arange(len(categories)) * 6.5, x1_h, bottom=x1_b, width = 6.0, color='red', alpha=0.3)
          # plt.bar(np.arange(len(categories)) * 6.5, x2_h, bottom=x2_b, width = 6.0, color='blue', alpha=0.3)
          # plt.bar(np.arange(len(categories)) * 6.5, ex1_b, bottom=ex1_b, width = 6.0, color='red', alpha=0.3)
          # plt.bar(np.arange(len(categories)) * 6.5, ex2_h, bottom=ex2_b, width = 6.0, color='blue', alpha=0.3)
          # plt.bar(np.arange(len(categories)) * 6.5, key_h, bottom=key_b, width = 6.0, color='green', alpha=0.3)
          # print("h1:{},b1:{},h2:{},b2:{},inter_h:{},inter_b:{}".format(x1_h[0],x1_b[0],x2_h[0],x2_b[0],key_b[0],key_b[0]))
          
          
          #   # 绘制重合度曲线
          # ax2 = plt.gca().twinx()
          # overlap_curve = calculate_overlap(heights1, bottoms1, heights2, bottoms2)
          # ax2.plot(np.arange(len(categories)) * 6.5, overlap_curve, color='blue', label='Overlap', alpha=0.3)
          # ax2.set_ylim(0, 1)
          
          plt.xticks([])
          plt.xlim(min(y_num)-1* 6.5,max(y_num)+1* 6.5)
          plt.ylim(min(min(bottoms1), min(bottoms2), 0), max(max(upper_bounds1), max(upper_bounds2)))
          plt.ylabel('Position')
          plt.xlabel('Dimension')
          # plt.xticks(np.arange(len(categories)) * 1.5, rotation=45)
          plt.savefig('/home/***/3DSSG/ssg/newest_draw_draw_max_slim{}.png'.format(max_index))
        print("old_max:{}".format(self.old_max)) 
        if min_value < self.old_min:
          self.old_min = min_value
          plt.figure(figsize=(40, 5),dpi=500)
          categories = list(range(256))
          y_num=np.arange(len(categories)* 6.5)
          lower_bounds1 = front_box.min_offset[min_index].cpu()
          upper_bounds1 = front_box.max_offset[min_index].cpu()
          heights1 = np.array(upper_bounds1) - np.array(lower_bounds1)
          bottoms1 = lower_bounds1
          
          lower_bounds2 = back_box.min_offset[min_index].cpu()
          upper_bounds2 = back_box.max_offset[min_index].cpu()
          heights2 = np.array(upper_bounds2) - np.array(lower_bounds2)
          bottoms2 = lower_bounds2
          
          plt.bar(np.arange(len(categories)) * 6.5, heights1, bottom=bottoms1, width = 5.5, color='red', alpha=0.4)
          plt.bar(np.arange(len(categories)) * 6.5, heights2, bottom=bottoms2, width = 5.5, color='green', alpha=0.4)
          
          # x1_h, x1_b, x2_h, x2_b, key_h, key_b, ex1_h, ex1_b, ex2_h, ex2_b = calculate_overlap(heights1, bottoms1, heights2, bottoms2)
          # plt.bar(np.arange(len(categories)) * 6.5, x1_h, bottom=x1_b, width = 6.0, color='red', alpha=0.3)
          # plt.bar(np.arange(len(categories)) * 6.5, ex1_b, bottom=ex1_b, width = 6.0, color='red', alpha=0.3)
          # plt.bar(np.arange(len(categories)) * 6.5, x2_h, bottom=x2_b, width = 6.0, color='blue', alpha=0.3)
          # plt.bar(np.arange(len(categories)) * 6.5, ex2_h, bottom=ex2_b, width = 6.0, color='blue', alpha=0.3)
          # plt.bar(np.arange(len(categories)) * 6.5, key_h, bottom=key_b, width = 6.0, color='green', alpha=0.3)
          
          #   # 绘制重合度曲线
          # ax2 = plt.gca().twinx()
          # overlap_curve = calculate_overlap(heights1, bottoms1, heights2, bottoms2)
          # ax2.plot(np.arange(len(categories)) * 6.5, overlap_curve, color='blue', label='Overlap', alpha=0.3)
          # ax2.set_ylim(0, 1)         
          
          plt.xticks([])
          plt.xlim(min(y_num)-1* 6.5,max(y_num)+1* 6.5)
          plt.ylim(min(min(bottoms1), min(bottoms2), 0), max(max(upper_bounds1), max(upper_bounds2)))
          plt.ylabel('Position')
          plt.xlabel('Dimension')
          plt.savefig('/home/***/3DSSG/ssg/newest_draw_draw_min_slim{}.png'.format(min_index))
        print("old_min:{}".format(self.old_min)) 
      new_dict.append({'rel_prob_list':rel_prob_list, 'node_to_node':[idx0,idx1], 'gt_rel':gt_rel, 
                       'front_translation':front_translation, 'front_scale':front_scale, 'back_translation':back_translation, 'back_scale':back_scale,
                       'front_box':front_box, 'back_box':back_box})
    return new_dict
        
  def forward(self, Box, gt_node, gt_edge, node_dict, Confusion_Memory_Block, node_to_node):
    '''
    input: node box
    output: 
    
          F_r1(Box_1) <-> B_r1(Box_2)   F_r1(Box_1) <-> B_r1(Box_3)   F_r1(Box_1) <-> B_r1(Box_4)   F_r1(Box_1) <-> B_r1(Box_5) ... F_r1(Box_1) <-> B_r1(Box_n)
          F_r1(Box_2) <-> B_r1(Box_1)   F_r1(Box_2) <-> B_r1(Box_3)   F_r1(Box_2) <-> B_r1(Box_4)   F_r1(Box_2) <-> B_r1(Box_5) ... F_r1(Box_2) <-> B_r1(Box_n)
          ...
          ...
          F_r1(Box_n) <-> B_r1(Box_1)   F_r1(Box_n) <-> B_r1(Box_2)   F_r1(Box_n) <-> B_r1(Box_3)   F_r1(Box_n) <-> B_r1(Box_4) ... F_r1(Box_n) <-> B_r1(Box_n-1)
          
          F_r2(Box_1) <-> B_r2(Box_2)   F_r2(Box_1) <-> B_r2(Box_3)   F_r2(Box_1) <-> B_r2(Box_4)   F_r2(Box_1) <-> B_r2(Box_5) ... F_r2(Box_1) <-> B_r2(Box_n)

          ...
    '''
    
    new_dict = self.new_relation_box(node_to_node, Box, Confusion_Memory_Block, self.gumbel_beta, self.euler_gamma, self.inv_softplus_temp, self.softplus_scale)
    return new_dict
    
    
    
    # front_box, front_translation, front_scale = self.transform_front_box(Box, Confusion_Memory_Block)
    # back_box, back_translation, back_scale = self.transform_back_box(Box, Confusion_Memory_Block)      
    # min_point, max_point, front_cls_gt, back_cls_gt, rel_ids, back_intersections_min_offsets, back_intersections_max_offsets, front_box_cls_pred, back_box_cls_pred = \
    #   CenterSigmoidBoxTensor.gumbel_intersection_relation_box(front_box, back_box, self.gumbel_beta, gt_node, self.REL_SIZE, self.BOX_DIM, node_dict)
    
    
    
    # log_intersection_vol = CenterSigmoidBoxTensor.log_soft_mutli_volume(min_point, 
    #                                                        max_point,
    #                                                        self.euler_gamma,
    #                                                        temp=self.inv_softplus_temp,
    #                                                        scale=self.softplus_scale,
    #                                                        gumbel_beta=self.gumbel_beta)
    # log_back_vol = CenterSigmoidBoxTensor.log_soft_mutli_volume(back_intersections_min_offsets,
    #                                                back_intersections_max_offsets,
    #                                                self.euler_gamma,
    #                                                temp=self.inv_softplus_temp,
    #                                                scale=self.softplus_scale,
    #                                                gumbel_beta=self.gumbel_beta)
    # rel_log_prob = log_intersection_vol - log_back_vol    
    # return rel_log_prob, front_cls_gt, back_cls_gt, rel_ids, front_box, back_box, front_box_cls_pred, back_box_cls_pred, front_translation, front_scale, back_translation, back_scale


class BCEWithLogProbLoss(nn.BCELoss):


  def log1mexp(self, x: torch.Tensor,
              split_point=math.log(0.5),
              exp_zero_eps=1e-7) -> torch.Tensor:
    """
    Computes log(1 - exp(x)).

    Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).

    = log1p(-exp(x)) when x <= log(1/2)
    or
    = log(-expm1(x)) when log(1/2) < x <= 0

    For details, see

    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    https://github.com/visinf/n3net/commit/31968bd49c7d638cef5f5656eb62793c46b41d76
    """
    logexpm1_switch = x > split_point
    Z = torch.zeros_like(x)
    # this clamp is necessary because expm1(log_p) will give zero when log_p=1,
    # ie. p=1
    logexpm1 = torch.log((-torch.expm1(x[logexpm1_switch])).clamp_min(1e-38))
    # hack the backward pass
    # if expm1(x) gets very close to zero, then the grad log() will produce inf
    # and inf*0 = nan. Hence clip the grad so that it does not produce inf
    logexpm1_bw = torch.log(-torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
    Z[logexpm1_switch] = logexpm1.detach() + (logexpm1_bw - logexpm1_bw.detach())
    #Z[1 - logexpm1_switch] = torch.log1p(-torch.exp(x[1 - logexpm1_switch]))
    Z[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]))

    return Z
  def _binary_cross_entropy(self,
                            input: torch.Tensor,
                            target: torch.Tensor,
                            weight: Optional[torch.Tensor] = None,
                            reduction: str = 'mean') -> torch.Tensor:
    """Computes binary cross entropy.

    This function takes log probability and computes binary cross entropy.

    Args:
      input: Torch float tensor. Log probability. Same shape as `target`.
      target: Torch float tensor. Binary labels. Same shape as `input`.
      weight: Torch float tensor. Scaling loss if this is specified.
      reduction: Reduction method. 'mean' by default.
    """
    loss = -target * input - (1 - target) * self.log1mexp(input)

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()

  def forward(self, input, target, weight=None):
    return self._binary_cross_entropy(input,
                                      target,
                                      weight=weight,
                                      reduction=self.reduction)