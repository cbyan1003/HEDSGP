import os
import copy
import torch
import time
import logging
import ssg
import time
import numpy as np
from tqdm import tqdm  # , tnrange
from collections import defaultdict
from codeLib.models import BaseTrainer
from codeLib.common import check_weights, check_valid, convert_torch_to_scalar
from ssg.utils.util_eva import EvalSceneGraphBatch, EvalUpperBound  # EvalSceneGraph,
import codeLib.utils.moving_average as moving_average
from codeLib.models import BaseTrainer
from tqdm import tqdm
import codeLib.utils.string_numpy as snp
from ssg.utils.util_data import match_class_info_from_two, merge_batch_mask2inst
from ssg import define
from ssg.utils.graph_vis import DrawSceneGraph
from ssg.trainer.eval_inst import EvalInst
import math
from ssg.models.box_decoder import CenterSigmoidBoxTensor, BoxTensor, BCEWithLogProbLoss
import cProfile
import random
from memory_profiler import profile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import datetime



logger_py = logging.getLogger(__name__)



class Trainer_SGFN(BaseTrainer, EvalInst):
    def __init__(self, cfg, model, node_cls_names: list, edge_cls_names: list,
                 device=None,  **kwargs):
        super().__init__(device)
        logger_py.setLevel(cfg.log_level)
        self.cfg = cfg
        self.model = model  # .to(self._device)
        # self.optimizer = optimizer
        self.w_node_cls = kwargs.get('w_node_cls', None)
        self.w_edge_cls = kwargs.get('w_edge_cls', None)
        self.node_cls_names = node_cls_names  # kwargs['node_cls_names']
        self.edge_cls_names = edge_cls_names  # kwargs['edge_cls_names']
        self.use_JointSG = self.cfg.model.use_JointSG
        self.REL_SIZE = 9
        self.gumbel_beta = 0.0036026463511690845
        self.euler_gamma = 0.57721566490153286060
        self.inv_softplus_temp = 1.2471085395024732
        self.softplus_scale = 1.
        
        trainable_params = filter(
            lambda p: p.requires_grad, model.parameters())
        self.optimizer = ssg.config.get_optimizer(cfg, trainable_params)

        if self.w_node_cls is not None:
            logger_py.info('train with weighted node class.')
            self.w_node_cls = self.w_node_cls.to(self._device)
        if self.w_edge_cls is not None:
            logger_py.info('train with weighted node class.')
            self.w_edge_cls = self.w_edge_cls.to(self._device)

        self.eva_tool = EvalSceneGraphBatch(
            self.node_cls_names, self.edge_cls_names,
            save_prediction=False,
            multi_rel_prediction=self.cfg.model.multi_rel,
            k=0, none_name=define.NAME_NONE)  # do not calculate topK in training mode
        self.loss_node_cls = torch.nn.CrossEntropyLoss(weight=self.w_node_cls)
        self.new_loss_node_cls = BCEWithLogProbLoss()
        if self.cfg.model.multi_rel:
            self.loss_rel_cls = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.w_edge_cls)
        else:
            self.loss_rel_cls = torch.nn.CrossEntropyLoss(
                weight=self.w_edge_cls)

    def zero_metrics(self):
        self.eva_tool.reset()

    def evaluate(self, val_loader, Confusion_Memory_Block, all_features, all_labels, topk):
        it_dataset = val_loader.__iter__()
        eval_tool = EvalSceneGraphBatch(
            self.node_cls_names, self.edge_cls_names,
            multi_rel_prediction=self.cfg.model.multi_rel,
            k=topk, save_prediction=True, none_name=define.NAME_NONE)
        eval_list = defaultdict(moving_average.MA)

        # time.sleep(2)# Prevent possible deadlock during epoch transition
        for data in tqdm(it_dataset, leave=False):
            eval_step_dict = self.eval_step(data, Confusion_Memory_Block, all_features, all_labels, eval_tool=eval_tool)

            for k, v in eval_step_dict.items():
                eval_list[k].update(v)
            
        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k, v in obj_.items():
            # print(k)
            eval_dict[k+'_node_cls'] = v
        for k, v in edge_.items():
            # print(k)
            eval_dict[k+'_edge_cls'] = v

        for k, v in eval_list.items():
            eval_dict[k] = v.avg

        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        del it_dataset
        return eval_dict, eval_tool

    def sample(self, dataloader):
        pass

    # @profile
    def train_step(self, data, Confusion_Memory_Block, all_features, all_labels, it=None):
        # pr = cProfile.Profile()
        # pr.enable()
        # self.model.train()
        self.optimizer.zero_grad()
        if not self.use_JointSG:
            logs, new_dict, node_pred_cls, edge_pred_cls = self.compute_loss(data, Confusion_Memory_Block, all_features, all_labels, it=it, eval_tool=self.eva_tool) #, edge_dict
            # if new_dict != None:
            #     self.update_confusion_memory(new_dict, Confusion_Memory_Block, data['node'].y, node_pred_cls, edge_pred_cls, mode='Train')
        else:
            logs = self.compute_loss(data, Confusion_Memory_Block, all_features, all_labels, it=it, eval_tool=self.eva_tool)
        torch.cuda.empty_cache()
        if 'loss' not in logs:
            return logs
        logs['loss'].backward(retain_graph=False)
        self.optimizer.step()
        # pr.disable()
        # pr.dump_stats('{}.prof'.format(self.cfg.name))
        
        # if not check_valid(logs):
        #     logs['loss'].backward()
        #     check_weights(self.model.state_dict())
        #     self.optimizer.step()
        # else:
        #     logger_py.info('skip loss backward due to nan occurs')
        return logs

    def eval_step(self, data, Confusion_Memory_Block, all_features, all_labels, eval_tool=None):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        eval_dict = {}
        if not self.use_JointSG:
            with torch.no_grad():
                eval_dict, eval_new_dict, eval_node_pred_cls, eval_edge_pred_cls= self.compute_loss(
                    data, Confusion_Memory_Block, all_features, all_labels, eval_mode=True, eval_tool=eval_tool) #, eval_edge_dict 
            # if eval_new_dict != None:
            #     self.update_confusion_memory(eval_new_dict, Confusion_Memory_Block, data['node'].y, eval_node_pred_cls, eval_edge_pred_cls, mode='Val')
        else:
            with torch.no_grad():
                eval_dict = self.compute_loss(data, Confusion_Memory_Block, all_features, all_labels, eval_mode=True, eval_tool=eval_tool)
        eval_dict = convert_torch_to_scalar(eval_dict)
        # for (k, v) in eval_dict.items():
        #     eval_dict[k] = v.item()
        return eval_dict
    # result_dict[1]
    def update_confusion_memory(self, edge_dict, Confusion_Memory_Block, gt_node, node_pred_cls, edge_pred_cls, mode=''):
        Confusion_Memory_Block.compute_IDM_BND()
        use_old = False
        if not use_old:
            if mode == 'Train':
                front_cls_gt_id = [gt_node[edge['node_to_node'][0].detach()].unsqueeze(0) for edge in edge_dict]
                front_cls_pred_id = [node_pred_cls[edge['node_to_node'][0].detach()].unsqueeze(0) for edge in edge_dict]
                front_translation = [edge['front_translation'].detach() for edge in edge_dict]
                front_scale = [edge['front_scale'].detach() for edge in edge_dict]

                back_cls_gt_id = [gt_node[edge['node_to_node'][1].detach()].unsqueeze(0) for edge in edge_dict]
                back_cls_pred_id = [node_pred_cls[edge['node_to_node'][1].detach()].unsqueeze(0) for edge in edge_dict]
                back_translation = [edge['back_translation'].detach() for edge in edge_dict]
                back_scale = [edge['back_scale'].detach() for edge in edge_dict]
                
                if edge_pred_cls != None:
                    rel_cls_pred_id = [edge_pred_cls[i].detach().unsqueeze(0) for i in range(len(edge_dict))]
                    rel_cls_gt_id = [edge['gt_rel'].detach().unsqueeze(0) for edge in edge_dict]

                    # 将列表组合成元组
                    front_box_list, back_box_list, relation_list = (
                        {'front_cls_gt_id': front_cls_gt_id, 'front_cls_pred_id': front_cls_pred_id,
                        'front_translation': front_translation, 'front_scale': front_scale},
                        {'back_cls_gt_id': back_cls_gt_id, 'back_cls_pred_id': back_cls_pred_id,
                        'back_translation': back_translation, 'back_scale': back_scale},
                        {'rel_cls_pred_id': rel_cls_pred_id, 'rel_cls_gt_id': rel_cls_gt_id}
                    )
                else:
                    front_box_list, back_box_list = (
                        {'front_cls_gt_id': front_cls_gt_id, 'front_cls_pred_id': front_cls_pred_id,
                        'front_translation': front_translation, 'front_scale': front_scale},
                        {'back_cls_gt_id': back_cls_gt_id, 'back_cls_pred_id': back_cls_pred_id,
                        'back_translation': back_translation, 'back_scale': back_scale}
                    )
                    relation_list = None
            elif mode == 'Val':
                front_box_min = [edge['front_box'].min_offset.detach() for edge in edge_dict]
                front_box_max = [edge['front_box'].max_offset.detach() for edge in edge_dict]
                front_cls_gt_id = [gt_node[edge['node_to_node'][0].detach()].unsqueeze(0) for edge in edge_dict]
                front_cls_pred_id = [node_pred_cls[edge['node_to_node'][0].detach()].unsqueeze(0) for edge in edge_dict]

                back_box_min = [edge['back_box'].min_offset.detach() for edge in edge_dict]
                back_box_max = [edge['back_box'].max_offset.detach() for edge in edge_dict]
                back_cls_gt_id = [gt_node[edge['node_to_node'][1].detach()].unsqueeze(0) for edge in edge_dict]
                back_cls_pred_id = [node_pred_cls[edge['node_to_node'][1].detach()].unsqueeze(0) for edge in edge_dict]

                rel_cls_pred_id = [edge_pred_cls[i].detach().unsqueeze(0) for i in range(len(edge_dict))]
                rel_cls_gt_id = [edge['gt_rel'].detach().unsqueeze(0) for edge in edge_dict]

                # 将列表组合成元组
                front_box_list, back_box_list, relation_list = (
                    {'front_box_min': front_box_min, 'front_box_max': front_box_max,
                    'front_cls_gt_id': front_cls_gt_id, 'front_cls_pred_id': front_cls_pred_id},
                    {'back_box_min': back_box_min, 'back_box_max': back_box_max,
                    'back_cls_gt_id': back_cls_gt_id, 'back_cls_pred_id': back_cls_pred_id},
                    {'rel_cls_pred_id': rel_cls_pred_id, 'rel_cls_gt_id': rel_cls_gt_id}
                )
            # if mode == 'Train':
            #     front_cls_gt_id = np.array([gt_node[edge['node_to_node'][0].detach()].unsqueeze(0) for edge in edge_dict])
            #     front_cls_pred_id = np.array([node_pred_cls[edge['node_to_node'][0].detach()].unsqueeze(0) for edge in edge_dict])
            #     front_translation = np.array([edge['front_translation'].detach() for edge in edge_dict])
            #     front_scale = np.array([edge['front_scale'].detach() for edge in edge_dict])

            #     back_cls_gt_id = np.array([gt_node[edge['node_to_node'][1].detach()].unsqueeze(0) for edge in edge_dict])
            #     back_cls_pred_id = np.array([node_pred_cls[edge['node_to_node'][1].detach()].unsqueeze(0) for edge in edge_dict])
            #     back_translation = np.array([edge['back_translation'].detach() for edge in edge_dict])
            #     back_scale = np.array([edge['back_scale'].detach() for edge in edge_dict])

            #     rel_cls_pred_id = np.array([edge_pred_cls[edge['node_to_node'][0].detach()].unsqueeze(0) for edge in edge_dict])
            #     rel_cls_gt_id = np.array([edge['gt_rel'].detach().unsqueeze(0) for edge in edge_dict])

            #     # 将数组组合成元组
            #     front_box_list, back_box_list, relation_list = (
            #         {'front_cls_gt_id': front_cls_gt_id, 'front_cls_pred_id': front_cls_pred_id,
            #         'front_translation': front_translation, 'front_scale': front_scale},
            #         {'back_cls_gt_id': back_cls_gt_id, 'back_cls_pred_id': back_cls_pred_id,
            #         'back_translation': back_translation, 'back_scale': back_scale},
            #         {'rel_cls_pred_id': rel_cls_pred_id, 'rel_cls_gt_id': rel_cls_gt_id}
            #     )
            # elif mode == 'Val':
            #     front_box_min = np.array([edge['front_box'].min_offset.detach() for edge in edge_dict])
            #     front_box_max = np.array([edge['front_box_max'].max_offset.detach() for edge in edge_dict])
            #     front_cls_gt_id = np.array([gt_node[edge['node_to_node'][0].detach()].unsqueeze(0) for edge in edge_dict])
            #     front_cls_pred_id = np.array([node_pred_cls[edge['node_to_node'][0].detach()].unsqueeze(0) for edge in edge_dict])
        
            #     back_box_min = np.array([edge['back_box'].min_offset.detach() for edge in edge_dict])
            #     back_box_max = np.array([edge['back_box'].max_offset.detach() for edge in edge_dict])
            #     back_cls_gt_id = np.array([gt_node[edge['node_to_node'][1].detach()].unsqueeze(0) for edge in edge_dict])
            #     back_cls_pred_id = np.array([node_pred_cls[edge['node_to_node'][1].detach()].unsqueeze(0) for edge in edge_dict])
               
            #     rel_cls_pred_id = np.array([edge_pred_cls[edge['node_to_node'][0].detach()].unsqueeze(0) for edge in edge_dict])
            #     rel_cls_gt_id = np.array([edge['gt_rel'].detach().unsqueeze(0) for edge in edge_dict])

            #     # 将数组组合成元组
            #     front_box_list, back_box_list, relation_list = (
            #         {'front_box_min': front_box_min, 'front_box_max': front_box_max,
            #         'front_cls_gt_id': front_cls_gt_id, 'front_cls_pred_id': front_cls_pred_id,},
            #         {'back_box_min': back_box_min, 'back_box_max': back_box_max,
            #         'back_cls_gt_id': back_cls_gt_id, 'back_cls_pred_id': back_cls_pred_id,},
            #         {'rel_cls_pred_id': rel_cls_pred_id, 'rel_cls_gt_id': rel_cls_gt_id}
            #     )
        else:
            front_box_list, back_box_list, relation_list = zip(*[
                        ({'front_box_min':edge_dict[edge]['front_box_min'].detach(), 
                        'front_box_max':edge_dict[edge]['front_box_max'].detach(), 
                        'front_cls_gt_id':edge_dict[edge]['front_cls_gt'].detach().unsqueeze(0),
                        'front_cls_pred_id':edge_dict[edge]['front_box_cls_pred'].detach().unsqueeze(0),
                        'front_translation':edge_dict[edge]['front_translation'].detach(),
                        'front_scale':edge_dict[edge]['front_scale'].detach()}, 
                        {'back_box_min':edge_dict[edge]['back_box_min'].detach(), 
                        'back_box_max':edge_dict[edge]['back_box_max'].detach(), 
                        'back_cls_gt_id':edge_dict[edge]['back_cls_gt'].detach().unsqueeze(0),
                        'back_cls_pred_id':edge_dict[edge]['back_box_cls_pred'].detach().unsqueeze(0),
                        'back_translation':edge_dict[edge]['back_translation'].detach(),
                        'back_scale':edge_dict[edge]['back_scale'].detach()}, 
                        {'rel_cls_pred_id':edge_dict[edge]['rel_cls_pred_id'].detach().unsqueeze(0),
                        'rel_cls_gt_id':edge_dict[edge]['rel_cls_gt_id'].detach()})
                        for edge in edge_dict])

            if mode == 'Train':
                '''build train block'''
                Confusion_Memory_Block.update(front_box_list, back_box_list, relation_list, mode = mode)
                '''merge mean to box decoder'''
                Confusion_Memory_Block.positive_mask()
                Confusion_Memory_Block.compute_mean_box()
            elif mode == 'Val':
                '''build eval block'''
                Confusion_Memory_Block.update(front_box_list, back_box_list, relation_list, mode = mode)
    
        
    def compute_loss(self, data, Confusion_Memory_Block, all_features, all_labels, eval_mode=False, it=None, eval_tool=None):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        # Initialize loss dictionary and other values
        logs = {}

        # Process data dictionary
        data = data.to(self._device)
        # data = self.process_data_dict(data)

        # Shortcuts
        # scan_id = data['scan_id']
        gt_node = data['node'].y
        gt_edge = data['node', 'to', 'node'].y

        for i in gt_edge:
            if i > 8:
                raise ValueError("error in gt_edge, please check the data.")

        if not self.use_JointSG:  
            ''' make forward pass through the network '''
            # node_cls_vol, edge_cls_vol, front_cls_gt, back_cls_gt, rel_cls_pred_id, front_box, back_box, front_box_cls_pred, back_box_cls_pred,\
            #     front_translation, front_scale, back_translation, back_scale = self.model(data, Confusion_Memory_Block)
        
            
            output, node_cls_vol, new_dict = self.model(data, Confusion_Memory_Block)
            
            # with torch.no_grad():
            #     features = output.cpu().numpy()
            #     all_features.append(features)
            #     label = data['node'].y
            #     all_labels.append(label.cpu().numpy())

            # if len(all_labels) % 800 == 0:
            #     print("len(all_labels){}".format(len(all_labels)))
            #     tsne = TSNE(n_components=2, random_state=42)
            #     train_tsne = tsne.fit_transform(np.concatenate(all_features))

            #     plt.figure(figsize=(12,12), dpi=300)
            #     for i in range(20):
            #         indices = np.concatenate(all_labels) == i
            #         plt.scatter(train_tsne[indices, 0], train_tsne[indices, 1], cmap='tab20', alpha=0.5, marker='+')
            #         center = np.mean(train_tsne[indices], axis=0)
            #         plt.annotate(f'{i}', center, fontsize=18, alpha=0.8, ha='center', va='center')
            #     # plt.legend()
            #     plt.axis('off')
            #     plt.savefig('./tsne/node_{}.png'.format(len(all_labels)//100))


        else:
            output, node_cls, edge_cls = self.model(data, Confusion_Memory_Block)
            
            # tsne draw
            with torch.no_grad():
                features = output.cpu().numpy()
                all_features.append(features)
                label = data['node'].y
                all_labels.append(label.cpu().numpy())
                # print("len(all_labels){}".format(len(all_labels)))


            if len(all_labels) % 800 == 0:
                # 获取当前日期和时间
                current_datetime = datetime.datetime.now()
                hour = current_datetime.hour
                minute = current_datetime.minute
                second = current_datetime.second

                print("len(all_labels), savefig at node_{}{}{}.png".format(hour,minute,second))
                tsne = TSNE(n_components=2, random_state=42)
                train_tsne = tsne.fit_transform(np.concatenate(all_features))

                plt.figure(figsize=(12,12), dpi=300)
                for i in range(20):
                    indices = np.concatenate(all_labels) == i
                    plt.scatter(train_tsne[indices, 0], train_tsne[indices, 1], cmap='tab20', alpha=0.5, marker='+')
                    center = np.mean(train_tsne[indices], axis=0)
                    plt.annotate(f'{i}', center, fontsize=18, alpha=0.8, ha='center', va='center')
                # plt.legend()
                plt.axis('off')
                plt.savefig('./tsne/new_draw_node_{}{}{}.png'.format(hour,minute,second))
        ''' calculate loss '''
        
        if not self.use_JointSG:  
            logs['loss'] = 0
            if self.cfg.training.lambda_mode == 'dynamic':
            # calculate loss ratio base on the number of node and edge
                batch_node = node_cls_vol.shape[0]
            self.cfg.training.lambda_node = 1

            if new_dict is not None:
                edge_cls_vol = []
                for i in new_dict:
                    edge_cls_vol.append(torch.stack(i['rel_prob_list']))
                edge_cls_vol = torch.stack(edge_cls_vol) 
                batch_edge = edge_cls_vol.shape[0]
                self.cfg.training.lambda_edge = batch_edge / batch_node
            if new_dict == None or len(gt_edge) == 0:
                node_cls, gt_node_cls = self.calc_node_loss(logs, node_cls_vol, gt_node, self.w_node_cls)
                edge_cls = None
            else:
                node_cls, gt_node_cls = self.calc_node_loss(logs, node_cls_vol, gt_node, self.w_node_cls)
                self.calc_edge_loss(logs, edge_cls_vol, gt_edge, weights = self.w_edge_cls)
                # self.calc_logic_loss(logs, new_dict, gt_node)
                
                # old version
                # edge_dict, edge_cls, gt_edge_cls = self.calc_edge_loss(logs, edge_cls_vol, gt_edge, front_cls_gt, back_cls_gt, rel_cls_pred_id, gt_node.shape[0], 
                #                                 front_box, back_box, front_box_cls_pred, back_box_cls_pred, 
                #                                 front_translation, front_scale, back_translation, back_scale, 
                #                                 data['node', 'to', 'node'].edge_index, data['node', 'to', 'node'].y, gt_node,
                #                                 self.w_edge_cls)
                # self.calc_logic_loss(logs, edge_dict)
            
                '''3. get metrics'''
                edge_cls = edge_cls_vol
                metrics = self.model.calculate_metrics(
                    node_cls_pred=node_cls,
                    node_cls_gt=gt_node_cls,
                    edge_cls_pred=edge_cls,
                    edge_cls_gt=gt_edge
                )
                for k, v in metrics.items():
                    logs[k] = v

                ''' eval tool '''
                if eval_tool is not None:
                    # node_cls = torch.softmax(node_cls.detach(), dim=1)
                    node_cls = torch.nn.functional.softmax(node_cls.detach(), dim = 1)
                    data['node'].pd = node_cls.detach()

                    if edge_cls is not None and gt_edge is not None:
                        edge_cls = torch.sigmoid(edge_cls.detach())
                        data['node', 'to', 'node'].pd = edge_cls.detach()
                        data['node', 'to', 'node'].y = gt_edge.detach()
                    eval_tool.add(data,
                                #   node_cls,gt_node,
                                #   edge_cls,gt_edge,
                                #   mask2instance,
                                #   edge_indices_node_to_node,
                                #   node_gt,
                                #   edge_index_node_gt
                                )
            if eval_mode == False:
                if new_dict == None or len(gt_edge) == 0:
                    return logs, new_dict, node_cls.max(1)[1], edge_cls
                else:
                    return logs, new_dict, node_cls.max(1)[1], edge_cls.max(1)[1]
      
            elif eval_mode ==True:
                if new_dict == None or len(gt_edge) == 0:
                    return logs, new_dict, node_cls.max(1)[1], edge_cls
                else:
                    return logs, new_dict, node_cls.max(1)[1], edge_cls.max(1)[1]
 
        else:
                
            logs['loss'] = 0

            if self.cfg.training.lambda_mode == 'dynamic':
                # calculate loss ratio base on the number of node and edge
                batch_node = node_cls.shape[0]
                self.cfg.training.lambda_node = 1

                if edge_cls is not None:
                    batch_edge = edge_cls.shape[0]
                    self.cfg.training.lambda_edge = batch_edge / batch_node

            ''' 1. node class loss'''
            self.calc_node_loss(logs, node_cls, gt_node, self.w_node_cls)

            ''' 2. edge class loss '''
            if edge_cls is not None:
                self.calc_edge_loss(logs, edge_cls, gt_edge, weights = self.w_edge_cls)

            '''3. get metrics'''
            metrics = self.model.calculate_metrics(
                node_cls_pred=node_cls,
                node_cls_gt=gt_node,
                edge_cls_pred=edge_cls,
                edge_cls_gt=gt_edge
            )
            for k, v in metrics.items():
                logs[k] = v

            ''' eval tool '''
            if eval_tool is not None:
                node_cls = torch.softmax(node_cls.detach(), dim=1)
                data['node'].pd = node_cls.detach()

                if edge_cls is not None:
                    edge_cls = torch.sigmoid(edge_cls.detach())
                    data['node', 'to', 'node'].pd = edge_cls.detach()
                eval_tool.add(data,
                            #   node_cls,gt_node,
                            #   edge_cls,gt_edge,
                            #   mask2instance,
                            #   edge_indices_node_to_node,
                            #   node_gt,
                            #   edge_index_node_gt
                            )

            # if check_valid(logs):
            #     raise RuntimeWarning()
            #     print('has nan')

            return logs
            # return loss if eval_mode else loss['loss']

    def calc_node_loss(self, logs, node_cls, node_cls_gt, weights=None):
        '''
        calculate node loss.
        can include
        classification loss
        attribute loss
        affordance loss
        '''
        if not self.use_JointSG:  
            if not self.cfg.model.use_BCE:
                loss_obj = self.loss_node_cls(node_cls, node_cls_gt)
                logs['loss'] += self.cfg.training.lambda_node * loss_obj
                logs['loss_obj'] = loss_obj
                return node_cls, node_cls_gt
            else:
                one_hot_node_gt = torch.nn.functional.one_hot(node_cls_gt, num_classes = 20)
                loss_obj = self.new_loss_node_cls(node_cls, one_hot_node_gt)
                logs['loss'] += self.cfg.training.lambda_node * loss_obj
                logs['loss_obj'] = loss_obj
                return node_cls, node_cls_gt
        else:
            loss_obj = self.loss_node_cls(node_cls, node_cls_gt)
            logs['loss'] += self.cfg.training.lambda_node * loss_obj
            logs['loss_obj'] = loss_obj
        

    def confirm_rel_pred(self, edge_cls_pred, front_cls_gt, back_cls_gt, rel_cls_pred_id, node_num, front_box, back_box, 
                         front_box_cls_pred, back_box_cls_pred, gt_edge_index, gt_edge_cls, gt_node,
                         front_translation, front_scale, back_translation, back_scale):
        def find_positive_n(result):
            a = 1
            b = -1
            c = -result
            
            # 计算根的公式
            root = math.sqrt(b ** 2 - 4 * a * c)
            n1 = (-b + root) / (2 * a)
            n2 = (-b - root) / (2 * a)
            
            # 返回绝对值为正数的解
            return abs(n1) if n1 > 0 else abs(n2)
        box_num = int(find_positive_n(torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor')))
        if box_num == node_num:        
            result_dict = {}
            front_gt_2_back_gt = []
            for i in range(torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor')):
                x = []
                y_gt = []
                z_gt = []
                for j in range(self.REL_SIZE):
                    # F_r1(Box_1) <-> B_r1(Box_2), F_r2(Box_1) <-> B_r2(Box_2), F_r3(Box_1) <-> B_r3(Box_2),...,F_r7(Box_1) <-> B_r7(Box_2)
                    # box1 和 box2 之间的置信度分数
                    x.append(edge_cls_pred[i + j * torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor')].clone().detach())
                    # y_gt  F_Box_[] gt
                    # z_gt  B_Box_[] gt
                    y_gt.append(front_cls_gt[i + j * torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor')].clone().detach())
                    z_gt.append(back_cls_gt[i + j * torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor')].clone().detach())
                x = torch.stack(x, dim=0)
                front_gt_2_back_gt.append(torch.unique((torch.t(torch.stack((torch.stack(y_gt, dim=0), torch.stack(z_gt, dim=0)), dim=0))), dim = 0).flatten().tolist())
                edge_idx = torch.t(gt_edge_index)
                
                # TODO: 明确 gt 和 node 的具体对应
                
                idx_0 = gt_node.tolist().index(front_gt_2_back_gt[i][0])
                idx_1 = gt_node.tolist().index(front_gt_2_back_gt[i][1])
                if [idx_0,idx_1] in edge_idx.tolist():
                    index = edge_idx.tolist().index([idx_0,idx_1])
                    rel_cls_gt_id = gt_edge_cls[index]
                else:
                    continue
                    # try to only use jointSG edge,
                    # rel_cls_gt_id = torch.tensor(0).cuda()
                    # gt_edge_index_list = gt_edge_index.tolist()
                    # gt_edge_index_list[0].append(idx_0)
                    # gt_edge_index_list[1].append(idx_1)
                    
                #     //   torch.div(a, b, rounding_mode='floor')
                                     
                value, index = torch.max(x, 0)
                if value == edge_cls_pred[i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))]:
                    front_cls_gt1 = front_cls_gt[i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))]
                    back_cls_gt1 = back_cls_gt[i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))]
                    
                    front_box_cls_pred1 = front_box_cls_pred[i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))]
                    back_box_cls_pred1 = back_box_cls_pred[i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))]
                    
                    rel_cls_pred_id1 = rel_cls_pred_id[i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))].cuda()
                    
                    edge_cls_vol1 = edge_cls_pred[i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))]
                    
                    if (i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))) % (node_num * (node_num - 1)) == 0:
                        rel_num = torch.div((i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))), (node_num * (node_num - 1)), rounding_mode='floor') - 1
                    else:
                        rel_num = torch.div((i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))), (node_num * (node_num - 1)), rounding_mode='floor')  # 前面有rel_num组，位于rel_num + 1               
                    
                    if ((i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))) - (index * node_num * (node_num - 1))) % (node_num - 1) == 0: # Determined box
                        rows = torch.div(((i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))) - (index * node_num * (node_num - 1))), (node_num - 1), rounding_mode='floor') - 1
                    else:
                        rows = torch.div(((i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))) - (index * node_num * (node_num - 1))), (node_num - 1), rounding_mode='floor') # 前面有rows行，位于rows + 1 行

                    front_box_min1 = front_box.min_offset[rel_num + rows * self.REL_SIZE]
                    front_box_max1 = front_box.max_offset[rel_num + rows * self.REL_SIZE]
                    
                    columns = ((i + index * (torch.div(edge_cls_pred.shape[0], self.REL_SIZE, rounding_mode='floor'))) - (index * node_num * (node_num - 1)) - rows * (node_num - 1)) % (node_num - 1) # 位于第columns列
                    
                    if (rows + 1) <= columns:
                        columns = columns + 1
                        
                    back_box_min1 = back_box.min_offset[rel_num + (columns - 1)* self.REL_SIZE]
                    back_box_max1 = back_box.max_offset[rel_num + (columns - 1) * self.REL_SIZE]
                    
                    rel_cls_gt_id1 = rel_cls_gt_id   
                    
                    
                    front_translation1 = front_translation[index]
                    front_scale1 = front_scale[index] 
                    back_translation1 = back_translation[index]
                    back_scale1 = back_scale[index]    
                else:
                    raise ValueError("error in rel loss, data location error, please check the data.")
                result_dict[i] = {
                'edge_cls_pred': value,
                'front_cls_gt': front_cls_gt1,
                'back_cls_gt': back_cls_gt1,
                'rel_cls_pred_id': rel_cls_pred_id1,
                'rel_cls_gt_id': rel_cls_gt_id1,
                'edge_cls_vol': edge_cls_vol1,
                'front_box_min':front_box_min1,
                'front_box_max':front_box_max1,
                'back_box_min':back_box_min1,
                'back_box_max':back_box_max1,
                'front_box_cls_pred':front_box_cls_pred1,
                'back_box_cls_pred':back_box_cls_pred1,
                'x': x,
                'y_gt': y_gt,
                'z_gt': z_gt,
                'front_translation':front_translation1,
                'front_scale':front_scale1,
                'back_translation':back_translation1,
                'back_scale':back_scale1,}
                
                
                
        else:
            raise ValueError("error in rel loss, node_num != box_num, please check the data.")
        return result_dict
        
   
            

    def calc_edge_loss(self, logs, edge_cls_pred, edge_cls_gt,\
        front_cls_gt=None, back_cls_gt=None, rel_cls_pred_id=None,  node_num=None,\
            front_box=None, back_box=None, front_box_cls_pred=None, back_box_cls_pred=None,\
                front_translation=None, front_scale=None, back_translation=None, back_scale=None, \
                    gt_edge_index=None, gt_edge_cls=None, gt_node=None, weights=None):
        '''
        input: 
            front_box:
            r1(Box_1), r2(Box_1), r3(Box_1),...,r7(Box_1), r1(Box_2),...
            edge_cls_pred: front_cls_gt, back_cls_gt
            F_r1(Box_1) <-> B_r1(Box_2)   F_r1(Box_1) <-> B_r1(Box_3)   F_r1(Box_1) <-> B_r1(Box_4)   F_r1(Box_1) <-> B_r1(Box_5) ... F_r1(Box_1) <-> B_r1(Box_n)
            F_r1(Box_2) <-> B_r1(Box_1)   F_r1(Box_2) <-> B_r1(Box_3)   F_r1(Box_2) <-> B_r1(Box_4)   F_r1(Box_2) <-> B_r1(Box_5) ... F_r1(Box_2) <-> B_r1(Box_n)
            ...
            ...
            F_r1(Box_n) <-> B_r1(Box_1)   F_r1(Box_n) <-> B_r1(Box_2)   F_r1(Box_n) <-> B_r1(Box_3)   F_r1(Box_n) <-> B_r1(Box_4) ... F_r1(Box_n) <-> B_r1(Box_n-1)
            
            F_r2(Box_1) <-> B_r2(Box_2)   F_r2(Box_1) <-> B_r2(Box_3)   F_r2(Box_1) <-> B_r2(Box_4)   F_r2(Box_1) <-> B_r2(Box_5) ... F_r2(Box_1) <-> B_r2(Box_n)

            ...
        output: 
        '''
        if not self.use_JointSG: 
            if self.cfg.model.multi_rel:
                loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
            else:
                loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
            logs['loss'] += self.cfg.training.lambda_edge * loss_rel
            logs['loss_rel'] = loss_rel
            
            
            
            
            # result_dict = self.confirm_rel_pred(edge_cls_pred, front_cls_gt, back_cls_gt, rel_cls_pred_id,  node_num, front_box, back_box, 
            #                                     front_box_cls_pred, back_box_cls_pred, gt_edge_index, gt_edge_cls, gt_node,
            #                                     front_translation, front_scale, back_translation, back_scale)
            
            # rel_list = list(range(self.REL_SIZE))
            # edge_vol = []
            # edge_gt = []
            # for index, i in enumerate(result_dict):
            #     each = result_dict[i]
            #     edge_vol.append(each['x'])
            #     edge_gt.append(each['rel_cls_gt_id'])
            # edge_cls = torch.nn.functional.softmax(torch.stack(edge_vol), dim = 1)
            # loss_rel = self.loss_rel_cls(edge_cls, torch.stack(edge_gt))
            # logs['loss'] += self.cfg.training.lambda_edge * loss_rel
            # logs['loss_rel'] = loss_rel
            # return result_dict, edge_cls, torch.stack(edge_gt)
        else:
            if self.cfg.model.multi_rel:
                loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
            else:
                loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
            logs['loss'] += self.cfg.training.lambda_edge * loss_rel
            logs['loss_rel'] = loss_rel
        

    def calc_logic_loss(self, logs, result_dict, gt_node):
        use_old = False
        if use_old:
            subsets = []

            # 使用字典的迭代器
            for r, info_r in result_dict.items():
                for r2, info_r2 in result_dict.items():
                    try:
                        # 提前判断条件
                        if r2 != r and info_r2['rel_cls_pred_id'] == info_r['rel_cls_pred_id'] and info_r2['rel_cls_gt_id'] == info_r['rel_cls_gt_id'] and info_r2['back_cls_gt'] == info_r['front_cls_gt'] and info_r2['front_cls_gt'] != info_r['back_cls_gt']:
                            # 使用列表解析
                            subsets.append([
                                [info_r2[key] for key in ['edge_cls_pred', 'front_cls_gt', 'back_cls_gt', 'rel_cls_pred_id', 'rel_cls_gt_id', 'front_box_min', 'front_box_max', 'back_box_min', 'back_box_max']],
                                [info_r[key] for key in ['edge_cls_pred', 'front_cls_gt', 'back_cls_gt', 'rel_cls_pred_id', 'rel_cls_gt_id', 'front_box_min', 'front_box_max', 'back_box_min', 'back_box_max']]
                            ])
                    except Exception as e:
                        print(f"Error in calc_logic_loss: {e}")
            
            
            # speed too slow
            # subsets = []
            # for r in result_dict:
            #     for r2 in result_dict:
            #         try:
            #             if r2 != r and result_dict[r2]['rel_cls_pred_id'] == result_dict[r]['rel_cls_pred_id'] and result_dict[r2]['rel_cls_gt_id'] == result_dict[r]['rel_cls_gt_id'] and result_dict[r2]['back_cls_gt'] == result_dict[r]['front_cls_gt'] and result_dict[r2]['front_cls_gt'] != result_dict[r]['back_cls_gt']:
            #                  subsets.append([[result_dict[r2]['edge_cls_pred'],
            #                             result_dict[r2]['front_cls_gt'],
            #                             result_dict[r2]['back_cls_gt'],
            #                             result_dict[r2]['rel_cls_pred_id'],
            #                             result_dict[r2]['rel_cls_gt_id'],
            #                             result_dict[r2]['front_box_min'],
            #                             result_dict[r2]['front_box_max'], 
            #                             result_dict[r2]['back_box_min'], 
            #                             result_dict[r2]['back_box_max']]
            #                         ,[result_dict[r]['edge_cls_pred'], 
            #                         result_dict[r]['front_cls_gt'], 
            #                         result_dict[r]['back_cls_gt'], 
            #                         result_dict[r]['rel_cls_pred_id'], 
            #                         result_dict[r]['rel_cls_gt_id'],
            #                         result_dict[r]['front_box_min'], 
            #                         result_dict[r]['front_box_max'], 
            #                         result_dict[r]['back_box_min'], 
            #                         result_dict[r]['back_box_max']]])
            #         except:
            #             print("error in calc_logic_loss")  
                        # TODO:result_dict运行多次之后，出现值丢失问题
                        
                        
            # subsets = [[[result_dict[r2]['edge_cls_pred'],
            #              result_dict[r2]['front_cls_gt'],
            #              result_dict[r2]['back_cls_gt'],
            #              result_dict[r2]['rel_cls_pred_id'],
            #              result_dict[r2]['rel_cls_gt_id'],
            #              result_dict[r2]['front_box_min'],
            #              result_dict[r2]['front_box_max'], 
            #              result_dict[r2]['back_box_min'], 
            #              result_dict[r2]['back_box_max']]
            #         ,[result_dict[r]['edge_cls_pred'], 
            #           result_dict[r]['front_cls_gt'], 
            #           result_dict[r]['back_cls_gt'], 
            #           result_dict[r]['rel_cls_pred_id'], 
            #           result_dict[r]['rel_cls_gt_id'],
            #           result_dict[r]['front_box_min'], 
            #           result_dict[r]['front_box_max'], 
            #           result_dict[r]['back_box_min'], 
            #           result_dict[r]['back_box_max']]]
            #         for r2 in result_dict 
            #         for r in result_dict 
            #         if r2 != r 
            #         and result_dict[r2]['rel_cls_pred_id'] == result_dict[r]['rel_cls_pred_id'] 
            #         and result_dict[r2]['back_cls_gt'] == result_dict[r]['front_cls_gt'] 
            #         and result_dict[r2]['front_cls_gt'] != result_dict[r]['back_cls_gt']]
            logic_loss = 0


            for i in random.sample(subsets, len(subsets) // 3):
                intersection_min, intersection_max = CenterSigmoidBoxTensor.gumbel_intersection_box(BoxTensor.from_zZ(i[0][7], i[0][8]), 
                                                                                                    BoxTensor.from_zZ(i[1][5], i[1][6]), 
                                                                                                    self.gumbel_beta)
                intersection_vol = CenterSigmoidBoxTensor.log_soft_volume(intersection_min, 
                                                        intersection_max,
                                                        self.euler_gamma,
                                                        temp=self.inv_softplus_temp,
                                                        scale=self.softplus_scale,
                                                        gumbel_beta=self.gumbel_beta)
                front_box_vol = CenterSigmoidBoxTensor.log_soft_volume(i[1][5], 
                                                        i[1][6],
                                                        self.euler_gamma,
                                                        temp=self.inv_softplus_temp,
                                                        scale=self.softplus_scale,
                                                        gumbel_beta=self.gumbel_beta)
                logic_loss += torch.norm(1 - torch.exp_(intersection_vol - front_box_vol))
                
            try:
                logic_loss = logic_loss / len(subsets)
            except ZeroDivisionError:
                logic_loss = logic_loss / 1
                
            logs['loss'] += self.cfg.training.lambda_logic * logic_loss
            logs['loss_logic'] = logic_loss
        else:
            prev_element = {} #存储（node0 ， node1）的第二位， 用于第二个边查找第一位
            after_element = {} #存储 (node0 ， node1）的第一位, 用于第二个边查找第二位
            subsets = []
            for i, each_edge in enumerate(result_dict):
                idx0 = each_edge['node_to_node'][0].tolist()
                idx1 = each_edge['node_to_node'][1].tolist()
                gt_cls0 = gt_node[idx0]
                gt_cls1 = gt_node[idx1]
                if idx0 in prev_element:
                    prev_edge_indices  = prev_element[idx0]
                    for prev_edge_index in prev_edge_indices:
                        if result_dict[prev_edge_index]['node_to_node'][0].tolist() != idx1:
                            subsets.append({'prev_back_gt':gt_node[result_dict[prev_edge_index]['node_to_node'][1]], 'after_front_gt':gt_cls0,
                                            'prev_front_id':result_dict[prev_edge_index]['node_to_node'][0].tolist(), 'prev_back_id':result_dict[prev_edge_index]['node_to_node'][1].tolist(), 
                                            'after_front_id':idx0, 'after_back_id':idx1,
                                        'prev_back_box':result_dict[prev_edge_index]['back_box'], 'after_front_box':each_edge['front_box']})
                if idx1 in after_element:
                    after_edge_indices = after_element[idx1]
                    for after_edge_index in after_edge_indices:
                        if result_dict[after_edge_index]['node_to_node'][1].tolist() != idx0:
                            subsets.append({'prev_back_gt':gt_cls1, 'after_front_gt':gt_node[result_dict[after_edge_index]['node_to_node'][0]],
                                            'prev_front_id':idx0, 'prev_back_id':idx1, 
                                            'after_front_id':result_dict[after_edge_index]['node_to_node'][0].tolist(), 'after_back_id':result_dict[after_edge_index]['node_to_node'][1].tolist(),
                                        'prev_back_box':each_edge['back_box'], 'after_front_box':result_dict[after_edge_index]['front_box']})     
                if idx1 not in prev_element:
                    prev_element[idx1] = [i]
                else:
                    prev_element[idx1].append(i)

                if idx0 not in after_element:
                    after_element[idx0] = [i]
                else:
                    after_element[idx0].append(i)

            logic_loss = 0
            for i in random.sample(subsets, min(30, len(subsets))):
                intersection_min, intersection_max = CenterSigmoidBoxTensor.gumbel_intersection_box(i['prev_back_box'], i['after_front_box'], self.gumbel_beta)
                intersection_vol = CenterSigmoidBoxTensor.log_soft_volume(intersection_min, 
                                                        intersection_max,
                                                        self.euler_gamma,
                                                        temp=self.inv_softplus_temp,
                                                        scale=self.softplus_scale,
                                                        gumbel_beta=self.gumbel_beta)
                after_front_box_vol = CenterSigmoidBoxTensor.log_soft_volume(i['after_front_box'].min_offset, 
                                                        i['after_front_box'].max_offset,
                                                        self.euler_gamma,
                                                        temp=self.inv_softplus_temp,
                                                        scale=self.softplus_scale,
                                                        gumbel_beta=self.gumbel_beta)
                logic_loss += torch.norm(1 - torch.exp_(intersection_vol - after_front_box_vol))
                
            try:
                logic_loss = logic_loss / len(subsets)
            except ZeroDivisionError:
                logic_loss = logic_loss / 1
                
            logs['loss'] += self.cfg.training.lambda_logic * logic_loss
            logs['loss_logic'] = logic_loss
            
    def visualize(self, eval_tool=None):
        if eval_tool is None:
            eval_tool = self.eva_tool
        node_confusion_matrix, edge_confusion_matrix = eval_tool.draw(
            plot_text=False,
            grid=False,
            normalize='log',
            plot=False
        )
        return {
            'node_confusion_matrix': node_confusion_matrix,
            'edge_confusion_matrix': edge_confusion_matrix
        }

    def get_log_metrics(self):
        output = dict()
        obj_, edge_ = self.eva_tool.get_mean_metrics()

        for k, v in obj_.items():
            output[k+'_node_cls'] = v
        for k, v in edge_.items():
            output[k+'_edge_cls'] = v
        return output
    
    def evaluate_inst_incre(self, dataset_seg, dataset_inst, topk):
        is_eval_image = self.cfg.model.method in ['imp']
        ignore_missing = self.cfg.eval.ignore_missing

        '''add a none class for missing instances'''
        (scanid2idx_seg, _, node_cls_names, edge_cls_names, noneidx_node_cls, noneidx_edge_cls,
            seg_valid_node_cls_indices, inst_valid_node_cls_indices,
            seg_valid_edge_cls_indices, inst_valid_edge_cls_indices) = \
            match_class_info_from_two(
                dataset_seg, dataset_inst, multi_rel=self.cfg.model.multi_rel)

        '''all'''
        eval_tool_all = EvalSceneGraphBatch(node_cls_names, edge_cls_names,
                                            multi_rel_prediction=self.cfg.model.multi_rel, k=topk, save_prediction=True,
                                            none_name=define.NAME_NONE, ignore_none=False)

        '''ignore none'''
        # eval_tool_ignore_none = EvalSceneGraphBatch(node_cls_names, edge_cls_names,
        #                                 multi_rel_prediction=self.cfg.model.multi_rel,k=topk,save_prediction=True,
        #                                 none_name=define.NAME_NONE,ignore_none=True)
        eval_tools = {'all': eval_tool_all,
                      #   'ignore_none': eval_tool_ignore_none
                      }

        # eval_upper_bound
        eval_UpperBound = EvalUpperBound(node_cls_names, edge_cls_names, noneidx_node_cls, noneidx_edge_cls,
                                         multi_rel=self.cfg.model.multi_rel, topK=topk, none_name=define.NAME_NONE)

        eval_list = defaultdict(moving_average.MA)

        ''' get scan_idx mapping '''
        scanid2idx_seg = dict()
        for index in range(len(dataset_seg)):
            scan_id = snp.unpack(dataset_seg.scans, index)  # self.scans[idx]
            scanid2idx_seg[scan_id] = index

        scanid2idx_inst = dict()
        for index in range(len(dataset_inst)):
            scan_id = snp.unpack(dataset_inst.scans, index)  # self.scans[idx]
            scanid2idx_inst[scan_id] = index

        '''start eval'''
        self.model.eval()
        for index in tqdm(range(len(dataset_inst))):
            data_inst = dataset_inst.__getitem__(index)
            scan_id_inst = data_inst['scan_id']

            if scan_id_inst not in scanid2idx_seg:
                data_seq_seq = None
            else:
                index_seg = scanid2idx_seg[scan_id_inst]
                data_seq_seq = dataset_seg.__getitem__(index_seg)

            '''process seg'''
            eval_dict = {}
            with torch.no_grad():
                logs = {}
                data_inst = self.process_data_dict(data_inst)

                # use the latest timestamp to calculate the upperbound
                if data_seq_seq is not None:
                    key_int = sorted([int(t) for t in data_seq_seq])
                    latest_t = max(key_int)
                    data_lastest = self.process_data_dict(
                        data_seq_seq[str(latest_t)])

                else:
                    data_lastest = None
                eval_UpperBound(data_lastest, data_inst, is_eval_image)
                # continue

                # Shortcuts
                # scan_id = data_inst['scan_id']
                inst_oids = data_inst['node'].oid
                # inst_mask2instance = data_inst['node'].idx2oid[0]#data_inst['mask2instance']
                inst_gt_cls = data_inst['node'].y  # data_inst['gt_cls']
                # data_inst['seg_gt_rel']
                inst_gt_rel = data_inst['node', 'to', 'node'].y
                # data_inst['node_edges']
                inst_node_edges = data_inst['node', 'to', 'node'].edge_index
                gt_relationships = data_inst['relationships']

                if data_seq_seq is None:
                    '''
                    If no target scan in dataset_seg is found, set all prediction to none
                    '''
                    # Nodes
                    node_pred = torch.zeros_like(torch.nn.functional.one_hot(
                        inst_gt_cls, len(node_cls_names))).float()
                    node_pred[:, noneidx_node_cls] = 1.0

                    # Edges
                    if not self.cfg.model.multi_rel:
                        edge_pred = torch.zeros_like(torch.nn.functional.one_hot(
                            inst_gt_rel, len(edge_cls_names))).float()
                        edge_pred[:, noneidx_edge_cls] = 1.0
                    else:
                        edge_pred = torch.zeros_like(inst_gt_rel).float()

                    # log
                    data_inst['node'].pd = node_pred.detach()
                    data_inst['node', 'to', 'node'].pd = edge_pred.detach()
                    for eval_tool in eval_tools.values():
                        eval_tool.add(data_inst)
                    continue

                predictions_weights = dict()
                predictions_weights['node'] = dict()
                predictions_weights['node', 'to', 'node'] = dict()
                merged_node_cls = torch.zeros(
                    len(inst_oids), len(node_cls_names)).to(self.cfg.DEVICE)
                merged_node_cls_gt = (torch.ones(
                    len(inst_oids), dtype=torch.long) * noneidx_node_cls).to(self.cfg.DEVICE)

                # convert them to list
                assert inst_node_edges.shape[0] == 2
                inst_node_edges = inst_node_edges.tolist()
                inst_oids = inst_oids.tolist()

                '''merge batched dict to one single dict'''
                # mask2seg= merge_batch_mask2inst(mask2seg)
                # inst_mask2inst=merge_batch_mask2inst(inst_mask2instance)

                # build search list for GT edge pairs
                inst_gt_pairs = set()
                # This collects "from" and "to" instances pair as key  -> predicate label
                inst_gt_rel_dict = dict()
                for idx in range(len(inst_gt_rel)):
                    src_idx, tgt_idx = inst_node_edges[0][idx], inst_node_edges[1][idx]
                    src_oid, tgt_oid = inst_oids[src_idx], inst_oids[tgt_idx]
                    inst_gt_pairs.add((src_oid, tgt_oid))
                    inst_gt_rel_dict[(src_oid, tgt_oid)] = inst_gt_rel[idx]
                inst_gt_pairs = [pair for pair in inst_gt_pairs]

                '''merge predictions'''
                merged_edge_cls = torch.zeros(
                    len(inst_gt_rel), len(edge_cls_names)).to(self.cfg.DEVICE)
                if not self.cfg.model.multi_rel:
                    merged_edge_cls_gt = (torch.ones(
                        len(inst_gt_rel), dtype=torch.long) * noneidx_edge_cls).to(self.cfg.DEVICE)
                else:
                    merged_edge_cls_gt = inst_gt_rel.clone().float()

                for timestamp in key_int:
                    timestamp = key_int[-1]
                    timestamp = str(timestamp)
                    data_seg = self.process_data_dict(data_seq_seq[timestamp])

                    assert data_seg['scan_id'] == data_inst['scan_id']

                    if not is_eval_image:
                        # seg_gt_cls = data_seg['node'].y
                        seg_gt_rel = data_seg['node', 'to', 'node'].y
                        seg_oids = data_seg['node'].oid
                        seg_node_edges = data_seg['node',
                                                  'to', 'node'].edge_index
                    else:
                        # seg_gt_cls = data_seg['roi'].y
                        seg_gt_rel = data_seg['roi', 'to', 'roi'].y
                        # mask2seg = data_seg['roi'].idx2oid[0]
                        seg_oids = data_seg['roi'].oid
                        seg_node_edges = data_seg['roi',
                                                  'to', 'roi'].edge_index
                        # seg2inst = data_seg['roi'].get('idx2iid',None)

                    ''' make forward pass through the network '''
                    node_cls, edge_cls = self.model(data_seg)

                    # convert them to list
                    assert seg_node_edges.shape[0] == 2
                    seg_node_edges = seg_node_edges.tolist()
                    seg_oids = seg_oids.tolist()

                    '''merge prediction from seg to instance (in case of "same part")'''
                    # use list bcuz may have multiple predictions on the same object instance
                    seg_oid2idx = defaultdict(list)
                    for idx in range(len(seg_oids)):
                        seg_oid2idx[seg_oids[idx]].append(idx)

                    '''merge nodes'''
                    merged_idx2oid = dict()
                    merged_oid2idx = dict()

                    for idx in range(len(inst_oids)):
                        oid = inst_oids[idx]
                        # merge predictions
                        if not ignore_missing:
                            merged_oid2idx[oid] = idx
                            merged_idx2oid[idx] = oid
                            # use GT class
                            merged_node_cls_gt[idx] = inst_gt_cls[idx]
                            if oid in seg_oid2idx:
                                '''merge nodes'''
                                predictions = node_cls[seg_oid2idx[oid]
                                                       ]  # get all predictions on that instance
                                node_cls_pred = torch.softmax(predictions, dim=1).mean(
                                    dim=0)  # averaging the probability

                                # Weighted Sum
                                if idx not in predictions_weights['node']:
                                    predictions_weights['node'][idx] = 0
                                merged_node_cls[idx, inst_valid_node_cls_indices] = \
                                    (merged_node_cls[idx, inst_valid_node_cls_indices] * predictions_weights['node'][idx] +
                                        node_cls_pred[seg_valid_node_cls_indices]
                                     ) / (predictions_weights['node'][idx]+1)
                                predictions_weights['node'][idx] += 1

                            else:
                                assert noneidx_node_cls is not None
                                # Only do this in the last estimation
                                if int(timestamp) == key_int[-1]:
                                    merged_node_cls[idx,
                                                    noneidx_node_cls] = 1.0
                        else:
                            raise NotImplementedError()
                            if inst not in inst2masks:
                                continue
                            merged_mask2instance[counter] = inst
                            merged_instance2idx[inst] = counter
                            predictions = node_cls[inst2masks[inst]]
                            node_cls_pred = torch.softmax(
                                predictions, dim=1).mean(dim=0)
                            merged_node_cls[counter,
                                            inst_valid_node_cls_indices] = node_cls_pred[seg_valid_node_cls_indices]
                            merged_node_cls_gt[counter] = inst_gt_cls[mask_old]
                            counter += 1
                    if ignore_missing:
                        raise NotImplementedError()
                        merged_node_cls = merged_node_cls[:counter]
                        merged_node_cls_gt = merged_node_cls_gt[:counter]

                    '''merge batched dict to one single dict'''
                    # For segment level
                    # map edge predictions on the same pair of instances.
                    merged_edge_cls_dict = defaultdict(list)
                    for idx in range(len(seg_gt_rel)):
                        src_idx, tgt_idx = seg_node_edges[0][idx], seg_node_edges[1][idx]
                        src_oid, tgt_oid = seg_oids[src_idx], seg_oids[tgt_idx]
                        pair = (src_oid, tgt_oid)
                        if pair in inst_gt_pairs:
                            merged_edge_cls_dict[pair].append(edge_cls[idx])
                        else:
                            # print('cannot find seg:{}(inst:{}) to seg:{}(inst:{}) with relationship:{}.'.format(src_seg_idx,src_inst_idx,tgt_seg_idx,tgt_inst_idx,relname))
                            pass

                    '''merge predictions'''
                    merged_node_edges = list()  # new edge_indices
                    for idx, pair in enumerate(inst_gt_pairs):
                        inst_edge_cls = inst_gt_rel_dict[pair]
                        if ignore_missing:
                            if pair[0] not in merged_oid2idx:
                                continue
                            if pair[1] not in merged_oid2idx:
                                continue
                        # merge edge index to the new mask ids
                        src_idx = merged_oid2idx[pair[0]]
                        tgt_idx = merged_oid2idx[pair[1]]
                        merged_node_edges.append([src_idx, tgt_idx])

                        if pair in merged_edge_cls_dict:
                            edge_pds = torch.stack(merged_edge_cls_dict[pair])
                            edge_pds = edge_pds[:, seg_valid_edge_cls_indices]
                            # seg_valid_edge_cls_indices

                            # ignore same part
                            if not self.cfg.model.multi_rel:
                                edge_pds = torch.softmax(
                                    edge_pds, dim=1).mean(0)
                            else:
                                edge_pds = torch.sigmoid(edge_pds).mean(0)

                            # Weighted Sum
                            if idx not in predictions_weights['node', 'to', 'node']:
                                predictions_weights['node',
                                                    'to', 'node'][idx] = 0
                            merged_edge_cls[idx, inst_valid_edge_cls_indices] = \
                                (merged_edge_cls[idx, inst_valid_edge_cls_indices]*predictions_weights['node', 'to', 'node'][idx] +
                                    edge_pds) / (predictions_weights['node', 'to', 'node'][idx]+1)
                            predictions_weights['node', 'to', 'node'][idx] += 1
                            # merged_edge_cls[counter,inst_valid_edge_cls_indices] = edge_pds
                        elif not self.cfg.model.multi_rel:
                            # Only do this in the last estimation
                            if int(timestamp) == key_int[-1]:
                                merged_edge_cls[idx, noneidx_edge_cls] = 1.0

                        if not self.cfg.model.multi_rel:
                            merged_edge_cls_gt[idx] = inst_edge_cls

                    if ignore_missing:
                        raise NotImplementedError()
                        merged_edge_cls = merged_edge_cls[:counter]
                        merged_edge_cls_gt = merged_edge_cls_gt[:counter]
                    merged_node_edges = torch.tensor(
                        merged_node_edges, dtype=torch.long)
                    break
                merged_node_edges = merged_node_edges.t().contiguous()

            data_inst['node'].pd = merged_node_cls.detach()
            data_inst['node'].y = merged_node_cls_gt.detach()
            data_inst['node', 'to', 'node'].pd = merged_edge_cls.detach()
            data_inst['node', 'to', 'node'].y = merged_edge_cls_gt.detach()
            data_inst['node', 'to', 'node'].edge_index = merged_node_edges
            data_inst['node'].clsIdx = torch.from_numpy(
                np.array([k for k in merged_idx2oid.values()]))
            for eval_tool in eval_tools.values():
                eval_tool.add(data_inst)

        eval_dict = dict()
        eval_dict['visualization'] = dict()
        for eval_type, eval_tool in eval_tools.items():

            obj_, edge_ = eval_tool.get_mean_metrics()
            for k, v in obj_.items():
                # print(k)
                eval_dict[eval_type+'_'+k+'_node_cls'] = v
            for k, v in edge_.items():
                # print(k)
                eval_dict[eval_type+'_'+k+'_edge_cls'] = v

            for k, v in eval_list.items():
                eval_dict[eval_type+'_'+k] = v.avg

            vis = self.visualize(eval_tool=eval_tool)

            vis = {eval_type+'_'+k: v for k, v in vis.items()}

            eval_dict['visualization'].update(vis)

        return eval_dict, eval_tools, eval_UpperBound.eval_tool

    def visualize_inst_incre(self, dataset_seg, topk):
        ignore_missing = self.cfg.eval.ignore_missing
        '''add a none class for missing instances'''
        node_cls_names = copy.copy(self.node_cls_names)
        edge_cls_names = copy.copy(self.edge_cls_names)
        if define.NAME_NONE not in self.node_cls_names:
            node_cls_names.append(define.NAME_NONE)
        if define.NAME_NONE not in self.edge_cls_names:
            edge_cls_names.append(define.NAME_NONE)
        # remove same part
        # if define.NAME_SAME_PART in edge_cls_names: edge_cls_names.remove(define.NAME_SAME_PART)

        noneidx_node_cls = node_cls_names.index(define.NAME_NONE)
        noneidx_edge_cls = edge_cls_names.index(define.NAME_NONE)

        '''
        Find index mapping. Ignore NONE for nodes since it is used for mapping missing instance.
        Ignore SAME_PART for edges.
        '''
        seg_valid_node_cls_indices = []
        inst_valid_node_cls_indices = []
        for idx in range(len(self.node_cls_names)):
            name = self.node_cls_names[idx]
            if name == define.NAME_NONE:
                continue
            seg_valid_node_cls_indices.append(idx)
        for idx in range(len(node_cls_names)):
            name = node_cls_names[idx]
            if name == define.NAME_NONE:
                continue
            inst_valid_node_cls_indices.append(idx)

        seg_valid_edge_cls_indices = []
        inst_valid_edge_cls_indices = []
        for idx in range(len(self.edge_cls_names)):
            name = self.edge_cls_names[idx]
            # if name == define.NAME_SAME_PART: continue
            seg_valid_edge_cls_indices.append(idx)
        for idx in range(len(edge_cls_names)):
            name = edge_cls_names[idx]
            # if name == define.NAME_SAME_PART: continue
            inst_valid_edge_cls_indices.append(idx)

        eval_tool = EvalSceneGraphBatch(
            node_cls_names, edge_cls_names,
            multi_rel_prediction=self.cfg.model.multi_rel, k=topk, save_prediction=True,
            none_name=define.NAME_NONE)
        eval_list = defaultdict(moving_average.MA)

        '''check'''
        # length
        # print('len(dataset_seg), len(dataset_inst):',len(dataset_seg), len(dataset_inst))
        print('ignore missing', ignore_missing)
        # classes

        ''' get scan_idx mapping '''
        scanid2idx_seg = dict()
        for index in range(len(dataset_seg)):
            scan_id = snp.unpack(dataset_seg.scans, index)  # self.scans[idx]
            scanid2idx_seg[scan_id] = index

        '''start eval'''
        acc_time = 0
        timer_counter = 0
        self.model.eval()
        for index in tqdm(range(len(dataset_seg))):
            # for data_inst in seg_dataloader:
            # data = dataset_seg.__getitem__(index)
            # scan_id_inst = data['scan_id'][0]
            # if scan_id_inst not in scanid2idx_seg: continue
            scan_id = '4acaebcc-6c10-2a2a-858b-29c7e4fb410d'
            index = scanid2idx_seg[scan_id]
            data_seg = dataset_seg.__getitem__(index)

            '''process seg'''
            eval_dict = {}
            with torch.no_grad():
                # logs = {}
                # data_inst = self.process_data_dict(data_inst)
                # Process data dictionary
                batch_data = data_seg
                if len(batch_data) == 0:
                    continue

                '''generate gt'''
                if isinstance(batch_data, list):
                    # the last one is the complete one
                    data_inst = self.process_data_dict(batch_data[-1])
                else:
                    data_inst = self.process_data_dict(batch_data)

                scan_id = data_inst['scan_id']
                graphDrawer = DrawSceneGraph(
                    scan_id, self.node_cls_names, self.edge_cls_names, debug=True)
                nodes_w = defaultdict(int)
                edges_w = defaultdict(int)
                nodes_pds_all = dict()
                edges_pds_all = dict()

                def fuse(old: dict, w_old: dict, new: dict):
                    for k, v in new.items():
                        if k in old:
                            old[k] = (old[k]*w_old[k]+new[k]) / (w_old[k]+1)
                            w_old[k] += 1
                        else:
                            old[k] = new[k]
                            w_old[k] = 1
                    return old, w_old

                def process(data):
                    data = self.process_data_dict(data)
                    # Shortcuts
                    scan_id = data['scan_id']
                    # gt_cls = data['gt_cls']
                    # gt_rel = data['gt_rel']
                    mask2seg = data['mask2instance']
                    node_edges_ori = data['node_edges']
                    data['node_edges'] = data['node_edges'].t().contiguous()
                    # seg2inst = data['seg2inst']

                    # check input valid
                    if node_edges_ori.ndim == 1:
                        return {}, {}, -1

                    ''' make forward pass through the network '''
                    tick = time.time()
                    node_cls, edge_cls = self.model(**data)
                    tock = time.time()

                    '''collect predictions on inst and edge pair'''
                    node_pds = dict()
                    edge_pds = dict()

                    '''merge prediction from seg to instance (in case of "same part")'''
                    # inst2masks = defaultdict(list)
                    mask2seg = merge_batch_mask2inst(mask2seg)
                    tmp_dict = defaultdict(list)
                    for mask, seg in mask2seg.items():
                        # inst = seg2inst[seg]
                        # inst2masks[inst].append(mask)

                        tmp_dict[seg].append(node_cls[mask])
                    for seg, l in tmp_dict.items():
                        if seg in node_pds:
                            raise RuntimeError()
                        pd = torch.stack(l, dim=0)
                        pd = torch.softmax(pd, dim=1).mean(dim=0)
                        node_pds[seg] = pd

                    tmp_dict = defaultdict(list)
                    for idx in range(len(node_edges_ori)):
                        src_idx, tgt_idx = data['node_edges'][0, idx].item(
                        ), data['node_edges'][1, idx].item()
                        seg_src, seg_tgt = mask2seg[src_idx], mask2seg[tgt_idx]
                        # inst_src,inst_tgt = seg2inst[seg_src],seg2inst[seg_tgt]
                        key = (seg_src, seg_tgt)

                        tmp_dict[key].append(edge_cls[idx])

                    for key, l in tmp_dict.items():
                        if key in edge_pds:
                            raise RuntimeError()
                        pd = torch.stack(l, dim=0)
                        pd = torch.softmax(pd, dim=1).mean(0)
                        edge_pds[key] = pd
                        # src_inst_idx, tgt_inst_idx = inst_mask2inst[src_idx], inst_mask2inst[tgt_idx]
                        # inst_gt_pairs.add((src_inst_idx, tgt_inst_idx))

                    return node_pds, edge_pds, tock-tick

                if isinstance(batch_data, list):
                    for idx, data in enumerate(batch_data):
                        fid = data['fid']
                        print(idx)
                        node_pds, edge_pds, pt = process(data)
                        if pt > 0:
                            acc_time += pt
                            timer_counter += 1

                            fuse(nodes_pds_all, nodes_w, node_pds)
                            fuse(edges_pds_all, edges_w, edge_pds)

                            inst_mask2instance = data_inst['mask2instance']
                            # data_inst['node_edges'] = data_inst['node_edges'].t().contiguous()
                            gts = None
                            # gts = dict()
                            # gt_nodes = gts['nodes'] = dict()
                            # gt_edges = gts['edges'] = dict()
                            # for mid in range(data_inst['gt_cls'].shape[0]):
                            #     idx = inst_mask2instance[mid]
                            #     gt_nodes[idx] = data_inst['gt_cls'][mid]
                            # for idx in range(len(data_inst['gt_rel'])):
                            #     src_idx, tgt_idx = data_inst['node_edges'][idx,0].item(),data_inst['node_edges'][idx,1].item()
                            #     src_inst_idx, tgt_inst_idx = inst_mask2instance[src_idx], inst_mask2instance[tgt_idx]
                            #     gt_edges[(src_inst_idx, tgt_inst_idx)] = data_inst['gt_rel'][idx]

                            g = graphDrawer.draw({'nodes': nodes_pds_all, 'edges': edges_pds_all},
                                                 gts)

                            # merged_node_cls, merged_node_cls_gt,  merged_edge_cls, \
                            #     merged_edge_cls_gt, merged_mask2instance, merged_node_edges  = \
                            #            merge_pred_with_gt(data_inst,
                            #            node_cls_names,edge_cls_names,
                            #            nodes_pds_all, edges_pds_all,
                            #            inst_valid_node_cls_indices,seg_valid_node_cls_indices,
                            #            inst_valid_edge_cls_indices,seg_valid_edge_cls_indices,
                            #            noneidx_node_cls,noneidx_edge_cls,
                            #            ignore_missing,self.cfg.DEVICE)

                            # eval_tool.add([scan_id],
                            #                       merged_node_cls,
                            #                       merged_node_cls_gt,
                            #                       merged_edge_cls,
                            #                       merged_edge_cls_gt,
                            #                       [merged_mask2instance],
                            #                       merged_node_edges)

                            # pds = process_pd(**eval_tool.predictions[scan_id]['pd'])
                            # gts = process_gt(**eval_tool.predictions[scan_id]['gt'])

                            # g =     draw_evaluation(scan_id, pds[0], pds[1], gts[0], gts[1], none_name = 'UN',
                            #         pd_only=False, gt_only=False)

                            g.render(os.path.join(
                                self.cfg['training']['out_dir'], self.cfg.name, str(fid)+'_graph'), view=True)
                # else:
                #     data = batch_data
                #     fid = data['fid']
                #     node_pds, edge_pds, pt = process(data)
                #     if pt>0:
                #         acc_time += pt
                #         timer_counter+=1

                #         fuse(nodes_pds_all,nodes_w,node_pds)
                #         fuse(edges_pds_all,edges_w,edge_pds)

                #         eval_tool.add([scan_id],
                #                                       merged_node_cls,
                #                                       merged_node_cls_gt,
                #                                       merged_edge_cls,
                #                                       merged_edge_cls_gt,
                #                                       [merged_mask2instance],
                #                                       merged_node_edges)

                #         pds = process_pd(**eval_tool.predictions[scan_id]['pd'])
                #         gts = process_gt(**eval_tool.predictions[scan_id]['gt'])

                #         g =     draw_evaluation(scan_id, pds[0], pds[1], gts[0], gts[1], none_name = 'UN',
                #                 pd_only=False, gt_only=False)

                #         g.render(os.path.join(self.cfg['training']['out_dir'], self.cfg.name, str(fid)+'_graph'),view=True)
            # if index > 10: break
            break
        print('time:', acc_time, timer_counter, acc_time/timer_counter)

        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k, v in obj_.items():
            # print(k)
            eval_dict[k+'_node_cls'] = v
        for k, v in edge_.items():
            # print(k)
            eval_dict[k+'_edge_cls'] = v

        for k, v in eval_list.items():
            eval_dict[k] = v.avg

        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        return eval_dict, eval_tool
