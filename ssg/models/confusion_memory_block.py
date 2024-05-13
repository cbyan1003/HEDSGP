import torch
import numpy as np
from sklearn.neighbors import KernelDensity
import pickle
import os
import json

class Confusion_Memory_Block():
    def __init__(self):
        super(Confusion_Memory_Block, self).__init__()
        self.REL_SIZE = 9
        self.ENTITY_SIZE = 20
        self.front_box = []
        self.back_box = []
        self.relation = []
        self.same_relation = {}
        self.mean_front_translation = 0
        self.mean_front_scale = 0
        self.mean_back_translation = 0
        self.mean_back_scale = 0
        self.val_front_box = []
        self.val_back_box = []
        self.val_relation = []
        self.Eval_front_box = []
        self.Eval_back_box = []
        self.Eval_relation = []
        
    def update(self, front_box, back_box, relation, mode=''):
        if mode == 'Train':
            self.front_box.extend(front_box)
            self.back_box.extend(back_box)
            self.relation.extend(relation)
            if len(self.front_box) > 1000:
                self.front_box = self.front_box[-1000:]
            if len(self.back_box) > 1000:
                self.back_box = self.back_box[-1000:]
            if len(self.relation) > 1000:
                self.relation = self.relation[-1000:]
            # self.check_front()
            # self.check_back()
            # self.check_relation()
        elif mode == 'Val':
            self.val_front_box.extend(front_box)
            self.val_back_box.extend(back_box)
            self.val_relation.extend(relation)
        elif mode == 'Eval':
            self.Eval_front_box.extend(front_box)
            self.Eval_back_box.extend(back_box)
            self.Eval_relation.extend(relation)

    def check_front(self):
        for each in self.front_box:
            if each['front_cls_gt_id'] != each['front_cls_pred_id']:
                each['mask'] = 0
            elif each['front_cls_gt_id'] == each['front_cls_pred_id']:
                each['mask'] = 1
            else:
                raise Exception('Error')
            
    def check_back(self):
        for each in self.back_box:
            if each['back_cls_gt_id'] != each['back_cls_pred_id']:
                each['mask'] = 0
            elif each['back_cls_gt_id'] == each['back_cls_pred_id']:
                each['mask'] = 1
            else:
                raise Exception('Error')
            
    def check_relation(self):
        for index, each in enumerate(self.relation):
            if each['rel_cls_pred_id'] != each['rel_cls_gt_id']:
                each['mask'] = 0
            elif each['rel_cls_pred_id'] == each['rel_cls_gt_id']:
                each['mask'] = 1
            else:
                raise Exception('Error')

    def positive_mask(self):
        same_relation = {}
        for index, each in enumerate(self.relation):
            data = each['rel_cls_gt_id'].squeeze(0).item()
            if data not in same_relation:
                same_relation[data] = []
            same_relation[data].append(index)
        self.same_relation = same_relation
    
    def compute_mean_box(self):
        front_translation = []
        front_scale = []
        back_translation = []
        back_scale = []
        for rel_id, index in self.same_relation.items():
            for i in index:
                front_translation.append(self.front_box[i]['front_translation'])
                front_scale.append(self.front_box[i]['front_scale'])
                back_translation.append(self.back_box[i]['back_translation'])
                back_scale.append(self.back_box[i]['back_scale'])
        
        mean_front_translation = torch.mean(torch.stack(front_translation, dim=0), dim=0)
        mean_front_scale = torch.mean(torch.stack(front_scale, dim=0), dim=0)
        mean_back_translation = torch.mean(torch.stack(back_translation, dim=0), dim=0)
        mean_back_scale = torch.mean(torch.stack(back_scale, dim=0), dim=0)

        self.mean_front_translation = mean_front_translation
        self.mean_front_scale = mean_front_scale
        self.mean_back_translation = mean_back_translation
        self.mean_back_scale = mean_back_scale
            
    def save_features(self, mode=''):
        path = os.getcwd()
        if mode == 'Val':
            front_val_features = [{'front_box_min':each['front_box_min'], 
                'front_box_max':each['front_box_max'], 
                'front_cls_gt_id':each['front_cls_gt_id'].squeeze(0).item(),
                'front_cls_pred_id':each['front_cls_pred_id'].squeeze(0).item()}
                    for each in self.val_front_box]
            try:
                with open(os.path.join(path, 'front_val_features.pkl'), 'wb') as file:
                    pickle.dump(front_val_features, file)
            except Exception as e:
                print(f"write error: {e}") 

            back_val_features = [{'back_box_min':each['back_box_min'], 
                'back_box_max':each['back_box_max'], 
                'back_cls_gt_id':each['back_cls_gt_id'].squeeze(0).item(),
                'back_cls_pred_id':each['back_cls_pred_id'].squeeze(0).item()}
                    for each in self.val_back_box]   
            try:
                with open(os.path.join(path, 'back_val_features.pkl'), 'wb') as file:
                    pickle.dump(back_val_features, file)
            except Exception as e:
                print(f"write error: {e}")        
                        
        elif mode == 'Eval':
            
            front_test_features = [{'front_box_min':each['front_box_min'], 
                'front_box_max':each['front_box_max'], 
                'front_cls_gt_id':each['front_cls_gt_id'].squeeze(0).item(),
                'front_cls_pred_id':each['front_cls_pred_id'].squeeze(0).item()}
                    for each in self.front_box]
            try:
                with open(os.path.join(path, 'front_test_features.pkl'), 'wb') as file:
                    pickle.dump(front_test_features, file)
            except Exception as e:
                print(f"write error: {e}") 
                
            back_test_features = [{'back_box_min':each['back_box_min'], 
                'back_box_max':each['back_box_max'], 
                'back_cls_gt_id':each['back_cls_gt_id'].squeeze(0).item(),
                'back_cls_pred_id':each['back_cls_pred_id'].squeeze(0).item()}
                    for each in self.back_box]
            try:
                with open(os.path.join(path, 'back_test_features.pkl'), 'wb') as file:
                    pickle.dump(back_test_features, file)
            except Exception as e:
                print(f"write error: {e}")         
        
    def compute_IDM_BND(self):
        '''
        input:
            front_test_features: list of dict, each dict contains 'front_cls_pred_id', 'front_cls_gt_id', 'front_box_min', 'front_box_max'
            front_val_features: list of dict, each dict contains 'front_cls_pred_id', 'front_cls_gt_id', 'front_box_min', 'front_box_max'
        '''
        try:
            with open('front_val_features.pkl', 'rb') as file:
                front_val_features = pickle.load(file)
            with open('back_val_features.pkl', 'rb') as file:
                back_val_features = pickle.load(file)
            with open('front_test_features.pkl', 'rb') as file:
                front_test_features = pickle.load(file)
            with open('back_test_features.pkl', 'rb') as file:
                back_test_features = pickle.load(file)
        except Exception as e:
                print(f"write error: {e}")
                
        # front part      
        front_corpus_density_matrix = {}
        pred_gt_id_cls_front_dict = {}
        
        # test dataset
        front_cls_pred_id_test_list = []
        front_cls_gt_id_test_list = []
        
        # val dataset 

        front_cls_pred_id_val_list = []
        front_cls_gt_id_val_list = []

        for i in range(len(front_val_features)):
            try:
                front_cls_pred_id_val_list.append(front_val_features[i]['front_cls_pred_id'])
                front_cls_gt_id_val_list.append(front_val_features[i]['front_cls_gt_id'])
                front_cls_pred_id_test_list.append(front_test_features[i]['front_cls_pred_id'])
                front_cls_gt_id_test_list.append(front_test_features[i]['front_cls_gt_id'])
                pred_gt_id_cls_front_dict[f"{front_val_features[i]['front_cls_gt_id']}_{front_val_features[i]['front_cls_pred_id']}"]
            except:
                pred_gt_id_cls_front_dict[f"{front_val_features[i]['front_cls_gt_id']}_{front_val_features[i]['front_cls_pred_id']}"] = []

            pred_gt_id_cls_front_dict[f"{front_val_features[i]['front_cls_gt_id']}_{front_val_features[i]['front_cls_pred_id']}"].append(\
                        torch.cat([front_val_features[i]['front_box_min'].squeeze(0), front_val_features[i]['front_box_max'].squeeze(0)], dim=0))

        front_corpus_density_mat = np.zeros((len(front_cls_pred_id_test_list), self.ENTITY_SIZE, self.ENTITY_SIZE)) - 99999999 
        front_corpus_density_weight = np.zeros((self.ENTITY_SIZE, self.ENTITY_SIZE))

        for i in range(len(front_val_features)):
            front_corpus_density_weight[front_cls_pred_id_val_list[i], front_cls_gt_id_test_list[i]] += 1.0/len(front_cls_gt_id_test_list)
        
        front_test_feature_list = []
        for i in front_test_features:
            front_test_feature_list.append(torch.cat([i['front_box_min'], i['front_box_max']], dim=0))
        front_test_feature_list = torch.stack(front_test_feature_list, dim=0)
        
        for i in range(self.ENTITY_SIZE): # gt
            for j in range(self.ENTITY_SIZE): # pred
                try:
                    data_ij = pred_gt_id_cls_front_dict[f"{front_val_features[i]['front_cls_gt_id']}_{front_val_features[j]['front_cls_pred_id']}"]
                    front_corpus_density_matrix[f"{front_val_features[i]['front_cls_gt_id']}_{front_val_features[j]['front_cls_pred_id']}"] = \
                    KernelDensity(kernel = 'gaussian', bandwidth = 1.0).fit(torch.stack(data_ij).detach().cpu().numpy())
                except:
                    front_corpus_density_matrix[f"{front_val_features[i]['front_cls_gt_id']}_{front_val_features[j]['front_cls_pred_id']}"] = -1
                
                kde_model = front_corpus_density_matrix[f"{front_val_features[i]['front_cls_gt_id']}_{front_val_features[j]['front_cls_pred_id']}"]
                if kde_model != -1:
                    front_corpus_density_mat[:, i, j] = kde_model.score_samples(front_test_feature_list.detach().cpu().numpy())
                             
        front_pred_diag_list = []

        for i in range(len(front_cls_pred_id_test_list)):
            max_front_diag_value = -99999999.0
            arg_front_max_idx = -1
            front_pred_diag_list.append(front_corpus_density_mat[i, front_test_features[i]['front_cls_pred_id'], front_test_features[i]['front_cls_pred_id']])#矩阵的对角线

        front_IDM_score = []
        front_BND_score = []
        for i in range(len(front_test_features)):
            front_IDM_score.append(np.clip(np.sum(np.exp(
                front_corpus_density_weight[:, front_test_features[i]['front_cls_pred_id']] * front_corpus_density_mat[i, :, front_test_features[i]['front_cls_pred_id']] - \
                    front_corpus_density_weight[front_test_features[i]['front_cls_gt_id'], front_test_features[i]['front_cls_pred_id']] * front_pred_diag_list[i])), -1, 100))
            front_BND_score.append(np.clip(np.sum(np.exp(
                np.diagonal(front_corpus_density_weight * front_corpus_density_mat[i]) - \
                    front_corpus_density_weight[front_test_features[i]['front_cls_gt_id'], front_test_features[i]['front_cls_pred_id']] * front_pred_diag_list[i])), -1, 100))

        # back part      
        back_corpus_density_matrix = {}
        pred_gt_id_cls_back_dict = {}
        
        # test dataset
        back_cls_pred_id_test_list = []
        back_cls_gt_id_test_list = []
        
        # val dataset 

        back_cls_pred_id_val_list = []
        back_cls_gt_id_val_list = []

        for i in range(len(back_val_features)):
            try:
                back_cls_pred_id_val_list.append(back_val_features[i]['back_cls_pred_id'])
                back_cls_gt_id_val_list.append(back_val_features[i]['back_cls_gt_id'])
                back_cls_pred_id_test_list.append(back_test_features[i]['back_cls_pred_id'])
                back_cls_gt_id_test_list.append(back_test_features[i]['back_cls_gt_id'])
                pred_gt_id_cls_back_dict[f"{back_val_features[i]['back_cls_gt_id']}_{back_val_features[i]['back_cls_pred_id']}"]
            except:
                pred_gt_id_cls_back_dict[f"{back_val_features[i]['back_cls_gt_id']}_{back_val_features[i]['back_cls_pred_id']}"] = []

            pred_gt_id_cls_back_dict[f"{back_val_features[i]['back_cls_gt_id']}_{back_val_features[i]['back_cls_pred_id']}"].append(\
                        torch.cat((back_val_features[i]['back_box_min'], back_val_features[i]['back_box_max'])))

        back_corpus_density_mat = np.zeros((len(back_cls_pred_id_test_list), self.ENTITY_SIZE, self.ENTITY_SIZE)) - 99999999 
        back_corpus_density_weight = np.zeros((self.ENTITY_SIZE, self.ENTITY_SIZE))

        for i in range(len(back_val_features)):
            back_corpus_density_weight[back_cls_pred_id_val_list[i], back_cls_gt_id_test_list[i]] += 1.0/len(back_cls_gt_id_test_list)
                    
        back_test_feature_list = []
        for i in back_test_features:
            back_test_feature_list.append(torch.cat([i['back_box_min'], i['back_box_max']], dim=0))
        back_test_feature_list = torch.stack(back_test_feature_list, dim=0)
                    
        for i in range(self.ENTITY_SIZE): # gt
            for j in range(self.ENTITY_SIZE): # pred
                try:
                    data_ij = pred_gt_id_cls_back_dict[f"{back_val_features[i]['back_cls_gt_id']}_{back_val_features[j]['back_cls_pred_id']}"]
                    back_corpus_density_matrix[f"{back_val_features[i]['back_cls_gt_id']}_{back_val_features[j]['back_cls_pred_id']}"] = \
                        KernelDensity(kernel = 'gaussian', bandwidth = 1.0).fit(torch.stack(data_ij).detach().cpu().numpy())
                except:
                    back_corpus_density_matrix[f"{back_val_features[i]['back_cls_gt_id']}_{back_val_features[j]['back_cls_pred_id']}"] = -1
                
                kde_model = back_corpus_density_matrix[f"{back_val_features[i]['back_cls_gt_id']}_{back_val_features[j]['back_cls_pred_id']}"]
                if kde_model != -1:
                    back_corpus_density_mat[:, i, j] = kde_model.score_samples(back_test_feature_list.detach().cpu().numpy())
                             
        back_pred_diag_list = []

        for i in range(len(back_cls_pred_id_test_list)):
            max_back_diag_value = -99999999.0
            arg_back_max_idx = -1
            back_pred_diag_list.append(back_corpus_density_mat[i, back_test_features[i]['back_cls_pred_id'], back_test_features[i]['back_cls_pred_id']])

        back_IDM_score = []
        back_BND_score = []
        for i in range(len(back_test_features)):
            back_IDM_score.append(np.clip(np.sum(np.exp(
                back_corpus_density_weight[:, back_test_features[i]['back_cls_pred_id']] * back_corpus_density_mat[i, :, back_test_features[i]['back_cls_pred_id']] - \
                    back_corpus_density_weight[back_test_features[i]['back_cls_gt_id'], back_test_features[i]['back_cls_pred_id']] * back_pred_diag_list[i])), -1, 100))
            back_BND_score.append(np.clip(np.sum(np.exp(
                np.diagonal(back_corpus_density_weight * back_corpus_density_mat[i]) - \
                    back_corpus_density_weight[back_test_features[i]['back_cls_gt_id'], back_test_features[i]['back_cls_pred_id']] * back_pred_diag_list[i])), -1, 100))
        
        path = os.getcwd()
        try:
            with open(os.path.join(path, 'front_IDM_score.json'), 'w') as file:
                json.dump(front_IDM_score, file)
            with open(os.path.join(path, 'front_BND_score.json'), 'w') as file:
                json.dump(front_BND_score, file)
            with open(os.path.join(path, 'back_IDM_score.json'), 'w') as file:
                json.dump(back_IDM_score, file)
            with open(os.path.join(path, 'back_BND_score.json'), 'w') as file:
                json.dump(back_BND_score, file)
        except Exception as e:
            print(f"write error: {e}")   
        
        
