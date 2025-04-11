# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations

Usage example: python ./batch_load_scannet_data.py
"""
import os
import sys
import datetime
import numpy as np
from load_scannet_data import export
import pdb

SCANNET_DIR = './scans'
TRAIN_SCAN_NAMES = [line.rstrip() for line in open('meta_data/scannet_train.txt')]
LABEL_MAP_FILE = 'meta_data/scannetv2-labels.combined.tsv'
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,17,18,19,24,25,28,32,33,34,36,39,40,41])  # 40: monitor 41: computer tower
# OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
MAX_NUM_POINT = 50000
OUTPUT_FOLDER = './scannet_train_detection_data_25classes'
# OUTPUT_FOLDER = '/home/www/PycharmProjects/votenet/scannet/scannet_train_detection_data_old'

def export_one_scan(scan_name, output_filename_prefix):    
    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.   
    mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = \
        export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None)

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask,:]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    num_instances = len(np.unique(instance_labels))
    print('Num of instances: ', num_instances)

    bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
    instance_bboxes = instance_bboxes[bbox_mask,:]
    print('Num of care instances: ', instance_bboxes.shape[0])

    N = mesh_vertices.shape[0]
    if N > MAX_NUM_POINT:
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        mesh_vertices = mesh_vertices[choices, :]
        semantic_labels = semantic_labels[choices]
        instance_labels = instance_labels[choices]

    np.save(output_filename_prefix+'_vert.npy', mesh_vertices)
    np.save(output_filename_prefix+'_sem_label.npy', semantic_labels)
    np.save(output_filename_prefix+'_ins_label.npy', instance_labels)
    np.save(output_filename_prefix+'_bbox.npy', instance_bboxes)

def batch_export():
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))                
        os.mkdir(OUTPUT_FOLDER)        
        
    for scan_name in TRAIN_SCAN_NAMES:
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name) 
        if os.path.isfile(output_filename_prefix+'_vert.npy'):
            print('File already exists. skipping.')
            print('-'*20+'done')
            continue
        try:            
            export_one_scan(scan_name, output_filename_prefix)
        except:
            print('Failed export scan: %s'%(scan_name))            
        print('-'*20+'done')


def get_3d_box_statistics():
    train_data = [line.rstrip() for line in open('meta_data/scannetv2_train.txt')]
    size_list = []
    class_list = []
    for scan in sorted(train_data):
        boxes = np.load(OUTPUT_FOLDER + '/' + scan + '_bbox.npy')
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            if box[6] not in OBJ_CLASS_IDS:
                continue
            size_list.append(box[3:6])
            class_list.append(box[6])

    median_boxes3d = np.array([0, 0, 0, 0])  # x, y, z, cnt
    for ob_class in OBJ_CLASS_IDS:
        box3d_list = []
        cnt = 0
        for i in range(len(class_list)):
            if class_list[i] == ob_class:
                box3d_list.append(size_list[i])
                cnt += 1
        median_box3d = np.median(box3d_list, 0)
        median_box3d = np.concatenate((median_box3d, [cnt]))
        print('%f %f %f %d' % (median_box3d[0], median_box3d[1], median_box3d[2], median_box3d[3]))
        median_boxes3d = np.row_stack((median_boxes3d, median_box3d))

    np.savez('./meta_data/scannet_means_25class', median_boxes3d[1:, 0:3])
    return 0


def ground_truth_boxes_vis():
    from model_util_scannet import ScannetDatasetConfig
    from pc_util import write_oriented_bbox
    DC = ScannetDatasetConfig()
    train_data = [line.rstrip() for line in open('meta_data/scannetv2_train.txt')]
    val_data = [line.rstrip() for line in open('meta_data/scannetv2_val.txt')]
    for train_scan in sorted(train_data):
        print('Visualizing ' + train_scan + '----------------------------------------------------\n')
        dump_dir = '../../demo_files/gt_boxes_vis_train/' + train_scan
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        boxes = np.load(OUTPUT_FOLDER + '/' + train_scan + '_bbox.npy')
        for i in range(boxes.shape[0]):
            box = np.concatenate((boxes[i, 0:6], np.array([0])))  # heading angle == 0
            box = box[np.newaxis, :]
            box_class = DC.class2type[DC.nyu40id2class[int(boxes[i, -1])]]
            write_oriented_bbox(box, os.path.join(dump_dir, 'id_%d_gt_box_' % i + box_class + '.ply'))

    for val_scan in sorted(val_data):
        print('Visualizing ' + val_scan + '----------------------------------------------------\n')
        dump_dir = '../../demo_files/gt_boxes_vis_val/' + val_scan
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        boxes = np.load(OUTPUT_FOLDER + '/' + val_scan + '_bbox.npy')
        for i in range(boxes.shape[0]):
            box = np.concatenate((boxes[i, 0:6], np.array([0])))  # heading angle == 0
            box = box[np.newaxis, :]
            box_class = DC.class2type[DC.nyu40id2class[int(boxes[i, -1])]]
            write_oriented_bbox(box, os.path.join(dump_dir, 'id_%d_gt_box_' % i + box_class + '.ply'))


def construct_rel_json():
    import json
    train_rel = []
    val_rel = []
    train_data = [line.rstrip() for line in open('meta_data/scannetv2_train.txt')]
    val_data = [line.rstrip() for line in open('meta_data/scannetv2_val.txt')]
    for train_scan in sorted(train_data):
        print('Scene id: ', train_scan, '-------------------------------------------------------')
        with open(os.path.join(SCANNET_DIR, train_scan + '/' + train_scan + '.txt')) as f:
            lines = f.readlines()
            scene_type = lines[-1].split(' = ')[1][0:-1]

        rel = {'scene_id': train_scan,
               'scene_type': scene_type,
               'core_objects_id': [],  # objects that are important for this scene
               'semantic_relationships': {
                   'identical': [{'object_id': [], 'name': ''}],
                   'same_set': [{'subject_id': [], 'subject_name': [''], 'object_id': [], 'object_name': ['']}],
                   'part of': [{'subject_id': [], 'subject_name': [''], 'object_id': 0, 'object_name': ''}]
               },
               'same_plane': [],
               'geometrical_relationships': [{
                   'relationship_id': 0,
                   'predicate': '',
                   'subject': {'object_id': 0, 'name': ''},
                   'object': {'object_id': 0, 'name': ''}}]
               }
        train_rel.append(rel)
    assert len(train_rel) == 1201
    train_rel_json = json.dumps(train_rel, indent=4)
    with open('./RelationScanNet_train.json', 'w') as train_f:
        train_f.write(train_rel_json)

    for val_scan in sorted(val_data):
        print('Scene id: ', val_scan, '-------------------------------------------------------')
        with open(os.path.join(SCANNET_DIR, val_scan + '/' + val_scan + '.txt')) as f:
            lines = f.readlines()
            scene_type = lines[-1].split(' = ')[1][0:-1]

        rel = {'scene_id': val_scan,
               'scene_type': scene_type,
               'core_objects_id': [],  # objects that are important for this scene
               'semantic_relationships': {
                   'identical': [{'object_id': [], 'name': ''}],
                   'same_set': [{'subject_id': [], 'subject_name': [''], 'object_id': [], 'object_name': ['']}],
                   'part of': [{'subject_id': [], 'subject_name': [''], 'object_id': 0, 'object_name': ''}]
               },
               'same_plane': [],
               'geometrical_relationships': [{
                   'relationship_id': 0,
                   'predicate': '',
                   'subject': {'object_id': 0, 'name': ''},
                   'object': {'object_id': 0, 'name': ''}}]
               }
        val_rel.append(rel)
    assert len(val_rel) == 312
    val_rel_json = json.dumps(val_rel, indent=4)
    with open('./RelationScanNet_val.json', 'w') as train_f:
        train_f.write(val_rel_json)


def generate_videos():
    import cv2
    fps = 40
    size = (1296, 968)
    train_data = [line.rstrip() for line in open('meta_data/scannetv2_train.txt')]
    val_data = [line.rstrip() for line in open('meta_data/scannetv2_val.txt')]
    for train_scan in sorted(train_data):
        print('Scene id: ', train_scan, '-------------------------------------------------------')
        image_dir = SCANNET_DIR + '/' + train_scan + '/sens_data/color/'
        images = os.listdir(image_dir)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(SCANNET_DIR + '/' + train_scan + '/sens_data/' + train_scan + '.mp4', fourcc, fps, size)
        for i in range(len(images)):
            frame = cv2.imread(image_dir + str(i) + '.jpg')
            videoWriter.write(frame)
        videoWriter.release()

    for val_scan in sorted(val_data):
        print('Scene id: ', val_scan, '-------------------------------------------------------')
        image_dir = SCANNET_DIR + '/' + val_scan + '/sens_data/color/'
        images = os.listdir(image_dir)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(SCANNET_DIR + '/' + val_scan + '/sens_data/' + val_scan + '.mp4', fourcc, fps,
                                      size)
        for i in range(len(images)):
            frame = cv2.imread(image_dir + str(i) + '.jpg')
            videoWriter.write(frame)
        videoWriter.release()


def generate_rel_label(dataset='train'):
    import itertools
    import json
    if dataset == 'train':
        with open('./RelationScanNet_train.json', 'r', encoding='utf-8') as f:
            rels = json.load(f)
    elif dataset == 'val':
        with open('./RelationScanNet_val.json', 'r', encoding='utf-8') as f:
            rels = json.load(f)
    else:
        print('Invalid dataset!')
        exit(-1)
    sem_rel_dict = {'background': 0, 'identical': 1, 'same_set': 2, 'part of': 3}
    geo_rel_dict = {'background': 0, 'support': 1, 'on': 2, 'below': 3, 'above': 4, 'near': 5, 'beside': 6,
                    'pushed in': 7, 'pulled out': 8, 'in': 9}
    same_plane_dict = {'background': 0, 'same_plane': 1}
    total_num_iden = 0
    total_num_same_set = 0
    total_num_part_of = 0
    total_num_same_plane = 0
    total_num_support_on = 0
    total_num_below_above = 0
    total_num_near = 0
    total_num_beside = 0
    total_num_push = 0
    total_num_pull = 0
    total_num_in = 0
    total_objects = 0
    for rel in rels:
        scan_name = rel['scene_id']
        if scan_name.split('_')[-1] != '00':
            print('Skip ' + scan_name + '--------------------\n')
            continue
        else:
            print('Start ' + scan_name + '--------------------')
        boxes3d = np.load(os.path.join(OUTPUT_FOLDER, scan_name)+'_bbox.npy')
        num_box = boxes3d.shape[0]
        total_objects += num_box
        sem_rel = rel['semantic_relationships']
        geo_rel = rel['geometrical_relationships']
        same_plane = rel['same_plane']

        # ----------Generate Semantic Relationship Labels----------
        sem_rel_label = np.zeros((num_box, num_box), dtype=np.int)
        num_iden = 0
        num_same_set = 0
        num_part_of = 0
        for identical_rel in sem_rel['identical']:
            object_id = identical_rel['object_id']
            if len(object_id) == 0:
                break
            pairs_id = list(itertools.permutations(object_id, 2))
            num_iden += len(pairs_id)
            for pair_id in pairs_id:
                sem_rel_label[pair_id[0], pair_id[1]] = sem_rel_dict['identical']

        for same_set_rel in sem_rel['same_set']:
            subject_id = same_set_rel['subject_id']
            if len(subject_id) == 0:
                break
            object_ids = same_set_rel['object_id']
            for obj_id in object_ids:
                sem_rel_label[subject_id, obj_id] = sem_rel_dict['same_set']
                sem_rel_label[obj_id, subject_id] = sem_rel_dict['same_set']
                num_same_set += 2

        for part_of_rel in sem_rel['part of']:
            subject_ids = part_of_rel['subject_id']
            if len(subject_ids) == 0:
                break
            object_id = part_of_rel['object_id']
            for sub_id in subject_ids:
                sem_rel_label[sub_id, object_id] = sem_rel_dict['part of']
                num_part_of += 1
        print('Number of objects: {}\nSemantic Relationships: \n\tnumber of identical: {}\n\tnumber of same_set: '
              '{}\n\tnumber of part_of: {}'.format(num_box, num_iden, num_same_set, num_part_of))
        total_num_iden += num_iden
        total_num_same_set += num_same_set
        total_num_part_of += num_part_of
        sem_rel_label = sem_rel_label * (np.ones([num_box, num_box]) - np.eye(num_box, num_box))
        np.save(os.path.join(OUTPUT_FOLDER, scan_name) + '_sem_rel.npy', sem_rel_label)

        # ----------Generate Same Plane Labels----------
        same_plane_label = np.zeros((num_box, num_box), dtype=np.int)
        num_same_plane = 0
        if len(same_plane) != 0:
            for same_plane_rel in same_plane:
                pairs_id = list(itertools.permutations(same_plane_rel, 2))
                num_same_plane += len(pairs_id)
                for pair_id in pairs_id:
                    same_plane_label[pair_id[0], pair_id[1]] = same_plane_dict['same_plane']
        print('Same Plane: \n\tnumber of same_plane: {}'.format(num_same_plane))
        total_num_same_plane += num_same_plane
        same_plane_label = same_plane_label * (np.ones([num_box, num_box]) - np.eye(num_box, num_box))
        np.save(os.path.join(OUTPUT_FOLDER, scan_name) + '_same_plane_rel.npy', same_plane_label)

        # ----------Generate Geometrical Relationship Labels----------
        geo_rel_label = np.zeros((num_box, num_box), dtype=np.int)
        num_support_on = 0
        num_below_above = 0
        num_near = 0
        num_beside = 0
        num_push = 0
        num_pull = 0
        num_in = 0
        for g_rel in geo_rel:
            predicate = g_rel['predicate']
            if predicate == '':
                break
            predicate_id = geo_rel_dict[predicate]
            sub_id = g_rel['subject']['object_id']
            obj_id = g_rel['object']['object_id']
            if predicate_id in [1, 3]:
                geo_rel_label[sub_id, obj_id] = predicate_id
                geo_rel_label[obj_id, sub_id] = predicate_id + 1
                if predicate_id == 1:
                    num_support_on += 2
                else:
                    num_below_above += 2
            elif predicate_id in [2, 4]:
                geo_rel_label[sub_id, obj_id] = predicate_id
                geo_rel_label[obj_id, sub_id] = predicate_id - 1
                if predicate_id == 2:
                    num_support_on += 2
                else:
                    num_below_above += 2
            elif predicate_id in [5, 6]:
                geo_rel_label[sub_id, obj_id] = predicate_id
                geo_rel_label[obj_id, sub_id] = predicate_id
                if predicate_id == 5:
                    num_near += 2
                else:
                    num_beside += 2
            else:
                geo_rel_label[sub_id, obj_id] = predicate_id
                if predicate_id == 7:
                    num_push += 1
                elif predicate_id == 8:
                    num_pull += 1
                else:
                    num_in += 1
        print('Geometrical Relationships: \n\tnumber of support/on: {}\n\tnumber of below/above: {}'
              '\n\tnumber of near: {}\n\tnumber of beside: {}\n\tnumber of pushed_in: {}\n\tnumber of pulled out: {}'
              '\n\tnumber of in: {}'.format(num_support_on, num_below_above, num_near, num_beside, num_push, num_pull, num_in))
        total_num_support_on += num_support_on
        total_num_below_above += num_below_above
        total_num_near += num_near
        total_num_beside += num_beside
        total_num_push += num_push
        total_num_pull += num_pull
        total_num_in += num_in
        geo_rel_label = geo_rel_label * (np.ones([num_box, num_box]) - np.eye(num_box, num_box))
        np.save(os.path.join(OUTPUT_FOLDER, scan_name) + '_geo_rel.npy', geo_rel_label)
        print('Finish ' + scan_name + '--------------------\n')
    print('Finish all scans!--------------------------------------------------')
    print('Relationship Statistics: \nNumber of objects: {}\nSemantic Relationships: \n\tnumber of identical: {}'
          '\n\tnumber of same_set: {}\n\tnumber of part_of: {}\nGeometrical Relationships: \n\tnumber of support/on: {}'
          '\n\tnumber of below/above: {}\n\tnumber of near: {}\n\tnumber of beside: {}\n\tnumber of pushed_in: {}'
          '\n\tnumber of pulled out: {}\n\tnumber of in: {}\nSame Plane: \n\tnumber of same_plane: {}'.format
          (total_objects, total_num_iden, total_num_same_set, total_num_part_of, total_num_support_on,
           total_num_below_above, total_num_near, total_num_beside, total_num_push, total_num_pull, total_num_in,
           total_num_same_plane))


if __name__ == '__main__':
    batch_export()

    """Calculate mean sizes"""
    # get_3d_box_statistics()

    """Visualize GT boxes"""
    # ground_truth_boxes_vis()

    """Construct relationship label file"""
    # construct_rel_json()

    """Generate RGB videos for each scene"""
    # generate_videos()

    """Generate Relationship Labels"""
    generate_rel_label('val')