from collections import defaultdict
import os
import json
import trimesh
import argparse
import pathlib
import open3d as o3d
import numpy as np
from pathlib import Path
from tqdm import tqdm
import codeLib
from ssg import define
from ssg.utils import util_3rscan, util_data, util_label, util_ply
from ssg.utils.util_search import SAMPLE_METHODS, find_neighbors
import h5py
import ast
import copy
import logging

def Parser(add_help=True):
    parser = argparse.ArgumentParser(description='Generate custom scene graph dataset from the 3RScan dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=add_help)
    parser.add_argument(
        '-c', '--config', default='./configs/config_default.yaml', required=False)
    parser.add_argument('-o', '--pth_out', type=str, default='../data/tmp',
                        help='pth to output directory', required=True)
    parser.add_argument('--target_scan', type=str, default='', help='')
    parser.add_argument('-l', '--label_type', type=str, default='3RScan160',
                        choices=['3RScan160', 'ScanNet20'], help='label', required=False)
    parser.add_argument('--only_support_type', action='store_true',
                        help='use only support type of relationship')

    parser.add_argument('--debug', action='store_true',
                        help='debug', required=False)
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite or not.')

    # constant
    parser.add_argument('--segment_type', type=str, default='GT',choices=['GT','INSEG','ORBSLAM'])
    return parser

if __name__ == '__main__':
    args = Parser().parse_args()
    cfg = codeLib.Config(args.config)
    lcfg = cfg.data.scene_graph_generation
    outdir = args.pth_out
    debug = args.debug
    debug = True

    '''create log'''
    pathlib.Path(outdir).mkdir(exist_ok=True, parents=True)
    name_log = os.path.split(__file__)[-1].replace('.py', '.log')
    path_log = os.path.join(outdir, name_log)
    logging.basicConfig(filename=path_log, level=logging.INFO, force=True)
    logger_py = logging.getLogger(name_log)
    logger_py.info(f'create log file at {path_log}')
    if debug:
        logger_py.setLevel('DEBUG')
    else:
        logger_py.setLevel('INFO')

    if lcfg.neighbor_search_method == 'BBOX':
        search_method = SAMPLE_METHODS.BBOX
    elif args.neighbor_search_method == 'KNN':
        search_method = SAMPLE_METHODS.RADIUS

    codeLib.utils.util.set_random_seed(cfg.SEED)

    '''create mapping'''
    label_names, * \
        _ = util_label.getLabelMapping(
            args.label_type, define.PATH_LABEL_MAPPING)

    '''get relationships'''
    target_relationships = sorted(codeLib.utils.util.read_txt_to_list(os.path.join(define.PATH_FILE, lcfg.relation + ".txt"))
                                  if not args.only_support_type else define.SUPPORT_TYPE_RELATIONSHIPS)
    if args.segment_type != "GT": # for est. seg., add "same part" in the case of oversegmentation.
        target_relationships.append(define.NAME_SAME_PART)

    ''' get all classes '''
    classes_json = list()
    for key, value in label_names.items():
        if value == '-':
            continue
        classes_json.append(value)

    ''' read target scan'''
    target_scan = []
    if args.target_scan != '':
        target_scan = codeLib.utils.util.read_txt_to_list(args.target_scan)

    '''filter scans according to the target type'''
    with open(os.path.join(cfg.data.path_3rscan_data, lcfg.relation + ".json"), "r") as read_file:
        data = json.load(read_file)
        filtered_data = list()
        '''skip scan'''
        for s in data["scans"]:
            scan_id = s["scan"]
            if len(target_scan) > 0 and scan_id not in target_scan:
                continue
            filtered_data.append(s)
            
    pth_relationships_json = os.path.join(
        args.pth_out, define.NAME_RELATIONSHIPS)
    
    h5f = h5py.File(pth_relationships_json, 'a')
    
    relation_len=0
    for s in tqdm(filtered_data):
        scan_id = s["scan"]
        relation_len += len(s["relationships"])
    print(relation_len)
