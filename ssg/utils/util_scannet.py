import json

def load_semseg(json_file, name_mapping_dict=None, mapping=True):
    '''
    Create a dict that maps instance id to label name.
    If name_mapping_dict is given, the label name will be mapped to a corresponding name.
    If there is no such a key exist in name_mapping_dict, the label name will be set to '-'

    Parameters
    ----------
    json_file : str
        The path to semseg.json file
    name_mapping_dict : dict, optional
        Map label name to its corresponding name. The default is None.
    mapping : bool, optional
        Use name_mapping_dict as name_mapping or name filtering.
        if false, the query name not in the name_mapping_dict will be set to '-'
    Returns
    -------
    instance2labelName : dict
        Map instance id to label name.

    '''
    instance2labelName = {}
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        for segGroups in data['segGroups']:
            # print('id:',segGroups["id"],'label', segGroups["label"])
            # if segGroups["label"] == "remove":continue
            labelName = segGroups["label"]
            if name_mapping_dict is not None:
                if mapping:
                    if not labelName in name_mapping_dict:
                        labelName = 'none'
                    else:
                        labelName = name_mapping_dict[labelName]
                else:
                    if not labelName in name_mapping_dict.values():
                        labelName = 'none'

            # segGroups["label"].lower()
            instance2labelName[segGroups["id"]] = labelName.lower()
    return instance2labelName

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def read_all_scan_ids(path:str):
    # get all splits
    splits = []
    splits.append(read_txt_to_list(path))
    scan_ids = []
    for v in splits:
        scan_ids += v
    # train_ids = read_txt_to_list(os.path.join(define.PATH_FILE,'train_scans.txt'))
    # val_ids = read_txt_to_list(os.path.join(define.PATH_FILE,'validation_scans.txt'))
    # test_ids = read_txt_to_list(os.path.join(define.PATH_FILE,'test_scans.txt'))
    return scan_ids

def read_scannet_info(pth):
    with open(pth, 'r') as f:
        lines = f.readlines()
        output = dict()
        for line in lines:
            split = line.rstrip().split(' = ')
            output[split[0]] = split[1]
    return output