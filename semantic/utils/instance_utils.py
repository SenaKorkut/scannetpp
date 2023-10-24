import os
import json

import numpy as np


def load_ids(filename):
    ids = np.loadtxt(filename, dtype=np.int32)
    return ids

# ------------ Instance Utils ------------ #

class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        # mesh_vert_instances: instance ids for all verts
        # instance_id: id of the current instance = sem * 1000 + inst
        if (instance_id == -1):
            return
        self.instance_id     = int(instance_id)
        self.label_id    = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        # divide by 1000 to get the sem class ID
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"]    = self.label_id
        dict["vert_count"]  = self.vert_count
        dict["med_dist"]    = self.med_dist
        dict["dist_conf"]   = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id     = int(data["instance_id"])
        self.label_id        = int(data["label_id"])
        self.vert_count      = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist    = float(data["med_dist"])
            self.dist_conf   = float(data["dist_conf"])

    def __str__(self):
        return "("+str(self.instance_id)+")"

def read_instance_prediction_file(filename, pred_path):
    lines = open(filename).read().splitlines()
    instance_info = {}
    abs_pred_path = os.path.abspath(pred_path)
    for line in lines:
        parts = line.split(' ')
        if len(parts) != 3:
            print('invalid instance prediction file. Expected (per line): [rel path prediction] [label id prediction] [confidence prediction]')
        if os.path.isabs(parts[0]):
            print('invalid instance prediction file. First entry in line must be a relative path')
        mask_file = os.path.join(os.path.dirname(filename), parts[0])
        mask_file = os.path.abspath(mask_file)
        # check that mask_file lives inside prediction path
        if os.path.commonprefix([mask_file, abs_pred_path]) != abs_pred_path:
            print('predicted mask {} in prediction text file {} points outside of prediction path.'.format(mask_file,filename))

        info            = {}
        info["label_id"] = int(float(parts[1]))
        info["conf"]    = float(parts[2])
        instance_info[mask_file]  = info
    return instance_info


def get_instances(ids, valid_class_ids, valid_class_labels, id2label):
    instances = {}
    # each class name
    for label in valid_class_labels:
        # instances in this class
        instances[label] = []
    # unique instance IDs
    instance_ids = np.unique(ids)
    # ignore instance ID 0!
    for id in instance_ids:
        if id == 0:
            continue
        # create new instance object
        inst = Instance(ids, id)
        if inst.label_id in valid_class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    # list of instances as dicts
    return instances
            


