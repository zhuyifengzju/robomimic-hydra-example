import json

import yaml
from robomimic.config import config_factory
import robomimic
import os

with open("experiment.json", "r") as f:
    ext_cfg = json.load(f)

with open('test.yaml', 'w') as yml:
    yaml.safe_dump(ext_cfg, yml, allow_unicode=True)

# exit()

# config = config_factory(ext_cfg["algo_name"])

# def get_dict_config(v):
#     if type(v) == robomimic.config.config.Config:
#         return v.to_dict()
#     return v

# dict_config = config.to_dict()

# os.makedirs("bc")

# # for k, v in dict_config.items():
# #     if type(v) == robomimic.config.config.Config:
