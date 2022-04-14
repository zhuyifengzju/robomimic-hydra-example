"""An example script to use hydra"""
import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger
from robomimic.utils.dataset import SequenceDataset

import wandb
import cv2

import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from easydict import EasyDict
import json
from hydra.experimental import compose, initialize
import pprint
from torch.utils.tensorboard import SummaryWriter
import kornia

from robosuite import load_controller_config
import robosuite.utils.transform_utils as T
from tqdm import trange

MP_ENABLED = False

import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method, Array
try:
    set_start_method('spawn')
    MP_ENABLED = True
except RuntimeError:
    pass


DEVICE = TorchUtils.get_torch_device(try_to_use_cuda=True)        


def custom_obs_processor(obs):
    # We add a channel dimension and normalize them to be in range [-1, 1]
    return ObsUtils.process_frame(frame=obs, channel_dim=6, scale=255.)

def custom_obs_unprocessor(obs):
    # We do the reverse
    return TensorUtils.to_uint8(ObsUtils.unprocess_frame(frame=obs, channel_dim=6, scale=255.))

def get_observation_for_policy(cfg, obs):
    state_image = np.concatenate((obs["agentview_image"],
                                  obs["robot0_eye_in_hand_image"]), axis=-1)


    obs_dict = {}
    obs_dict["stacked_rgb"] = custom_obs_processor(torch.from_numpy(state_image))
    if "gripper_states" in cfg.algo.observation.modalities.obs.low_dim:
        obs_dict["gripper_states"] = torch.from_numpy(obs["robot0_gripper_qpos"])
    if "joint_states" in cfg.algo.observation.modalities.obs.low_dim:
        obs_dict["joint_states"] = torch.from_numpy(obs["robot0_joint_pos"])
    if "ee_states" in cfg.algo.observation.modalities.obs.low_dim:
        obs_dict["ee_states"] = torch.from_numpy(obs["robot0_eef_pos"])
    obs_dict = TensorUtils.to_device(obs_dict, DEVICE)
    return obs_dict

def eval_loop(cfg, state_dir, model_checkpoint_name, n_eval, success_arr, env_args, rank):
    domain_name = env_args["domain_name"]
    task_name = env_args["task_name"]
    env_kwargs = env_args["env_kwargs"]
    
    env = TASK_MAPPING[domain_name](
        exp_name=task_name,
        **env_kwargs,
    )

    model = FileUtils.policy_from_checkpoint(ckpt_path=model_checkpoint_name)[0]
    
    num_success = 0
    utils.set_manual_seeds(rank * 77)
    if rank == 0:
        eval_range = trange(n_eval)
    else:
        eval_range = range(n_eval)

    for i in eval_range:
        if rank == 0:
            eval_range.set_description(f"Success rate: {num_success} / {i + 1}")
            eval_range.refresh()

        env.reset()
        initial_mjstate = env.sim.get_state().flatten()
        model_xml = env.sim.model.get_xml()
        xml = utils.postprocess_model_xml(model_xml, {"robot0_eye_in_hand": {"pos": "0.05 0.0 0.049999", "quat": "0.0 0.707108 0.707108 0.0"}})            
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(initial_mjstate)
        env.sim.forward()
        for _ in range(5):
            env.step([0.] * 3 + [-1.])

        obs = env._get_observations()

        done = False
        max_steps = cfg.eval.max_steps
        steps = 0
        gripper_history = []

        record_states = []
        while not done and steps < max_steps:

            record_states.append(env.sim.get_state().flatten())
            steps += 1

            obs_dict = get_observation_for_policy(cfg, obs)
            with torch.no_grad():
                action = model(obs_dict)
            obs, reward, done, info = env.step(action)
            done = env._check_success()

            # gripper_history.pop(0)
            # gripper_history.append(obs["robot0_gripper_qpos"])

            if cfg.eval.visualization:
                img = offscreen_visualization(env, use_eye_in_hand=cfg.algo.use_eye_in_hand)

        if done:
            num_success += 1
            for _ in range(10):
                record_states.append(env.sim.get_state().flatten())

        with h5py.File(f"{state_dir}/eval_run_ep_{i}_{done}_rank{rank}.hdf5", "w") as state_file:
        
        # state_file = h5py.File(f"{state_dir}/eval_run_{eval_run_idx}_ep_{i}_{done}.hdf5", "w")
            state_file.attrs["env_name"] = cfg.data.env_name
            state_file.attrs["model_file"] = env.sim.model.get_xml()
            state_file.create_dataset("states", data=np.array(record_states))

    del model
    success_arr[rank] = num_success
    return num_success

def eval_stats(cfg, model_checkpoint_name, eval_num=3, n_eval=30, final=False):

    state_dir = f"{cfg.model_dir.output_dir}/record_states"
    os.makedirs(state_dir, exist_ok=True)

    with open(os.path.join(state_dir, "cfg.json"), "w") as f:
        json.dump(cfg, f, cls=utils.NpEncoder, indent=4)

    data_path = cfg.data.params.data_file_name
    with h5py.File(data_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])

    num_successes = []

    if MP_ENABLED:
        num_procs = 5
    else:
        num_procs = 1

    num_eval = int(n_eval / num_procs)
    processes = []
    
    for eval_run_idx in range(eval_num):
        success_arr = Array('i', range(num_procs))

        if MP_ENABLED:
            for rank in range(num_procs):
                p = mp.Process(target=eval_loop, args=(cfg, state_dir, model_checkpoint_name, num_eval, success_arr, env_args, rank))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            num_success = np.sum(success_arr[:])
        else:
            num_success = eval_loop(cfg, state_dir, model_checkpoint_name, num_eval, success_arr, env_args, 0)
        
        num_successes.append(num_success / n_eval)
        print(f"Total success rate: {num_success} / {n_eval}")

    return np.mean(num_successes), np.std(num_successes)


@hydra.main(config_path="./config", config_name="config")
def main(hydra_cfg):

    # ----------------------------------------------------------------
    # Initialize configs
    # ----------------------------------------------------------------
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # robomimic_ext_cfg = dict(cfg.algo)
    # robomimic_config = config_factory(robomimic_ext_cfg["algo_name"])
    # with robomimic_config.values_unlocked():
    #     robomimic_config.update(robomimic_ext_cfg)
        
    with open("test_hydra.json", "w") as json_file:
        json.dump(dict(cfg), json_file)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)
    exit()
    # file paths for saving models
    model_dir = robomimic_bc_model_dir(cfg)
    cfg.model_dir = model_dir
    output_parent_dir = model_dir.output_dir
    os.makedirs(output_parent_dir, exist_ok=True)

    cfg.hostname = socket.gethostname()
    if robomimic_config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(output_parent_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # Default mode
    # ----------------------------------------------------------------
    # Custom processing of frames
    # ----------------------------------------------------------------
        
    ObsUtils.ImageModality.set_obs_processor(processor=custom_obs_processor)
    ObsUtils.ImageModality.set_obs_unprocessor(unprocessor=custom_obs_unprocessor)
    ObsUtils.initialize_obs_utils_with_config(robomimic_config)

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=cfg.data.params.data_file_name,
        all_obs_keys=robomimic_config.all_obs_keys,
        verbose=True
    )

    obs_normalization_stats = None
    if robomimic_config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # ----------------------------------------------------------------
    # Initialize model
    # ----------------------------------------------------------------
        
    model = algo_factory(
        algo_name=robomimic_config.algo_name,
        config=robomimic_config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=DEVICE
    )
    print(model)

    # ----------------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------------
    
    def get_dataset(filter_by_attribute=None):
        return SequenceDataset(
            hdf5_path=cfg.data.params.data_file_name,
            obs_keys=shape_meta["all_obs_keys"],
            dataset_keys=robomimic_config.train.dataset_keys,
            load_next_obs=False,
            frame_stack=1,
            seq_length=robomimic_config.train.seq_length,                  # length-10 temporal sequences
            pad_frame_stack=True,
            pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
            get_pad_mask=False,
            goal_mode=robomimic_config.train.goal_mode,
            hdf5_cache_mode=robomimic_config.train.hdf5_cache_mode,          # cache dataset in memory to avoid repeated file i/o
            hdf5_use_swmr=robomimic_config.train.hdf5_use_swmr,
            hdf5_normalize_obs=robomimic_config.train.hdf5_normalize_obs,
            filter_by_attribute=filter_by_attribute,       # can optionally provide a filter key here
        )

    dataset = get_dataset()
    train_loader = DataLoader(
        dataset=dataset,
        sampler=dataset.get_dataset_sampler(),       # no custom sampling logic (uniform sampling)
        batch_size=robomimic_config.train.batch_size,     # batches of size 100
        shuffle=True,
        num_workers=robomimic_config.train.num_data_workers,
        drop_last=True# don't provide last batch in dataset pass if it's less than 100 in size
    )
    tmp_checkpoint_name = cfg.model_dir.model_name_prefix + "_tmp.pth"
    env_meta = {
        "env_name": "",
        "type": 1,
        "env_kwargs": {}
    }




    # ----------------------------------------------------------------
    # Start training
    # ----------------------------------------------------------------
    
    
    best_training_loss = None
    best_validation_loss = None
    best_success = None
    model_checkpoint_name = cfg.model_dir.model_name_prefix + ".pth"
    best_model_checkpoint_name = model_checkpoint_name.replace(".pth", "_best.pth")
    for epoch in range(1, robomimic_config.train.num_epochs + 1):
    # for epoch in range(0, 1 + 1):
        train_step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=robomimic_config.experiment.epoch_every_n_steps)
        model.on_epoch_end(epoch)

        training_loss = train_step_log["Loss"]
        if best_training_loss is None or training_los > best_training_loss:
            TrainUtils.save_model(
                model=model,
                config=robomimic_config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=model_checkpoint_name,
                obs_normalization_stats=obs_normalization_stats,
            )
        print(training_loss)

        
        if epoch % 5 == 0:
            validation_step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=robomimic_config.experiment.epoch_every_n_steps, validate=True)
            TrainUtils.save_model(
                model=model,
                config=robomimic_config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=model_checkpoint_name.replace(".pth", "_best_validation.pth"),
                obs_normalization_stats=obs_normalization_stats,
            )
        
        if epoch >= robomimic_config.experiment.rollout.warmstart and epoch % 20 == 0:
            # # ----------------------------------------------------------------
            # # Evaluation
            # # ----------------------------------------------------------------

            model.set_eval()
            # eval_policy = RolloutPolicy(policy=model, obs_normalization_stats=obs_normalization_stats)    
            TrainUtils.save_model(
                model=model,
                config=robomimic_config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=tmp_checkpoint_name,
                obs_normalization_stats=obs_normalization_stats,
            )

            eval_mean, eval_std = eval_stats(cfg, model_checkpoint_name=tmp_checkpoint_name, eval_num=1, n_eval=cfg.algo.experiment.rollout.n, final=False)
            model.set_train()
            
            if best_success is None or eval_mean > best_success:
                best_success = eval_mean
                TrainUtils.save_model(
                    model=model,
                    config=robomimic_config,
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    ckpt_path=best_model_checkpoint_name,
                    obs_normalization_stats=obs_normalization_stats,
                )

    model.set_eval()
    eval_mean, eval_std = eval_stats(cfg, model_checkpoint_name=best_model_checkpoint_name, n_eval=cfg.eval.n_eval, final=True)

if __name__ == "__main__":
    main()
