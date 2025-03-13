from datetime import datetime
from model.network import NetworkFlowModel
from model.crew import MetroCrew
from model.solver import ColumnGenerationSolver
from model.recorder import Recorder
from model.benchmark import BenchmarkAlgorithmSolver
from utils.log_utils import Tee

import json
import argparse
import os

from pathlib import Path


if __name__ == "__main__":

    now = datetime.now()
    time_stamp_str = now.date().strftime("%Y-%m-%d")

    ### parse command line args
    default_cfg_file = os.path.join("getUp_sh_3d", "L3_back_use_sh_d3.json")


    parser = argparse.ArgumentParser(description="Input config file path and rand seed.")
    parser.add_argument("-p", type=str, help="config file path (without path suffix)", required=False, default=default_cfg_file)
    parser.add_argument("-r", type=int, help="rand seed default 0", required=False, default=0)
    parser.add_argument("-R", type=int, help="number of crew", required=False, default=0)
    parser.add_argument("-H", type=int, help="duty option interval", required=False, default=0)
    parser.add_argument("-a", type=float, help="double ratio", required=False, default=0)
    args = parser.parse_args()
    seed = args.r

    ### load config file
    cfg = Path("code_repository/config/") / args.p
    with open(cfg) as f:
        config = json.load(f)
    assert str(cfg).endswith(".json"), f"Wrong config path format: {cfg} should be json file."
    print("Using config file '{}'".format(cfg))

    ### set R
    if args.R != 0:
        config["crew"]["num_of_drivers"] = args.R
    if args.H != 0:
        config["system"]["h"] = args.H
    if args.a > 1e-3:
        config["crew"]["double_ratio"] = args.a

    ### set seed 
    seed = args.r
    config["seed"] = seed
    print(f"Using rand seed {seed}.")

    ### set up data save path
    try:
        data_dir_path =  Path("code_repository/output/data/") / ("data_" + args.p[:-5] + f"_R{config['crew']['num_of_drivers']}_da{config['crew']['double_ratio']}_h{config['system']['h']}_pi{config['work']['pi_ratio']}_lo{config['crew']['preference_settings']['lambda_o']}") / f"r{seed}"
        data_dir_path_abs = os.path.join(os.getcwd(), data_dir_path)
        if not os.path.exists(data_dir_path_abs): 
            print("make dir {} for data saving.".format(data_dir_path_abs))
            Path(data_dir_path_abs).mkdir(parents=True)
        print("Using data saving path '{}'".format(data_dir_path))
        config["data_save_dir"] = data_dir_path

        ### set up log path
        log_path = Path("code_repository/output/log/") / time_stamp_str / ("log_" + args.p[:-5] + f"_R{config['crew']['num_of_drivers']}_da{config['crew']['double_ratio']}_h{config['system']['h']}_pi{config['work']['pi_ratio']}_lo{config['crew']['preference_settings']['lambda_o']}_r{seed}" + ".txt") 
        log_path_abs = os.path.join(os.getcwd(), log_path)
        print(log_path_abs)
        if not os.path.exists(os.path.dirname(log_path_abs)):
            print("make dir {} for logging.".format(os.path.dirname(log_path_abs)))
            Path(os.path.dirname(log_path_abs)).mkdir(parents=True)
        logger = Tee(log_path_abs, "w") ### tee operation
    except FileExistsError:
        print("File exists.")
    
    now = datetime.now()
    time_stamp_str = now.date().strftime("%Y-%m-%d") + '-' + str(now.hour) + '-' + str(now.minute)
    print(f"Now: {time_stamp_str}, for rescheduling...")

    ### BEGIN!
    crew = MetroCrew(config)
    crew.display_crew_info()

    recorder = Recorder(config["data_save_dir"], config)

    bm_solver = BenchmarkAlgorithmSolver(crew, config, recorder)
    bm_solver.run(display=True, data_saving=True)

    network = NetworkFlowModel(config)
    network.create()
    CG_solver = ColumnGenerationSolver(network, crew, config, recorder)

    add_one_step = (config["crew"]["num_of_drivers"] <= 1050) 
    CG_solver.run(display=True, data_saving=True, add_one_step=add_one_step)
    
    recorder.save_json(add_time=False)
    recorder.save_summary_json(add_time=False)

    CG_solver.save_final_result(add_time=False)
    
    recorder.save_config_json(add_time=False)
    
    print("main ok")



