from model.network import NetworkFlowModel
from model.crew import MetroCrew
from model.recorder import Recorder
from utils.net_utils import transform_individual_plan

import networkx as nx
import copy
from collections import defaultdict
from itertools import combinations
import time
import numpy as np
import pandas as pd
import os
from multiprocessing import Process
import multiprocessing as mp
import json

import pyomo.environ as pyo
from pathlib import Path
from datetime import datetime



class ColumnGenerationSolver():

    def __init__(self, network: NetworkFlowModel, crew: MetroCrew, config:dict, recorder:Recorder):

        self.network = network
        self.crew = crew
        self.recorder = recorder
        self.config = config

        self.G = network.G

        self.lines = config["system"]["lines"]
        self.line2idx = {l:i for i, l in enumerate(self.lines)}

        print("\n\n******************************************************************************")
        print("Start Solver......")

        self.ts2idx = dict() # {day-depot-start_time: k(train service index)}
        self.lts2idx = [list() for _ in range(len(self.lines))] # [[k1, k2, ...], [k101, k102, ...]]
        self._map_train_services()

        self.init_path_num = self.crew.n_drivers

        self.n_path_lists = [] # updating
        self.n_label_lists = [] # updating
        self.n_a_lists = [] # updating

        self.n_io_lists = [] # for interday counts
        self.n_io_depots_list = [] # for interday counts

        self.m = np.array(self.crew.m)


        ### params
        self.parallel = config["solver"]["parallel"]
        self.max_iter = config["solver"]["max_iter"]
        self.Q = config["solver"]["Q"]
        self.early_stop = config["solver"]["early_stop"]

        self.save_data = config["save_data"]
        self.save_dir = config["data_save_dir"]


        self.n_path = 0 # updating
        ### cheat
        self.Q = 3

        ### preference heterogeneity - only for paths selected
        self.selected_path_idxes = []
        self.cross_line = dict() # dict of n_r booleans {path_id: bool}
        self.in_out_depots = dict() # dict of n_r lists {path_id: list}

        self.path_pools = dict()
        for q in self.crew.qualifications:
            self.path_pools[q] = []
        self.s_path_pools = dict()
        for q in self.crew.qualifications:
            self.s_path_pools[q] = []

        print(f"There are {len(self.ts2idx)} train services in total to be covered.")

        self.final_planning_result = None # for rescheduling

    def run_noco(self, display=True, data_saving=True):
        self.run_CG_without_cross_operations(display, data_saving)

    def run(self, display=True, data_saving=True, add_one_step=True) -> str:
        """
        run the whole algorithm:
        comparing heuristics and CG
        """

        h_obj, h_coverage, h_x, heuristics_time = self.run_old_heuristics(display, data_saving)

        stuff_for_ref = self.run_CG(display, data_saving)

        if add_one_step:
            self.run_CG_one_step(data_saving)


        if self.save_data:
            obj_val_records, min_reduced_costs, add_gammas, mus, nu_sums, xi_sums, optim_flag, cg_time, final_obj, final_coverage, final_x_opt, obj_val, transfer_arc_counts, upper_obj, upper_coverage, upper_x, x, iter = stuff_for_ref
            
            iterating_records = {"obj": obj_val_records, "rc": min_reduced_costs, "gamma": add_gammas, "mu": mus, "nus": nu_sums, "xis": xi_sums}
            iterating_df = pd.DataFrame(iterating_records)

            now = datetime.now()
            time_stamp_str = now.date().strftime("%Y-%m-%d") + '-' + str(now.hour) 
            iterating_df.to_excel(Path(self.save_dir) / "iterating_df.xlsx")
            result_file_path = Path(self.save_dir) / "res.txt"
            with open(result_file_path, "w") as f:
                f.write("Basic info:\n|W|={}, Tw={}, |R|={}, D={}\nlicenses_cnt={}\nmax_iter={}, early_stop={}, optim_flag={}\n".format(\
                    self.network.W, self.network.Tw, self.crew.n_drivers, self.network.D, self.m, self.max_iter, self.early_stop, optim_flag))

                f.write("Time elapsed:  Heuristic {}     CG {} ({} iterations)\n".format(heuristics_time, cg_time, iter))
                f.write("Summary:\n heuristic {:.5f}, {:.5f}, {}\n ub(res) {:.5f}, {:.5f}, {} \n lb {:.5f}\n Gap={:.5f} \nImprove: of_obj: {:.6f} of_coverage: {:.6f}\n".format(\
                    h_obj, h_coverage[-1][-1], h_coverage[-1][-2]-h_coverage[-1][-3], final_obj, final_coverage[-1][-1], final_coverage[-1][-2]-final_coverage[-1][-3], obj_val, \
                        (final_obj - obj_val) / obj_val, \
                            (-final_obj + h_obj)/h_obj, (final_coverage[-1][-1] - h_coverage[-1][-1]) / h_coverage[-1][-1]))

                f.write("intra_transfer_count: {}\n".format(transfer_arc_counts))
                f.write("\nHeuristic:\n")
                f.write("{:.5f}\t{}\n{}({}, {})\n".format(h_obj, h_coverage, h_x, len(h_x), sum(h_x)))
                f.write("First_UB:\n")
                f.write("{:.5f}\t{}\n{}({}, {})\n".format(upper_obj, upper_coverage, upper_x, len(upper_x), sum(upper_x)))
                f.write("LB:\n")
                f.write("{:.5f}\t{}\n{}({}, {})\n".format(obj_val, final_coverage, x, len(x), sum(x)))
                f.write("Result(UB as the best feasible solution):\n")
                f.write("{:.5f}\t{}\n{}({}, {})\n".format(final_obj, final_coverage, final_x_opt, len(final_x_opt), sum(final_x_opt)))

            print("data saved in {}".format(self.save_dir))
        
        if data_saving:
            record_name = "conf"
            self.recorder.create_record("conf")
            conf_info = {
                "W": self.network.W,
                "Tw": self.network.Tw,
                "R": self.crew.n_drivers,
                "double_ratio": self.crew.double_ratio
            }
            self.recorder.set_data_dict(record_name, conf_info)

        return

    def save_final_result(self, add_time=False):
        """
        save the resulting paths as json
        """
        assert self.final_planning_result is not None, "no final result generated"

        result = dict()

        for driver_id, plan in self.final_planning_result.items():
            result[driver_id] = plan.to_dict()
        

        if add_time:
            now = datetime.now()

            time_stamp_str = now.date().strftime("%Y-%m-%d") + '-' + str(now.hour)

            file_name = "final_result" + '-' + f"{self.config['crew']['num_of_drivers']}-" + time_stamp_str + ".json"
            dir_name = Path(self.save_dir, time_stamp_str)
            if not dir_name.exists():
                dir_name.mkdir(parents=True)
        else:
            file_name = "final_result" + '-' + f"{self.config['crew']['num_of_drivers']}" + ".json"
            dir_name = Path(self.save_dir)
        with open(dir_name / file_name, "w") as f:
            json.dump(result, f)

        print(f"Save final results json in file {dir_name / file_name}")
        


    def run_old_heuristics(self, display, data_saving:bool=True):
        

        print("#################################")
        run_start = time.time()
        print("Begin Path-based Heuristics......")
        self.gamma, self.theta, self.a, self.ms = self._path_based_heuristics(display=display)

        coverage = []
        total_ts = 0
        total_cvg = 0
        for i, (l, lts) in enumerate(zip(self.lines, self.lts2idx)):
            covered_train_theta = self.theta[:, total_ts:total_ts+len(lts)]
            n_covered = 0
            total_ts += len(lts)
            for j in range(len(self.gamma)):
                n_covered += sum(covered_train_theta[j, :])
            print(f"For line {l}, train service coverage {n_covered} / {len(lts)} = {n_covered/len(lts):.3f}")
            total_cvg += n_covered
            coverage.append((n_covered, len(lts), round(n_covered/len(lts), 3)))
        
        h_obj, h_coverage, h_x = sum(self.gamma)+self.network.pi_summation, coverage, [1]*len(self.gamma)
        print("In total, train coverage: {} / {} = {}".format(total_cvg, total_ts, round(total_cvg / total_ts, 3)))
        coverage.append((total_cvg, total_ts, round(total_cvg / total_ts, 3)))
        heuristics_time = round(time.time() - run_start, 2)
        print("Finish heuristics with elapsed time {}s...".format(heuristics_time))
        print("Heuristic results:")
        print("h_val", h_obj, "  h_coverage", h_coverage)
        print("heuristic path selection:", h_x)


        if data_saving:
            record_name = "old_path_based_heuristics"
            self.recorder.create_record(record_name)

            self.recorder.set_data(record_name, "obj", h_obj)
            self.recorder.set_data(record_name, "cvg", h_coverage)
            self.recorder.set_data(record_name, "t", heuristics_time)

            record_name = "Summary"
            self.recorder.create_record(record_name)
            self.recorder.set_data(record_name, f"one_line_obj", h_obj)
            self.recorder.set_data(record_name, f"one_line_cvg", h_coverage[-1][-1])
            self.recorder.set_data(record_name, f"one_line_t", heuristics_time)
        
        return h_obj, h_coverage, h_x, heuristics_time


    def run_CG(self, display, data_saving):
        ##### begin CG algorithm
        print("#################################")
        print("Begin Column Generation......")
        run_start = time.time()
        self.gamma, self.theta, self.a, self.ms = self._construct_initial_path_set(init_path_num=self.init_path_num, display=display)
        print(f"Final param m(line liscence pattern counts) for {list(self.crew.pat2idx.keys())}: {self.m}")

        obj_val_records = []
        min_reduced_costs = []
        add_gammas = []
        mus = []
        nu_sums = []
        xi_sums = []
        optim_flag = False
        upper_obj = 0
        upper_x = None
        upper_coverage = None
        print("Begin iterating with parallel mode on: {}".format(self.parallel))

        transfer_arc_counts = 0
        interday_cross_counts = 0

        for iter in range(self.max_iter):
            print(f"\n\nCG iteration {iter}:")

            ### RLMP
            obj_val, dual_solutions, x, coverage = self._solve_RLMP(iter)
            print(f"Obj val for iter {iter}: {obj_val}")
            print("xi: ", dual_solutions["xi"])
            print("mu: ", dual_solutions["mu"])

            if iter == 0:
                upper_obj = obj_val
                upper_x = x
                upper_coverage = coverage
            
            ### early stop
            if iter > self.early_stop[0]:
                if abs(obj_val - obj_val_records[-self.early_stop[1]]) < 1e-3: # no improvement
                    print("Early stop triggered at iteration {} (for obj val {:.3f} vs. {:.3f})".format(iter, obj_val, obj_val_records[-self.early_stop[1]]))
                    break

            obj_val_records.append(round(obj_val, 3))
            mus.append(round(dual_solutions["mu"], 3))
            nu_sums.append(round(sum(dual_solutions["nu"]), 3))
            xi_sums.append(round(sum(dual_solutions["xi"]),3))
            
            ### Sub-P
            shortest_path_label, theta_p, a_p, ms_p, gamma_p, io_depots = self._solve_Sub_P(dual_solutions, iter, display)
            
            # optimality check
            min_reduced_costs.append(round(shortest_path_label[0], 3))
            add_gammas.append(round(gamma_p, 3))
            
            if shortest_path_label[0] >= 0 or abs(shortest_path_label[0] - 0) < 1e-4:
            # if shortest_path_label[0] > 1e-4: # buggy - non-stop
                optim_flag = True
                print("Optimal condition is met.")
                print("Path count: ", self.n_path)
                print(dual_solutions)
                break

            # update paths
            self.n_path_lists.append(shortest_path_label[2])
            self.n_path += 1
            self.n_label_lists.append(shortest_path_label[1])
            self.n_a_lists.append(np.sum(a_p))
            self.n_io_lists.append(len(io_depots))
            self.n_io_depots_list.append(io_depots)

            # updata params
            theta_p, a_p, ms_p = np.array(theta_p), np.array(a_p), np.array(ms_p)
            theta_p, a_p, ms_p = theta_p[np.newaxis, :], a_p[np.newaxis, :], ms_p[np.newaxis, :]
            self.gamma = np.append(self.gamma, values=gamma_p)
            self.theta = np.concatenate([self.theta, theta_p], axis=0)
            self.a = np.concatenate([self.a, a_p], axis=0)
            self.ms = np.concatenate([self.ms, ms_p], axis=0)
            assert len(self.gamma) == self.n_path and self.theta.shape[0] == self.n_path, "path num error."
        
        print("Obj val records: ", f"{obj_val_records}")
        print("Min reduced cost records: ", f"{min_reduced_costs}")
        print("add gamma records: ", f"{add_gammas}")
        print("mu records: ", f"{mus}")
        print("nu sum records: ", f"{nu_sums}")
        print("xi sum records: ", f"{xi_sums}")
        print("||||||||\n")

        # finally P
        cg_prep_time = round(time.time() - run_start, 3)
        self.cg_prep_time = cg_prep_time

        print("Reach optimality: {}".format(optim_flag))
        final_obj, final_x_opt, final_coverage = self._solve_P()
        print("Final obj val: ", f"{final_obj}")
        print("Final x_opt: ", f"{final_x_opt}")
        cg_time = round(time.time() - run_start, 2)
        print("Total time elapsed for running solver: ", cg_time)

        # analysis final solution
        assert len(final_x_opt) == len(self.n_path_lists), "size wrong."
        assert len(final_x_opt) == len(self.n_io_depots_list)
        for path_id, p_x in enumerate(final_x_opt):
            if abs(p_x-0) < 1e-3: 
                continue
            self.selected_path_idxes.append(path_id)
            transfer_arc_counts += self.n_label_lists[path_id][1]
            if self.n_io_lists[path_id] > 1:
                interday_cross_counts += 1

            ### here record path info for preference heterogeneity
            selected_path = self.n_path_lists[path_id]
            ## record the sign-in and sign-out depots && as well as the lines required
            depots_used = []
            path_lines_traversed = set()
            for node1, node2 in zip(selected_path[:-1], selected_path[1:]):
                arc_type = self.network.G.edges[node1, node2]["label"]
                if arc_type == "si":
                    n2s = node2.split("-")
                    depots_used.append(int(n2s[4]))
                    path_lines_traversed.add(int(n2s[3]))
                elif arc_type == "so":
                    n1s = node1.split("-")
                    depots_used.append(int(n1s[4]))
                    path_lines_traversed.add(int(n1s[3]))
                elif arc_type == "w":
                    n1s = node1.split("-")
                    path_lines_traversed.add(int(n1s[3]))
                elif arc_type == "t":
                    n1s = node1.split("-") # here........
                    n2s = node2.split("-")
                    path_lines_traversed.add(int(n1s[3]))
                    path_lines_traversed.add(int(n2s[3]))
                    

            cross_line = (len(set(depots_used)) > 2)
            # record the sign-in and sign-out depots
            self.in_out_depots[path_id] = depots_used
            self.cross_line[path_id] = cross_line
            # add path index to the corresponding path pool
            for line_num in range(1, len(path_lines_traversed)+1):
                for pattern in combinations(sorted(list(path_lines_traversed)), line_num):
                    self.path_pools[tuple(pattern)].append(path_id)

            ### RESTART：here new path pool
            self.s_path_pools[tuple(sorted(path_lines_traversed))].append(path_id)  
            if len(tuple(sorted(path_lines_traversed))) == 3:
                print(f"Path id {path_id} {path_lines_traversed}")
                print(self.n_path_lists[path_id])

        for pool, pool_paths in self.path_pools.items():
            print(f"New Path pool {pool}: {self.s_path_pools[pool]} -> New Crew pool {pool}: {self.crew.s_crew_pools[pool]}")
            print(f"New Path pool {pool} len: {len(self.s_path_pools[pool])} -> New Crew pool {pool} len: {len(self.crew.s_crew_pools[pool])}")

        print("Total transfer arc counts: {}".format(transfer_arc_counts))
        print("Total interday cross crew counts: {}".format(interday_cross_counts))
        print("Upper results in initialized case:")
        print("ub_obj_val", upper_obj, "  coverage", upper_coverage)
        print("init path selection:", upper_x)

        

        version=0
        assignment_cost, allocation = self.run_assignment_after_CG(version, data_saving)

        cg_time = round(time.time() - run_start, 3)
        print("Total time elapsed for running solver: ", cg_time)
        
        if data_saving:
            record_name = f"CG_version_{version}"
            self.recorder.create_record(record_name)
            
            self.recorder.set_data(record_name, "obj", final_obj)
            self.recorder.set_data(record_name, "cvg", final_coverage)
            self.recorder.set_data(record_name, "t", cg_time)
            self.recorder.set_data(record_name, "iter", iter)
            self.recorder.set_data(record_name, "optim_flag", optim_flag)

            self.recorder.set_data(record_name, "init_set_obj", upper_obj)
            self.recorder.set_data(record_name, "init_set_cvg", upper_coverage)

            self.recorder.set_data(record_name, "linear_obj", obj_val_records[-1])

            self.recorder.set_data(record_name, "interday_cnt", interday_cross_counts)
            self.recorder.set_data(record_name, "intraday_cnt", transfer_arc_counts)

            self.recorder.set_data(record_name, "final_x", list(final_x_opt))
            self.recorder.set_data(record_name, "init_x", list(upper_x))
            self.recorder.set_data(record_name, "gamma", list(self.gamma))

            self.recorder.set_data(record_name, "obj_val_records", obj_val_records)
            self.recorder.set_data(record_name, "min_reduced_costs", min_reduced_costs)



            self.recorder.set_data(record_name, "assign_cost", assignment_cost)
            self.recorder.set_data(record_name, "allocation", allocation)


            self.recorder.set_data(record_name, "CG_total_obj", assignment_cost + final_obj)
            
            record_name = "Summary"
            self.recorder.create_record(record_name)
            self.recorder.set_data(record_name, f"CG_prep_time", cg_prep_time)
            self.recorder.set_data(record_name, f"CG_{version}_obj", assignment_cost + final_obj)
            self.recorder.set_data(record_name, f"CG_{version}_cvg", final_coverage[-1][-1])
            self.recorder.set_data(record_name, f"CG_{version}_t", cg_time)
            self.recorder.set_data(record_name, f"CG_{version}_solver_t", cg_time - cg_prep_time)
            if self.recorder.has_data(record_name, "one_line_obj") and self.recorder.has_data(record_name, "one_line_cvg"):
                one_line_obj = self.recorder.get_data(record_name, f"one_line_obj")
                one_line_cvg = self.recorder.get_data(record_name, f"one_line_cvg")
                self.recorder.set_data(record_name, f"CG_{version}_obj_delta", (one_line_obj - (assignment_cost + final_obj)) / one_line_obj)
                self.recorder.set_data(record_name, f"CG_{version}_cvg_delta", (final_coverage[-1][-1] - one_line_cvg) / one_line_cvg)

            if self.recorder.has_data(record_name, "bm_obj") and self.recorder.has_data(record_name, "bm_cvg"):
                bm_obj = self.recorder.get_data(record_name, f"bm_obj")
                bm_cvg = self.recorder.get_data(record_name, f"bm_cvg")
                self.recorder.set_data(record_name, f"CG_{version}_obj_delta_bm", (bm_obj - (assignment_cost + final_obj)) / bm_obj)
                self.recorder.set_data(record_name, f"CG_{version}_cvg_delta_bm", (final_coverage[-1][-1] - bm_cvg) / bm_cvg)

            if self.recorder.has_data(record_name, "one_line_obj") and self.recorder.has_data(record_name, "one_line_cvg") and self.recorder.has_data(record_name, "bm_obj") and self.recorder.has_data(record_name, "bm_cvg"):
                bm_obj = self.recorder.get_data(record_name, f"bm_obj")
                bm_cvg = self.recorder.get_data(record_name, f"bm_cvg")
                one_line_obj = self.recorder.get_data(record_name, f"one_line_obj")
                one_line_cvg = self.recorder.get_data(record_name, f"one_line_cvg")
                self.recorder.set_data(record_name, f"H_obj_delta_bm", (bm_obj - one_line_obj) / bm_obj)
                self.recorder.set_data(record_name, f"H_cvg_delta_bm", (one_line_cvg - bm_cvg) / bm_cvg)


        stuff_for_ref = [obj_val_records, min_reduced_costs, add_gammas, mus, nu_sums, xi_sums, optim_flag, cg_time, final_obj, final_coverage, final_x_opt, obj_val, transfer_arc_counts, upper_obj, upper_coverage, upper_x, x, iter]
    

        ########################################
        # save path data for rescheduling 241114
        ########################################
        final_planning_result = dict() # {driver_id : IndividualCrewPlan}
        for path_id, driver_id in allocation.items():
            final_planning_result[driver_id] = transform_individual_plan(self.network.G, self.n_path_lists[path_id], self.config)
            
        self.final_planning_result = final_planning_result
        
        return stuff_for_ref

    def run_CG_one_step(self, data_saving):
        """
        need to be run after run_CG
        """
        assert self.cg_prep_time is not None and self.n_path >= self.init_path_num

        obj, coverage, t = self._solve_P_one_step()
        t = t + self.cg_prep_time

        if data_saving:
            record_name = f"CG_one_step"
            self.recorder.create_record(record_name)
            
            self.recorder.set_data(record_name, "obj", obj)
            self.recorder.set_data(record_name, "cvg", coverage)
            self.recorder.set_data(record_name, "t", t)
            
            record_name = f"Summary"
            self.recorder.create_record(record_name)
            
            self.recorder.set_data(record_name, "CG_os_obj", obj)
            self.recorder.set_data(record_name, "CG_os_cvg", coverage[-1][-1])
            self.recorder.set_data(record_name, "CG_os_t", t)
            self.recorder.set_data(record_name, "CG_os_solver_t", t-self.cg_prep_time)
            
            cg_obj = self.recorder.get_data(record_name, f"CG_0_obj")
            cg_cvg = self.recorder.get_data(record_name, f"CG_0_cvg")
            cg_t = self.recorder.get_data(record_name, f"CG_0_t")
            cg_solver_t = self.recorder.get_data(record_name, f"CG_0_solver_t")
            self.recorder.set_data(record_name, "CG_os_obj_delta", (cg_obj - obj) / cg_obj)
            self.recorder.set_data(record_name, "CG_os_cvg_delta", (coverage[-1][-1] - cg_cvg) / cg_cvg)
            self.recorder.set_data(record_name, "CG_os_t_delta", (t - cg_t) / cg_t)
            self.recorder.set_data(record_name, "CG_os_solver_t_delta", (t - cg_t) / cg_solver_t)

    def _init_CG_savings(self):

        self.init_path_num = self.crew.n_drivers

        self.n_path_lists = [] # updating
        self.n_label_lists = [] # updating
        self.n_a_lists = [] # updating

        self.n_io_lists = [] # for interday counts
        self.n_io_depots_list = [] # for interday counts

        self.m = np.array(self.crew.m)

        ### preference heterogeneity - only for paths selected
        self.selected_path_idxes = []
        self.cross_line = dict() # dict of n_r booleans {path_id: bool}
        self.in_out_depots = dict() # dict of n_r lists {path_id: list}

        self.path_pools = dict()
        for q in self.crew.qualifications:
            self.path_pools[q] = []
        self.s_path_pools = dict()
        for q in self.crew.qualifications:
            self.s_path_pools[q] = []

    def run_CG_without_cross_operations(self, display, data_saving):
        print("#################################")
        print("Begin CG_without_CO......")
        run_start = time.time()
        self._init_CG_savings()

        
        # # adjust crew 
        # for i  in range(self.crew.n_drivers):
        #     self.crew.drivers[i].license = self.crew.drivers[i].license[0:1]
        self.gamma, self.theta, self.a, self.ms = self._construct_initial_path_set(init_path_num=self.init_path_num, display=display, noco=True)

        obj_val_records = []
        min_reduced_costs = []
        add_gammas = []
        mus = []
        nu_sums = []
        xi_sums = []
        optim_flag = False
        upper_obj = 0
        upper_x = None
        upper_coverage = None

        transfer_arc_counts = 0
        interday_cross_counts = 0

        for iter in range(self.max_iter):
            print(f"\n\nCG noco iteration {iter}:")

            ### RLMP
            obj_val, dual_solutions, x, coverage = self._solve_RLMP(iter)
            print(f"Obj val for iter {iter}: {obj_val}")
            print("xi: ", dual_solutions["xi"])
            print("mu: ", dual_solutions["mu"])

            if iter == 0:
                upper_obj = obj_val
                upper_x = x
                upper_coverage = coverage
            
            ### early stop
            if iter > self.early_stop[0]:
                if abs(obj_val - obj_val_records[-self.early_stop[1]]) < 1e-3: # no improvement
                    print("Early stop triggered at iteration {} (for obj val {:.3f} vs. {:.3f})".format(iter, obj_val, obj_val_records[-self.early_stop[1]]))
                    break

            obj_val_records.append(round(obj_val, 3))
            mus.append(round(dual_solutions["mu"], 3))
            nu_sums.append(round(sum(dual_solutions["nu"]), 3))
            xi_sums.append(round(sum(dual_solutions["xi"]),3))
            
            ### Sub-P
            shortest_path_label, theta_p, a_p, ms_p, gamma_p, io_depots = self._solve_Sub_P(dual_solutions, iter, display, noco=True)
            
            # optimality check
            min_reduced_costs.append(round(shortest_path_label[0], 3))
            add_gammas.append(round(gamma_p, 3))
            
            if shortest_path_label[0] >= 0 or abs(shortest_path_label[0] - 0) < 1e-4:
            # if shortest_path_label[0] > 1e-4: # buggy - non-stop
                optim_flag = True
                print("Optimal condition is met.")
                print("Path count: ", self.n_path)
                print(dual_solutions)
                break

            # update paths
            self.n_path_lists.append(shortest_path_label[2])
            self.n_path += 1
            self.n_label_lists.append(shortest_path_label[1])
            self.n_a_lists.append(np.sum(a_p))
            self.n_io_lists.append(len(io_depots))
            self.n_io_depots_list.append(io_depots)

            # updata params
            theta_p, a_p, ms_p = np.array(theta_p), np.array(a_p), np.array(ms_p)
            theta_p, a_p, ms_p = theta_p[np.newaxis, :], a_p[np.newaxis, :], ms_p[np.newaxis, :]
            self.gamma = np.append(self.gamma, values=gamma_p)
            self.theta = np.concatenate([self.theta, theta_p], axis=0)
            self.a = np.concatenate([self.a, a_p], axis=0)
            self.ms = np.concatenate([self.ms, ms_p], axis=0)
            assert len(self.gamma) == self.n_path and self.theta.shape[0] == self.n_path, "path num error."
        

        # finally P
        cg_prep_time = round(time.time() - run_start, 3)
        self.cg_prep_time = cg_prep_time

        print("Reach noco optimality: {}".format(optim_flag))
        final_obj, final_x_opt, final_coverage = self._solve_P()
        print("Final obj val: ", f"{final_obj}")
        print("Final x_opt: ", f"{final_x_opt}")
        cg_time = round(time.time() - run_start, 2)
        print("Total time elapsed for running noco solver: ", cg_time)

        # analysis final solution
        assert len(final_x_opt) == len(self.n_path_lists), "size wrong."
        assert len(final_x_opt) == len(self.n_io_depots_list)
        for path_id, p_x in enumerate(final_x_opt):
            if abs(p_x-0) < 1e-3: 
                continue
            self.selected_path_idxes.append(path_id)
            transfer_arc_counts += self.n_label_lists[path_id][1]
            if self.n_io_lists[path_id] > 1:
                interday_cross_counts += 1

            ### here record path info for preference heterogeneity
            selected_path = self.n_path_lists[path_id]
            ## record the sign-in and sign-out depots && as well as the lines required
            depots_used = []
            path_lines_traversed = set()
            for node1, node2 in zip(selected_path[:-1], selected_path[1:]):
                arc_type = self.network.G.edges[node1, node2]["label"]
                if arc_type == "si":
                    n2s = node2.split("-")
                    depots_used.append(int(n2s[4]))
                    path_lines_traversed.add(int(n2s[3]))
                elif arc_type == "so":
                    n1s = node1.split("-")
                    depots_used.append(int(n1s[4]))
                    path_lines_traversed.add(int(n1s[3]))
                elif arc_type == "w":
                    n1s = node1.split("-")
                    path_lines_traversed.add(int(n1s[3]))
                elif arc_type == "t":
                    n1s = node1.split("-") # here........
                    n2s = node2.split("-")
                    path_lines_traversed.add(int(n1s[3]))
                    path_lines_traversed.add(int(n2s[3]))
                    

            cross_line = (len(set(depots_used)) > 2)
            # record the sign-in and sign-out depots
            self.in_out_depots[path_id] = depots_used
            self.cross_line[path_id] = cross_line
            # add path index to the corresponding path pool
            for line_num in range(1, len(path_lines_traversed)+1):
                for pattern in combinations(sorted(list(path_lines_traversed)), line_num):
                    self.path_pools[tuple(pattern)].append(path_id)

            ### RESTART：here new path pool
            self.s_path_pools[tuple(sorted(path_lines_traversed))].append(path_id)  
            if len(tuple(sorted(path_lines_traversed))) == 3:
                print(f"Path id {path_id} {path_lines_traversed}")
                print(self.n_path_lists[path_id])



        
        

    def run_assignment_after_CG(self, CG_version, data_saving):
        
        data = {
            "selected_path_idxes": self.selected_path_idxes, \
            "transfer_costs": [self.crew.drivers[cid].transfer_cost for cid in range(self.crew.n_drivers)], \
            "prefered_depots": [self.crew.drivers[cid].prefered_depots for cid in range(self.crew.n_drivers)], \

            "qualifications": self.crew.qualifications, \
            "Qs": self.crew.Qs, \
            "c_crew_pools": [(k, v) for k, v in self.crew.crew_pools.items()], \
            "s_crew_pools": [(k, v) for k, v in self.crew.s_crew_pools.items()], \
            "c_path_pools": [(k, v) for k, v in self.path_pools.items()], \
            "s_path_pools": [(k, v) for k, v in self.s_path_pools.items()], \
            
            "in_out_depots": self.in_out_depots, \
            "cross_line": self.cross_line
        }

        asmodel = AssignmentSolver(data, self.config)

        assignment_cost, allocation = asmodel.solve()

        for pid, cid in allocation.items():
            print(f"Driver {cid} prefs: {self.crew.drivers[cid].prefered_depots}")
            print(f"Path {pid} traversed depots: {self.n_io_depots_list[pid]}")
        
        return assignment_cost, allocation





    def _construct_initial_path_set(self, init_path_num, interday_ablation=False, display=False, noco=False):
        """
        Requires Further Design:
        for different crews - different paths
        - considering pattern counts 20240101

        1. determine init_path_sum max
        2. get init paths and init a, theta params
        3. reconstruct edge weights
        
        """

        assert len(self.n_path_lists) == 0 and len(self.n_label_lists) == 0 and len(self.n_io_depots_list) == 0

        self.n_path = init_path_num
        theta_init = np.zeros((init_path_num, len(self.ts2idx)))
        a_init = np.zeros((init_path_num, len(self.crew.patterns)))
        ms_init = np.zeros((init_path_num, len(self.crew.patterns)))
        gamma_init = np.zeros(init_path_num)


        ### generating initial path set and initialize parameters
        for path_id in range(int(init_path_num)):
            print(f"Generating initial path {path_id} with driver_id={self.crew.drivers_reordered[path_id].driver_id}...")


            ### considering patterns
            pattern = self.crew.drivers_reordered[path_id].license

            if interday_ablation == "inter" or noco:
                sub_G = self.network.subnetworks[tuple(pattern)]
                shortest_path_label, theta, a, ms, gamma, io_depots = None, None, None, None, None, None
                line_selected = None
                for one_line in pattern:
                    inter_sub_G = copy.deepcopy(sub_G)
                    # infinitize sign-in arcs for lines outside one_line
                    for u, v, attrs in inter_sub_G.edges(data=True):
                        # if attrs["label"] == "si":
                        if attrs["label"] == "si":
                            if v.split("-")[3] != str(one_line):
                                inter_sub_G[u][v]["c"] = float("inf")
                        elif  attrs["label"] == "so":
                            if u.split("-")[3] != str(one_line):
                                inter_sub_G[u][v]["c"] = float("inf")

                    shortest_path_label_, theta_, a_, ms_, gamma_, io_depots_  = self._outer_labeling_shortest_path_topo(inter_sub_G, display=display)
                    if gamma is None or gamma_ < gamma:
                        shortest_path_label, theta, a, ms, gamma, io_depots = shortest_path_label_, theta_, a_, ms_, gamma_, io_depots_
                        line_selected = one_line



            else:
                sub_G = self.network.subnetworks[tuple(pattern)]
                shortest_path_label, theta, a, ms, gamma, io_depots = self._outer_labeling_shortest_path_topo(sub_G, display=display)
                print("Choose to work on lines with a={}\n".format(a))

            theta_init[path_id,:] = theta
            a_init[path_id, :] = a
            ms_init[path_id, :] = ms
            gamma_init[path_id] = gamma

            self.n_path_lists.append(shortest_path_label[2])
            self.n_label_lists.append(shortest_path_label[1])
            self.n_a_lists.append(np.sum(a))
            self.n_io_lists.append(len(io_depots))
            self.n_io_depots_list.append(io_depots)
            # need to change weights for all subgraphs
            for subgraph in self.network.subnetworks.values():
                _ = self._alter_working_arc_weights(subgraph, shortest_path_label[2])

        return gamma_init, theta_init, a_init, ms_init
    
    def _path_based_heuristics(self, display=False):
        """
        Fooled construct init path set
        """
        init_path_num = self.crew.n_drivers
        self.n_path = init_path_num

        theta_init = np.zeros((init_path_num, len(self.ts2idx)))
        a_init = np.zeros((init_path_num, len(self.crew.patterns)))
        ms_init = np.zeros((init_path_num, len(self.crew.patterns)))
        gamma_init = np.zeros(init_path_num)


        ### generating initial path set and initialize parameters
        total_depots_violated = 0
        for path_id in range(init_path_num):
            print(f"Generating heuristic path {path_id}...with driver_id={self.crew.drivers[path_id].driver_id}...")


            ### considering patterns
            pattern = self.crew.drivers[path_id].license
            print("With pattern: {}".format(pattern))
            shortest_path_label, theta, a, ms, gamma, io_depots, path_depots_violated = None, None, None, None, None, None, 0
            for one_line in pattern:
                pat = [one_line]
                sub_G = self.network.subnetworks_c2[tuple(pat)]
                shortest_path_label_, theta_, a_, ms_, gamma_, io_depots_ = self._outer_labeling_shortest_path_topo(sub_G, display=display)

                ### additional irrational preference costs emmm no better choice
                depots_used = []
                pat_depots_violated = 0
                for node1, node2 in zip(shortest_path_label_[2][:-1], shortest_path_label_[2][1:]):
                    arc_type = self.network.G.edges[node1, node2]["label"]
                    if arc_type == "si":
                        n2s = node2.split("-")
                        depots_used.append(int(n2s[4]))
                    elif arc_type == "so":
                        n1s = node1.split("-")
                        depots_used.append(int(n1s[4]))
                assert len(depots_used) <= self.network.n_si * 2 and len(depots_used) % 2 == 0, "Heuristics: wrong depots used..."
                for d in depots_used:
                    if d not in self.crew.drivers[path_id].prefered_depots:
                        gamma_ += self.crew.lambda_o
                        pat_depots_violated += 1
                    

                if gamma is None or gamma_ < gamma:
                    shortest_path_label, theta, a, ms, gamma, io_depots = shortest_path_label_, theta_, a_, ms_, gamma_, io_depots_
                    path_depots_violated = pat_depots_violated
            total_depots_violated += path_depots_violated

            theta_init[path_id,:] = theta
            a_init[path_id, :] = a
            ms_init[path_id, :] = ms
            gamma_init[path_id] = gamma

            # self.n_path_lists.append(shortest_path_label[2])
            # need to change weights for all subgraphs
            for subgraph in self.network.subnetworks_c2.values():
                _ = self._alter_working_arc_weights(subgraph, shortest_path_label[2])
        
        ### recover working arc weights - no need now
        self.network.subnetworks_c2 = None

        return gamma_init, theta_init, a_init, ms_init


    def _solve_Sub_P(self, dual:dict, CG_iter:int, display=False, ablation_strategy=None, noco=False):
        print('Begin solving Sub-P in iteration {}...'.format(CG_iter))
        sstart = time.time()

        ### parallel mode - Abandoned. after 0525
        self_data = {"Q":self.Q, "pat2idx":self.crew.pat2idx, \
                     "lines": self.lines, "line2idx": self.line2idx, \
                        "ts2idx": self.ts2idx, "n_si": self.network.n_si, \
                        "Qs": self.crew.Qs, \
                            "n_t": self.network.n_t, "W": self.network.W}

        results = list()
        for line_num in range(len(self.lines), 0, -1):
            for pattern in combinations(self.lines, line_num):
                if pattern not in self.crew.valid_patterns: continue
                if noco:
                    if len(pattern) > 1: continue
                results.append(self._one_subnetwork_labeling_topo(self_data, self.network.subnetworks[pattern], pattern, dual, False, save_place=None))

        selected_label, selected_pattern = min(results, key=lambda x:x[0][0])
        
        print("cost: ", round(selected_label[0], 3), "resource vector: ", selected_label[1])
        print(f"Time elapsed for parallel labeling algorithm: {time.time() - sstart}s")
        if display:
            self._pprint_path(self.G, selected_label[2])


        ## extract a, theta, gamma
        shortest_path_label = selected_label
        G = self.G
        gamma = 0
        a = [0] * len(self.crew.patterns) # (list of Boolean)
        ms = [0] * len(self.crew.patterns) # (list of Boolean)
        theta = [0] * len(self.ts2idx)

        io_depots = list()
        lines_traversed = set()
        for node1, node2 in zip(shortest_path_label[2][:-1], shortest_path_label[2][1:]):
            gamma += G.edges[node1, node2]["c"]
            arc_type, c = G.edges[node1, node2]["label"], G.edges[node1, node2]["c"]
            if arc_type == "w": # get a and theta
                line = int(node1.split("-")[3])
                lines_traversed.add(line)
                a[self.line2idx[line]] = 1
                train_service_idx = self._map_working_arc(node1)
                theta[train_service_idx] = 1

            elif arc_type == "si":
                io_depots.append(int(node2.split("-")[3]))
                line = int(node2.split("-")[3])
                lines_traversed.add(line)

            elif arc_type == "so":
                io_depots.append(int(node1.split("-")[3]))
                line = int(node1.split("-")[3])
                lines_traversed.add(line)
                
            elif arc_type == "t":
                line1 = int(node1.split("-")[3])
                line2 = int(node2.split("-")[3])
                lines_traversed.add(line1)
                lines_traversed.add(line2)

    
        lines_traversed = sorted(list(lines_traversed))
        for line_num in range(1, len(lines_traversed)+1):
            for pattern in combinations(lines_traversed, line_num):
                pat_i = self.crew.pat2idx[pattern]
                a[pat_i] = 1
        if len(lines_traversed) > 0:
            one_pat_i = self.crew.pat2idx[tuple(lines_traversed)]
            ms[one_pat_i] = 1

        return shortest_path_label, theta, a, ms, gamma, io_depots
    
    @staticmethod
    def _one_subnetwork_labeling_topo(self_data:dict, G:nx.DiGraph, sub_lines:list, dual:dict, display=False, save_place=None, strategy:int=0):
        """
        used in solving sub_P
        """
        
        Q = self_data["Q"] ### capacity for labels for one node
        sub_start = time.time()
        print("Labeling begin for pattern {}...".format(sub_lines))


        labels_dict = defaultdict(list) # {node: [label1, label2, ...]} - Q set for each node

        # dual cost for subnetwork selection
        # emmm Here is actually not a right way for putting dual value "xi", since the sub_lines are not necessarily passed
        # but, for now, just code like this
        dc = -dual["mu"]
        # RE: rewrite objective for sub-P
        for sid, s_qs in enumerate(self_data["Qs"]):
            for q in s_qs:
                if q == tuple(sorted(sub_lines)): # "sorted" returns a list...
                    dc -= dual["xi"][sid]
                    # print(f"xi for {q} in {sid}") # for debug
                    break

        # for line_num in range(1, len(sub_lines)+1):
        #     for s_qs in combinations(sub_lines, line_num):
        #         pat_i = self_data["pat2idx"][pattern]
        #         dc -= dual["xi"][pat_i]
                
        labels_dict["sc"].append([dc, [[0]*self_data["W"], 0, [0,]*len(self_data["lines"]), [0]*self_data["W"]], ["sc",]]) # [cost, resource vector(list) - [sign-in-cnt, transfer_cnt, [...]], path(list)]

        topo_id = 0
        print("subnetwork with {} nodes".format(G.number_of_nodes()))
        for n1 in nx.topological_sort(G):
            topo_id += 1
            nbrs = G[n1]
            for label1 in labels_dict[n1]:
                ### 2 extending node labels
                for n2, attrs in nbrs.items():
                    ### 2.1.1 create new vector
                    new_vec = copy.deepcopy(label1[1])
                    if attrs["label"] == "si":
                        w = int(n1.split("-")[1])
                        new_vec[0][w] += 1
                    elif attrs["label"] == "t":
                        new_vec[1] += 1
                    elif attrs["label"] == "w":
                        l = int(n1.split("-")[3])
                        l_idx = self_data["line2idx"][l]
                        new_vec[2][l_idx] += 1
                    ### 2.1.2 feasibility check
                    elif attrs["label"] == "m":
                        w = int(n1.split("-")[1])
                        if new_vec[3][w] == 1: continue ### can only have one meal a day
                        new_vec[3][w] += 1

                    new_path = label1[2] + [n2,]

                    if sum(new_vec[0]) > self_data["n_si"] or new_vec[1] > self_data["n_t"]:
                        continue
                    if n2[0] == 'o': ### only check meal break constraint at sign-out nodes
                        if sum(new_vec[0]) != sum(new_vec[3]): continue

                    ### 2.2.1 update cost
                    new_cost = label1[0] + attrs["c"]
                    if attrs["label"] == "w":
                        ts_idx = self_data["ts2idx"][f"{n1.split('-')[1]}-{n1.split('-')[4]}-{n1.split('-')[-1]}"]
                        nu_k = dual["nu"][ts_idx]
                        new_cost -= nu_k
                    ### 2.2.2 get new label
                    new_label = [new_cost, new_vec, new_path]

                    


                    prune_tags = [False] * (len(labels_dict[n2])+1)
                    strategy=1
                    if strategy == 0: 
                        for i, label_n2 in enumerate(labels_dict[n2]):
                            if label_n2[0] >= new_label[0] and sum(label_n2[1][0]) >= sum(new_label[1][0]) and label_n2[1][1] >= new_label[1][1] and sum(label_n2[1][3]) == sum(new_label[1][3]):                                 prune_tags[i] = True
                            elif label_n2[0] <= new_label[0] and sum(label_n2[1][0]) <= sum(new_label[1][0]) and label_n2[1][1] <= new_label[1][1] and sum(label_n2[1][3]) == sum(new_label[1][3]): 
                                prune_tags[-1] = True
                        n2_labels = [labels_dict[n2][i] for i in range(len(labels_dict[n2])) if prune_tags[i] is False]
                        if prune_tags[-1] is False:
                            n2_labels.append(new_label)
                    elif strategy == 1: ### add L_max
                        for i, label_n2 in enumerate(labels_dict[n2]):
                            if label_n2[0] >= new_label[0] and sum(label_n2[1][0]) >= sum(new_label[1][0]) and label_n2[1][1] >= new_label[1][1] and sum(label_n2[1][3]) == sum(new_label[1][3]): 
                                prune_tags[i] = True
                            elif label_n2[0] <= new_label[0] and sum(label_n2[1][0]) <= sum(new_label[1][0]) and label_n2[1][1] <= new_label[1][1] and sum(label_n2[1][3]) == sum(new_label[1][3]): 
                                prune_tags[-1] = True
                        n2_labels = [labels_dict[n2][i] for i in range(len(labels_dict[n2])) if prune_tags[i] is False]
                        if prune_tags[-1] is False:
                            n2_labels.append(new_label)
                        n2_labels = sorted(n2_labels, key=lambda x:x[0])
                        n2_labels = n2_labels[:Q]                 

                    ### 2.4 update node2 labels
                    labels_dict[n2] = n2_labels

        sk_labels = labels_dict["sk"]
        shortest_path_label = min(sk_labels, key=lambda x:x[0])

        print(f"Cost for {sub_lines}: ", round(shortest_path_label[0], 3), "resource vector: ", shortest_path_label[1])
        print(f"Time elapsed for {sub_lines} labeling algorithm: {time.time() - sub_start}s")
        if display:
            print(f"Iter for {sub_lines}: ", iter, end=" ")
            print(f"Get {len(sk_labels)} labels after {sub_lines} pruned labeling.")
            # _pprint_path(G, shortest_path_label[2])

        ### parallel mode
        if save_place is not None:
            save_place.append((shortest_path_label, sub_lines))
            return None
        ### sequential mode
        else:
            return (shortest_path_label, sub_lines)

    def _outer_labeling_shortest_path_topo(self, G:nx.DiGraph, display:bool=False, strategy:int=0):
        """
        for construct initial path set (NOT SUB_P)
        Using TOPOLOGICAL SORT
        strategy: 0 (Zhou's paper), 1 (add capacity Q)
        """
        sstart = time.time()
        print("fast labeling topologically for one network/subnetwork...")

        Q = self.Q ### capacity for labels for one node

        labels_dict = defaultdict(list) # {node: [label1, label2, ...]}
        labels_dict["sc"].append([0, [[0]*self.network.W, 0, [0]*len(self.lines), [0]*self.network.W], ["sc"]]) # [cost, resource vector(list), path(list)]

        topo_id = 0
        for n1 in nx.topological_sort(G):
            topo_id += 1
            nbrs = G[n1]
            for label1 in labels_dict[n1]:
                ### 2 extending node labels
                for n2, attrs in nbrs.items():
                    ### 2.1 create new label
                    new_vec = copy.deepcopy(label1[1])
                    if attrs["label"] == "si":
                        w = int(n1.split("-")[1])
                        new_vec[0][w] += 1
                    elif attrs["label"] == "t":
                        new_vec[1] += 1
                    elif attrs["label"] == "w":
                        l = int(n1.split("-")[3])
                        l_idx = self.line2idx[l]
                        new_vec[2][l_idx] += 1
                    ### 2.1.2 feasibility check
                    elif attrs["label"] == "m":
                        w = int(n1.split("-")[1])
                        if new_vec[3][w] == 1: 
                            continue ### can only have one meal a day
                        new_vec[3][w] += 1


                    if sum(new_vec[0]) > self.network.n_si or new_vec[1] > self.network.n_t:
                        continue
                    if n2[0] == 'o': ### only check meal break constraint at sign-out nodes
                        if sum(new_vec[0]) != sum(new_vec[3]): 
                            continue
                        
                    ### 2.2.1 update cost
                    new_cost = label1[0] + attrs["c"]
                    ### 2.2.2 get new label
                    new_path = label1[2] + [n2,]
                    new_label = [new_cost, new_vec, new_path]

                    ### 2.3 dominance pruning TODO - this is important
                    prune_tags = [False] * (len(labels_dict[n2])+1)
                    strategy=0 
                    if strategy == 0:
                        for i, label_n2 in enumerate(labels_dict[n2]):
                            if label_n2[0] >= new_label[0] and sum(label_n2[1][0]) >= sum(new_label[1][0]) and label_n2[1][1] >= new_label[1][1] and sum(label_n2[1][3]) == sum(new_label[1][3]):
                                prune_tags[i] = True
                            elif label_n2[0] <= new_label[0] and sum(label_n2[1][0]) <= sum(new_label[1][0]) and label_n2[1][1] <= new_label[1][1] and sum(label_n2[1][3]) == sum(new_label[1][3]):
                                prune_tags[-1] = True
                        n2_labels = [labels_dict[n2][i] for i in range(len(labels_dict[n2])) if prune_tags[i] is False]
                        if prune_tags[-1] is False:
                            n2_labels.append(new_label)
                    elif strategy == 1: ### add L_max
                        for i, label_n2 in enumerate(labels_dict[n2]):
                            if label_n2[0] >= new_label[0] and sum(label_n2[1][0]) >= sum(new_label[1][0]) and label_n2[1][1] >= new_label[1][1] and sum(label_n2[1][3]) == sum(new_label[1][3]): 
                                prune_tags[i] = True
                            elif label_n2[0] <= new_label[0] and sum(label_n2[1][0]) <= sum(new_label[1][0]) and label_n2[1][1] <= new_label[1][1] and sum(label_n2[1][3]) == sum(new_label[1][3]):
                                prune_tags[-1] = True
                        n2_labels = [labels_dict[n2][i] for i in range(len(labels_dict[n2])) if prune_tags[i] is False]
                        if prune_tags[-1] is False:
                            n2_labels.append(new_label)
                        n2_labels = sorted(n2_labels, key=lambda x:x[0])
                        n2_labels = n2_labels[:Q]

                    ### 2.4 update node2 labels
                    labels_dict[n2] = n2_labels


        sk_labels = labels_dict["sk"]

        shortest_path_label = min(sk_labels, key=lambda x:x[0])


        print("cost: ", round(shortest_path_label[0], 3), "resource vector: ", shortest_path_label[1])
        if display:
            print(f"Get {len(sk_labels)} labels after pruned labeling.")
            print(f"Time elapsed for labeling algorithm: {time.time() - sstart}s")
            self._pprint_path(G, shortest_path_label[2])

        ### extract a, theta, gamma
        gamma = 0
        a = [0] * len(self.crew.patterns) # (list of Boolean)
        ms = [0] * len(self.crew.patterns) # (list of Boolean)
        theta = [0] * len(self.ts2idx)

        io_depots = list()
        lines_traversed = set()
        for node1, node2 in zip(shortest_path_label[2][:-1], shortest_path_label[2][1:]):
            gamma += G.edges[node1, node2]["c"]
            arc_type, c = G.edges[node1, node2]["label"], G.edges[node1, node2]["c"]
            if arc_type == "w": # get a and theta
                line = int(node1.split("-")[3])
                lines_traversed.add(line)
                a[self.line2idx[line]] = 1
                train_service_idx = self._map_working_arc(node1)
                theta[train_service_idx] = 1

            elif arc_type == "si":
                io_depots.append(int(node2.split("-")[4]))
                line = int(node2.split("-")[3])
                lines_traversed.add(line)

            elif arc_type == "so":
                io_depots.append(int(node1.split("-")[4]))
                line = int(node1.split("-")[3])
                lines_traversed.add(line)
                
            elif arc_type == "t":
                line1 = int(node1.split("-")[3])
                line2 = int(node2.split("-")[3])
                lines_traversed.add(line1)
                lines_traversed.add(line2)

    
        lines_traversed = sorted(list(lines_traversed))
        for line_num in range(1, len(lines_traversed)+1):
            for pattern in combinations(lines_traversed, line_num):
                pat_i = self.crew.pat2idx[pattern]
                a[pat_i] = 1
        if len(lines_traversed) > 0:
            one_pat_i = self.crew.pat2idx[tuple(lines_traversed)]
            ms[one_pat_i] = 1

        print(f"add ms: {ms}")
        return shortest_path_label, theta, a, ms, gamma, io_depots






    #########################################################################################
    ## Solver Part
    #########################################################################################
    def _solve_RLMP(self, iter):
        start_time = time.time()
        print('Begin solving RLMP in iteration {}...'.format(iter))
        
        m = pyo.ConcreteModel()

        ### Sets
        m.K = pyo.Set(initialize = list(range(len(self.ts2idx))))
        m.R = pyo.Set(initialize = list(range(self.crew.n_drivers)))
        m.Q = pyo.Set(initialize = list(range(self.n_path)))
        m.I = pyo.Set(initialize = list(range(len(self.crew.patterns))))
        m.S = pyo.Set(initialize = list(range(len(self.crew.Qs))))

        ### Params
        assert self.theta.shape[0] == self.n_path and self.a.shape[0] == self.n_path \
            and self.ms.shape[0] == self.n_path, "path num not match"
        m.m = pyo.Param(m.I, initialize=self.m)
        def init_theta(m, p, k): # initialize 2-dim variables
            return self.theta[p, k]
        m.theta = pyo.Param(m.Q, m.K, initialize=init_theta)
        m.gamma = pyo.Param(m.Q, initialize=self.gamma, within=pyo.Reals)
        def init_a(m, p, i): # initialize 2-dim variables
            return self.a[p, i]
        m.a = pyo.Param(m.Q, m.I, initialize=init_a)
        # m.R = pyo.Param(within=pyo.Integers, initialize=self.crew.n_drivers)

        # NEW mm compared to a
        def init_ms(m, q, i):
            return self.ms[q, i]
        m.ms = pyo.Param(m.Q, m.I, initialize=init_ms, within=pyo.Binary)


        ### Vars
        m.x = pyo.Var(m.Q, within=pyo.NonNegativeReals, initialize=0)

        ### Extract Dual
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        ### Obj
        def obj_rule(m):
            total_train_penalties = self.network.pi * len(self.ts2idx) if not self.network.real_case else self.network.pi_summation
            return total_train_penalties + pyo.summation(m.gamma, m.x)
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        ### Constraints
        def constrs1(m):
            return pyo.summation(m.x) <= self.crew.n_drivers
        m.constrs1 = pyo.Constraint(rule=constrs1)

        def constrs2(m, k):
            return sum(m.theta[q, k] * m.x[q] for q in m.Q) <= 1 
        m.constrs2 = pyo.Constraint(m.K, rule=constrs2)

        # def constrs3(m, i):
        #     return sum(m.a[q, i] * m.x[q] for q in m.Q) <= m.m[i]
        # m.constrs3 = pyo.Constraint(m.I, rule=constrs3)
        def constrs3(m, s):
            s_set = self.crew.Qs[s]
            np = self.crew.Qs_np[tuple(s_set)]
            sum_m_s = 0
            for pattern in s_set:
                sum_m_s += sum(m.ms[q, self.crew.pat2idx[pattern]] * m.x[q] for q in m.Q)
            return sum_m_s <= np
        m.constrs3 = pyo.Constraint(m.S, rule=constrs3)


        opt = pyo.SolverFactory(os.environ.get('OPTIMIZER', "gurobi"))
        
        solutions = opt.solve(m)
        print(f'Finish solving RLMP, using time {time.time() - start_time:.5}s')
        # solutions.write()

        status = solutions.solver.status
        print(f"solver status: {status}")
        termination_condition = solutions.solver.termination_condition
        print(f"solver termination condition: {termination_condition}")
        obj_value = pyo.value(m.obj)
        print(f"optimal value: {obj_value}")

        x_opt = np.array([pyo.value(m.x[p]) for p in m.Q])
        print(f"Optimal path selection: {x_opt} ({np.sum(x_opt)})")
        ### TODO: print # transfer arcs...

        ### get dual values
        dual_solutions = dict()
        for c in m.component_objects(pyo.Constraint, active=True):
            if c.name == "constrs1":
                dual_solutions["mu"] = m.dual[c]
            elif c.name == "constrs2":
                dual_solutions["nu"] = list()
                for i in c:
                    dual_solutions["nu"].append(m.dual[c[i]]) 
                assert len(dual_solutions["nu"]) == len(self.ts2idx), "nu dual value shape error"
            elif c.name == "constrs3":
                dual_solutions["xi"] = list()
                for i in c:
                    dual_solutions["xi"].append(m.dual[c[i]]) 
                assert len(dual_solutions["xi"]) == len(self.crew.Qs), "xi dual value shape error"

        ### check train services coverage
        total_ts = 0
        
        total_cvg = 0
        coverage = []
        for i, (l, lts) in enumerate(zip(self.lines, self.lts2idx)):
            covered_train_theta = self.theta[:, total_ts:total_ts+len(lts)]
            n_covered = 0
            total_ts += len(lts)
            for j, x in enumerate(x_opt):
                if abs(x - 1) < 1e-5:
                    n_covered += sum(covered_train_theta[j, :])
            print(f"For line {l}, train service coverage {n_covered} / {len(lts)} = {n_covered/len(lts):.3f}")
            total_cvg += n_covered
            coverage.append((n_covered, len(lts), round(n_covered/len(lts), 3)))
        print("In total, train coverage: {} / {} = {}".format(total_cvg, total_ts, round(total_cvg / total_ts, 3)))
        coverage.append((total_cvg, total_ts, round(total_cvg / total_ts, 3)))

        return obj_value, dual_solutions, x_opt, coverage


    def _solve_P(self):
        """
        0-1 integer programming
        """
        start_time = time.time()
        print('Begin solving P ...')
        
        m = pyo.ConcreteModel()

        ### Sets
        m.K = pyo.Set(initialize = list(range(len(self.ts2idx))))
        m.R = pyo.Set(initialize = list(range(self.crew.n_drivers)))
        m.Q = pyo.Set(initialize = list(range(self.n_path)))
        m.I = pyo.Set(initialize = list(range(len(self.crew.patterns))))
        m.S = pyo.Set(initialize = list(range(len(self.crew.Qs))))


        def init_theta(m, p, k): # initialize 2-dim variables
            return self.theta[p, k]
        m.theta = pyo.Param(m.Q, m.K, initialize=init_theta)
        m.gamma = pyo.Param(m.Q, initialize=self.gamma, within=pyo.Reals)
        def init_a(m, p, i): # initialize 2-dim variables
            return self.a[p, i]
        m.a = pyo.Param(m.Q, m.I, initialize=init_a)
        # m.R = pyo.Param(within=pyo.Integers, initialize=self.crew.n_drivers)

        # NEW mm compared to a
        def init_ms(m, q, i):
            return self.ms[q, i]
        m.ms = pyo.Param(m.Q, m.I, initialize=init_ms, within=pyo.Binary)


        ### Vars
        m.x = pyo.Var(m.Q, within=pyo.Binary, initialize=0)

        ### Obj
        def obj_rule(m):
            total_train_penalties = self.network.pi * len(self.ts2idx) if not self.network.real_case else self.network.pi_summation
            return total_train_penalties + pyo.summation(m.gamma, m.x)
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        ### Constraints
        def constrs1(m):
            return pyo.summation(m.x) <= self.crew.n_drivers
        m.constrs1 = pyo.Constraint(rule=constrs1)

        def constrs2(m, k): 
            return sum(m.theta[q, k] * m.x[q] for q in m.Q) <= 1
        m.constrs2 = pyo.Constraint(m.K, rule=constrs2)

        # RE-new constraints for qualifications
        def constrs3(m, s):
            s_set = self.crew.Qs[s]
            np = self.crew.Qs_np[tuple(s_set)]
            sum_m_s = 0
            for pattern in s_set:
                sum_m_s += sum(m.ms[q, self.crew.pat2idx[pattern]] * m.x[q] for q in m.Q)
            return sum_m_s <= np
        m.constrs3 = pyo.Constraint(m.S, rule=constrs3)



        opt = pyo.SolverFactory(os.environ.get('OPTIMIZER', "gurobi"), solver_io="python")
        
        solutions = opt.solve(m)
        print(f'Finish solving P, using time {time.time() - start_time:.5}s')
        # solutions.write()

        status = solutions.solver.status
        print(f"solver status: {status}")
        termination_condition = solutions.solver.termination_condition
        print(f"solver termination condition: {termination_condition}")
        obj_value = pyo.value(m.obj)
        # print(f"optimal value: {obj_value}")

        x_opt = np.array([pyo.value(m.x[p]) for p in m.Q])
        print(f"Optimal path selection: {x_opt} ({np.sum(x_opt)})")
        ### check train services coverage
        print("Finally: ")
        coverage = []
        total_ts = 0
        total_cvg = 0
        for i, (l, lts) in enumerate(zip(self.lines, self.lts2idx)):
            covered_train_theta = self.theta[:, total_ts:total_ts+len(lts)]
            n_covered = 0
            total_ts += len(lts)
            for j, x in enumerate(x_opt):
                if abs(x - 1) < 1e-5:
                    n_covered += sum(covered_train_theta[j, :])
            print(f"For line {l}, train service coverage {n_covered} / {len(lts)} = {n_covered/len(lts):.3f}")
            total_cvg += n_covered
            coverage.append((n_covered, len(lts), round(n_covered/len(lts), 3)))
        print("In total, train coverage: {} / {} = {}".format(total_cvg, total_ts, round(total_cvg / total_ts, 3)))
        coverage.append((total_cvg, total_ts, round(total_cvg / total_ts, 3)))
        return obj_value, x_opt, coverage

    def _solve_P_one_step(self):
        """
        0-1 integer programming
        """
        start_time = time.time()
        print('Begin solving P all at once with x and y...')
        
        m = pyo.ConcreteModel()

        ### Sets
        m.K = pyo.Set(initialize = list(range(len(self.ts2idx))))
        m.R = pyo.Set(initialize = list(range(self.crew.n_drivers)))
        m.Q = pyo.Set(initialize = list(range(self.n_path)))
        m.I = pyo.Set(initialize = list(range(len(self.crew.patterns))))
        m.S = pyo.Set(initialize = list(range(len(self.crew.Qs))))


        def init_theta(m, p, k): # initialize 2-dim variables
            return self.theta[p, k]
        m.theta = pyo.Param(m.Q, m.K, initialize=init_theta)
        m.gamma = pyo.Param(m.Q, initialize=self.gamma, within=pyo.Reals)
        def init_a(m, p, i): # initialize 2-dim variables
            return self.a[p, i]
        m.a = pyo.Param(m.Q, m.I, initialize=init_a)

        def init_ms(m, q, i):
            return self.ms[q, i]
        m.ms = pyo.Param(m.Q, m.I, initialize=init_ms, within=pyo.Binary)

        def init_qr(m, i, r):
            path_pattern = set(self.crew.qualifications[i])
            crew_qualification = set(self.crew.drivers[r].license)
            return int(path_pattern.issubset(crew_qualification))
        m.qr = pyo.Param(m.I, m.R, initialize=init_qr, within=pyo.Binary)

        def init_pp(m, q, r): # preference penalties
            preference_penalty = 0
            prefered_depots = self.crew.drivers[r].prefered_depots
            for o in self.n_io_depots_list[q]:
                if o not in prefered_depots:
                    preference_penalty += self.config["crew"]["preference_settings"]["lambda_o"]
            return preference_penalty
        m.pp = pyo.Param(m.Q, m.R, initialize=init_pp, within=pyo.Reals)
        

        ### Vars
        m.x = pyo.Var(m.Q, within=pyo.Binary, initialize=0)
        m.y = pyo.Var(m.Q, m.R, within=pyo.Binary, initialize=0)

        ### Obj
        def obj_rule(m):
            total_train_penalties = self.network.pi_summation
            total_path_cost = pyo.summation(m.gamma, m.x)
            total_preference_penalties = pyo.summation(m.pp, m.y)
            return total_train_penalties + total_path_cost + total_preference_penalties
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        ### Constraints
        def constrs1(m):
            return pyo.summation(m.x) <= self.crew.n_drivers
        m.constrs1 = pyo.Constraint(rule=constrs1)

        def constrs2(m, k): 
            return sum(m.theta[q, k] * m.x[q] for q in m.Q) <= 1
        m.constrs2 = pyo.Constraint(m.K, rule=constrs2)

        # RE-new constraints for qualifications
        def constrs3(m, s):
            s_set = self.crew.Qs[s]
            np = self.crew.Qs_np[tuple(s_set)]
            sum_m_s = 0
            for pattern in s_set:
                sum_m_s += sum(m.ms[q, self.crew.pat2idx[pattern]] * m.x[q] for q in m.Q)
            return sum_m_s <= np
        m.constrs3 = pyo.Constraint(m.S, rule=constrs3)


        def constrs4(m, q):
            return sum(m.y[q, r] for r in m.R) == m.x[q]
        m.constrs4 = pyo.Constraint(m.Q, rule=constrs4)

        #### each driver is assigned less than once
        def constrs5(m, r):
            return sum(m.y[q, r] for q in m.Q) <= 1
        m.constrs5 = pyo.Constraint(m.R, rule=constrs5)

        #### driver qualification assurance
        def constrs6(m, q, r, i):
            return m.ms[q, i]*m.y[q, r] <= m.qr[i, r]
        m.constrs6 = pyo.Constraint(m.Q, m.R, m.I, rule=constrs6)


        opt = pyo.SolverFactory(os.environ.get('OPTIMIZER', "gurobi"), solver_io="python")
        
        solutions = opt.solve(m)
        time_elapsed = round(time.time() - start_time, 3)

        print(f'Finish solving P in one step, using time {time.time() - start_time:.5}s')
        # solutions.write()

        status = solutions.solver.status
        print(f"solver status: {status}")
        termination_condition = solutions.solver.termination_condition
        print(f"solver termination condition: {termination_condition}")
        obj_value = pyo.value(m.obj)
        print(f"optimal value: {obj_value}")

        x_opt = np.array([pyo.value(m.x[p]) for p in m.Q])
        print(f"Optimal path selection: {x_opt} ({np.sum(x_opt)})")
        ### check train services coverage
        print("Finally: ")
        coverage = []
        total_ts = 0
        total_cvg = 0
        for i, (l, lts) in enumerate(zip(self.lines, self.lts2idx)):
            covered_train_theta = self.theta[:, total_ts:total_ts+len(lts)]
            n_covered = 0
            total_ts += len(lts)
            for j, x in enumerate(x_opt):
                if abs(x - 1) < 1e-5:
                    n_covered += sum(covered_train_theta[j, :])
            print(f"For line {l}, train service coverage {n_covered} / {len(lts)} = {n_covered/len(lts):.3f}")
            total_cvg += n_covered
            coverage.append((n_covered, len(lts), round(n_covered/len(lts), 3)))
        print("In total, train coverage: {} / {} = {}".format(total_cvg, total_ts, round(total_cvg / total_ts, 3)))
        coverage.append((total_cvg, total_ts, round(total_cvg / total_ts, 3)))
        return obj_value, coverage, time_elapsed
    
    










    ##################################################
    # just util functions
    ##################################################
    def _map_train_services(self):
        k = 0
        for depot in self.crew.depots:
            l = self.line2idx[int(depot//2)+1]
            if not self.network.consider_weekends:
                for service in self.network.train_services[depot]:
                    for w in range(self.network.W):
                        self.ts2idx[f"{w}-{depot}-{service[0]}"] = k
                        self.lts2idx[l].append(k)
                        k += 1
            else:
                for w in range(self.network.W):
                    for service in self.network.train_services_w[w][depot]:
                        self.ts2idx[f"{w}-{depot}-{service[0]}"] = k
                        self.lts2idx[l].append(k)
                        k += 1


        # asseret sequential mapping for sequential lines
        total_ts = 0
        for i, lts in enumerate(self.lts2idx):
            n_lts = len(lts)
            total_ts += n_lts
            max_ltx = max(lts)
            assert max_ltx == total_ts-1, "sequential train service mapping seems wrong..."
            print(f"Finish building {n_lts} train service indexes for line {self.lines[i]}.")
            
        print(f"Finish building train service indexes: {len(self.ts2idx)} services in total.")
    
    def _map_working_arc(self, start_node:str) -> int:
        n1s = start_node.split("-")
        service = f"{n1s[1]}-{n1s[4]}-{n1s[-1]}"
        return self.ts2idx[service]

    @staticmethod
    def _pprint_path(G, path, detailed=False):
        """
        path: list of nodes
        """
        for node1, node2 in zip(path[:-1], path[1:]):
            arc_type, c = G.edges[node1, node2]["label"], G.edges[node1, node2]["c"]
            n1s, n2s = node1.split("-"), node2.split("-")
            if arc_type == "s":
                print(f"start(w={n2s[-2]}, t={n2s[-1]})", end=" -> \n")
            if arc_type == "e":
                print(f"end", end=".\n")
            elif arc_type == "si":
                print(f"sign-in(d={n1s[1]}, t={n1s[-1]},t'={n2s[-1]}, depot={n2s[4]})", end=" -> \n")
            elif arc_type == "so":
                print(f"sign-out(d={n1s[1]}, t={n1s[-1]}, t'={n2s[-1]}, depot={n1s[4]})", end=" -> \n")
            elif arc_type == "w":
                print(f"working(t={n1s[-1]},t'={n2s[-1]}, depot={n1s[4]}, depot'={n2s[4]}, c={c})", end=" -> \n")
            elif arc_type == "m":
                print(f"meal(t={n1s[-1]},t'={n2s[-1]}, depot={n1s[4]}, c={c:.2f})", end=" -> \n")
            elif arc_type == "t":  
                print(f"transfering(t={n1s[-1]}, t'={n2s[-1]}, line={n1s[3]}, depot={n1s[4]}, line'={n2s[3]}, depot'={n2s[4]}, c={c})", end=" -> \n")
            elif arc_type == "shifting arc":
                print(f"shifting(w={n2s[1]})", end=" -> \n")
            elif arc_type == "n":
                print(f"sleep and sleep", end=" .\n")
            
            if detailed:
                if arc_type == "r":  
                    print(f"resting(t={n1s[-1]}, t'={n2s[-1]}, depot={n1s[-3]}, line={n2s[-4]}, c={c})", end=" -> \n")
                elif arc_type == "a":
                    print(f"waiting(t={n1s[-1]}, t'={n2s[-1]}, depot={n1s[-3]}, line={n2s[-4]}, c={c})", end=" -> \n")



    def _alter_working_arc_weights(self, G, path:list):
        """
        all traversed working arcs now have weight = /inf
        return original weights for network reconstruction after initialize path sets
        """
        working_weights = dict() # {(u, v): weight}  for reconstruction weights
        for node1, node2 in zip(path[:-1], path[1:]):
            if not G.has_edge(node1, node2): continue
            arc_type, c = G.edges[node1, node2]["label"], G.edges[node1, node2]["c"]
            if arc_type == "w":
                ### here need to set "inf" all working arcs (in different duty windows) associated with this train service
                n1s, n2s = node1.split("-"), node2.split("-") 
                service_start, service_end = int(n1s[-1]), int(n2s[-1]) - self.network.beta  # bug fixed - minus resting time
                # service_start, service_end = int(n1s[-1]), int(n2s[-1])
                if not self.network.consider_weekends:
                    for d in self.network.train_services_C_set[(service_start, service_end)]:
                        n1 = "-".join([n1s[0], n1s[1], str(d), n1s[3], n1s[4], n1s[5], n1s[6]])
                        n2 = "-".join([n2s[0], n2s[1], str(d), n2s[3], n2s[4], n2s[5], n2s[6]])
                        working_weights[(n1, n2)] = c
                        G.edges[n1, n2]["c"] = float("inf") 
                else:
                    day_id = int(n1s[1])
                    for d in self.network.train_services_C_set_w[day_id][(service_start, service_end)]:
                        n1 = "-".join([n1s[0], n1s[1], str(d), n1s[3], n1s[4], n1s[5], n1s[6]])
                        n2 = "-".join([n2s[0], n2s[1], str(d), n2s[3], n2s[4], n2s[5], n2s[6]])
                        working_weights[(n1, n2)] = c
                        G.edges[n1, n2]["c"] = float("inf") 

        return working_weights


#############################################################################################################
#############################################################################################################
#############################################################################################################

class AssignmentSolver():

    def __init__(self, data:dict, config:dict, method="solver", ablation:str=None):
        """
        @method: "dfs", "Hungarian", "solver"
        """
        
        assert "preference_settings" in config["crew"], "Error: no preference included, no need for assignment solver..."
        self.lambda_o = config["crew"]["preference_settings"]["lambda_o"]
        self.save_dir = config["data_save_dir"]
        self.ablation = ablation

        self.method = method
        assert method == "solver", f"Error: assignment algorithm {method} not implemented..."

        self.selected_path_idxes = data["selected_path_idxes"]
        self.transfer_costs = data["transfer_costs"]
        self.prefered_depots = data["prefered_depots"]
        self.qualifications = [tuple(q) for q in data["qualifications"]]
        self.Qs = [tuple(qs) for qs in data["Qs"]]
        self.c_crew_pools = {tuple(k):v for k, v in data["c_crew_pools"]}
        self.s_crew_pools = {tuple(k):v for k, v in data["s_crew_pools"]}
        self.c_path_pools = {tuple(k):v for k, v in data["c_path_pools"]}
        self.s_path_pools = {tuple(k):v for k, v in data["s_path_pools"]}
        self.in_out_depots = data["in_out_depots"]
        self.cross_line = data["cross_line"]

        self.for_check = {tuple(k):(len(v1), len(v2)) for (k, v1), (k, v2) in zip(data["s_crew_pools"], data["s_path_pools"])}



        print("\n####################################################\nLoad {} crew data and {} path data.".format(len(self.transfer_costs), len(self.selected_path_idxes)))
        print("Finish loading and parsing data for assignment problem...")



    def solve(self):
        """
        Hungarian Algorithm Theory: https://www.bilibili.com/video/BV1P54y1i7Ka/?share_source=copy_web&vd_source=b1636646ae975436511cbade32ddc089- TODO: coding implementation

        """
        print("Begin solving assignment problem using {}...".format(self.method))
        
        solve_start = time.time()

        # solve
        if self.method == "solver":
            allocation, cost = self._solve_by_solver()
        print(f"With cost {cost}")
        print(f"Allocation {allocation}")

        # check correctness
        for pid, cid in allocation.items():
            p_qualification = None
            for q, path_list in self.s_path_pools.items():
                if pid in path_list:
                    p_qualification = q
                    break
            pc_qualification = None
            for q, crew_list in self.s_crew_pools.items():
                if cid in crew_list:
                    pc_qualification = q
                    break
            for l in p_qualification:
                assert l in pc_qualification, f"wrong assignment pair discovered, path qualification {p_qualification}, crew qualification{pc_qualification}"

        
        
        assert len(allocation) == len(self.selected_path_idxes), "Wrong matching..."
        solve_time_elapsed = round(time.time() - solve_start, 3)
        print("Finish solving assignment problem by {} using time {} for {} path-crew pairs.".format(self.method, solve_time_elapsed, len(allocation)))
        print("Total assignment cost: {}".format(cost))
    
        # save results
        now = datetime.now()
        time_stamp_str = now.date().strftime("%Y-%m-%d") + '-' + str(now.hour) 
        result_file_path = self.save_dir / "res_asm.txt" if self.ablation is None else self.save_dir / "res_asm_{}.txt".format(self.ablation)
        # result_file_path = self.save_dir / time_stamp_str / "res_asm.txt" if self.ablation is None else self.save_dir / "res_asm_{}.txt".format(self.ablation)
        # if not Path(self.save_dir, time_stamp_str).exists():
        #     Path(self.save_dir, time_stamp_str).mkdir(parents=True)

        with open(result_file_path, "w") as f:
            f.write(f"Cost: {cost}\n")
            f.write(f"Time elapsed(s): {solve_time_elapsed}\n\n")
            f.write(f"Final allocation: {allocation}\n\n")
            f.write(f"Selected Paths: {self.selected_path_idxes}\n")
            f.write(f"Crew qualifications: {self.s_crew_pools}\n")
        print("Finish saving results for assignment problem.")


        return cost, allocation 
    

    def _solve_by_solver(self) -> tuple:
        """
        one-time solving - a big matrix
        """

        # data
        crew_pool = list(range(len(self.transfer_costs)))
        path_pool = self.selected_path_idxes

        # result storage
        res_path_allocation = [-1] * len(path_pool)

        # generate the cost matrix with shape [len(path_pool), len(crew_pool)]
        cost_matrix = []
        for pid in path_pool:
            row = []
            for cid in crew_pool:
                ## qualification constraint
                p_qualification = None
                for q, path_list in self.s_path_pools.items():
                    if pid in path_list:
                        p_qualification = q
                        # break
                assert p_qualification is not None, "buggy path qualification..."
                
                
                if cid not in self.c_crew_pools[p_qualification]:
                    c = 1e9 
                    # c = float("inf") #gurobipy.GurobiError: Coefficient is Nan or Inf
                    for q, crew_list in self.s_crew_pools.items():
                        if cid in crew_list:
                            pc_qualification = q
                    
                    is_sublist = True
                    for l in p_qualification:
                        if l not in pc_qualification:
                            is_sublist = False
                    assert is_sublist is False
                    
                else:
                    ## preference costs 
                    c = 0
                    for depot in self.in_out_depots[pid]: # preference term 2
                        if depot not in self.prefered_depots[cid]:
                            c += self.lambda_o
                row.append(c)
            cost_matrix.append(row)
        

        # solver part
        m = pyo.ConcreteModel()

        ## Sets:
        m.C = pyo.Set(initialize=list(range(len(crew_pool))))
        m.P = pyo.Set(initialize=list(range(len(path_pool))))

        ## Params
        def init_matrix(m, p, c):
            return cost_matrix[p][c]
        m.matrix = pyo.Param(m.P, m.C, initialize=init_matrix, within=pyo.Reals)

        ## Vars
        m.x = pyo.Var(m.P, m.C, within=pyo.Binary, initialize=0)

        ## Obj
        def obj_rule(m):
            return sum(sum(m.x[p, c] * m.matrix[p, c] for p in m.P) for c in m.C)
            # return pyo.summation(m.x, m.matrix) 

        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        ## Constraints
        def constrs1(m, p):
            return sum(m.x[p, c] for c in m.C) == 1
        m.constrs1 = pyo.Constraint(m.P, rule=constrs1)

        def constrs2(m, c):
            return sum(m.x[p, c] for p in m.P) <= 1
        m.constrs2 = pyo.Constraint(m.C, rule=constrs2)

        opt = pyo.SolverFactory(os.environ.get('OPTIMIZER', "gurobi"), solver_io="python")
        
        solutions = opt.solve(m)
        obj_value = pyo.value(m.obj)
        print(f"optimal value: {obj_value}")

        x_opt = np.zeros((len(path_pool), len(crew_pool)))
        for pid, p in enumerate(m.P):
            x_opt[p, :] = np.array([pyo.value(m.x[p, c]) for c in m.C])
            for cid, x_pc in enumerate(x_opt[p, :]):
                if abs(x_pc - 1) < 1e-5:
                    assert res_path_allocation[p] == -1
                    res_path_allocation[p] = cid

        
        
        assert -1 not in res_path_allocation, "Error: certain paths are not allocated..."

        return {path_pool[pi]: crew_pool[aid] for pi, aid in enumerate(res_path_allocation)}, obj_value


    
    