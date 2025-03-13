from collections import defaultdict
import copy
import math
import time

import numpy as np

from model.renetwork import ReNetworkFlowModel
from model.rebenchmark import ReBenchmarkAlgorithmSolver
from model.crew import MetroCrew
from model.recorder import Recorder
from pathlib import Path
import json

import networkx as nx
from itertools import combinations


class Rescheduler():
    
    def __init__(self, final_planning_path:Path, config:dict, crew:MetroCrew, recorder:Recorder) -> None:



        self.config = config
        self.days = [config["reschedule"]["re_d"]]
        self.crew = crew
        self.recorder = recorder

        with open(final_planning_path, "r") as f:
            final_planning = json.load(f)
        print(f"{len(final_planning.keys())} drivers.")
        

        self.duty_groups = []
        self._generate_duty_groups()

        self.daily_plannings_dict = defaultdict(dict) # {day_id: {driver_id: list(planning quintuple tasks)}}
        for driver_id, indiv_plan in final_planning.items():
            driver_id = int(driver_id)  # transform "driver_id" and "day_id" strings to int (due to the json format)
            for day_id, day_plan in indiv_plan.items():
                if day_id == "depots": continue
                day_id = int(day_id)
                self.daily_plannings_dict[day_id][driver_id] = day_plan

                
        self.involved_planning = dict() #{driver_id: daily_planning}
        self.virtual_duty_windows = dict() #{driver_id: [start, end, meal_eaten, cur_depot]}
        self.crew_status_dict = dict() #{driver_id: status}  status 1-at work now  2-about to work later today (3-off work today - excluded already)


                

    def reschedule(self, data_saving:bool=True, no_transfer:bool=False):

        start_t = time.time()

        if no_transfer:
            self.config["schedule"]["D"] = 1e10
        else:
            self.config["schedule"]["D"] = 1
        
        # 1 cut the rescheduling plans
        re_d, re_t =  self.config["reschedule"]["re_d"], self.config["reschedule"]["re_t"]
        self.set_t_and_duty_windows(re_d, re_t)

        print(f"{len(self.virtual_duty_windows.keys())} drivers to reschedule.")
        self.n_rescheduled = len(self.virtual_duty_windows.keys())

        record_name = "basic_info"
        self.recorder.create_record(record_name)
        self.recorder.set_data(record_name, "n_reschedued", self.n_rescheduled)
        
        
        
        # the sorting process
        self.driver_windows_at_work = {k:v for k, v in self.virtual_duty_windows.items() if self.crew_status_dict[k] == 1}
        self.driver_windows_to_work = {k:v for k, v in self.virtual_duty_windows.items() if self.crew_status_dict[k] == 2}
        self.sorted_drivers_1 = sorted(self.driver_windows_at_work.keys(), key=lambda id : self.driver_windows_at_work[id][1])
        self.sorted_drivers_2 = sorted(self.driver_windows_to_work.keys(), key=lambda id : self.driver_windows_to_work[id][0], reverse=True)
        self.sorted_drivers = self.sorted_drivers_1 + self.sorted_drivers_2

        # 2 reconstruct nets
        self.reconstruct_net()

        # 3 rescheduling
        total_cost = 0
        n_total_transfer, n_total_violated_depots = 0, 0
        
        n_path = len(self.virtual_duty_windows.keys()) 
        theta_init = np.zeros((n_path, len(self.ts2idx)))
        a_init = np.zeros((n_path, len(self.crew.patterns)))
        ms_init = np.zeros((n_path, len(self.crew.patterns)))
        gamma_init = np.zeros(n_path)

        ### added for special tasks
        completed_special_task_set = set()
        completed_special_task_cnt = 0
        
        
        
        for i, driver_id  in enumerate(self.sorted_drivers):
            duty_setting = self.virtual_duty_windows[driver_id]
            print(f"Reschedule for driver {driver_id} ({i}) with license {self.crew.drivers[driver_id].license} and state {duty_setting}...")
            shortest_path_label, theta, a, ms, gamma, io_lines, n_transfer, n_violated = self.reschedule_one_driver(driver_id, duty_setting)
            total_cost += gamma
            n_total_transfer += n_transfer
            n_total_violated_depots += n_violated

            # record count of special work arcs
            for node1, node2 in zip(shortest_path_label[2][:-1], shortest_path_label[2][1:]):
                if self.net.G.edges[node1, node2]["label"] == "w":
                    if "special" in self.net.G[node1][node2]:
                        completed_special_task_cnt += 1
                        completed_special_task_set.add((node1, node2))            


            theta_init[i,:] = theta
            a_init[i, :] = a
            ms_init[i, :] = ms
            gamma_init[i] = gamma

        total_cost += self.net.pi_summation

        # 4 summary
        print("Finally: ")
        print(f"Total cost: {total_cost}")
        print(f"Total n transfer: {n_total_transfer}")
        print(f"Total n violated depots: {n_total_violated_depots}")
        coverage = []
        total_ts = 0
        total_cvg = 0
        for i, (l, lts) in enumerate(zip(self.lines, self.lts2idx)):
            covered_train_theta = theta_init[:, total_ts:total_ts+len(lts)]
            n_covered = 0
            total_ts += len(lts)
            for j in range(n_path):
                n_covered += sum(covered_train_theta[j, :])
            print(f"For line {l}, train service coverage {n_covered} / {len(lts)} = {n_covered/len(lts):.3f}")
            total_cvg += n_covered
            coverage.append((n_covered, len(lts), round(n_covered/len(lts), 3)))
        print("In total, train coverage: {} / {} = {}".format(total_cvg, total_ts, round(total_cvg / total_ts, 3)))
        coverage.append((total_cvg, total_ts, round(total_cvg / total_ts, 3)))
        
        ### coverage of special tasks
        special_task_coverage = (completed_special_task_cnt, self.net.special_task_cnt, round(completed_special_task_cnt / self.net.special_task_cnt, 3))
        
        
        time_elapsed = round(time.time() - start_t, 3)
        print(f"Time elapsed: {time_elapsed}")

        if data_saving:

            record_name = "path_heuristic"
            if no_transfer:
                record_name += "_no_trans"
            self.recorder.create_record(record_name)

            self.recorder.set_data(record_name, "obj", total_cost)
            self.recorder.set_data(record_name, "cvg", coverage)
            self.recorder.set_data(record_name, "t", time_elapsed)
            self.recorder.set_data(record_name, "n_transfer", n_total_transfer)
            self.recorder.set_data(record_name, "special_cvg", special_task_coverage)


        
    def reschedule_benchmark(self, data_saving=True):
        """
        should be run after running self.reschedule
        """

        print("Reschedule benchmark.")
        bm_solver = ReBenchmarkAlgorithmSolver(self.crew, self.config, self.recorder)
        bm_obj, bm_cvg_total, bm_t, bm_cvg, bm_special_cvg, bm_transfer_cnt = bm_solver.run_for_reschedule(self.duty_groups, self.net.train_services_w, self.virtual_duty_windows, self.sorted_drivers, display=True)
        bm_obj += self.net.pi_summation

        if data_saving:
            record_name = "bm"
            self.recorder.create_record(record_name)
            self.recorder.set_data(record_name, "obj", bm_obj)
            self.recorder.set_data(record_name, "cvg", bm_cvg)
            self.recorder.set_data(record_name, "special_cvg", bm_special_cvg)
            self.recorder.set_data(record_name, "cvg_total", bm_cvg_total)
            self.recorder.set_data(record_name, "t", bm_t)
            self.recorder.set_data(record_name, "bm_transfer_cnt", bm_transfer_cnt)


    def calc_result_comparison(self):

        record_name = "Summary"
        self.recorder.create_record(record_name)
        assert self.recorder.has_data("path_heuristic", "obj") and self.recorder.has_data("path_heuristic", "cvg")
        f_obj, f_cvg = self.recorder.get_data("path_heuristic", "obj"), self.recorder.get_data("path_heuristic", "cvg")[-1][-1]
        f_scvg = self.recorder.get_data("path_heuristic", "special_cvg")[-1]

        if self.recorder.has_data("path_heuristic_no_trans", "obj") and self.recorder.has_data("path_heuristic_no_trans", "cvg"):
            fn_obj, fn_cvg = self.recorder.get_data("path_heuristic_no_trans", "obj"), self.recorder.get_data("path_heuristic_no_trans", "cvg")[-1][-1]
            fn_scvg = self.recorder.get_data("path_heuristic_no_trans", "special_cvg")[-1]
            self.recorder.set_data(record_name, f"h_trans_obj_delta", (fn_obj - f_obj) / fn_obj)
            self.recorder.set_data(record_name, f"h_trans_cvg_delta", (f_cvg - fn_cvg) / fn_cvg)
            self.recorder.set_data(record_name, f"h_trans_scvg_delta", (f_scvg - fn_scvg) / fn_scvg)

        if self.recorder.has_data("bm", "obj") and self.recorder.has_data("bm", "cvg_total"):
            b_obj, b_cvg = self.recorder.get_data("bm", "obj"), self.recorder.get_data("bm", "cvg_total")
            b_scvg = self.recorder.get_data("bm", "special_cvg")[-1]
            self.recorder.set_data(record_name, f"h_bm_obj_delta", (b_obj - f_obj) / b_obj)
            self.recorder.set_data(record_name, f"h_bm_cvg_delta", (f_cvg - b_cvg) / b_cvg)
            self.recorder.set_data(record_name, f"h_bm_scvg_delta", (f_scvg - b_scvg) / b_scvg)
            if self.recorder.has_data("path_heuristic_no_trans", "obj") and self.recorder.has_data("path_heuristic_no_trans", "cvg"):
                self.recorder.set_data(record_name, f"hn_bm_obj_delta", (b_obj - fn_obj) / b_obj)
                self.recorder.set_data(record_name, f"hn_bm_cvg_delta", (fn_cvg - b_cvg) / b_cvg)
                self.recorder.set_data(record_name, f"hn_bm_scvg_delta", (fn_scvg - b_scvg) / b_scvg)
        




###########################################################
###########################################################
###########################################################


    def set_t_and_duty_windows(self, re_d:int, re_t:int):
        self.d, self.t = re_d, re_t 

        # get all involved drivers and corresponding daily planning
        for driver_id, day_plan in self.daily_plannings_dict[re_d].items():
            if day_plan["duty_window"][1] > re_t:
                self.involved_planning[driver_id] = day_plan

        for driver_id, day_plan in self.involved_planning.items():
            inside_task = None
            meal_eaten = False
            for task in day_plan["tasks"]:

                if re_t <= task[2]:
                    inside_task = task
                    break
                if task[0] == "m":
                    meal_eaten = True
            
            if inside_task is None:
                raise ValueError("empty.")
            
            
            if inside_task[0] == "w":
                new_duty_start = inside_task[2]
                self.crew_status_dict[driver_id] = 1
            elif inside_task[0] == "si":
                new_duty_start = inside_task[2]
                self.crew_status_dict[driver_id] = 2
            elif inside_task[0] == "so":
                continue
            else:
                new_duty_start = re_t
                self.crew_status_dict[driver_id] = 1

            cur_depot = inside_task[4] 

            self.virtual_duty_windows[driver_id] = [new_duty_start, day_plan["duty_window"][1], meal_eaten, cur_depot]
        
    def reconstruct_net(self):
        """
        just reconstruct net for one day
        
        FOR NOW, add all sign-in arcs for the entire crew with cost "inf"
        
        """

        net = ReNetworkFlowModel(self.config)
        print(self.virtual_duty_windows)
        net.create(self.virtual_duty_windows)
        
        self.net = net
        self.lines = self.net.lines
        self.line2idx = {l:i for i, l in enumerate(self.net.lines)}
        self.ts2idx = dict() # {day-depot-start_time: k(train service index)}
        self.lts2idx = [list() for _ in range(len(self.net.lines))] # [[k1, k2, ...], [k101, k102, ...]]
        self._map_train_services()
    
    def reschedule_one_driver(self, driver_id:int, duty_setting:list):
        
        sub_G = self.net.subnetworks[tuple(self.crew.drivers[driver_id].license)]
        
        self.net.activate_driver_path(driver_id, sub_G)

        shortest_path_label, theta, a, ms, gamma, io_lines = self.conduct_topo_labeling_for_one_driver(sub_G, duty_setting[2])
        print("Choose to work on lines with a={}, ms={}".format(a, ms))

        # add depot preference penalties and count transfer arcs
        n_transfer = 0
        n_violated = 0
        for node1, node2 in zip(shortest_path_label[2][:-1], shortest_path_label[2][1:]):
            arc_type, c = sub_G.edges[node1, node2]["label"], sub_G.edges[node1, node2]["c"]
            if arc_type == "si":
                depot = int(node2.split("-")[4])
                if depot not in self.crew.drivers[driver_id].prefered_depots:
                    gamma += self.config["crew"]["preference_settings"]["lambda_o"]
                    n_violated += 1
            elif arc_type == "so":
                depot = int(node1.split("-")[4])
                if depot not in self.crew.drivers[driver_id].prefered_depots:
                    gamma += self.config["crew"]["preference_settings"]["lambda_o"]
                    n_violated += 1
            elif arc_type == "t":
                n_transfer += 1
        print(f"n_transfer: {n_transfer}, n_violated: {n_violated}")
        print()

        # alternate working arc weights and deactivate driver_id
        for subgraph in self.net.subnetworks.values():
            _ = self._alter_working_arc_weights(subgraph, shortest_path_label[2])
        self.net.deactivate_driver_path(driver_id, sub_G)

        return shortest_path_label, theta, a, ms, gamma, io_lines, n_transfer, n_violated
        

        
    def conduct_topo_labeling_for_one_driver(self, G, meal_eaten:bool):


        labels_dict = defaultdict(list) # {node: [label1, label2, ...]}
        labels_dict["sc"].append([0, [[0], 0, [0]*len(self.net.lines), [0]], ["sc"]]) # [cost, resource vector(list), path(list)]

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


                    if sum(new_vec[0]) > self.net.n_si or new_vec[1] > self.net.n_t:
                        continue
                    if n2[0] == 'o': ### only check meal break constraint at sign-out nodes
                        required_meal = 1 if not meal_eaten else 0
                        if required_meal != sum(new_vec[3]): 
                            continue
                        
                    ### 2.2.1 update cost
                    new_cost = label1[0] + attrs["c"]
                    ### 2.2.2 get new label
                    new_path = label1[2] + [n2,]
                    new_label = [new_cost, new_vec, new_path]

                    prune_tags = [False] * (len(labels_dict[n2])+1)
                    for i, label_n2 in enumerate(labels_dict[n2]):
                        if label_n2[0] >= new_label[0] and sum(label_n2[1][0]) >= sum(new_label[1][0]) and label_n2[1][1] >= new_label[1][1] and sum(label_n2[1][3]) == sum(new_label[1][3]):
                            prune_tags[i] = True
                        elif label_n2[0] <= new_label[0] and sum(label_n2[1][0]) <= sum(new_label[1][0]) and label_n2[1][1] <= new_label[1][1] and sum(label_n2[1][3]) == sum(new_label[1][3]): 
                            prune_tags[-1] = True
                    n2_labels = [labels_dict[n2][i] for i in range(len(labels_dict[n2])) if prune_tags[i] is False]
                    if prune_tags[-1] is False:
                        n2_labels.append(new_label)

                    ### 2.4 update node2 labels
                    labels_dict[n2] = n2_labels


        sk_labels = labels_dict["sk"]

        shortest_path_label = min(sk_labels, key=lambda x:x[0])


        print("cost: ", round(shortest_path_label[0], 3), "resource vector: ", shortest_path_label[1])

        ### extract a, theta, gamma
        gamma = 0
        a = [0] * len(self.crew.patterns) # (list of Boolean)
        ms = [0] * len(self.crew.patterns) # (list of Boolean)
        theta = [0] * len(self.ts2idx)

        io_lines = set()
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
                io_lines.add(int(node2.split("-")[3]))
                line = int(node2.split("-")[3])
                lines_traversed.add(line)

            elif arc_type == "so":
                io_lines.add(int(node1.split("-")[3]))
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

        return shortest_path_label, theta, a, ms, gamma, io_lines







#######################################################################################
#######################################################################################
#######################################################################################



    def _generate_duty_groups(self):
        """
        determine [s, e] of each duty group

        definition: 
            duty_groups: just time interval in each day
            duty: time interval for each line
            number of duty_groups * number of lines = number of duties
    
        """
        Tw = self.config["system"]["Tw"]
        alpha = self.config["system"]["alpha"]
        h = self.config["system"]["h"]
        n_duty_groups = math.ceil((Tw - alpha) / h) + 1
        print(f"Finish generating {n_duty_groups} duty_groups in one day for one line for one station.")

        duty_start, duty_end = 0, alpha
        for _ in range(n_duty_groups-1):
            self.duty_groups.append([duty_start, duty_end])
            duty_start, duty_end = duty_start + h, duty_end + h
        self.duty_groups.append([Tw - alpha, Tw])
        
        print(f"Duty_groups in one day: {self.duty_groups}")
        print("****************************************************************")
        return

    def _map_train_services(self):
        k = 0
        for depot in self.crew.depots:
            l = self.line2idx[int(depot//2)+1]
            for w in [0]:
                for service in self.net.train_services_w[w][depot]:
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
                service_start, service_end = int(n1s[-1]), int(n2s[-1]) - self.net.beta  # bug fixed - minus resting time
                # service_start, service_end = int(n1s[-1]), int(n2s[-1])
                
                day_id = int(n1s[1])
                for d in self.net.train_services_C_set_w[day_id][(service_start, service_end)]:
                    n1 = "-".join([n1s[0], n1s[1], str(d), n1s[3], n1s[4], n1s[5], n1s[6]])
                    n2 = "-".join([n2s[0], n2s[1], str(d), n2s[3], n2s[4], n2s[5], n2s[6]])
                    working_weights[(n1, n2)] = c
                    G.edges[n1, n2]["c"] = float("inf") 

        return working_weights