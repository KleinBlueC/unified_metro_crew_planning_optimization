from collections import defaultdict
import json
import networkx as nx
from pathlib import Path

def calc_transfer_min_time_dict(transfer_setting_path:Path, timetable_setting_path:Path, tr:int):
    """
    Args:
        transfer_setting_path (Path): path of json file
        timetable_setting_path (Path): path of json file
        tr (int): number of in-station transfer time units
        
    Return:
        transfer_dict: {"depot_id1-depot_id2":t}
    
    Note: now the number of items are redundant - twice the needed (since o1-o2 and o2-o1 are the same in value)
    """

    with open(transfer_setting_path, encoding="utf-8") as f:
        transfer_settings = json.load(f)
    with open(timetable_setting_path, encoding="utf-8") as f:
        timetable_settings = json.load(f)
        
    transfer_dict = dict()
    
    for line_str, line_st in transfer_settings.items():
        line = int(line_str[4:])
        duration = timetable_settings[line_str]["duration"]

        num_stations = line_st["num_stations"]

        #(line - to_line)
        for to_line_str, to_line_st in transfer_settings[line_str]["transfer"].items():  
            to_line = int(to_line_str[7:])

            to_duration = timetable_settings[to_line_str[3:]]["duration"]
            ### from_line: ratios
            ratio_from_0 = float(to_line_st["dist_from_depot0"] / (num_stations-1))
            ratio_from_1 = 1 - ratio_from_0
            ratios_from = [ratio_from_0, ratio_from_1]
            ### to_line: ratios
            ratio_to_0 = float(transfer_settings[to_line_str[3:]]["transfer"]["to_" + line_str]["dist_from_depot0"] / ( transfer_settings[to_line_str[3:]]["num_stations"] - 1 ))
            ratio_to_1 = 1 - ratio_to_0
            ratios_to = [ratio_to_0, ratio_to_1]
            
            ### four types of transferings
            from_depots = [(line-1)*2, (line-1)*2+1]
            to_depots = [(to_line-1)*2, (to_line-1)*2+1]
            for i, from_depot in enumerate(from_depots):
                for j, to_depot in enumerate(to_depots): # the arriving depot of the transfer 
                    transfer_time = duration * ratios_from[i] + tr + to_duration * ratios_to[j]

                    transfer_dict[f"{from_depot}-{to_depot}"] = transfer_time
                
    print(f"Build transfer dict with {len(transfer_dict)} items.")
    return transfer_dict


class IndividualCrewPlan():
    
    def __init__(self, duty_window_dict:dict, tasks_dict:dict) -> None:
        self.duty_windows = duty_window_dict
        self.tasks = tasks_dict

        self.validate()
    
    def validate(self):
        traversed_depots = set()
        
        for d in self.duty_windows.keys():
            cur_time = self.duty_windows[d][0]
            cur_depot = -1

            for task in self.tasks[d]:# {day: list of list [type, start_t, end_t, start_depot, end_depot]}
                if task is None: break
                assert task[1] == cur_time, f"t mismatch, {cur_time}, {task}, {self.tasks}"
                assert task[3] == cur_depot, f"depot mismatch,{cur_time}, {task}, {self.tasks}"
                cur_time = task[2]
                cur_depot = task[4]

                if cur_depot != -1:
                    traversed_depots.add(cur_depot)
        self.traversed_depots = traversed_depots

    def to_dict(self):
        res = defaultdict(dict)
        res["depots"] = list(self.traversed_depots)
        for d in self.duty_windows.keys():
            res[d]["duty_window"] = self.duty_windows[d]
            res[d]["tasks"] = self.tasks[d]
        return res

def transform_individual_plan(G:nx.DiGraph, path:list, config:dict)-> IndividualCrewPlan:

    duty_windows = dict() # {day: tuple}
    tasks_set = defaultdict(list) # {day: list of list [type, start_t, end_t, start_depot, end_depot]}
    for node1, node2 in zip(path[:-1], path[1:]):
        arc_type, c = G.edges[node1, node2]["label"], G.edges[node1, node2]["c"]
        n1s, n2s = node1.split("-"), node2.split("-")

        if arc_type == "si":
            day_start_time = int(n1s[-1])
            duty_windows[int(n1s[1])] = (day_start_time, min(day_start_time+config["system"]["alpha"], config["system"]["Tw"]))
            tasks_set[int(n1s[1])].append([arc_type, int(n1s[-1]), int(n2s[-1]), -1, int(n2s[4])])
        elif arc_type == "so":
            tasks_set[int(n1s[1])].append([arc_type, int(n1s[-1]), int(n2s[-1]), int(n1s[4]), -1])
        elif arc_type == "w":
            tasks_set[int(n1s[1])].append([arc_type, int(n1s[-1]), int(n2s[-1]), int(n1s[4]), int(n2s[4])])
        elif arc_type == "t":  
            tasks_set[int(n1s[1])].append([arc_type, int(n1s[-1]), int(n2s[-1]), int(n1s[4]), int(n2s[4])])
        elif arc_type == "n":
            tasks_set[int(n1s[1])].append(None)
        elif arc_type == "m":
            tasks_set[int(n1s[1])].append([arc_type, int(n1s[-1]), int(n2s[-1]), int(n1s[-3]), int(n1s[-3])])
        elif arc_type == "a":
            tasks_set[int(n1s[1])].append([arc_type, int(n1s[-1]), int(n2s[-1]), int(n1s[-3]), int(n1s[-3])])
        
    return IndividualCrewPlan(duty_windows, tasks_set)


            
                    


def check_net_info(G:nx.DiGraph):
    n_start, n_end = 0, 0
    n_sign_in, n_sign_out = 0, 0
    n_working = 0
    n_transfering = 0
    n_waiting, n_day_shifting, n_non_working = 0, 0, 0
    for edge in G.edges(data=True):
        l = edge[2]["label"]
        if l == "s":
            n_start += 1
        if l == "e":
            n_end += 1
        if l == "si":
            n_sign_in += 1
        if l == "so":
            n_sign_out += 1
        if l == "w":
            n_working += 1
        if l == "t":
            n_transfering += 1
        if l == "a":
            n_waiting += 1
        if l == "f":
            n_day_shifting += 1
        if l == "n":
            n_non_working += 1
    
    print(f"G has {n_start} starting-arcs, {n_end} ending-arcs.")
    print(f"G has {n_sign_in} sign-in-arcs, {n_sign_out} sign-out-arcs.")
    print(f"G has {n_working} working arcs.")
    print(f"G has {n_transfering} transfering arcs.")
    print(f"G has {n_waiting} waiting arcs.")
    print(f"G has {n_day_shifting} day-shifting arcs.")
    print(f"G has {n_non_working} non-working arcs.")
    print(f"In total: {G.number_of_edges()} edges inserted, and {G.number_of_nodes()} nodes in use.")
        
