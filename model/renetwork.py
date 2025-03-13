import sys
sys.path.append(r"code_repository") 

import networkx as nx
import math
from collections import defaultdict
import time
from itertools import combinations, permutations
import json
from functools import reduce
import copy

import random
from random import randint
import platform

class ReNetworkFlowModel():
    def __init__(self, config:dict):
        """
        input: 
        1. model parameter configurations
        2. train timetable database
        3. crew database
        """

        ### rand seed
        seed = config["seed"]
        random.seed(seed)
        ### parse configuration parameters

        self.config = config

        self.data_paths = config["data_paths"]

        ##### system configurations
        self.W = config["system"]["W"] ### W: number of planning days
        self.Tw = config["system"]["Tw"]  ### Tw: number time units in one day  20h 5:00 - 1:00  time unit idx [0, 1200]  1200+1 units
        self.alpha = config["system"]["alpha"] ### alpha: maximum working time units in one day
        self.h = config["system"]["h"]  ### h: time intervals between optional duty_groups

        
        self.days = [config["reschedule"]["re_d"]]


        ##### train configurations
        self.real_case = config["schedule"]["real_case"] ### real_case: use real train schedules (Shanghai) or synthetic schedules
        self.lines = config["system"]["lines"]
        self.depots = {l: [(l-1)*2, (l-1)*2+1] for l in self.lines} ### map line number to depot ids


        self.consider_weekends = False
        if "consider_weekends" in config["schedule"]:
            self.consider_weekends = config["schedule"]["consider_weekends"]

        ##### parse crew stuff

        self.delta = config["work"]["delta"] ### delta: allowed early-leave time


        self.eps1 = config["work"]["eps1"]###  sign-in time length
        self.eps2 = config["work"]["eps2"] ###  sign-out time length
        self.beta = config["work"]["beta"] ### minimum rest time

        self.gb = config["work"]["gb"] ### start of meal interval  
        self.ge = config["work"]["ge"] ### end of meal interval  
        self.g = config["work"]["g"] ### (minimum) meal time 

        self.tr = config["work"]["tr"] ### transfer time length

        self.D = config["schedule"]["D"]

        if self.real_case is False:
            self.syn_data = config["schedule"]["syn_data"]
        else:
            self.add_rand_real = config["schedule"]["add_rand_real"]
            if self.add_rand_real:
                self.rand_range_real = config["schedule"]["rand_range_real"]
        
        if "pi_ratio" in config["work"]:
            self.pi_ratio = config["work"]["pi_ratio"] ### penalty same for all for now
            self.pi_summation = 0
        else:
            self.pi_ratio = None

        ### params for constraint shortest path
        self.n_si = config["work"]["n_si"]
        self.n_t = config["work"]["n_t"]

        #### path cost
        self.arc_types = ["s", "e", "si", "so", "w", "t", "r", "a", "f", "n"]
        self.type_cost = config["work"]["type_cost"]
        self.pi = config["work"]["pi"] ### penalty same for all for now
        self.tPi = config["work"]["tpi"] ### transfer fixed cost

        ### for reschedule
        self.driver2start = dict() # {driver_id: (sign_in_node, work_start_node)}
        self.sign_in_node, self.work_start_node = None, None
        self.special_task_cnt = 0




    def create(self, virtual_duty_windows:dict, ablation_strategy="-"):
        """
        create network flow model
        """
        
        assert ablation_strategy == "-" or ablation_strategy == "inter" or ablation_strategy == "intra", "Error: unknown ablation strategy."
        if ablation_strategy == "intra": self.D = 10000000000

        begin = time.time()


        self.train_services_w = list()
        self.special_train_services_w = list()
        self.transfer_options_w = list()
        for _ in self.days:
            self.train_services_w.append(defaultdict(list))
            self.special_train_services_w.append(defaultdict(list))
            self.transfer_options_w.append(defaultdict(list))


        self._generate_train_services()
        self._generate_transfer_options()


        ##### generate duty_groups ASSUME duty groups are NOT day-aware
        self.duty_groups = []
        self._generate_duty_groups()
        ##### match services & options to duty_groups
        self.service_duty_group_match = defaultdict(list) ### {depot_id: [ [duty_group_id1, duty_group_id_2, ...], [duty_group_id1, duty_group_id_2, ...], ... ]}   one service (start_end) for one list [duty_group_id1, duty_group_id_2, ...] (sequentially match self.train_services)
        self.train_services_C_set = defaultdict(set) ### {(start_time, end_time): (duty_group_id1, duty_group_id2, ...)}
        self.transfer_duty_group_match = defaultdict(list) ### {(depot_id_1, depot_id_2): [ [duty_group_id1, duty_group_id_2, ...], [duty_group_id1, duty_group_id_2, ...], ... ]}   one option (start_end) for one list [duty_group_id1, duty_group_id_2, ...] (sequentially match self.train_services)

    
        self.service_duty_group_match_w = list()
        self.train_services_C_set_w = list()
        self.transfer_duty_group_match_w = list()
        for _ in range(self.W):
            self.service_duty_group_match_w.append(defaultdict(list))
            self.train_services_C_set_w.append(defaultdict(set))
            self.transfer_duty_group_match_w.append(defaultdict(list))

        self._match_service_duty_groups()
        self._match_transfer_duty_groups()

        ##### generate network
        self.G = nx.DiGraph()

        
        # self._create_edges()
        self._create_edges_for_reschedule(virtual_duty_windows)
        self._clear_up_nodes()

        #### generate all sub network
        self.subnetworks = dict() ### {(lines): G}
        self._create_subnetworks()
        

        print(f"Finish creating network model, with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        print(f"Time elapsed: {time.time() - begin:.5} s")
        print("****************************************************************")

        return

    #######################
    #  set up train schedules
    #######################
    def _generate_train_services(self):
        """
        two types of json timetables
        """
        print("****************************************************************")
        # for linux:
        schedule_path = self.config["reschedule"]["reschedule_table"]
        if platform.system() == "Linux":
            schedule_path = schedule_path.replace("\\", "/")
        
        with open(schedule_path, encoding="utf-8") as f:
            settings = json.load(f)

        for day_id in [0]:
            day = str(day_id + 1)
            day_pi_summation = 0
            for line in self.lines:
                line_sts = settings["line"+str(line)]
                duration = line_sts["duration"]


                for d_idx, depot in enumerate(self.depots[line]):

                    # fecth the day-aware data structures
                    depart_intervals = line_sts["depart_intervals"][str(depot)]
                    last_starts = line_sts["last_train_depart_time"]

                    if self.add_rand_real:
                        rand_range = self.rand_range_real

                    first_start = self._time_trans(depart_intervals[0][0])
                    last_start = self._time_trans(last_starts[d_idx]) # here just use weekday data
                    
                    # add train services for one depot
                    s_start = first_start
                    s_end = first_start + duration
                    itv_idx = 0
                    _, itv_end, itv, pi, normal_flag = depart_intervals[itv_idx]
                    itv_end = self._time_trans(itv_end)

                    while s_start < last_start and s_end < self.Tw - self.eps2 - self.beta:
                        if normal_flag == 2: #special ones
                            self.special_train_services_w[day_id][depot].append((s_start, s_end, pi, normal_flag))     
                            self.special_task_cnt += 1
                            
                    
                        ### for next service
                        ## in normal intervals
                        day_pi_summation += (s_end - s_start) * pi 
                        s_start += itv
                        if self.add_rand_real and itv > 3: # here just adjust for cases with intervals > 3
                            s_start += randint(-rand_range, rand_range)

                        s_end = s_start + duration

                        if s_start > itv_end: # change time section
                            itv_idx += 1
                            _, itv_end, itv, pi, normal_flag = depart_intervals[itv_idx]
                            itv_end = self._time_trans(itv_end)

                print(f"Finish generating train services for line {line} in day {day}, {len(self.train_services_w[day_id][2*(line - 1)])} (backward) + {len(self.train_services_w[day_id][2*(line - 1) + 1])} (forward) = {len(self.train_services_w[day_id][2*(line - 1)]) + len(self.train_services_w[day_id][2*(line - 1)+1])} services.")

            ### finish generating services for one day
            print(f"->In total (for day {day}): {sum(len(self.train_services_w[day_id][2*(line - 1)]) + len(self.train_services_w[day_id][2*(line - 1)+1]) for line in self.lines)} train services among all lines.")
            self.pi_summation += day_pi_summation

            for line in self.lines:
                print("--> In total, for all days and line {}: {} train services.".format(line, len(self.train_services_w[0][2*(line - 1)]) + len(self.train_services_w[0][2*(line - 1)+1])))
            print("---> In total, for all days and all lines: {} train services.".format(sum(len(self.train_services_w[0][2*(l - 1)]) + len(self.train_services_w[0][2*(l - 1)+1]) for l in self.lines)))
            print("****************************************************************")

        return

    def _time_trans(self, start_time:str) -> int:
        """
        from real start time str (e.g., 6:30) to time unit index
        currently: assume start_time is before 24:00
        """
        hour, minute = int(start_time.split(':')[0]), int(start_time.split(':')[1])
        return 60*(hour - 5) + minute


    #######################
    #  set up transfer arcs
    #######################
    def _generate_transfer_options(self):
        """
        for this part, it has to be simplified... 比如，两个线路之间，可能有多个重合的站点，也就是多个换乘站
        1. assume one transfer station between two lines - consider line 1, 2, 9
        2. 
        info_source: http://sh.bendibao.com/ditie/linemap.shtml
        """

        ###### set up transfer conditions
        transfer_path = self.data_paths["transfer_settings"]
        if platform.system() == "Linux":
            transfer_path = transfer_path.replace("\\", "/")
        with open(transfer_path, encoding="utf-8") as f:
            transfer_settings = json.load(f)
        
        for line_str, line_st in transfer_settings.items():
            line = int(line_str[4:])
            if line not in self.lines:
                continue
            num_stations = line_st["num_stations"]
            for to_line_str, to_line_st in transfer_settings[line_str]["transfer"].items():
                to_line = int(to_line_str[7:])
                if to_line not in self.lines:
                    continue

                ### from_line: ratios
                ratio_from_0 = float(to_line_st["dist_from_depot0"] / (num_stations-1))
                ratio_from_1 = 1 - ratio_from_0
                ratios_from = [ratio_from_0, ratio_from_1]
                ### to_line: ratios
                ratio_to_0 = float(transfer_settings[to_line_str[3:]]["transfer"]["to_" + line_str]["dist_from_depot0"] / ( transfer_settings[to_line_str[3:]]["num_stations"] - 1 ))
                ratio_to_1 = 1 - ratio_to_0
                ratios_to = [ratio_to_0, ratio_to_1]


                # decrease the num of transfer arcs
                D = self.D

                drop = 0
                ### four types of transferings
                for i, from_depot in enumerate(self.depots[line]):
                    for j, to_depot_ in enumerate(self.depots[to_line]): 
                        ### DAY-AWARE
                        for day_id in [0]:
                            from_services = self.train_services_w[day_id][from_depot]  ### [(start, end), (start, end), ...]
                            to_services = self.train_services_w[day_id][to_depot_] ### [(start, end), (start, end), ...]
                            
                            arrive_depot = to_depot_ + 1 if j == 0 else to_depot_ - 1 ### the arriving station / depot

                            cur_to_services_id = 0
                            total_to_services = len(to_services)
                            for from_service in from_services: ### for each train service in the from line depot  - assign a matching train service to form a tranfering option!
                                drop += 1
                                if drop % D != 0: continue
                                option_start = from_service[0]
                                t1 = from_service[0] + ratios_from[i] * (from_service[1] - from_service[0]) ### time unit for arriving at transfer station
                                t2 = t1 + self.tr ### arriving at the other line
                                t3 = t2 + ratios_to[1-j] * (to_services[0][1] - to_services[0][0]) ### earliest time unit for transfer service final arrival;  j: services begin at j; 1-j: the depot id for transfer arc arrival
                                for t_s_id in range(cur_to_services_id, total_to_services):
                                    if to_services[t_s_id][1] >= t3: ### find the earliest matching to_service
                                        cur_to_services_id = t_s_id
                                        option_end = to_services[t_s_id][1]
                                        self.transfer_options_w[day_id][(from_depot, arrive_depot)].append((option_start, option_end))
                                        break
                            print(f"Finish generating transfer options from depot {from_depot} to depot {arrive_depot} in day {day_id+1}: {len(self.transfer_options_w[day_id][(from_depot, arrive_depot)])} options.")
                        print(f"-> Finish generating transfer options from depot {from_depot} to depot {arrive_depot} in all days: {len(self.transfer_options_w[0][(from_depot, arrive_depot)])} options.")



        for d in [0]:
            print(f"-->In total (for day {d+1}): {sum(len(v) for v in self.transfer_options_w[0].values())} transfer options.") 
        print(f"--->In total (for all days): {sum(len(v) for v in self.transfer_options_w[0].values())} transfer options.") 

        print("****************************************************************")

    #######################
    #  match each services / transfers to several duty groups (using time interval inclusion)
    #######################

    def _match_service_duty_groups(self):
        """
        {depot_id: [ [duty_group_id1, duty_group_id_2, ...], [duty_group_id1, duty_group_id_2, ...], ... ]}   
        one service (start_end) for one list [duty_group_id1, duty_group_id_2, ...] (sequentially match self.train_services)
        
        in fact, just (start_time, end_time) is enough to determine all duty groups - no need to consider depot additionally
        {(start_time, end_time): (duty_group_id1, duty_group_id2, ...)}  use set(can be duplicates)
        
        """
    
        working_arcs_cnt = 0
        for day_id in [0]:
            services_match_total_count = defaultdict(int)
            for o, services in self.train_services_w[day_id].items():
                for s, service in enumerate(services):
                    self.service_duty_group_match_w[day_id][o].append([])
                    for d, duty_group in enumerate(self.duty_groups):
                        if self._determine_match(service[:2], duty_group):
                            self.service_duty_group_match_w[day_id][o][s].append(d)
                            self.train_services_C_set_w[day_id][(service[0], service[1])].add(d)

                    one_service_match_cnt = len(self.service_duty_group_match_w[day_id][o][s])
                    assert one_service_match_cnt, f"service {service} in depot {o} no matching duty group?"
                    services_match_total_count[o] += one_service_match_cnt
            print("Finish matching train services to duty group indexes in day {}.".format(day_id+1))
            print(f"Counting working arcs for {len(services_match_total_count.keys())} depots in day {day_id+1}: {sum(n for n in services_match_total_count.values())}.")
            working_arcs_cnt += sum(n for n in services_match_total_count.values())
        print(f"-> Counting working arcs for {len(services_match_total_count.keys())} depots in all days: {working_arcs_cnt}.")


        return


    def _match_transfer_duty_groups(self):
        
        transfer_arcs_cnt = 0
        for day_id in [0]:
            transfers_match_total_count = defaultdict(int)
            for od, options in self.transfer_options_w[day_id].items():
                for s, option in enumerate(options):
                    self.transfer_duty_group_match_w[day_id][od].append([])
                    for d, duty_group in enumerate(self.duty_groups):
                        if self._determine_match(option, duty_group):
                            self.transfer_duty_group_match_w[day_id][od][s].append(d)
                    
                    one_option_match_cnt = len(self.transfer_duty_group_match_w[day_id][od][s])
                    assert one_option_match_cnt, f"option {option} in depot-depot {od} no matching duty group ?"
                    transfers_match_total_count[od] += one_option_match_cnt
            print("Finish matching transfer options to duty group indexes.")
            print(f"Counting transfering arcs for {len(transfers_match_total_count.keys())} depot pairs in day {day_id+1}: {sum(n for n in transfers_match_total_count.values())}.")
            transfer_arcs_cnt += sum(n for n in transfers_match_total_count.values())
        print(f"-> Counting transfer arcs for {len(transfers_match_total_count.keys())} depots in all days: {transfer_arcs_cnt}.")

        print("****************************************************************\n\n\n")

        return
    
    def _determine_match(self, task:tuple, duty_group:tuple):
        """
        decide if a task(train service or transfer service) can be included in one duty_group
        one duty must begin with a sign-in and end with a rest + a sign-out  (here a rest is mandatory for sign-out, for convenience)

        """
        t1, t2 = task
        d1, d2 = duty_group
        d1_ = d1 + self.eps1
        d2_ = d2 - self.eps2 - self.beta
        return t1 >= d1_ and t2 <= d2_
        

    #######################
    #  set up crews
    #######################
    def _generate_duty_groups(self):
        """
        determine [s, e] of each duty group

        definition: 
            duty_groups: just time interval in each day
            duty: time interval for each line
            number of duty_groups * number of lines = number of duties
    
        """
        Tw = self.Tw
        alpha = self.alpha
        h = self.h
        n_duty_groups = math.ceil((self.Tw - self.alpha) / self.h) + 1
        print(f"Finish generating {n_duty_groups} duty_groups in one day for one line for one station.")

        duty_start, duty_end = 0, self.alpha
        for _ in range(n_duty_groups-1):
            self.duty_groups.append([duty_start, duty_end])
            duty_start, duty_end = duty_start + h, duty_end + h
        self.duty_groups.append([Tw - alpha, Tw])
        
        print(f"Duty_groups in one day: {self.duty_groups}")
        print("****************************************************************")
        return



    #######################
    #  create nodes -> clear up nodes
    #######################
    def _clear_up_nodes(self):
        """
        create node indexes for each type
        
        index system:
        1. "sc", "sk"
        2. in-nodes and out-nodes  
            count: Tw * W in-nodes and out-nodes
            name: "i-d-t", "o-d-t"
                # in-nodes are the fixed to the first time unit in each duty;  - "i-d-t"
                # out-nodes are all possible time units within 'delta' range - "o-d-t"
        3. state nodes
            count: W(Days)  * D(duty_groups)* L(lines) * 2(depot1, depot2) * 2(arrive, leave) * alpha(duty length)
            name: "s-w-d-l-o-0/1-t"    0- (just) arrive, 1- (about to) leave
        """
        # print("****************************************************************")
        s_nodes = ["sc", "sk"]

        days_nodes = [list() for _ in range(self.W)]

        in_nodes, out_nodes = [], []
        for w in range(self.W):
            for t in range(self.Tw+1):
                if self.G.has_node(f"i-{w}-{t}"):
                    in_nodes.append(f"i-{w}-{t}")
                    days_nodes[w].append(f"i-{w}-{t}")

        assert len(self.duty_groups) != 0, "duty_groups not generated yet !"
        state_nodes = defaultdict(list)
        for w in range(self.W):
            for _t in range(self.alpha+1): ### +1
                for l in self.lines:  
                    for depot in [(l-1)*2, (l-1)*2+1]:
                        for state in [0, 1]:
                            for d in range(len(self.duty_groups)):
                                duty_start = self.duty_groups[d][0]
                                if self.G.has_node(f"s-{w}-{d}-{l}-{depot}-{state}-{_t+duty_start}"):
                                    state_nodes[l].append(f"s-{w}-{d}-{l}-{depot}-{state}-{_t+duty_start}")
                                    days_nodes[w].append(f"s-{w}-{d}-{l}-{depot}-{state}-{_t+duty_start}")
        
        for w in range(self.W):
            for t in range(self.Tw+1):
                if self.G.has_node(f"o-{w}-{t}"):
                    out_nodes.append(f"o-{w}-{t}")
                    days_nodes[w].append(f"o-{w}-{t}")

        self.out_nodes = out_nodes
        all_shared_nodes = s_nodes + in_nodes + out_nodes
        all_working_nodes = reduce(lambda a, b:a+b, state_nodes.values()) 


        self.line_state_nodes = state_nodes # dict {int:list}
        self.shared_nodes = all_shared_nodes # list

        print(f"Finish clearing up {len(in_nodes)} in-nodes, {len(out_nodes)} out-nodes and {len(all_working_nodes)} state-nodes")
        print("****************************************************************")



    #######################
    #  create edges
    #######################
    def _create_edges_for_reschedule(self, virtual_duty_windows:dict):
        
        ### !!!!!!!! starting and sign-in arcs
        for driver_id, duty_setting in virtual_duty_windows.items():
            work_start_t, duty_end, _, start_depot = duty_setting
            duty_id = None
            for i, (s, t) in enumerate(self.duty_groups):
                if t == duty_end:
                    duty_id = i
                    break
            sign_in_node = f"i-0-{work_start_t-self.eps1}"
            work_start_node = f"s-0-{duty_id}-{int(start_depot / 2) + 1}-{start_depot}-1-{work_start_t}"
            self.G.add_edge("sc", sign_in_node, label="s", c=self.type_cost["s"])
            self.G.add_edge(sign_in_node, work_start_node, label="si", c=float("inf"))
            self.driver2start[driver_id] = (sign_in_node, work_start_node)
            
        

        ##### ending arcs
        for w in [0]:
            for d in range(len(self.duty_groups)):
                duty_end = self.duty_groups[d][1]
                for _t in range(self.delta+1):  # here +1 for now
                    self.G.add_edge(f"o-{w}-{duty_end - _t}", "sk", label="e", c=self.type_cost["e"])


        ##### sign-out arcs
        for w in [0]:
            for d in range(len(self.duty_groups)):
                duty_start, duty_end = self.duty_groups[d]
                for l in self.lines:
                    for depot in self.depots[l]:
                        for _t in range(self.delta+1):  # here +1 for now
                            self.G.add_edge(f"s-{w}-{d}-{l}-{depot}-1-{duty_end - _t - self.eps2}", f"o-{w}-{duty_end - _t}", label="so", c=self.type_cost["so"]*self.eps2)

        ##### working arcs
        for w in [0]:
            for l in self.lines:
                for o in self.depots[l]:
                    zip_services = zip(self.train_services[o], self.service_duty_group_match[o]) if not self.consider_weekends else zip(self.train_services_w[w][o], self.service_duty_group_match_w[w][o])
                    for service_id, (service, duty_group_ids) in enumerate(zip_services):
                        for d in duty_group_ids:
                            ### working arc
                            arrive_depot_id = o + 1 if o % 2 == 0 else o - 1
                            pi_s = (service[1]-service[0]) * service[2]
                            self.G.add_edge(f"s-{w}-{d}-{l}-{o}-{1}-{service[0]}", f"s-{w}-{d}-{l}-{arrive_depot_id}-{1}-{service[1] + self.beta}", label="w", c=self.type_cost["w"]*(service[1]-service[0]) - pi_s + self.type_cost["r"]*self.beta)
                            # added for special tasks
                            if service[3] == 2:
                                self.G.add_edge(f"s-{w}-{d}-{l}-{o}-{1}-{service[0]}", f"s-{w}-{d}-{l}-{arrive_depot_id}-{1}-{service[1] + self.beta}", label="w", c=self.type_cost["w"]*(service[1]-service[0]) - pi_s + self.type_cost["r"]*self.beta, special=True)

        ##### meal arcs
        for w in [0]:
            for d in range(len(self.duty_groups)):
                duty_start, duty_end = self.duty_groups[d]
                meal_last_start = duty_end - self.ge - self.g
                for l in self.lines:
                    for o in self.depots[l]:
                        for t in range(duty_start + self.gb, meal_last_start):
                            self.G.add_edge(f"s-{w}-{d}-{l}-{o}-1-{t}", f"s-{w}-{d}-{l}-{o}-1-{t + self.g}", label="m", c=self.type_cost["m"]*self.g)


        ##### transfering arcs
        depot_ods = self.transfer_options_w[0].keys() if self.consider_weekends else self.transfer_options.keys()
        for w in [0]:
            for od in depot_ods:
                zip_options = zip(self.transfer_options[od], self.transfer_duty_group_match[od]) if not self.consider_weekends else zip(self.transfer_options_w[w][od], self.transfer_duty_group_match_w[w][od])
                for option, duty_group_ids in zip_options:
                    for d in duty_group_ids:
                        ### transfering arc
                        from_depot_id, arrive_depot_id = od
                        from_line, arrive_line =int(from_depot_id / 2 + 1), int(arrive_depot_id / 2 + 1) 
                        self.G.add_edge(f"s-{w}-{d}-{from_line}-{from_depot_id}-{1}-{option[0]}", f"s-{w}-{d}-{arrive_line}-{arrive_depot_id}-{1}-{option[1] + self.beta}", label="t", c=self.type_cost["t"]*(option[1] - option[0]) + self.tPi + self.type_cost["r"]*self.beta, lines = (from_line, arrive_line))


        ##### waiting arcs
        for w in [0]:
            for d in range(len(self.duty_groups)):
                duty_start = self.duty_groups[d][0]
                for l in self.lines:
                    for depot in self.depots[l]:
                        n1 = None
                        for _t in range(self.alpha):  # all possible waits
                            n2 = f"s-{w}-{d}-{l}-{depot}-1-{duty_start + _t}"
                            if self.G.has_node(n2):
                                if n1 is not None: # just set init cost
                                    t1, t2 = int(n1.split("-")[-1]), int(n2.split("-")[-1])
                                    self.G.add_edge(n1, n2, label="a", c=(t2 - t1) * self.type_cost["a"])
                                n1 = n2

        ###### non-working arcs
        self.G.add_edge("sc", "sk", label="n", c=self.type_cost["n"])
         

        ##### counts
        n_start, n_end = 0, 0
        n_sign_in, n_sign_out = 0, 0
        n_working = 0
        n_meal = 0
        n_transfering = 0
        n_waiting, n_day_shifting, n_non_working = 0, 0, 0
        for edge in self.G.edges(data=True):
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
            if l == "m":
                n_meal += 1
            if l == "t":
                n_transfering += 1
            if l == "a":
                n_waiting += 1
            if l == "f":
                n_day_shifting += 1
            if l == "n":
                n_non_working += 1
        
        print("****************************************************************")
        print(f"Finish generating {n_start} starting-arcs, {n_end} ending-arcs.")
        print(f"Finish generating {n_sign_in} sign-in-arcs, {n_sign_out} sign-out-arcs.")
        print(f"Finish generating {n_working} working arcs.")
        print(f"Finish generating {n_meal} meal arcs.")
        print(f"Finish generating {n_transfering} transfering arcs.")
        print(f"Finish generating {n_waiting} waiting arcs.")
        print(f"Finish generating {n_day_shifting} day-shifting arcs.")
        print(f"Finish generating {n_non_working} non-working arcs.")
        print(f"In total: {self.G.number_of_edges()} edges inserted, and {self.G.number_of_nodes()} nodes in use.")
            
        print("****************************************************************")


    #######################################
    #  create sub-networks: space for time
    #######################################
    def _create_one_subnetwork(self, lines_select:list):
        """
        return the subgraph where only nodes in 'lines_select' are included for faster shortest path finding

        Mind the copy problem!
        Currently this function: frozen graph (graph view), part of self.G, no deepcopy

        return: a copy of subgraph, and a subgraph view

        """
        nodes_select = copy.deepcopy(self.shared_nodes)
        for l in lines_select:
            nodes_select += self.line_state_nodes[l]
        return self.G.subgraph(nodes_select).copy()

    def _create_subnetworks(self):
        """
        all possible line combinations are included
        """
        for line_cnt in range(1, 3):
            for lines_select in combinations(self.lines, line_cnt):
                subnetwork = self._create_one_subnetwork(lines_select)
                print(f"Finish generating subnetwork for lines {lines_select}, with {subnetwork.number_of_nodes()} nodes and {subnetwork.number_of_edges()} edges.")
                self.subnetworks[lines_select] = subnetwork

    def _pprint_path(self, G, path, detailed=False):
        """
        path: list of nodes
        """
        for node1, node2 in zip(path[:-1], path[1:]):
            arc_type, c = G.edges[node1, node2]["label"], G.edges[node1, node2]["c"]
            n1s, n2s = node1.split("-"), node2.split("-")
            if arc_type == "s":
                print(f"start(w={n2s[-2]}, t={n2s[-1]})", end=" -> \n")
            if arc_type == "e":
                print(f"end", end=".\n\n")
            elif arc_type == "si":
                print(f"sign-in(w={n1s[1]}, t={n1s[-1]},t'={n2s[-1]}, depot={n2s[4]}, line={n2s[3]})", end=" -> \n")
            elif arc_type == "so":
                print(f"sign-out(t={n1s[-1]}, t'={n2s[-1]}, depot={n1s[4]}, line={n1s[3]})", end=" -> \n")
            elif arc_type == "w":
                print(f"working(t={n1s[-1]},t'={n2s[-1]}, depot={n1s[4]}, depot'={n2s[4]}, c={c})", end=" -> \n")
            elif arc_type == "t":  
                print(f"transfering(t={n1s[-1]}, t'={n2s[-1]}, line={n1s[3]}, depot={n1s[4]}, line'={n2s[3]}, depot'={n2s[4]}, c={c})", end=" -> \n")
            elif arc_type == "f":
                print(f"shifting(w={n1s[1]}, w'={n2s[1]})", end=" -> \n")
            elif arc_type == "n":
                print(f"sleep and sleep", end=" .\n")
            
            if detailed:
                if arc_type == "r":  
                    print(f"resting(t={n1s[-1]}, t'={n2s[-1]}, depot={n1s[-3]}, line={n2s[-4]}, c={c})", end=" -> \n")
                elif arc_type == "a":
                    print(f"waiting(t={n1s[-1]}, t'={n2s[-1]}, depot={n1s[-3]}, line={n2s[-4]}, c={c})", end=" -> \n")




#######################################################################################
## for reschedule
#######################################################################################

    def activate_driver_path(self, driver_id:int, G):
        u, v = self.driver2start[driver_id]
        G.edges[u, v]["c"] = self.type_cost["si"]*self.eps1

    def deactivate_driver_path(self, driver_id:int, G):
        u, v = self.driver2start[driver_id]
        G.edges[u, v]["c"] = float("inf")
