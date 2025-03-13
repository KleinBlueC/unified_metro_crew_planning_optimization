from collections import defaultdict
import json
import math
import random
from random import randint
import time
from model.crew import MetroCrew, Driver
from model.recorder import Recorder
from pathlib import Path
from pprint import pprint

from utils.net_utils import calc_transfer_min_time_dict

class TrainTaskSet():
    
    def __init__(self, data:list) -> None:
        """
        provide the ranked task sets
        
        Args:
            data (list): list of dict of list
        """
        self.n_days = len(data)
        self.days = list(range(self.n_days))
        self.data = {i:data[i] for i in range(self.n_days)} # dict of dict of list {day:line:[]}

        self.original_task_cnt_dict = defaultdict(int) # {"total": x, "line_x": x}

        self.depot2line_fn = lambda o : int(o/2)+1
        for d in self.days:
            day_tasks = self.data[d]
            for o, day_depot_tasks in day_tasks.items():
                self.original_task_cnt_dict[f"line_{self.depot2line_fn(o)}"] += len(day_depot_tasks)
                self.original_task_cnt_dict["total"] += len(day_depot_tasks)
        
        # states for updating
        self.task_cnt_dict= defaultdict(int) 
        self.task_cvg_dict = defaultdict(float)
        
        self.violated_depot_cnt = 0
        
    def get_filtered_data_and_ranked_days(self, license:list) -> tuple:
        
        filtered_data = self._get_filtered_data(license)
        ranked_days = self._rank_days(filtered_data)

        return filtered_data, ranked_days
    
    def remove_task(self, day:int, depot:int, task:tuple):
        self.data[day][depot].remove(task)
    
    def update_state(self):
        self.task_cnt_dict.clear()
        for d in self.days:
            day_tasks = self.data[d]
            for o, day_depot_tasks in day_tasks.items():
                self.task_cnt_dict[f"line_{self.depot2line_fn(o)}"] += len(day_depot_tasks)
                self.task_cnt_dict["total"] += len(day_depot_tasks)

        
        for k, v in self.task_cnt_dict.items():
            self.task_cvg_dict[k] = round((self.original_task_cnt_dict[k] - v) / self.original_task_cnt_dict[k], 3)


    
    def _get_filtered_data(self, license:list):
        # get the valid depots
        valid_depots = []
        for line in license:
            valid_depots += [(line-1)*2, (line-1)*2+1]
        
        filtered_data = defaultdict(dict)

        for day in self.days:
            for depot in valid_depots:
                filtered_data[day][depot] = self.data[day][depot]
        
        return filtered_data
    
    def _rank_days(self, filtered_data:dict) -> list:
        day_cnts = [self._count_day_tasks(filtered_data, i) for i in range(self.n_days)]
        sorted_days = sorted(range(self.n_days), key=lambda i: day_cnts[i], reverse=True)
        return sorted_days
        
    
    def _count_day_tasks(self, filtered_data:dict, day:int):
        """
        day(int) is the index of working day (start from 0)
        """
        assert day in filtered_data.keys(), "wrong day number"
        return sum(len(line_tasks) for line_tasks in filtered_data[day].values())

    def __len__(self):
        """
        return the total count of  remained tasks
        """
        task_cnt = 0
        for d in self.days:
            day_tasks = self.data[d]
            task_cnt += sum(len(day_line_tasks) for day_line_tasks in day_tasks.values())
        return task_cnt
    
        
        

class IndividualPlan():
    """
    plan for one driver based on leg generation
    """

    def __init__(self, driver:Driver, n_days, config:dict):

        self.id = driver.driver_id
        self.lic = driver.license
        self.n_days = n_days

        self.config = config

        self.planned_legs = dict()

        self.total_cost = 0
    
        self.prefered_depots = driver.prefered_depots
        self.violated_depots = []

    def create_leg(self, day:int, duty_window:list=None, empty:bool=False):
        if empty:
            self.planned_legs[day] = None
            return
        self.planned_legs[day] = Leg(duty_window, self.config)
    
    def add_event(self, day:int, type:str, start:int, end:int, cost:float, depot_begin:int, depot_end:int):
        self.planned_legs[day].add( (type, (start, depot_begin), (end, depot_end), cost) )
    
    def close_leg(self, day:int):
        self.total_cost += self.planned_legs[day].summarize()
        #  add preference penalty
        for depot in self.planned_legs[day].sign_in_out_depots():
            if depot not in self.prefered_depots:
                self.violated_depots.append(depot)
                self.total_cost += self.config["crew"]["preference_settings"]["lambda_o"]
    
    def print_brief_info(self):
        for d, leg in self.planned_legs.items():
            assert leg is None or leg.summarized, f"leg {leg.tasks} not summarized"
            if leg is not None:
                print(f"-Day {d}: {leg.tasks}")
            else:
                print(f"-Day {d}: Sleep.")
                
        
    def __str__(self):
        ret = ""
        ret += f"Driver {self.id}:\n"
        for d, leg in self.planned_legs.items():
            assert leg.summarized, "leg not summarized"
            ret += f"-Day {d}\n"
            if leg is None:
                ret += "\tSleep.\n"
                continue
            for task in leg.tasks:
                ret += f"\t{task[0]}\t: t={task[1][0]}(depot {task[1][1]}) -> t={task[2][0]}(depot {task[2][1]})\tc={task[3]}\n"
        ret += f"-Violated depots {self.violated_depots}.\n"
        ret += f"-Total cost: {self.total_cost}.\n"
        return ret
        


class Leg():
    """
    the overall arrangement for one crew in one day, within one duty window of course
    """

    def __init__(self, duty_window, config):
        self.duty_start, self.duty_end = duty_window
        self.config = config
        self.tasks = []


        self.cost = False

        self.summarized = False

    def add(self, task:tuple):
        """
        task: ( type(str), (start_t, start_depot), (end_t, end_depot), cost) )
        """
        self.tasks.append(task)

    def summarize(self):
        """
        1 validate tasks
        2 determing waiting time
        3 summarize total cost
        """
        cur_time = self.duty_start
        cur_depot = -1
        
        for task in self.tasks:
            assert task[1][0] >= cur_time and task[1][1] == cur_depot, f"Task bug {task} in {self.tasks}" # validate

            if task[1][0] > cur_time:
                self.cost += self.config["work"]["type_cost"]["a"] * (task[1][0] - cur_time)
            
            self.cost += task[3]
            cur_time = task[2][0]
            cur_depot = task[2][1]

        assert self.tasks[0][0] == "si" and self.tasks[-1][0] == "so", "wrong sign-in and sign-out"
        self.summarized = True
        return self.cost

    def sign_in_out_depots(self):
        assert self.summarized is True, "without summarizing"
        return [self.tasks[0][2][1], self.tasks[-1][1][1]]
                



class BenchmarkAlgorithmSolver():
    
    """
    leg-generation based greedy heuristics
    
    """
    def __init__(self, crew:MetroCrew, config:dict, recorder:Recorder):
        self.config = config
        self.crew = crew

        self.recorder = recorder
        
        self._load_config_params()


    def run(self, display:bool=True, data_saving:bool=True):

        begin = time.time()

        self.prepare()
        self.greedy_leg_generation(display)

        end = time.time()
        self.bm_t = round(end-begin, 3)
        print(f"Time elapsed for bm: {self.bm_t}")

        if data_saving:
            record_name = "Summary"
            self.recorder.create_record(record_name)
            self.recorder.set_data(record_name, f"bm_obj", self.bm_obj)
            self.recorder.set_data(record_name, f"bm_cvg", self.bm_cvg)
            self.recorder.set_data(record_name, f"bm_t", self.bm_t)

    def run_for_reschedule(self, duty_gropups:list, train_services:list, virtual_duty_windows:dict, sorted_drivers:list, display:bool=True):

        print("Run leg-based greedy benchmark for rescheduling.")
        start_t = time.time()
        # prepare
        self.duty_groups = duty_gropups
        self.train_services = train_services
        self.task_set_manager = TrainTaskSet(self.train_services)
        self.transfer_dict = calc_transfer_min_time_dict(self.data_paths["transfer_settings"], self.data_paths["train_schedules"], self.tr)

        self.greedy_leg_generation(display=display, virtual_duty_windows=virtual_duty_windows, sorted_drivers=sorted_drivers)

        self.bm_t = round(time.time() - start_t, 3)
        
        return self.bm_obj, self.bm_cvg, self.bm_t, self.task_set_manager.task_cvg_dict
        

        

    ########################################################################################################
    ########################################################################################################
    ########################################################################################################

    def prepare(self):
        
        ### define optional duties
        self.duty_groups = []
        self._generate_duty_groups()
        
        ### define working train service tasks for each day, each line
        assert self.consider_weekends is True, "no task generation on a daily base ?"
        self.train_services = list()
        for _ in range(self.W):
            self.train_services.append(defaultdict(list))
        self._generate_train_services()
        self.task_set_manager = TrainTaskSet(self.train_services)

        self.transfer_dict = calc_transfer_min_time_dict(self.data_paths["transfer_settings"], self.data_paths["train_schedules"], self.tr)

        # pprint(self.train_services)
        
        
    def greedy_leg_generation(self, display:bool=True, virtual_duty_windows:dict=None, sorted_drivers:list=None):
        """
        main algorithm
        """

        result_plan = dict()
        greedy_total_cost = self.pi_summation
        
        print(f"Task cnt at the beginning: {self.task_set_manager.original_task_cnt_dict}")
        n_tasks_left = len(self.task_set_manager)
        n_violated_depots  = 0

        planned_drivers = self.crew.drivers if sorted_drivers is None else [self.crew.drivers[id] for id in sorted_drivers]
        
        for driver in planned_drivers:
            # RUN
            reschedule_plan = None if virtual_duty_windows is None else virtual_duty_windows[driver.driver_id]
            result_plan[driver.driver_id] = self.greedy_leg_for_one_driver(driver, reschedule_plan=reschedule_plan)
            if display:
                result_plan[driver.driver_id].print_brief_info()
                print(f"Finish {n_tasks_left - len(self.task_set_manager)} tasks, and {len(self.task_set_manager)} tasks left after driver {driver.driver_id}.")
            n_tasks_left = len(self.task_set_manager)
            self.task_set_manager.update_state()

            n_violated_depots += len(result_plan[driver.driver_id].violated_depots)
            greedy_total_cost += result_plan[driver.driver_id].total_cost

        print("Summary for benchmark leg-generation based heuristc:")
        print(f"Total cost without pi_summation: {greedy_total_cost}")
        pprint(f"Total cvg: {dict(self.task_set_manager.task_cvg_dict)}")
        pprint(f"Total violated depot count: {n_violated_depots} -> additional cost {n_violated_depots*self.config['crew']['preference_settings']['lambda_o']}")

        self.bm_obj, self.bm_cvg = greedy_total_cost, self.task_set_manager.task_cvg_dict["total"]
        

    
    def greedy_leg_for_one_driver(self, driver:Driver, reschedule_plan:list=None) -> IndividualPlan:
        """
        Core rule-based greedy heuristics

        
        reschedule_plan: [start, end, meal_eaten, cur_depot]
        """
        
        print(f"Driver id {driver.driver_id}:")
        lic = driver.license
        driver_plan = IndividualPlan(driver, self.n_si, self.config)

        filtered_tasks, ranked_days = self.task_set_manager.get_filtered_data_and_ranked_days(lic)
        
        selected_days = ranked_days[:self.n_si]
        
        
        # two util func
        alternate_depot_fn = lambda o : o+1 if o % 2 == 0 else o-1
        def _check_meal(begin, end, selected_duty_window) -> int:
            meal_allowed_start = selected_duty_window[1] - self.alpha + self.gb
            if min(end, selected_duty_window[1] - self.ge) - max(begin, meal_allowed_start) < self.g: # not suitable for a meal
                return None
            return max(begin, meal_allowed_start)


        # begin leg generation
        for d in selected_days:
            day_tasks = filtered_tasks[d]
            meal_eaten = False 
            
            if reschedule_plan is None: # normal benchmark
                ## find earliest task in any line
                first_task, first_line, first_depot = None, None, None # (start, end), int, int
                for l in lic:
                    for o in [(l-1)*2, (l-1)*2+1]:
                        if len(day_tasks[o]) != 0 and (first_task is None or first_task[0] > day_tasks[o][0][0]):
                            first_task = day_tasks[o][0]
                            first_line, first_depot = l, o
                if first_task is None:
                    print(f"No task available for driver {driver.driver_id} on day {d}.")
                    driver_plan.create_leg(d, empty=True)
                    continue
                    
                ## determine duty window (according to the first task selected)
                for duty_window in self.duty_groups[::-1]:
                    if duty_window[0] + self.eps1 <= first_task[0]:
                        selected_duty_window = duty_window
                        break
                driver_plan.create_leg(d, selected_duty_window)
            
                    
                ## begin adding tasks
                cur_time_unit = selected_duty_window[0]
                cur_depot = -1
                cur_line = -1
                ### sign-in
                driver_plan.add_event(d, "si", selected_duty_window[0], selected_duty_window[0]+self.eps1, 
                                        self.type_cost["si"]*self.eps1, cur_depot, first_depot)
                cur_time_unit += self.eps1
                cur_depot = first_depot
                cur_line = first_line


                ### check meal
                if not meal_eaten:
                    meal_time = _check_meal(cur_time_unit, first_task[0], selected_duty_window)
                    if meal_time is not None:
                        driver_plan.add_event(d, "m", meal_time, meal_time+self.g, 
                                            self.type_cost["m"]*self.g, cur_depot, cur_depot)
                        cur_time_unit = meal_time + self.g 
                        meal_eaten = True
                    
                ### first task
                driver_plan.add_event(d, "w", first_task[0], first_task[1] + self.beta, 
                                        (self.type_cost["w"] - self.pi_ratio)*(first_task[1] - first_task[0]) + self.type_cost["r"]*self.beta, 
                                        first_depot, alternate_depot_fn(first_depot)) # work and rest
                self.task_set_manager.remove_task(d, first_depot, first_task)
                # filtered_tasks[d][first_depot].remove(first_task) # shallow copy - remove once is ok
                cur_time_unit = first_task[1] + self.beta
                cur_depot = alternate_depot_fn(first_depot)


            else:
                determined_window = None
                meal_already_eaten, determined_first_depot = False, None
                if reschedule_plan is not None:
                    determined_window = [reschedule_plan[0]-self.eps1, reschedule_plan[1]]
                    meal_already_eaten = reschedule_plan[2]
                    determined_first_depot = reschedule_plan[3]
                selected_duty_window = determined_window
                meal_eaten = meal_already_eaten
                first_task, first_line, first_depot = None, int(determined_first_depot / 2)+1, determined_first_depot # (start, end), int, int

                driver_plan.create_leg(d, selected_duty_window)

                ## begin adding tasks
                cur_time_unit = selected_duty_window[0]
                cur_depot = -1
                cur_line = -1

                ### sign-in
                driver_plan.add_event(d, "si", selected_duty_window[0], selected_duty_window[0]+self.eps1, 
                                        self.type_cost["si"]*self.eps1, cur_depot, first_depot)
                cur_time_unit += self.eps1
                cur_depot = first_depot
                cur_line = first_line

                ### add meal if needed
                if not meal_eaten:
                    meal_time = _check_meal(cur_time_unit, selected_duty_window[1]-self.eps2, selected_duty_window) # have meal right now
                    if meal_time is not None:
                        driver_plan.add_event(d, "m", meal_time, meal_time+self.g, 
                                            self.type_cost["m"]*self.g, cur_depot, cur_depot)
                        cur_time_unit = meal_time + self.g 
                        meal_eaten = True


            #########################################################################################################################
            ## CORE GREEDY LOGIC
            ### option 1: closest train service task at the current depot
            ### option 2: if meets the time delay, the closest train service task at other depots in other available lines
            ### option 3: if meal is allowed, just eat
            #########################################################################################################################

            ready_to_sign_out = False
            while cur_time_unit <= selected_duty_window[1]:
            
                ### check meal
                if not meal_eaten:
                    meal_time = _check_meal(cur_time_unit, selected_duty_window[1]-self.eps2, selected_duty_window) # have meal right now
                    if meal_time is not None:
                        driver_plan.add_event(d, "m", meal_time, meal_time+self.g, 
                                            self.type_cost["m"]*self.g, cur_depot, cur_depot)
                        cur_time_unit = meal_time + self.g 
                        meal_eaten = True
                
                ### next train task - option 1:
                train_task_this_line = None
                for task in filtered_tasks[d][cur_depot]:
                    if task[0] > cur_time_unit:
                        train_task_this_line = task
                        break
                ### next train task - option 2
                train_task_other_line, next_line, next_depot = None, None, None
                for l in lic:
                    if l == cur_line:
                        continue
                    for o in [(l-1)*2, (l-1)*2+1]:
                        if f"{cur_depot}-{o}" not in self.transfer_dict: 
                            continue
                        transfer_time = self.transfer_dict[f"{cur_depot}-{o}"]
                        for task in filtered_tasks[d][o]:
                            if task[0] > cur_time_unit + transfer_time:
                                if train_task_other_line is None or train_task_other_line[0] > task[0]: # find the earliest other-line task
                                    train_task_other_line, next_line, next_depot, selected_transfer_time = task, l, o, transfer_time
                                break
                
                ### select next train task
                if train_task_this_line is not None and (train_task_other_line is None or train_task_this_line[0] < train_task_other_line[0]):
                    if train_task_this_line[1] + self.beta > selected_duty_window[1] - self.eps2:  # ready to sign-out
                        ready_to_sign_out = True
                    else:
                        driver_plan.add_event(d, "w", train_task_this_line[0], train_task_this_line[1] + self.beta, 
                                            (self.type_cost["w"] - self.pi_ratio)*(train_task_this_line[1] - train_task_this_line[0]) + self.type_cost["r"]*self.beta, 
                                            cur_depot, alternate_depot_fn(cur_depot)) # work and rest
                        self.task_set_manager.remove_task(d, cur_depot, train_task_this_line)
                        cur_time_unit = train_task_this_line[1] + self.beta
                        cur_depot = alternate_depot_fn(cur_depot)
                elif train_task_other_line is not None: # option 2
                    if train_task_other_line[1] + self.beta > selected_duty_window[1] - self.eps2: # ready to sign-out
                        ready_to_sign_out = True
                    else:
                        driver_plan.add_event(d, "t", cur_time_unit, cur_time_unit+selected_transfer_time, 
                                            self.type_cost["t"]*selected_transfer_time, 
                                            cur_depot, next_depot) # work and rest
                        driver_plan.add_event(d, "w", train_task_other_line[0], train_task_other_line[1] + self.beta, 
                                            (self.type_cost["w"] - self.pi_ratio)*(train_task_other_line[1] - train_task_other_line[0]) + self.type_cost["r"]*self.beta, 
                                            next_depot, alternate_depot_fn(next_depot)) # work and rest
                        self.task_set_manager.remove_task(d, next_depot, train_task_other_line)
                        cur_time_unit = train_task_other_line[1] + self.beta
                        cur_depot = alternate_depot_fn(next_depot) 
                        cur_line = next_line
                else: # no more task
                    ready_to_sign_out = True
                
                if ready_to_sign_out is True:
                    assert meal_eaten, f"Driver is hungury. {driver_plan.planned_legs[d].tasks} - reschedule_plan={reschedule_plan}" 

                    if reschedule_plan is not None:
                        is_halfway_duty = False
                        if selected_duty_window[1] - selected_duty_window[0] != self.alpha:
                            is_halfway_duty = True
                        has_work, has_meal = False, False
                        for task in driver_plan.planned_legs[d].tasks:
                            if task[0] == "w":
                                has_work = True
                            if task[0] == "m":
                                has_meal = True
                        if not has_work and ( (not is_halfway_duty) or (is_halfway_duty and not has_meal) ):
                            driver_plan.create_leg(d, empty=True)
                            break
                            
                    
                    driver_plan.add_event(d, "so", max(cur_time_unit, selected_duty_window[1]-self.delta-self.eps2), max(cur_time_unit, selected_duty_window[1]-self.delta-self.eps2)+self.eps2, 
                                        self.type_cost["so"] * self.eps2, cur_depot, -1) # work and rest
                    driver_plan.close_leg(d)
                    break
        
        return driver_plan
                        
                            
                
                
                


                






                
                
                
                    
                        
                    
                    
                

                

                    



                        
                        

                


            
            
        
        
        
        
        
    
        

        
        

         
        
        
        
        
        
        
    
    ############################################################################
    ############################################################################
    ## old dame codes...    
    ############################################################################
    ############################################################################
        
    def _load_config_params(self):
        config = self.config
        ### load params
        seed = config["seed"]
        random.seed(seed)
        ### parse configuration parameters

        self.data_paths = config["data_paths"]

        ##### system configurations
        self.W = config["system"]["W"] ### W: number of planning days
        self.Tw = config["system"]["Tw"]  ### Tw: number time units in one day  20h 5:00 - 1:00  time unit idx [0, 1200]  1200+1 units
        self.alpha = config["system"]["alpha"] ### alpha: maximum working time units in one day
        self.h = config["system"]["h"]  ### h: time intervals between optional duty_groups


        ##### train configurations
        self.lines = config["system"]["lines"]
        self.depots = {l: [(l-1)*2, (l-1)*2+1] for l in self.lines} ### map line number to depot ids

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

        self.add_rand_real = config["schedule"]["add_rand_real"]
        if self.add_rand_real:
            self.rand_range_real = config["schedule"]["rand_range_real"]
        
        self.pi_ratio = config["work"]["pi_ratio"] ### penalty same for all for now
        self.pi_summation = 0


        ### params for constraint shortest path
        self.n_si = config["work"]["n_si"]
        self.n_t = config["work"]["n_t"]

        self.meal_arc_interval = config["work"]["meal_arc_interval"]

        #### path cost
        self.arc_types = ["s", "e", "si", "so", "w", "t", "r", "a", "f", "n"]
        self.type_cost = config["work"]["type_cost"]
        self.tPi = config["work"]["tpi"] ### transfer fixed cost

        
        

        ### for intraday ablation - weekend-aware is not considered TODO
        self.special_intervals = None
        self.special_lines_included = []
        self.sp_services = defaultdict(list) ### {depot: [special_service_id...]}
        self.sp_services_pi = defaultdict(dict) ### {depot: {special_service_id: sp_pi, ...}, ...}
        if "add_special_intervals" in config["schedule"] and config["schedule"]["add_special_intervals"] is True:
            self.special_intervals = config["schedule"]["special_intervals"]
            self.special_lines_included = [int(line_str[-1]) for line_str in self.special_intervals.keys()]  # here assume line < 10


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
        # print(f"-> {n_duty_groups * 2 * len(self.lines) * self.cfg['W']} duty_groups in total")

        duty_start, duty_end = 0, self.alpha
        for _ in range(n_duty_groups-1):
            self.duty_groups.append([duty_start, duty_end])
            duty_start, duty_end = duty_start + h, duty_end + h
        self.duty_groups.append([Tw - alpha, Tw])
        
        print(f"Duty_groups in one day: {self.duty_groups}")
        print("****************************************************************")
        return

    def _generate_train_services(self, save_excel: bool=False):
        """
        two types of json timetables
        """
        print("****************************************************************")

        ### new version -  lines 1, 2, 9
        assert self.W == 3 or self.W == 7, "TODO for day W"

        # load json
        schedule_path = Path.cwd() / Path(self.data_paths["train_schedules"])
        
        with open(schedule_path, encoding="utf-8") as f:
            settings = json.load(f)
        for day_id in range(self.W):
            day = str(day_id + 1)
            day_pi_summation = 0
            for line in self.lines:
                line_sts = settings["line"+str(line)]
                duration = line_sts["duration"]


                for d_idx, depot in enumerate(self.depots[line]):

                    # fecth the day-aware data structures
                    for days in line_sts["depart_intervals"][str(depot)].keys():
                        if day in days:
                            print(line_sts["depart_intervals"])
                            depart_intervals = line_sts["depart_intervals"][str(depot)][days]
                            break
                    else:
                        raise ValueError("No day matched in train schedule table for depart intervals...")
                    for days in line_sts["last_train_depart_time"].keys():
                        if day in days:
                            last_starts = line_sts["last_train_depart_time"][days]
                            break
                    else:
                        raise ValueError("No day matched in train schedule table for last train depart time...")

                    if self.add_rand_real:
                        rand_range = self.rand_range_real

                    first_start = self._time_trans(line_sts["first_train_depart_time"][d_idx])
                    last_start = self._time_trans(last_starts[d_idx]) # here just use weekday data
                    
                    # add train services for one depot
                    s_start = first_start
                    s_end = first_start + duration
                    itv_idx = 0
                    _, itv_end, itv, pi, normal_flag = depart_intervals[itv_idx]
                    itv_end = self._time_trans(itv_end)



                    while s_start < last_start and s_end < self.Tw - self.eps2 - self.beta:
                        self.train_services[day_id][depot].append((s_start, s_end))     ### here changed for the day-aware version
                    
                        ### for next service
                        ## in normal intervals
                        day_pi_summation += (s_end - s_start) * pi ### here changed for the day-aware version
                        s_start += itv
                        ### add rand (mind that for ablation study, this should be canceled at least for intraday cases)
                        if self.add_rand_real:
                            s_start += randint(-rand_range, rand_range)

                        s_end = s_start + duration

                        if s_start > itv_end: # change time section
                            itv_idx += 1
                            _, itv_end, itv, pi, normal_flag = depart_intervals[itv_idx]
                            itv_end = self._time_trans(itv_end)

                print(f"Finish generating train services for line {line} in day {day}, {len(self.train_services[day_id][2*(line - 1)])} (backward) + {len(self.train_services[day_id][2*(line - 1) + 1])} (forward) = {len(self.train_services[day_id][2*(line - 1)]) + len(self.train_services[day_id][2*(line - 1)+1])} services.")

            ### finish generating services for one day
            print(f"->In total (for day {day}): {sum(len(self.train_services[day_id][2*(line - 1)]) + len(self.train_services[day_id][2*(line - 1)+1]) for line in self.lines)} train services among all lines.")
            self.pi_summation += day_pi_summation

            for line in self.lines:
                print("--> In total, for all days and line {}: {} train services.".format(line, sum(len(self.train_services[d][2*(line - 1)]) + len(self.train_services[d][2*(line - 1)+1]) for d in range(self.W))))
            print("---> In total, for all days and all lines: {} train services.".format(sum(sum(len(self.train_services[d][2*(l - 1)]) + len(self.train_services[d][2*(l - 1)+1]) for d in range(self.W)) for l in self.lines)))    
            print("****************************************************************")

        return


    def _time_trans(self, start_time:str) -> int:
        """
        from real start time str (e.g., 6:30) to time unit index
        currently: assume start_time is before 24:00
        """
        hour, minute = int(start_time.split(':')[0]), int(start_time.split(':')[1])
        return 60*(hour - 5) + minute