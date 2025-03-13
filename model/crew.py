import random
import copy
from itertools import combinations
from collections import defaultdict
import random
from utils.crew_utils import get_power_set, get_superset_group, get_union_supeset_collections

class Driver():
    def __init__(self, driver_id:int, begin_depot_id:int, begin_line:int, license:list, transfer_cost:float, prefered_depots:list):
        self.driver_id = driver_id
        self.begin_depot_id = begin_depot_id
        self.begin_line = begin_line
        self.license = license
        self.transfer_cost = transfer_cost
        self.prefered_depots = prefered_depots
        
class MetroCrew():

    def __init__(self, config:dict, preference_strategy:int=0):
        """
        input:
        @num_of_drivers: int (R)
        @lines: lines of int e.g., [1, 2, 9] line 1, line 2, line 9
        """

        seed = config["seed"]
        random.seed(seed)

        self.n_drivers = config["crew"]["num_of_drivers"]
        self.driver_ids = list(range(self.n_drivers))
        self.drivers = []

        
        ######## add proportions 20240607
        self.double_ratio = 0.4
        if "double_ratio" in config["crew"]:
            self.double_ratio = config["crew"]["double_ratio"]

        self.lines = config["system"]["lines"]
        self.line2idx = {l:i for i, l in enumerate(self.lines)}
        self.depots = []
        for l in self.lines:
            self.depots += [(l-1)*2, (l-1)*2+1] 

        self.all_master = config["crew"]["master_all"]

        
        ###########################
        # self.all_master = True


        ### preference heterogeneity
        self.consider_preferences = False
        if "preference_settings" in config["crew"]:
            print("MIND: Considering the preference heterogeneity...")
            self.consider_preferences = True
            self.cw_max = config["crew"]["preference_settings"]["cw_max"]
            self.prefered_depot_cnt = config["crew"]["preference_settings"]["prefered_depot_cnt"]
            assert self.prefered_depot_cnt == 2, "TODO: consider more or less than two prefered depot cnts"
            self.lambda_o = config["crew"]["preference_settings"]["lambda_o"]

        
        # for the ablation asm data problem... 20240530
        self.preference_strategy = preference_strategy
        if self.preference_strategy == 1:
            assert self.consider_preferences and self.prefered_depot_cnt == 2


        ### generate crew now
        generate_method = config["crew"]["generate_method"]
        if generate_method == "uniform":
            self._generate_uniform_crew()
        else:
            raise ValueError("TODO for crew generation")

        ### get patterns
        self.m = [0] * (2 ** len(self.lines) - 1) # all possible certificate patterns
        self.patterns = [] # this is the patterns
        self.pat2idx = dict() #(tuple, int)
        i = 0
        for line_num in range(1, len(self.lines)+1):
            for pattern in combinations(sorted(self.lines), line_num):
                self.pat2idx[pattern] = i
                self.patterns.append(pattern)
                i += 1

        self.qualifications = [tuple(pat) for pat in self.patterns]
        print(f"Qualifications: {self.qualifications}")
        self.crew_pools = dict()  # {qualification tuple: [driver indexes]}
        for q in self.qualifications:
            self.crew_pools[q] = []


        ### get param m_i and crew pools
        for driver_id, driver in enumerate(self.drivers):
            for line_num in range(1, len(driver.license)+1):
                for pattern in combinations(sorted(driver.license), line_num):
                    pat_i = self.pat2idx[pattern]
                    self.m[pat_i] += 1
                    self.crew_pools[tuple(pattern)].append(driver_id)
                    
        print(f"Final param m(line license pattern counts) for {list(self.pat2idx.keys())}: {self.m}")

        self.valid_patterns = set()
        for pat, pat_cnt in zip(self.patterns, self.m):
            if pat_cnt > 0:
                self.valid_patterns.add(pat)     


        # reconstruct Q and P
        self.s_crew_pools = dict() # one crew one pool
        for q in self.qualifications:
            self.s_crew_pools[q] = []
        for driver_id, driver in enumerate(self.drivers):
            self.s_crew_pools[tuple(driver.license)].append(driver_id)
        for q, pool in self.s_crew_pools.items():
            print(f"Single crew pool {q} - ({len(pool)})")
        self.np = {q: len(pool) for q, pool in self.s_crew_pools.items()}

        def rank_q(q):
            return len(q)*10000000000 + q[0]

        Q = [] # list of tuples (q)
        for i in range(1, 2+1): # just take double-line crew
            for q_i in combinations(sorted(self.lines), i): # Q_i
                Q.append(q_i)
        print(f"Q ({len(Q)}):{Q}")
        E = [] # list of tuples (Eq)
        for q in Q:
            E.append(get_superset_group(q, Q))
        print(f"E ({len(E)}):{E}")
        S = set() 
        j=1
        for i in range(1, len(E)+1):
            for partEs in combinations(sorted(E), i): # Q_i
                SE = set() # as the union set of part
                for Eq in partEs:
                    SE = SE.union(Eq)
                j += 1
                S.add(tuple(sorted(SE, key=rank_q)))
        print(j)
        bS = sorted(S, key=len)
        print(f"bS ({len(bS)}):")
        for s in bS:
            print(s)


        self.Qs_np = dict()
        self.Qs = bS
        for S in self.Qs:
            np_S = 0
            for q in S:
                np_S += self.np[q]
            self.Qs_np[tuple(S)] = np_S
        for S, np_S in self.Qs_np.items():
            print("S {}: {}".format(S, np_S))
            
        #######################################################################
        


    def _generate_uniform_crew(self):

        ### generate begin_depot_ids uniformly
        n_group, n_left = len(self.driver_ids) // len(self.depots), len(self.driver_ids) % len(self.depots)
        begin_depot_ids = self.depots * n_group + self.depots[:n_left] # list

        ### generate licenses uniformly  1-2;1-3;2-3;1-2-3
        licenses = [[(depot_id // 2) + 1] for depot_id in begin_depot_ids]
        begin_lines = copy.deepcopy(licenses)

        driverNumUnit = 100  
        license_group_size = (len(self.driver_ids)) // driverNumUnit  
        two_qualified_group_num = int(driverNumUnit * self.double_ratio)
        two_qualified_groups = list(range(0, two_qualified_group_num // 2)) + list(range(int(driverNumUnit/2), int(driverNumUnit/2) + two_qualified_group_num - two_qualified_group_num // 2))
        for group in range(driverNumUnit):
            for driver_id in range(group * license_group_size, (group + 1) * license_group_size):
                if group in two_qualified_groups: #two qualified lines
                    begin_line_idx = self.lines.index(licenses[driver_id][0])
                    additional_line = self.lines[(begin_line_idx + 1) % len(self.lines)]
                    licenses[driver_id].append(additional_line) # master two lines
                else: continue

        ### generate transfer cost randomly
        transfer_costs = [0] * len(self.driver_ids)

        ### ADD: generate preferred depots
        prefered_depots = []
        for driver_id in self.driver_ids:
            allowed_depots = []
            for l in licenses[driver_id]:
                allowed_depots += [(l-1)*2, (l-1)*2+1]
            if self.consider_preferences:
                if self.preference_strategy == 0:
                    prefered_depots.append(random.sample(allowed_depots, self.prefered_depot_cnt))
                elif self.preference_strategy == 1: 
                    if len(allowed_depots) == 2:
                        prefered_depots.append(allowed_depots)
                    else: # for crew with 2 qualified lines, generate one random prefered depot for each line
                        assert len(allowed_depots) == 4
                        r_prefered = []
                        for l in licenses[driver_id]:
                            r_prefered.append((l-1)*2 + random.randint(0, 1))
                        prefered_depots.append(r_prefered)             
            else:
                prefered_depots.append(allowed_depots)
            
        ### generate drivers
        for driver_id in self.driver_ids:
            self.drivers.append(Driver(driver_id, begin_depot_ids[driver_id], begin_lines[driver_id], sorted(licenses[driver_id]), transfer_costs[driver_id], sorted(prefered_depots[driver_id])))
        self.drivers_reordered = sorted(self.drivers, key=lambda x:len(x.license), reverse=False)
        
        print(f"Finish generating {len(self.drivers)} drivers in a uniform manner.")

    
    def display_crew_info(self):
        assert len(self.drivers) != 0, "No driver generated."

        print("--------------- drivers on roster ---------------")
        for driver in self.drivers:
            if self.consider_preferences:
                print(f"id: {driver.driver_id}\tlicense:{driver.license}\t\ttransfer_cost:{driver.transfer_cost}\t\tprefered_depots:{driver.prefered_depots}")
            else:
                print(f"id: {driver.driver_id}\tstart_depot:{driver.begin_depot_id}\tstart_line:{driver.begin_line}\tlicense:{driver.license}\t\ttransfer_cost:{driver.transfer_cost}")
        print("-----------------------------------------------")



