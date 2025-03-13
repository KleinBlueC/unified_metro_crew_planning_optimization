import json
from pathlib import Path
from collections import defaultdict
import numpy as np


class RandSummarizer():

    def __init__(self, sum_dir:Path) -> None:
        self.data_as_lists = defaultdict(list) 
        self.sum_dir = sum_dir

        self.save_cnt = 0
        
        self.summarized_dict = None
    
    def fetch_data(self, data:dict):
        for k, v in data.items():
            self.data_as_lists[k].append(v)
        self.save_cnt += 1

    
    def summarize(self, save:bool=True, file_name="rand_summary"):

        file_name = self.sum_dir / f"{file_name}.json"

        summarized_dict = dict()
        for k, v in self.data_as_lists.items():
            summarized_dict[k] = dict()
            v = np.array(v)
            
            summarized_dict[k]["cnt"] = len(v)
            summarized_dict[k]["mean"] = np.mean(v)
            summarized_dict[k]["std"] = np.std(v)
            summarized_dict[k]["max"] = np.max(v)
            summarized_dict[k]["min"] = np.min(v)
        
        self.summarized_dict = summarized_dict

        if save:
            with open(file_name, "w") as f:
                json.dump(summarized_dict, f)
            print(f"Save rand summarizer results at {file_name}")
    
    @property
    def data(self):
        return self.summarized_dict

        
        
        
        