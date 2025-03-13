import json
from pathlib import Path
from pprint import pprint
from datetime import datetime

class Recorder():
    
    def __init__(self, save_dir:Path, config) -> None:

        self.data = dict()

        self.R = config["crew"]["num_of_drivers"]
        self.save_dir = save_dir
        self.config = config

        print(Path.cwd())
    
    def create_record(self, record_name:str):
        if record_name not in self.data:
            self.data[record_name] = dict()
    
    def set_data(self, record_name, key:str, v):
        self.data[record_name][key] = v

    def set_data_dict(self, record_name, data_dict:dict):
        for k, v in data_dict.items():
            self.data[record_name][k] = v

    def has_data(self, record_name, key):
        if record_name not in self.data or key not in self.data[record_name]:
            return False
        return True
    
    def get_data(self, record_name, key):
        if not self.has_data(record_name, key):
            raise ValueError("Value not in Recorder.")
        return self.data[record_name][key]

    def print(self):
        pprint(self.data)
    
    def save_json(self, name:str="result", add_time=True):

        if add_time:
            now = datetime.now()
            time_stamp_str = now.date().strftime("%Y-%m-%d") + '-' + str(now.hour)
            file_name = name + '-' + f"{self.R}-" + time_stamp_str + ".json"
            dir_name = Path(self.save_dir, time_stamp_str)
            if not dir_name.exists():
                dir_name.mkdir(parents=True)
        else:
            file_name = name + '-' + f"{self.R}" + ".json"
            dir_name = Path(self.save_dir)

        with open(dir_name / file_name, "w") as f:
            json.dump(self.data, f)

        print(f"Save json in file {dir_name / file_name}")

    def save_config_json(self, add_time=True):
        if add_time:
            now = datetime.now()
            time_stamp_str = now.date().strftime("%Y-%m-%d") + '-' + str(now.hour)
            file_name = "config" + '-' + f"{self.R}-" + time_stamp_str + ".json"
            dir_name = Path(self.save_dir, time_stamp_str)
            if not dir_name.exists():
                dir_name.mkdir(parents=True)
        else:
            file_name = "config" + '-' + f"{self.R}" + ".json"
            dir_name = Path(self.save_dir)

        
        self.config["data_save_dir"] = str(self.config["data_save_dir"])
        with open(dir_name / file_name, "w") as f:
            json.dump(self.config, f)

        print(f"Save used config json in file {dir_name / file_name}")
        
        
        
    def save_summary_json(self, name:str="summary_result", add_time=True):
        
        assert "Summary" in self.data, "No summary in self.data."

        if add_time:
            now = datetime.now()
            time_stamp_str = now.date().strftime("%Y-%m-%d") + '-' + str(now.hour)
            file_name = name + '-' + f"{self.R}-" + time_stamp_str + ".json"
            dir_name = Path(self.save_dir, time_stamp_str)
            if not dir_name.exists():
                dir_name.mkdir(parents=True)
        else:
            file_name = name + '-' + f"{self.R}" + ".json"
            dir_name = Path(self.save_dir)
            
        with open(dir_name / file_name, "w") as f:
            json.dump(self.data["Summary"], f)

        print(f"Save summary json in file {dir_name / file_name}")
    