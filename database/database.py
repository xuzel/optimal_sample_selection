import typing

from algorithms import Result
from pathlib import Path
import json
import os


class Database:
    def __init__(self, root_path: str = './'):
        os.makedirs(root_path, exist_ok=True)
        self.database_path = os.path.join(root_path, 'database.json')
        mode = 'r+' if not os.path.exists(self.database_path) else 'r+'
        self.database = open(self.database_path, mode)
        try:
            self.data = json.load(self.database)
        except json.JSONDecodeError:
            self.data = {}

    def search(self, **kwargs) -> typing.Dict:
        matching_dict = dict()
        for key, value in self.data.items():
            parts = key.split('-')
            match = True
            for part, expected_value in kwargs.items():
                expected_value = str(expected_value)
                if part in ['m', 'n', 'k', 'j', 's', 'alg', 'i']:  # 确保part合法
                    index = ['m', 'n', 'k', 'j', 's', 'alg', 'i'].index(part)
                    if parts[index] != expected_value:
                        match = False
                        break
            if match:
                matching_dict[key] = value
        return matching_dict

    def append_data(self, new_data: Result):
        have_value = self.search(
            m=new_data.input_parm[0],
            n=new_data.input_parm[1],
            k=new_data.input_parm[2],
            j=new_data.input_parm[3],
            s=new_data.input_parm[4],
            alg=new_data.algorithm,
        )
        for key in have_value.keys():
            self.data.pop(key)
        for index, (key, value) in enumerate(have_value.items()):
            self.data[f"{key[-2]}{index}"] = value
        self.data[
            f"{new_data.input_parm[0]}-{new_data.input_parm[1]}-{new_data.input_parm[2]}-{new_data.input_parm[3]}-{new_data.input_parm[4]}-{new_data.algorithm}-{len(have_value)}"] = {
            'solution': new_data.solution,
            'num_solution': new_data.solution_num,
            'time': new_data.run_time,
        }
        json.dump(self.data, self.database, indent=4)

    def delete_data(self, **kwargs):
        pass


if __name__ == '__main__':
    database = Database('./')
    database.data = {
        "1-2-3-4-5-algo1-1": "Value1",
        "1-3-3-4-6-algo2-2": "Value2",
        "1-2-3-4-0-algo1-3": "Value3",
        "2-2-3-4-5-algo1-4": "Value4"
    }
    print(database.search())
