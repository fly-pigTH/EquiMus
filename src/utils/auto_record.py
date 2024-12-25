# experiment auto-record bot
import csv
import os
import datetime

def record_experiment(config, output_file="./log/experiment_log.csv"):
    '''
        config: {
            "id": "exp-1",
            "start_time": 实验时间,
            "notes": "some notes"
        }
        end_time: 自动计算
    '''
    # 确保日志文件存在
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Experiment ID", "Start Time", "End Time", "DataFileName", "Notes"])
    
    # 写入实验记录
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            config["id"],
            config["start_time"],
            datetime.datetime.now(),
            config["dataFileName"],
            config["notes"]
        ])


