import json
import os

MLDOC_DIR = "/export/b10/amueller/bert/data/mldoc"
OUT_DIR = "/export/b10/amueller/bert/data/mldoc_json"


with open(os.path.join(OUT_DIR, "mldoc.json"), 'w') as outfile:
    for filename in os.listdir(MLDOC_DIR):
        with open(os.path.join(MLDOC_DIR, filename), 'r') as dataset:
            for line in dataset:
                json_dict = {}
                split_line = line.split("\t")
                json_dict["label"] = split_line[0].strip()
                json_dict["body"] = split_line[1].strip()
                json_dict["lang"] = filename.split(".")[0]
                json_dict["fold"] = filename.split(".")[1]
                outfile.write(json.dumps(json_dict)+"\n")
