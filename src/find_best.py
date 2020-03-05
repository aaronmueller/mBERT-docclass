import os
import sys

#LANG = sys.argv[1]
LANG_MODELS = "model/tobacco-new/tuneall/one-model-sepclass"

max_f1 = 0.0
best_model = "bs1-lr2e-5-ep16.0/model.pth"

for subdir, dirs, files in os.walk(LANG_MODELS):
    if subdir.startswith(LANG_MODELS+"/bs") and not subdir.endswith("eval"): #and not subdir.endswith(LANG):
        with open(os.path.join(subdir, "eval_results.txt"), 'r') as results:
            f1 = 0.0
            for line in results:
                tokens = line.strip().split()
                if tokens[0] == "dev-f1":
                    f1 = float(tokens[2])
            if f1 > max_f1:
                best_model = subdir.split("/")[-1]+"/model.pth"
                max_f1 = f1

BEST_DIR = LANG_MODELS + "/best"
if not os.path.exists(BEST_DIR):
    os.makedirs(BEST_DIR)
os.symlink("../" + best_model, BEST_DIR+"/model.pth")

