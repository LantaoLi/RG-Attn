# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import subprocess
import time

def run_script(param):
    try:
        print(f"Starting train.py as set in folder {param}")
        #python opencood/tools/train.py -y None --model_dir opencood/logs/HEAL_m1_based/stage3/m2_80partial_trained_2 -p 0.8
        subprocess.run(["python", "opencood/tools/train.py", "-y", "None", "--model_dir", param, "-p", "0.8"])
        print(f"Finished training in folder {param}")
    except:
        print(f"Error with traning as set in folder {param}")


def main():
    params = ["opencood/logs/HEAL_m1_based/stage3/m2_80partial_trained_3", "opencood/logs/HEAL_m1_based/stage3/m2_80partial_trained_4", "opencood/logs/HEAL_m1_based/stage3/m2_80partial_trained_5"]
    for param in params:
        run_script(param)
        print("Waiting for 1 minute before the next round!")
        time.sleep(60)
    print("All training finished!")

if __name__ == '__main__':
    main()
