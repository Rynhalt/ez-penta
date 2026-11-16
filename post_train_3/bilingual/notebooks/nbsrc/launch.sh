#!/bin/bash
# usage: launch.sh NB OUTNAME "-p K V -p K V ..."   (runs on VM via colab exec)
S=/private/tmp/claude-501/-Users-isaac--claude-spares-spare-nTajOk/12cfbb78-b17a-4999-9758-ec0b366c49df/scratchpad
F=$(mktemp $S/run.XXXXXX)
cat > $F <<PY
import subprocess, shlex
NB, OUT, PARAMS = "$1", "$2", shlex.split("""$3""")
running = subprocess.run(["pgrep", "-af", "papermill"], capture_output=True, text=True).stdout
if f"/content/pt3bi/nb/{OUT}.ipynb" in running:
    print("ALREADY RUNNING", OUT)
else:
    cmd = ["nohup", "papermill", "--log-output", f"/content/pt3bi/nb/{NB}.ipynb", f"/content/pt3bi/nb/{OUT}.ipynb", *PARAMS]
    log = open(f"/content/pt3bi/logs/{OUT}.log", "w")
    subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd="/content/pt3bi/nb")
    print("LAUNCHED", OUT)
PY
timeout 120 ~/.local/bin/colab exec -s pt3-bi -f $F --timeout 90
