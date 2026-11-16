#!/bin/bash
# VM-side sequential runner. Each line of /content/pt3bi/queue.txt: NB OUT [papermill -p args]
# Lines starting with PAR run as a parallel group: "PAR NB OUT args ;; NB OUT args ;; ..."
cd /content/pt3bi/nb
while pgrep -f "B1_train_out_ja" >/dev/null; do sleep 15; done
while read -r line; do
  [ -z "$line" ] && continue
  [ -f /content/pt3bi/STOP ] && { echo "STOP $(date -u)" >> /content/pt3bi/logs/queue.log; break; }
  if [[ "$line" == PAR* ]]; then
    IFS=';;' read -ra parts <<< "${line#PAR }"
    pids=()
    for part in "${parts[@]}"; do
      part=$(echo "$part" | xargs); [ -z "$part" ] && continue
      set -- $part; NB=$1; OUT=$2; shift 2
      [ -f /content/pt3bi/logs/$OUT.ok ] && continue
      echo "START $OUT $(date -u)" >> /content/pt3bi/logs/queue.log
      (papermill --log-output $NB.ipynb $OUT.ipynb "$@" > /content/pt3bi/logs/$OUT.log 2>&1 && touch /content/pt3bi/logs/$OUT.ok || touch /content/pt3bi/logs/$OUT.fail) &
      pids+=($!); sleep 20
    done
    wait "${pids[@]}"
  else
    set -- $line; NB=$1; OUT=$2; shift 2
    [ -f /content/pt3bi/logs/$OUT.ok ] && continue
    echo "START $OUT $(date -u)" >> /content/pt3bi/logs/queue.log
    papermill --log-output $NB.ipynb $OUT.ipynb "$@" > /content/pt3bi/logs/$OUT.log 2>&1 && touch /content/pt3bi/logs/$OUT.ok || touch /content/pt3bi/logs/$OUT.fail
  fi
  echo "END $OUT $(date -u)" >> /content/pt3bi/logs/queue.log
done < /content/pt3bi/queue.txt
echo "QUEUE DONE $(date -u)" >> /content/pt3bi/logs/queue.log
