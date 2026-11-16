#!/bin/bash
# Local: refresh LOCK, forward STOP, pull outputs/checkpoints/executed notebooks from the VM to Drive / repo.
S=/private/tmp/claude-501/-Users-isaac--claude-spares-spare-nTajOk/12cfbb78-b17a-4999-9758-ec0b366c49df/scratchpad
P="/Users/isaac/Library/CloudStorage/GoogleDrive-test-ezpenta/My Drive/ez-penta/post_train_3/bilingual"
NBL=/Users/isaac/App_Files/Code/ez-penta/post_train_3/bilingual/notebooks
date -u +%Y-%m-%dT%H:%M:%SZ > "$P/LOCK"
if [ -f "$P/STOP" ]; then $S/vm.sh 'touch /content/pt3bi/STOP' >/dev/null; echo "STOP forwarded"; fi
F=$(mktemp $S/run.XXXXXX)
cat > $F <<'PY'
import os, glob
for d in sorted(glob.glob("/content/pt3bi/out/adapters/*")) + sorted(glob.glob("/content/pt3bi/out/eval/*")):
    if os.path.isdir(d): print("OUTDIR", d, os.path.exists(d + "/DONE"))
for d in sorted(glob.glob("/content/pt3bi/work/*")): print("WORK", d)
for f in sorted(glob.glob("/content/pt3bi/logs/*.ok") + glob.glob("/content/pt3bi/logs/*.fail")): print("FIN", f)
if os.path.exists("/content/pt3bi/out/eval/reuse_checks.json"): print("RC")
PY
OUT=$(timeout 120 ~/.local/bin/colab exec -s pt3-bi -f $F --timeout 90)
echo "$OUT" | grep '^OUTDIR' | while read _ d done; do
  rel=${d#/content/pt3bi/out/}
  [ "$done" = "True" ] && [ -f "$P/$rel/DONE" ] && continue
  $S/pull.sh "$d" "$P/$rel"
done
echo "$OUT" | grep '^WORK' | while read _ d; do $S/pull.sh "$d" "$P/work/${d##*/}" >/dev/null; done
echo "$OUT" | grep -q '^RC' && timeout 300 ~/.local/bin/colab download -s pt3-bi /content/pt3bi/out/eval/reuse_checks.json "$P/eval/reuse_checks.json" >/dev/null 2>&1
echo "$OUT" | grep '^FIN' | while read _ f; do
  b=$(basename $f); n=${b%.*}
  [ -f "$NBL/$n.ipynb" ] && continue
  timeout 300 ~/.local/bin/colab download -s pt3-bi /content/pt3bi/nb/$n.ipynb "$NBL/$n.ipynb" >/dev/null 2>&1 && echo "nb $n (${b##*.})"
done
$S/pull.sh /content/pt3bi/logs "$P/logs" >/dev/null
tail -n 4 "$P/logs/queue.log" 2>/dev/null
