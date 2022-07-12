from collections import defaultdict
from glob import glob
import re
from subprocess import run, PIPE

results = defaultdict(list)

for filename in glob("exps/*/*/*7.log"):
    result = run(['tail', '-n1', filename], stdout=PIPE).stdout.decode()
    try:
        expname = re.findall(r'exps/(.+?)/.+', filename)[0]
        acc = float(re.findall(r'best: ([.\d]+)', result)[0])
    except IndexError:
        continue
    results[expname].append(acc)

for exp, vals in results.items():
    if len(vals) != 4:
        continue
    print(','.join(['{}']*6).format(exp, *vals, sum(vals) / 4))
