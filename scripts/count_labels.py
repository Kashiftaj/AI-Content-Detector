import csv
import sys
from pathlib import Path

path = Path(r"c:\Deep Learning Project\data\processed\test.csv")
if not path.exists():
    print(f"ERROR: file not found: {path}")
    sys.exit(2)

c0 = 0
c1 = 0
other = 0
total = 0

with path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        print("ERROR: CSV is empty")
        sys.exit(3)
    # find 'label' column if present
    try:
        li = header.index('label')
    except ValueError:
        li = len(header) - 1

    for row in reader:
        total += 1
        if len(row) <= li:
            other += 1
            continue
        v = row[li].strip()
        if v in ('0', '0.0'):
            c0 += 1
        elif v in ('1', '1.0'):
            c1 += 1
        else:
            # try to coerce numeric
            try:
                fv = float(v)
                if int(fv) == 0:
                    c0 += 1
                elif int(fv) == 1:
                    c1 += 1
                else:
                    other += 1
            except Exception:
                other += 1

print(f"label_index={li}")
print(f"count_0={c0}")
print(f"count_1={c1}")
print(f"other={other}")
print(f"total_rows={total}")
