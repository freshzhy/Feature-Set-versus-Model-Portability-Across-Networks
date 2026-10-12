# data/raw

Download the four NetFlow v2 datasets from the University of Queensland NIDS
dataset repository (https://staff.itee.uq.edu.au/marius/NIDS_datasets/) and
place the CSV files here, unchanged:

```
data/raw/NF-UNSW-NB15-v2.csv
data/raw/NF-CSE-CIC-IDS2018-v2.csv
data/raw/NF-ToN-IoT-v2.csv
data/raw/NF-BoT-IoT-v2.csv
```

The files are not redistributed in this repository. `code/rq3_diagnostics.py`
also reads them to obtain the original class counts.
