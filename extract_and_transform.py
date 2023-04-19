from torch import load as torch_load
from utils.extract import ExtractData
from utils.transform import split_records, save_records
from pickle import HIGHEST_PROTOCOL, dump as save_pickle

VERBOSE = True

ed = ExtractData()

CSV_DIR = "data/physionet.org/files/ephnogram/1.0.0/ECGPCGSpreadsheet.csv"
ed.filter_records(
    notes_file_dir=CSV_DIR,
    record_id_col="Record Name",
    ecg_col="ECG Notes",
    # ecg_contains="Good",
    ecg_contains="",
    pcg_col="PCG Notes",
    # pcg_contains="Good",
    pcg_contains="",
    verbose=VERBOSE,
)

ed.read_records_dir(
    import_dir="data/physionet.org/files/ephnogram/1.0.0/WFDB",
    verbose=VERBOSE,
    # test=True,
)

ed.same_shape(verbose=VERBOSE)

save_records(
    records=ed.records,
    export_dir="data/tensors",
    export_name="raw_records_5sec",
    verbose=VERBOSE,
)

raw_records = torch_load("data/tensors/raw_records_5sec.pt")

resized, names_repeated = split_records(
    raw_records, ed.records_names, seconds=5, verbose=VERBOSE
)

save_records(
    records=resized,
    export_dir="data/tensors",
    export_name="resized_records_5sec",
    verbose=VERBOSE,
)

with open("data/tensors/records_names_5sec.xz", "wb") as f:
    save_pickle(names_repeated, f, protocol=HIGHEST_PROTOCOL)
