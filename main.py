from torch import load as torch_load
from utils.extract import ExtractData
from utils.transform import split_records, save_records

VERBOSE = True

ed = ExtractData()


CSV_DIR = "data/physionet.org/files/ephnogram/1.0.0/ECGPCGSpreadsheet.csv"
ed.filter_records(
    notes_file_dir=CSV_DIR,
    record_id_col="Record Name",
    ecg_col="ECG Notes",
    ecg_cotains="Good",
    pcg_col="PCG Notes",
    pcg_cotains="Good",
    verbose=VERBOSE,
)

ed.read_records_dir(
    import_dir="data/physionet.org/files/ephnogram/1.0.0/WFDB",
    verbose=VERBOSE,
    # test=True
)

ed.same_shape(verbose=VERBOSE)

save_records(
    records=ed.records,
    export_dir="data/tensors",
    export_name="raw_records",
    verbose=VERBOSE,
)

raw_records = torch_load("data/tensors/raw_records.pt")
resized = split_records(raw_records, seconds=2, verbose=VERBOSE)
save_records(
    records=resized,
    export_dir="data/tensors",
    export_name="resized_records",
    verbose=VERBOSE,
)
