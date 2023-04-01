from utils.extract import ExtractData

ed = ExtractData()


CSV_DIR = "data/physionet.org/files/ephnogram/1.0.0/ECGPCGSpreadsheet.csv"
ed.filter_records(
    notes_file_dir=CSV_DIR,
    record_id_col="Record Name",
    ecg_col="ECG Notes",
    ecg_cotains="Good",
    pcg_col="PCG Notes",
    pcg_cotains="Good",
    verbose=True,
)

ed.extract_n_export(
    import_dir="data/physionet.org/files/ephnogram/1.0.0/WFDB",
    export_dir="data",
    verbose=True,
)
