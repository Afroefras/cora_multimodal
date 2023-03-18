import wfdb

FILE_DIR = "data/physionet.org/files/ephnogram/1.0.0/WFDB/ECGPCG0004"

record = wfdb.rdrecord(FILE_DIR) 
wfdb.plot_wfdb(record=record) 