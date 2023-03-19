from wfdb import rdrecord, plot_wfdb

record = rdrecord("data/physionet.org/files/ephnogram/1.0.0/WFDB/ECGPCG0004")
plot_wfdb(record=record)
