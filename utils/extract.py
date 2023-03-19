from pathlib import Path
from wfdb import processing, rdrecord, plot_wfdb

from torch import Tensor


class ExtractData:
    def __init__(self) -> None:
        pass

    def read_record(self, file_dir: str, just_signal: bool) -> Tensor:
        record = rdrecord(file_dir)
        if just_signal:
            record = Tensor(record.p_signal)

        return record

    def read_records_folder(self, folder_dir: str, just_signal: bool) -> list:
        folder_dir = Path(folder_dir)
        files_list = folder_dir.glob("*[!.html]")
        unique_filenames = set(map(lambda x: x.stem, files_list))

        records = []
        for filename in unique_filenames:
            file_dir = folder_dir.joinpath(filename)
            record = self.read_record(file_dir, just_signal)
            records.append(record)

        return records


ed = ExtractData()

FOLDER_DIR = "data/physionet.org/files/ephnogram/1.0.0/WFDB"
ed.read_records_folder(FOLDER_DIR, just_signal=True)


# qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)
# print('\nGQRS detect: ', qrs_inds)
# plot_wfdb(record=record)
