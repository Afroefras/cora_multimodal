from pathlib import Path
from wfdb import processing, rdrecord, plot_wfdb

from torch import Tensor, stack, cat


class ExtractData:
    def __init__(self) -> None:
        pass

    def read_record(self, file_dir: str) -> list:
        record = rdrecord(file_dir)
        record = Tensor(record.p_signal)
        record = record.unsqueeze(0)
        return record

    def read_records_dir(
        self, dir: str, verbose: bool = False, test: bool = False
    ) -> None:
        dir = Path(dir)
        files_list = dir.glob("*[!.html]")
        unique_filenames = set(map(lambda x: x.stem, files_list))

        self.raw_records = []
        for i, filename in enumerate(unique_filenames):
            file_dir = dir.joinpath(filename)
            record = self.read_record(file_dir)
            self.raw_records.append(record)

            if verbose:
                print(f"#{i+1}: {filename} imported")

            if test and i > 20:
                break

    def same_shape(self, verbose: bool) -> None:
        min_shape1 = min(map(lambda x: x.shape[1], self.raw_records))
        min_shape2 = min(map(lambda x: x.shape[2], self.raw_records))

        if verbose:
            print(f"\nMin shape: {min_shape1, min_shape2}")

        self.records = Tensor()
        for i, record in enumerate(self.raw_records):
            shape_before = record.shape

            if record.shape[2] > min_shape2:
                record = record[:, :, :min_shape2].clone()

            if record.shape[1] > min_shape1:
                to_stack = record.split(min_shape1, dim=1)
                record = stack(to_stack).squeeze()

            self.records = cat((self.records, record))

            if verbose:
                print(f"#{i+1}: {shape_before} --> {record.shape}")


ed = ExtractData()

DIR = "data/physionet.org/files/ephnogram/1.0.0/WFDB"
ed.read_records_dir(DIR, verbose=True, test=True)

ed.same_shape(verbose=True)

# qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)
# print('\nGQRS detect: ', qrs_inds)
# plot_wfdb(record=record)
