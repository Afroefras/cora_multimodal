from pathlib import Path
from wfdb import processing, rdrecord, plot_wfdb

from torch import Tensor, stack, cat, save as torch_save


class ExtractData:
    def __init__(self) -> None:
        pass

    def read_record(self, file_dir: str) -> list:
        record = rdrecord(file_dir)
        record = Tensor(record.p_signal)
        record = record.unsqueeze(0)
        return record

    def read_records_dir(self, import_dir: str, verbose: bool, test: bool = False) -> None:
        import_dir = Path(import_dir)
        files_list = import_dir.glob("*[!.html]")
        unique_filenames = set(map(lambda x: x.stem, files_list))

        self.raw_records = []
        for i, filename in enumerate(unique_filenames):
            file_dir = import_dir.joinpath(filename)
            record = self.read_record(file_dir)
            self.raw_records.append(record)

            if verbose:
                print(f"#{i+1}: {filename} imported")

            if test and i > 7:
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

    def save_records(self, dir_to_save: str, verbose: bool) -> None:
        dir_to_save = Path(dir_to_save)
        dir_to_save = dir_to_save.joinpath("tensors")
        dir_to_save.mkdir(exist_ok=True)

        to_save = dir_to_save.joinpath("records.pt")

        torch_save(self.records, to_save)

        if verbose:
            print(f"\nTensor 'records.pt' is now saved at:\n{dir_to_save}")

    def extract_n_export(self, import_dir: str, export_dir: str, verbose: bool=False) -> None:
        self.read_records_dir(import_dir, verbose, test=True)
        self.same_shape(verbose)
        self.save_records(export_dir, verbose)



ed = ExtractData()

IMPORT_DIR = "data/physionet.org/files/ephnogram/1.0.0/WFDB"
ed.extract_n_export(
    import_dir=IMPORT_DIR,
    export_dir='data',
    verbose=True
)

# qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)
# print('\nGQRS detect: ', qrs_inds)
# plot_wfdb(record=record)
