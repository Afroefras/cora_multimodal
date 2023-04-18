from pathlib import Path
from wfdb import rdrecord
from pandas import read_csv
from torch import Tensor, stack, cat


class ExtractData:
    def __init__(self) -> None:
        pass

    def filter_records(
        self,
        notes_file_dir: str,
        record_id_col: str,
        ecg_col: str,
        ecg_cotains: str,
        pcg_col: str,
        pcg_cotains: str,
        verbose: bool = False,
    ) -> None:
        df = read_csv(notes_file_dir)
        self.filtered = df[
            (df[ecg_col].str.contains(ecg_cotains))
            & (df[pcg_col].str.contains(pcg_cotains))
        ].copy()

        self.filtered = set(self.filtered[record_id_col])

        if verbose:
            print(f"\n{len(self.filtered)}/{len(df)} records filtered from:")
            print(f"\t'{notes_file_dir}'")
            print(f"where {ecg_col} column contains '{ecg_cotains}'")
            print(f"and {pcg_col} column contains '{pcg_cotains}'\n")

    def read_record(self, file_dir: Path) -> list:
        record_name = file_dir.stem
        record = rdrecord(file_dir)
        record = Tensor(record.p_signal)
        record = record.unsqueeze(0)
        return record, record_name
    
    def read_records_dir(
        self, import_dir: str, verbose: bool, test: bool = False
    ) -> None:
        import_dir = Path(import_dir)
        files_list = import_dir.glob("*[!.html]")
        unique_filenames = set(map(lambda x: x.stem, files_list))

        valid_filenames = unique_filenames.intersection(self.filtered)

        self.raw_records = []
        self.raw_records_names = []

        for i, filename in enumerate(valid_filenames):
            file_dir = import_dir.joinpath(filename)
            record, record_name = self.read_record(file_dir)

            self.raw_records.append(record)
            self.raw_records_names.append(record_name)

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
        self.records_names = []

        for i, record in enumerate(self.raw_records):
            shape_before = record.shape

            if record.shape[2] > min_shape2:
                record = record[:, :, :min_shape2].clone()

            if record.shape[1] > min_shape1:
                to_stack = record.split(min_shape1, dim=1)
                record = stack(to_stack).squeeze()

            self.records = cat((self.records, record))

            to_append = [self.raw_records_names[i]] * record.shape[0]
            self.records_names.extend(to_append)

            if verbose:
                print(f"#{i+1}: {shape_before} --> {record.shape}")
