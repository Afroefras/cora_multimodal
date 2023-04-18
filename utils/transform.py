from numpy import repeat
from pathlib import Path
from torch import Tensor, stack, save as torch_save


def split_records(
    records: Tensor, records_names: list, seconds: int, verbose: bool
) -> Tensor:
    new_duration = 8000 // seconds
    seconds *= 8000

    to_stack = records.split(seconds, dim=1)
    to_reshape = stack(to_stack, dim=1)

    resized = to_reshape.view(-1, new_duration, 2)

    times_to_repeat = records.shape[1] // new_duration
    names_repeated = repeat(records_names, times_to_repeat)
    names_repeated = dict(enumerate(names_repeated))

    if verbose:
        print(f"Before: {records.shape} --> After: {resized.shape}")

    return resized, names_repeated


def save_records(
    records: Tensor, export_dir: str, export_name: str, verbose: bool
) -> None:
    export_dir = Path(export_dir)
    to_save = export_dir.joinpath(f"{export_name}.pt")

    torch_save(records, to_save)
    if verbose:
        print(f"\nTensor '{export_name}.pt' exported at:\n{export_dir}\n")
