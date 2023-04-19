from torch import Tensor
from pandas import read_csv
from numpy.random import shuffle as np_shuffle
from torch import load as torch_load
from pickle import load as load_pickle
from torch.utils.data import Dataset, DataLoader


def get_sounds(records_dir: str, names_dir: str) -> tuple:
    records = torch_load(records_dir)
    sounds = records[:, :, 1:].clone()

    with open(names_dir, "rb") as f:
        names = load_pickle(f)

    return sounds, names


def get_notes(csv_dir: str, id_col: str, notes_col: str, names: dict) -> list:
    df = read_csv(csv_dir)
    df.set_index(id_col, inplace=True)

    sound_notes = []
    for x in names.values():
        note = df.loc[x, notes_col]
        sound_notes.append(note)

    return sound_notes


def sound_note_contains(sound_notes: list, contains: str) -> Tensor:
    contains = contains.lower()

    note_contains = []
    for note in sound_notes:
        note_lower = note.lower()
        to_match = contains in note_lower
        note_contains.append(to_match * 1)

    note_contains = Tensor(note_contains)
    return note_contains


def train_test_split(
    data: Tensor, labels: Tensor, to_train: float = 0.8, shuffle: bool = True
) -> tuple:
    data_len = len(labels)
    idx = list(range(data_len))

    if shuffle:
        np_shuffle(idx)

    n_to_train = int(to_train * data_len)
    train_idx = idx[:n_to_train]
    test_idx = idx[n_to_train:]

    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    return X_train, y_train, X_test, y_test


class NoiseDataset(Dataset):
    def __init__(self, sounds_tensor: Tensor, labels_list: list) -> None:
        self.sounds = sounds_tensor
        self.labels = labels_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sound = self.sounds[idx]
        label = self.labels[idx]
        return sound, label


def get_data_loader(X: Tensor, y: Tensor, **kwargs) -> DataLoader:
    dataset = NoiseDataset(X, y)
    data_loader = DataLoader(dataset, **kwargs)
    return data_loader
