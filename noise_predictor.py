from utils.noise_utils import (
    get_sounds,
    get_notes,
    sound_note_contains,
    train_test_split,
    get_data_loader,
)
from torch import device as torch_device, cuda

sounds, names = get_sounds(
    records_dir="data/tensors/resized_records_5sec.pt",
    names_dir="data/tensors/records_names_5sec.xz",
)

sound_notes = get_notes(
    csv_dir="data/physionet.org/files/ephnogram/1.0.0/ECGPCGSpreadsheet.csv",
    id_col="Record Name",
    notes_col="PCG Notes",
    names=names,
)
labels = sound_note_contains(sound_notes, contains="Good")

X_train, y_train, X_test, y_test = train_test_split(sounds, labels)
train = get_data_loader(X_train, y_train, batch_size=256, shuffle=True)
test = get_data_loader(X_test, y_test, batch_size=1024, shuffle=True)

device = torch_device("cuda" if cuda.is_available() else "cpu")
