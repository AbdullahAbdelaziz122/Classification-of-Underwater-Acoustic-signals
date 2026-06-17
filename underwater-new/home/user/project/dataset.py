from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config import AudioConfig, ProjectConfig, TrainingConfig
from utils import AudioPreprocessor

AudioSample = Tuple[Path, int]

WHALE_SPECIES = [
    'Humpback Whale', 'Sperm Whale', 'Bowhead Whale',
    'Fin_ Finback Whale', 'Minke Whale', 'Killer Whale',
    'Northern Right Whale', 'Southern Right Whale',
    'Long-Finned Pilot Whale', 'Short-Finned _Pacific_ Pilot Whale',
    'Melon Headed Whale', 'False Killer Whale'
]

DOLPHIN_SPECIES = [
    'Spinner Dolphin', 'Fraser_s Dolphin', 'Striped Dolphin',
    'Pantropical Spotted Dolphin', 'Atlantic Spotted Dolphin',
    'Common Dolphin', 'Bottlenose Dolphin', 'Clymene Dolphin',
    'Rough-Toothed Dolphin', 'Grampus_ Risso_s Dolphin',
    'White-beaked Dolphin', 'White-sided Dolphin'
]

SEAL_SPECIES = ['Ross Seal', 'Harp Seal', 'Bearded Seal', 'Leopard Seal', 'Weddell Seal']
VESSEL_TYPES = ['Small vessels', 'Medium vessels', 'Large vessels', 'Passenger ferries']


def assign_family(class_name: str) -> str:
    if class_name in WHALE_SPECIES: return 'Whale'
    if class_name in DOLPHIN_SPECIES: return 'Dolphin'
    if class_name in SEAL_SPECIES: return 'Seal'
    if class_name in VESSEL_TYPES: return 'Vessel'

    mapping = {
        'Beluga White Whale': 'Beluga',
        'Narwhal': 'Narwhal',
        'Walrus': 'Walrus',
        'Background Noise': 'Background',
        'Torpedo': 'Torpedo',
        'Submarine': 'Submarine',
    }
    return mapping.get(class_name, 'Unknown')


class UnderwaterAudioDataset(Dataset[Tuple[Tensor, int]]):
    """Dataset for underwater acoustic classification from WAV files."""

    def __init__(self, samples: Sequence[AudioSample], audio_config: AudioConfig) -> None:
        self.samples: List[AudioSample] = list(samples)
        self.preprocessor = AudioPreprocessor(audio_config)

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """Load and preprocess a single audio file."""
        audio_path, label = self.samples[index]
        features = self.preprocessor(audio_path)
        return features, label


def collect_samples(project_config: ProjectConfig) -> List[AudioSample]:
    """Parse all 4 dataset directories and build file-label pairs."""

    class_to_idx = {name: idx for idx, name in enumerate(project_config.class_names)}
    raw_samples: List[Tuple[Path, str]] = []

    # 1. DS3500 (Vessels and Background)
    if project_config.ds3500_dir.exists():
        label_map = {
            "0": 'Small vessels', "1": 'Medium vessels',
            "2": 'Passenger ferries', "3": 'Large vessels', "4": 'Background Noise'
        }
        for wav_file in project_config.ds3500_dir.glob("*.wav"):
            first_char = wav_file.name[0]
            if first_char in label_map:
                raw_samples.append((wav_file, label_map[first_char]))

    # 2. NOAA (Background / Soundscape)
    if project_config.noaa_dir.exists():
        keep = {'soundscape', 'rain', 'sonar'}
        for wav_file in project_config.noaa_dir.rglob("*.wav"):
            if "data" in wav_file.parts:
                continue
            parts = wav_file.name.split('_')
            if len(parts) > 3 and parts[3].lower() in keep:
                raw_samples.append((wav_file, 'Background Noise'))

    # 3. Watkins (Marine Mammals)
    if project_config.watkins_dir.exists():
        for wav_file in project_config.watkins_dir.rglob("*.wav"):
            species_name = wav_file.parent.name
            raw_samples.append((wav_file, species_name))

    # 4. Threats (Torpedo / Submarine)
    if project_config.threat_dir.exists():
        for wav_file in project_config.threat_dir.rglob("*.wav"):
            basename = wav_file.stem
            parts = basename.split('_')
            if len(parts) >= 2:
                class_label = parts[-1].capitalize()
                if class_label in ['Submarine', 'Torpedo']:
                    raw_samples.append((wav_file, class_label))

    # Convert specific raw classes to target Families and map to integer IDs
    final_samples: List[AudioSample] = []
    for file_path, raw_class in raw_samples:
        family = assign_family(raw_class)
        if family != 'Unknown' and family in class_to_idx:
            final_samples.append((file_path, class_to_idx[family]))

    if not final_samples:
        raise ValueError("No matching WAV files found across the provided datasets.")

    return final_samples


def stratified_split(
        samples: Sequence[AudioSample],
        validation_ratio: float,
        seed: int,
) -> Tuple[List[AudioSample], List[AudioSample]]:
    """Create an 80/20 stratified split without external dependencies."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    per_class: Dict[int, List[AudioSample]] = defaultdict(list)
    for sample in samples:
        per_class[sample[1]].append(sample)

    generator_state = __import__("random").Random(seed)
    train_samples: List[AudioSample] = []
    val_samples: List[AudioSample] = []

    for class_id, class_samples in per_class.items():
        shuffled = list(class_samples)
        generator_state.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * validation_ratio)))
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1
        val_samples.extend(shuffled[:val_count])
        train_samples.extend(shuffled[val_count:])

        if not train_samples or not val_samples:
            continue

    if not train_samples or not val_samples:
        raise ValueError("Unable to create non-empty stratified train/validation splits.")

    return train_samples, val_samples


def build_dataloaders(
        project_config: ProjectConfig,
        audio_config: AudioConfig,
        training_config: TrainingConfig,
        batch_size: int,
) -> Tuple[DataLoader[Tuple[Tensor, int]], DataLoader[Tuple[Tensor, int]], Dict[str, int]]:
    """Build train and validation dataloaders."""
    samples = collect_samples(project_config)
    train_samples, val_samples = stratified_split(
        samples=samples,
        validation_ratio=training_config.validation_split,
        seed=training_config.seed,
    )

    train_dataset = UnderwaterAudioDataset(train_samples, audio_config)
    val_dataset = UnderwaterAudioDataset(val_samples, audio_config)
    class_to_idx = {class_name: index for index, class_name in enumerate(project_config.class_names)}

    effective_workers = max(0, training_config.num_workers)
    persistent_workers = effective_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, class_to_idx