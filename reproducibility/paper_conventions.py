"""Paper-wide dataset naming conventions fixed by the 20-dataset Table 2."""

from __future__ import annotations

from typing import Dict, Tuple


PAPER_DATASET_FILES: Tuple[str, ...] = (
    "chess_nowin_227_variant1",
    "monks_0_12_variant1",
    "mushroom_p_365_variant1",
    "tic_tac_toe_negative_12_variant1",
    "zoo_variant1",
    "autos_variant1",
    "15_Hepatitis",
    "heart270_2_16_variant1",
    "horse_1_12_variant1",
    "sick_sick_72_variant1",
    "14_glass",
    "25_musk",
    "26_optdigits",
    "29_Pima",
    "35_SpamBase",
    "38_thyroid",
    "41_Waveform",
    "satellite",
    "wdbc_M_39_variant1",
    "wpbc_variant1",
)

PAPER_DATASET_ABBREVIATIONS: Dict[str, str] = dict(
    zip(
        PAPER_DATASET_FILES,
        (
            "Chess",
            "Monks",
            "Mush",
            "Tic",
            "Zoo",
            "Auto",
            "Hepa",
            "Heart",
            "Horse",
            "Sick",
            "Glass",
            "Musk",
            "Opt",
            "Pima",
            "Spam",
            "Thyr",
            "Wave",
            "Sat",
            "WDBC",
            "WPBC",
        ),
    )
)
PAPER_DATASET_COUNT = len(PAPER_DATASET_FILES)
PAPER_DATASET_ORDER = {
    abbreviation: index
    for index, abbreviation in enumerate(PAPER_DATASET_ABBREVIATIONS.values())
}


def paper_dataset_abbreviation(name: object) -> str:
    """Map a raw benchmark filename or an existing abbreviation to Table 2."""

    text = str(name).strip()
    if text in PAPER_DATASET_ABBREVIATIONS:
        return PAPER_DATASET_ABBREVIATIONS[text]
    if text in PAPER_DATASET_ORDER:
        return text
    raise ValueError(f"Dataset {text!r} is not defined by the paper's Table 2.")


__all__ = [
    "PAPER_DATASET_ABBREVIATIONS",
    "PAPER_DATASET_COUNT",
    "PAPER_DATASET_FILES",
    "PAPER_DATASET_ORDER",
    "paper_dataset_abbreviation",
]
