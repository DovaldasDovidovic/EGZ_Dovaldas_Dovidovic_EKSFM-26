# AI naudojimo auditas

Šis failas fiksuoja, kaip projekte naudotas dirbtinis intelektas: svarbiausios užklausos, priimti ir pakeisti pasiūlymai, aptiktos klaidos / nepatikrintos prielaidos ir kaip jos patikrintos. Įrašai daromi tik tada, kai jie realiai įvyksta.


## Svarbiausios AI užklausos


| Data | Etapas | Užklausos santrauka |
|------|--------|---------------------|
| 2026-09-20 | 1 | Subject-disjoint UCI HAR duomenų paruošimas: oficialus lokalus datasetas, originalių train/test sujungimas, `GroupShuffleSplit` (`RANDOM_STATE=42`). |
| 2026-09-20 | 2 | Majority Class ir k-NN baseline: fiksuotas split, `GroupKFold` tik TRAIN, TEST nenaudotas hiperparametrų paieškai. |
| 2026-09-20 | 3 | SVM (`LinearSVC`) ir MLP įgyvendinimas tomis pačiomis sąlygomis; SVM formulės `f_k(x)=w_k^T x - b_k` susiejimas su `decision_function` + `argmax`. |
| 2026-09-21 | 4 | Požymių abliacija: accelerometer / gyroscope / all, be naujo GridSearch. |
| 2026-09-21 | 5 | Kontroliuojamas Gauso triukšmas standartizuotame TEST (robustness). |
| 2026-09-21 | 6 | Švaraus TEST klaidų analizė: confusion matrix, per-class/per-subject metrikos, realūs klaidingi pavyzdžiai. |
| 2026-09-21 | 7 | Galutinė suvestinė iš jau esamų rezultatų (naujų eksperimentų neprašyta). |
| 2026-09-21 | 8 | Atkuriamas `main.py`: viena komanda pakartoja pagrindinį eksperimentą ir palygina su išsaugota lentele. |

Papildomai: po 8 etapo paprašyta pataisyti README 7 etapo eilutę ir `pandas` versijos ribą `requirements.txt`, kad atitiktų realiai patikrintą aplinką.

## Priimti AI pasiūlymai

Svarbiausi priimti sprendimai ir kaip jie patikrinti:

| Data | Etapas | Pasiūlymas | Kaip patikrinta |
|------|--------|------------|-----------------|
| 2026-09-20 | 1 | Subject-disjoint split: `GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)` | Notebook / vėliau `main.py`: 10299 įrašai, 561 požymis, 30 subject, TRAIN 7144 / 21, TEST 3155 / 9, sankirta tuščia. |
| 2026-09-20 | 2–3 | Hiperparametrai tik TRAIN su `GroupKFold`; `StandardScaler` Pipeline viduje; metrikos Macro-F1 ir Balanced Accuracy | 2 ir 3 etapų notebook’ai paleisti; TEST nenaudotas paieškai. |
| 2026-09-20 | 3 | SVM `decision_function` + `argmax` = `predict()` | Assert: 3155/3155 sutapimas. |
| 2026-09-21 | 4 | Acc / Gyro / All abliacija su fiksuotais parametrais | 9 eksperimentai; All features sutapo su 2/3 etapų TEST. |
| 2026-09-21 | 5 | Gauso triukšmas tik standartizuotame TEST | σ=0 sutapo su ankstesniu clean TEST; visiems modeliams ta pati noisy matrica. |
| 2026-09-21 | 6–7 | Klaidų analizė ir galutinė suvestinė be naujo tuning | Metrikos perskaičiuotos / nuskaitomos iš CSV; 7 etapas naujų modelių nekūrė. |
| 2026-09-21 | 8 | `python main.py` + palyginimas su `final_model_comparison.csv` | Realus paleidimas: `REPRODUCIBILITY CHECK: PASSED`. |

## Atmesti arba pakeisti AI pasiūlymai / prielaidos

Išgalvotų „atmestų metodų“ (pvz. kito split ar kito klasifikatoriaus) nebuvo. Realūs pakeitimai:

| Data | Etapas | Kas buvo pasiūlyta / įrašyta | Kas pakeista ir kodėl |
|------|--------|------------------------------|------------------------|
| 2026-09-20 | 1 | Prielaida, kad datasetas yra paprastame kelyje `data/UCI HAR Dataset/` | Pakeista automatine paieška pagal `features.txt`, nes realus archyvas nested. |
| 2026-09-21 | 4 | Pradinėje `requirements.txt` nebuvo `matplotlib`, nors abliacijos notebook’ui grafiko reikėjo | Priklausomybė įrašyta po to, kai importas realiai nepavyko. |
| 2026-09-21 | 8 | `requirements.txt`: `pandas>=2.0,<3` | Pakeista į `pandas>=2.0,<4`, nes sėkmingas `python main.py` vyko su pandas 3.0.6. |

## AI klaidos arba nepatikrintos prielaidos ir jų patikra

### A) Dataset kelio / katalogų struktūros prielaida (1 ir 8 etapai)

**Kas buvo negerai.** AI rėmėsi paprastesne UCI HAR struktūra (`data/UCI HAR Dataset/`). Realiai atsisiųstas rinkinys buvo keliuose nested kataloguose, pvz. `data/human+activity+recognition+using+smartphones (2)/UCI HAR Dataset/UCI HAR Dataset/`.

**Kaip patikrinta.** Peržiūrėta reali `data/` failų struktūra (`features.txt`, `train/`, `test/`).

**Sprendimas.** Notebook’ai ir `main.py` datasetą suranda po `data/` pagal `features.txt` ir patikrina reikalingus failus (`activity_labels.txt`, `train/`, `test/`). Originalūs dataset failai neperkelti ir neištrinti.

### B) pandas dependency neatitikimas (8 etapas)

**Kas buvo negerai.** `requirements.txt` nurodė `pandas>=2.0,<3`, bet projektas realiai sėkmingai veikė su **pandas 3.0.6**.

**Kaip patikrinta.** Palyginta reali Python aplinka (`python -c` versijos) su `requirements.txt`.

**Sprendimas.** Constraint pakeistas į `pandas>=2.0,<4`. Po pakeitimo realiai paleista `python main.py` ir gauta `REPRODUCIBILITY CHECK: PASSED`.

### C) Papildomas atvejis: matplotlib nebuvo `requirements.txt` (4 etapas)

**Kas buvo negerai.** Abliacijos notebook’as naudojo `matplotlib.pyplot`, bet priklausomybės dar nebuvo `requirements.txt`. Pirmas paleidimas baigėsi `ModuleNotFoundError: No module named 'matplotlib'`.

**Kaip patikrinta.** Realus notebook paleidimas (nbconvert) ir `import matplotlib`.

**Sprendimas.** Įdiegta matplotlib, įrašyta į `requirements.txt`; notebook paleistas iš naujo, rezultatai gauti.

## Registravimo taisyklės (toliau)

1. Neregistruoti išgalvotų klaidų ar atmestų pasiūlymų.
2. Netyčinis proceso nutraukimas (pvz. Ctrl+C) nelaikomas AI klaida.
