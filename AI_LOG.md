# AI naudojimo auditas

Šis failas fiksuoja, kaip projekte naudotas dirbtinis intelektas: kokie pasiūlymai priimti, kokie atmesti ir kokios klaidos pastebėtos. Įrašai daromi tik tada, kai jie realiai įvyksta.

## Meta

- Dalykas: Intelektualiosios sistemos
- Projektas: Human Activity Recognition (UCI HAR)
- Studentas: Dovaldas Dovidovič, EKSFM-26

## Registravimo taisyklės

1. Neregistruoti išgalvotų klaidų ar atmestų pasiūlymų.
2. Kiekvienam įrašui nurodyti datą, etapą ir trumpą faktą.
3. Jei AI pasiūlymas priimtas be pakeitimų — taip ir pažymėti.
4. Jei studentas pataisė ar atmetė — aprašyti priežastį.

## Priimti AI pasiūlymai

| Data | Etapas | Pasiūlymas | Pastaba |
|------|--------|------------|---------|
| 2026-09-20 | 1. Duomenų paruošimas | Projekto struktūra, `01_data_preparation.ipynb`, subject-disjoint skaidymas su `GroupShuffleSplit` (`RANDOM_STATE = 42`) | Priimta. Notebook paleistas lokaliai; skaičiai gauti iš realių UCI failų. |
| 2026-09-20 | 2. Baseline modeliai | `02_baselines.ipynb`: Majority Class ir k-NN `Pipeline` (`StandardScaler` + `KNeighborsClassifier`), `GroupKFold` tik TRAIN, TEST tik galutiniam įvertinimui | Priimta. Naudotas jau esamas `subject_disjoint_split.npz`; naujas split nekurtas. |
| 2026-09-20 | 3. SVM ir MLP | `03_intelligent_models.ipynb`: `LinearSVC` ir `MLPClassifier` su `Pipeline`+`GroupKFold`, SVM `decision_function`+`argmax` sutikrinta su `predict()` | Priimta. Tas pats split; baseline CSV tik užkrautas, neperrašytas. |
| 2026-09-21 | 4. Požymių abliacija | `04_ablation.ipynb`: Acc/Gyro/All grupės iš `feature_names`; k-NN, SVM, MLP su ankstesniais parametrais, be naujo GridSearch | Priimta. All features sutapo su 2/3 etapų TEST; `matplotlib` įrašytas į `requirements.txt`. |
| 2026-09-21 | 5. Triukšmo robustness | `05_robustness_noise.ipynb`: Gauso triukšmas tik standartizuotame TEST, `sigma` 0.0–0.5, tas pats noisy TEST visiems modeliams | Priimta. `sigma=0` sutapo su 2/3 etapų TEST; hiperparametrai neperrinkti. |
| 2026-09-21 | 6. Klaidų analizė | `06_error_analysis.ipynb`: confusion matrices, per-class/per-subject metrikos, realūs klaidingi TEST pavyzdžiai | Priimta. TEST metrikos sutapo su 2/3 etapais; 1–5 notebook’ai nekeisti. |
| 2026-09-21 | 7. Galutinė suvestinė | `07_final_analysis.ipynb` ir `results/final_analysis_summary.md`: 1–6 etapų rezultatų surinkimas, ribos ir rizikos | Priimta. Naujų modelių ir eksperimentų nebuvo; Majority klaidos perskaičiuotos iš to paties split. |
| 2026-09-21 | 8. Atkuriamumas | `main.py`: viena komanda atkuria split ir clean TEST palyginimą; palyginimas su `final_model_comparison.csv` | Priimta. Be GridSearch; dataset ieškomas po `data/`; `.gitignore` leidžia trackinti svarbiausius CSV. |

## Atmesti AI pasiūlymai

| Data | Etapas | Pasiūlymas | Priežastis |
|------|--------|------------|------------|
| — | — | — | Kol kas nėra. |

## Pastebėtos AI klaidos

| Data | Etapas | Kas buvo negerai | Kaip pataisyta |
|------|--------|-------------------|----------------|
| 2026-09-20 | 1. Duomenų paruošimas | Promptas rėmėsi keliu `data/UCI HAR Dataset/`, bet lokaliai rinkinys yra `data/human+activity+recognition+using+smartphones (2)/UCI HAR Dataset/UCI HAR Dataset/` | Notebook ieško `features.txt` po `data/` ir naudoja rastą realų kelią. Originalūs dataset failai neperkelti ir neištrinti. |
