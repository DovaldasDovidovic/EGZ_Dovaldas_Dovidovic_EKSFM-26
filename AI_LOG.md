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

## Atmesti AI pasiūlymai

| Data | Etapas | Pasiūlymas | Priežastis |
|------|--------|------------|------------|
| — | — | — | Kol kas nėra. |

## Pastebėtos AI klaidos

| Data | Etapas | Kas buvo negerai | Kaip pataisyta |
|------|--------|-------------------|----------------|
| 2026-09-20 | 1. Duomenų paruošimas | Promptas rėmėsi keliu `data/UCI HAR Dataset/`, bet lokaliai rinkinys yra `data/human+activity+recognition+using+smartphones (2)/UCI HAR Dataset/UCI HAR Dataset/` | Notebook ieško `features.txt` po `data/` ir naudoja rastą realų kelią. Originalūs dataset failai neperkelti ir neištrinti. |
