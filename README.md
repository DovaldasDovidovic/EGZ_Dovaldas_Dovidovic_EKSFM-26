# Žmogaus veiklos atpažinimas (HAR) — Intelektualiosios sistemos

Magistrantūros dalyko **Intelektualiosios sistemos** galutinio egzamino projektas.

## Tikslas

Parengti atkuriamą žmogaus veiklos atpažinimo (Human Activity Recognition, HAR) eksperimentą su **fiksuotu TRAIN/TEST skaidymu pagal žmones (subject ID)**. Skaidymas fiksuojamas 1 etape ir vėliau nekuriamas iš naujo.

Eksperimento tikslas — įvertinti modelių gebėjimą atpažinti **mokymo metu nematyto žmogaus** veiklą.

Šiuo metu baigti etapai:

1. Duomenų paruošimas ir fiksuotas subject-disjoint TRAIN/TEST skaidymas.
2. Baseline modeliai: Majority Class ir k-NN.
3. Intelektualieji metodai: linijinis daugiaklasis SVM (`LinearSVC`) ir MLP (be SOM, abliacijos ir robustness).



## Duomenų rinkinys

Naudojamas oficialus **UCI Human Activity Recognition Using Smartphones** duomenų rinkinys.

- Oficiali nuoroda: [https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- Šaltinis: UCI Machine Learning Repository (ne Kaggle ir ne GitHub veidrodžiai).



## Kur lokaliai turi būti datasetas

Datasetas jau yra projekto aplanke (UCI archyvo išpakavimo kelias):

```text
data/human+activity+recognition+using+smartphones (2)/UCI HAR Dataset/UCI HAR Dataset/
```

Ten turi būti `features.txt`, `activity_labels.txt`, `train/` ir `test/` failai. Notebook šį kelią randa automatiškai. Duomenų failų į GitHub kelti nereikia (žr. `.gitignore`).

## Atkuriamumas

- Fiksuotas atsitiktinumo sėklos parametras: `RANDOM_STATE = 42`.
- Galutinis train/test skaidymas atliekamas **pagal subject ID**, ne pagal atsitiktinius įrašus (`sklearn.model_selection.GroupShuffleSplit`, apie 70 % / 30 % žmonių).
- Tas pats žmogus negali būti ir TRAIN, ir TEST dalyje.
- k-NN, SVM ir MLP hiperparametrai parenkami tik TRAIN dalyje su `GroupKFold` pagal subject ID. `StandardScaler` visada yra `Pipeline` viduje.



## Projekto struktūra

```text
data/          # lokalus UCI HAR datasetas (gitignore)
notebooks/     # duomenų paruošimo, baseline ir SVM/MLP notebook'ai
results/       # skaidymas ir etapų rezultatų lentelės
src/           # pagalbinis kodas (vėlesniems etapams)
```



## Paleidimas

```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/01_data_preparation.ipynb
jupyter notebook notebooks/02_baselines.ipynb
jupyter notebook notebooks/03_intelligent_models.ipynb
```

Kiekvieną notebook paleiskite nuo pradžios iki pabaigos (Run All). 2 etapui reikia `results/subject_disjoint_split.npz`, 3 etapui — dar ir `results/baseline_results.csv`.