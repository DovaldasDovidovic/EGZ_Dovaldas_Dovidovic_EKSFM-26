# Žmogaus veiklos atpažinimas (HAR) — Intelektualiosios sistemos

Magistrantūros dalyko **Intelektualiosios sistemos** galutinio egzamino projektas.

## Tikslas

Parengti atkuriamą žmogaus veiklos atpažinimo (Human Activity Recognition, HAR) eksperimentą su **fiksuotu TRAIN/TEST skaidymu pagal žmones (subject ID)**. Šiame etape atliekama tik duomenų paruošimo grandinė: originalių UCI dalių sujungimas ir subject-disjoint skaidymas. Klasifikavimo modeliai šiame etape nekuriami.

Eksperimento tikslas vėlesniuose etapuose — įvertinti modelių gebėjimą atpažinti **mokymo metu nematyto žmogaus** veiklą.

## Duomenų rinkinys

Naudojamas oficialus **UCI Human Activity Recognition Using Smartphones** duomenų rinkinys.

- Oficiali nuoroda: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
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

## Projekto struktūra

```text
data/          # lokalus UCI HAR datasetas (gitignore)
notebooks/     # duomenų paruošimo ir vėlesnių etapų notebook'ai
results/       # išsaugoti skaidymo rezultatai
src/           # pagalbinis kodas (vėlesniems etapams)
```

## Paleidimas

```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/01_data_preparation.ipynb
```

Arba visą notebook paleiskite nuo pradžios iki pabaigos (Run All).
