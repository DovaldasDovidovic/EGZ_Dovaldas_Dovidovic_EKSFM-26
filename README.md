# Žmogaus veiklos atpažinimas (HAR) — Intelektualiosios sistemos

Magistrantūros dalyko **Intelektualiosios sistemos** galutinio egzamino projektas.

## Tikslas

Parengti atkuriamą žmogaus veiklos atpažinimo (Human Activity Recognition, HAR) eksperimentą su **fiksuotu TRAIN/TEST skaidymu pagal žmones (subject ID)**. Skaidymas fiksuojamas 1 etape ir vėliau nekuriamas iš naujo.

Eksperimento tikslas — įvertinti modelių gebėjimą atpažinti **mokymo metu nematyto žmogaus** veiklą.

Šiuo metu baigti etapai:

1. Duomenų paruošimas ir fiksuotas subject-disjoint TRAIN/TEST skaidymas.
2. Baseline modeliai: Majority Class ir k-NN.
3. Intelektualieji metodai: linijinis daugiaklasis SVM (`LinearSVC`) ir MLP.
4. Požymių abliacija: accelerometer-only, gyroscope-only ir visi 561 požymiai (k-NN, SVM, MLP).
5. Robustness: Gauso triukšmas standartizuotame TEST (`sigma ∈ {0.0, 0.1, 0.2, 0.3, 0.5}`).
6. Klaidų analizė švariame TEST: confusion matrix, per-class metrikos, klaidų pavyzdžiai, per-subject Macro-F1.
7. Galutinė rezultatų analizė: modelių palyginimas, abliacijos ir robustness rezultatų suvestinė, klaidų tendencijos, generalizacijos ribos, rizikos ir praktinis tinkamumas.
8. Atkuriamumas: `main.py` viena komanda pakartoja pagrindinį švaraus TEST palyginimą.



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
- k-NN, SVM ir MLP hiperparametrai parenkami tik TRAIN dalyje su `GroupKFold` pagal subject ID. 2–4 etapuose `StandardScaler` yra `Pipeline` viduje; 5 etape scaler `fit` daromas aiškiai ant švaraus TRAIN, kad Gauso triukšmą būtų galima pridėti jau standartizuotam TEST.



## Projekto struktūra

```text
data/          # lokalus UCI HAR datasetas (gitignore)
notebooks/     # etapų notebook'ai (nuo duomenų iki galutinės suvestinės)
results/       # lentelės ir grafikai (svarbiausi CSV/SVG skirti Git)
src/           # pagalbinis kodas (vėlesniems etapams)
main.py        # vienos komandos pagrindinis eksperimentas
```



## Paleidimas

Notebook'ai (detalūs etapai) ir pagrindinė vienos komandos eiga aprašyti žemiau.

## Reproducibility / How to run

Pagrindinis eksperimentas atkuriamas **viena komanda**, be rankinio kelių notebook'ų paleidimo. `main.py` **nevykdo** GridSearchCV: hiperparametrai jau parinkti 2–3 etapuose tik TRAIN (`GroupKFold`). Čia pakartojamas tik galutinis švaraus TEST palyginimas su fiksuotais parametrais (`RANDOM_STATE = 42`).

`python main.py` **nepriklauso** nuo `results/subject_disjoint_split.npz`, Jupyter kernel būsenos ar išsaugotų modelių. Vienintelė išorinė sąlyga — oficialus UCI HAR dataset išarchyvuotas į `data/`.

1. **Python.** Rekomenduojama Python 3.10+ (šiame kompiuteryje tikrinant naudota 3.14).
2. **Virtual environment (rekomenduojama):**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate
```

3. **Priklausomybės:**

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` nurodo version ranges. Šioje aplinkoje tikrinant naudota: Python 3.14.7, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2.

4. **Datasetas — tik oficialus šaltinis** (ne Kaggle / ne GitHub veidrodžiai):

- UCI Human Activity Recognition Using Smartphones
- DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K)
- https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Duomenų failų kopijų į Git teikti nereikia.

5. **Išarchyvuokite** rinkinį į `data/` (bet kuris nested gylis tinka, jei viduje yra `UCI HAR Dataset` su `features.txt`, `activity_labels.txt`, `train/` ir `test/`). Programa pati suranda tą katalogą po `data/`, net jei archyvas turi papildomą aplanką.

Pavyzdys (gali skirtis pagal archyvą):

```text
data/human+activity+recognition+using+smartphones (2)/UCI HAR Dataset/UCI HAR Dataset/
```

6. **Pagrindinė komanda:**

```bash
python main.py
```

7. **Ką daro `python main.py`:** randa lokalų UCI HAR; sujungia originalias train/test dalis; atkuria subject-disjoint split (`GroupShuffleSplit`, `test_size=0.30`, `random_state=42`); patikrina 10299 / 561 / 30 / 7144 / 3155 / 21 / 9 ir tuščią subject sankirtą; apmoko Majority, k-NN, SVM ir MLP su fiksuotais parametrais (`StandardScaler` Pipeline viduje, `fit` tik TRAIN); spausdina lentelę; įrašo `results/reproduced_main_results.csv`; palygina su `results/final_model_comparison.csv` (jei failas yra).

8. **Orientyriniai** TEST rezultatai (programa juos **perskaičiuoja**, nehardcodina): Majority ≈ 0.053188 / 0.166667 / 2556 klaidos; k-NN ≈ 0.844324 / 0.842074 / 491; SVM ≈ 0.897130 / 0.893666 / 347; MLP ≈ 0.896616 / 0.894957 / 340. Sėkmės žinutė: `REPRODUCIBILITY CHECK: PASSED`.

### Notebook'ai (papildomi etapai)

Detali abliacija, robustness ir klaidų analizė lieka notebook'uose:

```bash
jupyter notebook notebooks/01_data_preparation-duomenu_paruosimas.ipynb
jupyter notebook notebooks/02_baselines-baziniai_metodai.ipynb
jupyter notebook notebooks/03_intelligent_models-SVM_MLP.ipynb
jupyter notebook notebooks/04_ablation-pozymiu_abliacija.ipynb
jupyter notebook notebooks/05_robustness_noise-atsparumas_triuksmui.ipynb
jupyter notebook notebooks/06_error_analysis-klaidu_analize.ipynb
jupyter notebook notebooks/07_final_analysis-galutine_analize.ipynb
```

Kiekvieną notebook paleiskite nuo pradžios iki pabaigos (Run All). Jie nėra būtini `python main.py`.