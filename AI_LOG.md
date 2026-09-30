# AI naudojimo žurnalas


## Pagrindinės AI užklausos ir atlikti darbai

## Užduotis
Paruošti UCI HAR duomenis ir padaryti TRAIN/TEST skaidymą pagal asmenis (`GroupShuffleSplit`, `RANDOM_STATE=42`), kad TEST žmonės nebūtų matyti mokymo metu.

## Rezultatas
Sujungtos originalios UCI train/test dalys; gautas subject-disjoint skaidymas: TRAIN 21 asmuo, TEST 9 asmenys, sankirta tuščia.

## Patikra
Notebook’e ir vėliau `main.py` patikrinta: 10299 įrašai, 561 požymis, 30 asmenų; TRAIN 7144 / 21, TEST 3155 / 9, sankirta tuščia.


## Užduotis
Įgyvendinti Majority Class baseline, k-NN, SVM (`LinearSVC`) ir MLP tomis pačiomis subject-disjoint sąlygomis.

## Rezultatas
Visi keturi metodai mokyti tik TRAIN. Hiperparametrai parinkti tik TRAIN su `GroupKFold`. `StandardScaler` naudojamas Pipeline viduje. TEST naudotas tik galutiniam vertinimui.

## Patikra
Notebook’ai paleisti; TEST nenaudotas paieškai. Pagrindinės metrikos: Macro-F1 ir Balanced Accuracy.


## Užduotis
Patikrinti metrikas ir SVM sprendimo taisyklę `f_k(x) = w_k^T x - b_k` su `decision_function` ir `argmax`.

## Rezultatas
SVM `decision_function` + `argmax` sutapo su `predict()`.

## Patikra
Assert visiems 3155 TEST įrašams: 3155/3155 sutapimas.


## Užduotis
Atlikti požymių abliaciją: visos savybės, tik akselerometras, tik giroskopas; be naujo GridSearch.

## Rezultatas
Devyni eksperimentai su jau parinktais parametrais. Visų požymių variantas sutapo su ankstesniu TEST.

## Patikra
Rezultatai palyginti su 2–3 etapų TEST metrikėmis.


## Užduotis
Patikrinti atsparumą kontroliuojamam Gauso triukšmui standartizuotame TEST.

## Rezultatas
σ = 0 sutapo su švariu TEST. Visiems modeliams naudota ta pati triukšmo matrica.

## Patikra
Triukšmas pridėtas standartizuotiems testavimo duomenims, o mokymo duomenys nekeisti. σ = 0 palygintas su ankstesniu švariu TEST.


## Užduotis
Išanalizuoti švaraus TEST klaidas: painiavos matrica, metrikos pagal klasę ir asmenį.

## Rezultatas
Nustatytos pagrindinės painiavos (ypač SITTING / STANDING). Nauji modeliai nekurtos.

## Patikra
Metrikos perskaičiuotos ir palygintos su išsaugotais CSV.


## Užduotis
Įgyvendinti GRRF principu paremtą požymių atranką Python aplinkoje (ne originalų R RRF paketą) ir palyginti su SVM visiems 561 požymiui.

## Rezultatas
Pirmas variantas TEST: 254 požymiai, Macro-F1 0.881137. Parametrai G1–G5 parinkti tik TRAIN GroupKFold; pasirinktas G5. Galutinis G5 TEST: 523 požymiai (pašalinti 38 požymiai, 6,77 %). Macro-F1 0.894047. Klasifikavimas šiek tiek pablogėjo, todėl pagerėjimo neskelbiama.

## Patikra
Naudoti tik išsaugoti CSV; TEST nenaudotas naujam derinimu. Skaičiai sutikrinti pagal `grrf_inspired_vs_svm.csv`, `grrf_inspired_groupkfold_tuning.csv` ir `grrf_g5_vs_svm_test.csv`.


## Užduotis
Paruošti galutinę ataskaitą ir atkuriamumą: `main.py`, README ir `report/index.html`.

## Rezultatas
`python main.py` atkuria pagrindinį eksperimentą ir palygina su `results/final_model_comparison.csv`. HTML ataskaita papildo esamus rezultatus GRRF lentelėmis iš CSV, be naujo mokymo.

## Patikra
Realus `python main.py` paleidimas: `REPRODUCIBILITY CHECK: PASSED`. HTML skaičiai sutikrinti pagal CSV.

## 3. AI klaidos arba nepatikrintos prielaidos

### Problema
AI pateikė prielaidą apie standartinę UCI HAR katalogo struktūrą (`data/UCI HAR Dataset/`). Realiai rinkinys buvo nested kataloguose, pvz. `data/human+activity+recognition+using+smartphones (2)/UCI HAR Dataset/UCI HAR Dataset/`.

### Kaip patikrinta
Peržiūrėta reali `data/` struktūra (`features.txt`, `train/`, `test/`).

### Pataisymas
Notebook’ai ir `main.py` datasetą randa po `data/` pagal `features.txt`. Originalūs failai neperkelti.


### Problema
Pradinėje `requirements.txt` nebuvo `matplotlib`, nors abliacijos notebook’ui grafiko reikėjo.

### Kaip patikrinta
Notebook paleidimas baigėsi `ModuleNotFoundError: No module named 'matplotlib'`.

### Pataisymas
`matplotlib` įdiegta ir įrašyta į `requirements.txt`; notebook paleistas iš naujo.


### Problema
`requirements.txt` nurodė `pandas>=2.0,<3`, nors aplinka veikė su pandas 3.0.6.

### Kaip patikrinta
Palygintos realios Python paketų versijos su `requirements.txt`.

### Pataisymas
Ribos pakeistos į `pandas>=2.0,<4`. Po to `python main.py` vėl davė `REPRODUCIBILITY CHECK: PASSED`.

## 4. Galutinė patikra

- Paleista `python main.py` (`RANDOM_STATE = 42`); atkuriamumo patikra praėjo.
- Pagrindinių modelių TEST metrikos sutampa su išsaugota lentele.
- Hiperparametrai (k-NN, SVM, MLP ir GRRF G1–G5) parinkti tik TRAIN duomenyse arba TRAIN vidinio kryžminio patikrinimo metu. TEST naudotas tik galutiniam vertinimui, ne naujai paieškai.
