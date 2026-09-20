# HAR galutinė rezultatų santrauka (1–6 etapai)
Santrauka sugeneruota 7 etape iš jau egzistuojančių results/ failų.
## Protokolas
- UCI HAR; 10299 įrašai; 561 požymiai; 30 subject; 6 klasės.
- GroupShuffleSplit, RANDOM_STATE=42; TRAIN 7144/21; TEST 3155/9; sankirta tuščia.
- Hiperparametrai tik TRAIN (GroupKFold); TEST nenaudotas tuning.
## TEST palyginimas
| Model | Macro-F1 | Balanced Accuracy | TEST klaidos |
|---|---|---|---|
| Majority Class | 0.053188 | 0.166667 | 2556 |
| k-NN | 0.844324 | 0.842074 | 491 |
| SVM | 0.897130 | 0.893666 | 347 |
| MLP | 0.896616 | 0.894957 | 340 |

Pagal Macro-F1 SVM didžiausias; MLP nežymiai geresnis BA ir mažiau klaidų.
## Abliacija
All > Accelerometer-only > Gyroscope-only visiems trims modeliams (šis eksperimentas).
- k-NN: All 0.844324; Acc 0.809685 (Δ -0.034639); Gyro 0.690462 (Δ -0.153862).
- SVM: All 0.897130; Acc 0.855685 (Δ -0.041445); Gyro 0.797924 (Δ -0.099206).
- MLP: All 0.896616; Acc 0.861582 (Δ -0.035033); Gyro 0.750806 (Δ -0.145809).
## Robustness (Gauso triukšmas standartizuotame TEST)
- k-NN: σ=0 0.844324; σ=0.5 0.841872; kritimas 0.002451.
- SVM: σ=0 0.897130; σ=0.5 0.859410; kritimas 0.037720.
- MLP: σ=0 0.896616; σ=0.5 0.878703; kritimas 0.017913.
Viena realizacija kiekvienam sigma; ne IMU fizikinis modelis.
## Klaidos
Dažna SITTING↔STANDING; taip pat LAYING→STANDING ir dalis walking porų.
- k-NN: TEST 6, subject 16, STANDING → SITTING.
- SVM: TEST 0, subject 16, STANDING → SITTING.
- MLP: TEST 0, subject 16, STANDING → SITTING.
## Nematytų subject Macro-F1
- k-NN: 0.7262–0.9613.
- SVM: 0.7155–1.0000.
- MLP: 0.7433–0.9918.
Bendras TEST balas neslepia šios variacijos.
## Išvada
Hipotezė pagal Macro-F1 (SVM didžiausias) šiame eksperimente pasitvirtino; SVM ir MLP skirtumas mažas, MLP nežymiai geresnis pagal BA ir klaidų skaičių.
