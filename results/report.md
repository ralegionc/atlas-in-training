# Report

places in the analysis: 15058

## Scaling: median error, km

model                        spelling      70m     160m     410m       1b     1.4b     2.8b
region                                                                                     
Australia & New Zealand       13780.0  14740.0  14800.0  14260.0  13510.0  11690.0  11370.0
Central America & Caribbean    3980.0   5250.0   4830.0   3680.0   3350.0   3010.0   2720.0
Central Asia                   2510.0   2920.0   2840.0   2790.0   2660.0   2340.0   1850.0
East Asia                      3010.0   4840.0   4030.0   2600.0   1870.0   1560.0   1350.0
Middle East & North Africa     1780.0   1980.0   1880.0   1650.0   1650.0   1450.0   1230.0
North America                  6470.0   6570.0   6000.0   4930.0   3980.0   3480.0   3130.0
Northern & Western Europe      1760.0   1940.0   1650.0   1420.0   1350.0   1190.0    960.0
Pacific Islands               11570.0  13250.0  12860.0  12260.0  10810.0  11000.0   9630.0
South America                  5380.0   7050.0   6310.0   5190.0   4290.0   3930.0   3260.0
South Asia                     2070.0   2540.0   2180.0   1890.0   1670.0   1550.0   1360.0
Southeast Asia                 4600.0   6060.0   5290.0   4500.0   3340.0   2920.0   2360.0
Southern & Eastern Europe      1740.0   1790.0   1660.0   1410.0   1330.0   1200.0    940.0
Sub-Saharan Africa             4070.0   5050.0   4890.0   4260.0   3600.0   3320.0   2880.0
World                          2680.0   3290.0   2940.0   2460.0   2140.0   1880.0   1560.0

## Replication of Godey et al. (2024), R^2 x 100

model        r2  layer
 160m 56.502381     10
   1b 68.634112     15

## Template check, 410m final: median km and cut beyond spelling

prompt                          bare     name
region                                       
Australia & New Zealand      13889.0  14259.0
Central America & Caribbean   3557.0   3678.0
Central Asia                  2693.0   2791.0
East Asia                     2139.0   2597.0
Middle East & North Africa    1688.0   1648.0
North America                 4556.0   4931.0
Northern & Western Europe     1349.0   1424.0
Pacific Islands              12409.0  12256.0
South America                 4743.0   5186.0
South Asia                    1822.0   1889.0
Southeast Asia                4279.0   4495.0
Southern & Eastern Europe     1268.0   1414.0
Sub-Saharan Africa            4102.0   4257.0
World                         2314.0   2459.0
prompt                        bare   name
region                                   
Australia & New Zealand      0.005 -0.014
Central America & Caribbean  0.091  0.080
Central Asia                 0.008 -0.033
East Asia                    0.231  0.146
Middle East & North Africa   0.111  0.128
North America                0.227  0.176
Northern & Western Europe    0.229  0.200
Pacific Islands              0.064  0.045
South America                0.197  0.149
South Asia                   0.132  0.114
Southeast Asia               0.150  0.105
Southern & Eastern Europe    0.251  0.209
Sub-Saharan Africa           0.015  0.002
World                        0.145  0.116
rank correlation of regional cuts between templates: 0.94

## 410m: formation

world median km by step: {0: 3797, 128: 3795, 512: 3736, 1000: 3525, 2000: 3212, 3000: 3132, 4000: 3047, 6000: 2900, 8000: 2956, 11000: 2821, 16000: 2782, 23000: 2678, 32000: 2623, 45000: 2559, 64000: 2547, 90000: 2527, 113000: 2476, 143000: 2459}

half of each region's improvement reached at step (95% CI):
  Central Asia                        668  [130, 1,296]  3,614 -> 2,791 km
  South America                     1,225  [966, 1,542]  8,916 -> 5,186 km
  Central America & Caribbean       1,393  [999, 2,455]  7,496 -> 3,678 km
  South Asia                        1,458  [1,198, 1,831]  3,323 -> 1,889 km
  Northern & Western Europe         1,817  [1,631, 2,321]  2,216 -> 1,424 km
  Southern & Eastern Europe         2,494  [1,918, 3,445]  2,038 -> 1,414 km
  North America                     2,871  [1,659, 4,380]  7,429 -> 4,931 km
  World                             3,047  [nan, nan]  3,797 -> 2,459 km
  East Asia                         3,495  [1,995, 3,966]  5,944 -> 2,597 km
  Southeast Asia                    3,758  [1,745, 5,688]  6,881 -> 4,495 km
  Pacific Islands                   4,614  [1,883, 45,546]  13,778 -> 12,256 km
  Middle East & North Africa        4,704  [495, 6,867]  2,231 -> 1,648 km
  Australia & New Zealand           7,131  [209, 31,222]  14,962 -> 14,259 km
  Sub-Saharan Africa               14,621  [9,917, 30,050]  5,059 -> 4,257 km

world gain beyond spelling by step: {0: 0.0, 128: 0.2, 512: 0.1, 1000: -0.2, 2000: 0.4, 3000: 1.5, 4000: 1.6, 6000: 2.4, 8000: 2.2, 11000: 4.3, 16000: 4.9, 23000: 6.8, 32000: 7.4, 45000: 8.5, 64000: 9.6, 90000: 9.4, 113000: 10.8, 143000: 11.6}

final gain beyond spelling by region (95% CI):
  North America                  +21.2%  [+17.6, +23.8]  model 4,931 km, spelling 6,472 km
  East Asia                      +20.1%  [+14.8, +23.5]  model 2,597 km, spelling 3,008 km
  Northern & Western Europe      +16.7%  [+13.5, +19.1]  model 1,424 km, spelling 1,764 km
  Southern & Eastern Europe      +13.2%  [+10.6, +16.3]  model 1,414 km, spelling 1,740 km
  World                          +11.6%  [+10.4, +12.9]  model 2,459 km, spelling 2,681 km
  South Asia                     +11.1%  [+6.3, +14.3]  model 1,889 km, spelling 2,067 km
  Central America & Caribbean    +10.5%  [+5.9, +14.5]  model 3,678 km, spelling 3,984 km
  South America                  +10.1%  [+6.6, +14.5]  model 5,186 km, spelling 5,379 km
  Middle East & North Africa      +9.1%  [+5.1, +12.7]  model 1,648 km, spelling 1,777 km
  Southeast Asia                  +8.5%  [+4.9, +13.7]  model 4,495 km, spelling 4,601 km
  Pacific Islands                 +5.8%  [+0.0, +11.3]  model 12,256 km, spelling 11,572 km
  Sub-Saharan Africa              +5.4%  [+3.2, +7.9]  model 4,257 km, spelling 4,065 km
  Australia & New Zealand         -0.7%  [-4.9, +1.7]  model 14,259 km, spelling 13,779 km
  Central Asia                    -3.6%  [-17.2, +2.6]  model 2,791 km, spelling 2,506 km

letters first by region (n 13): early rho 0.81 (p 0.0007), partial 0.81; late rho 0.39 (p 0.19), partial 0.39 (p 0.19)

letters first by iso (n 117): early rho 0.69 (p 5e-18), partial 0.70; late rho 0.12 (p 0.19), partial 0.17 (p 0.065)

country gain vs country mentions: rho 0.28 (p 0.0027, n 117)
lowest: Spain -23%, Mongolia -22%, Armenia -16%, Peru -14%, Angola -12%, Algeria -11%
highest: Thailand +40%, Vietnam +36%, Japan +33%, Slovakia +31%, Bolivia +30%, Romania +28%

## 1.4b: formation

world median km by step: {0: 3649, 512: 3579, 1000: 3323, 2000: 3042, 4000: 2889, 8000: 2703, 16000: 2418, 23000: 2333, 32000: 2205, 45000: 2104, 64000: 2059, 90000: 1986, 143000: 1881}

half of each region's improvement reached at step (95% CI):
  Central America & Caribbean       1,653  [974, 2,580]  7,132 -> 3,010 km
  South America                     1,812  [989, 3,648]  8,626 -> 3,933 km
  Central Asia                      2,015  [76, 18,165]  3,466 -> 2,339 km
  South Asia                        2,534  [1,558, 4,438]  3,107 -> 1,546 km
  East Asia                         3,028  [2,374, 4,424]  5,922 -> 1,556 km
  Northern & Western Europe         3,587  [2,620, 5,063]  2,120 -> 1,189 km
  Middle East & North Africa        4,493  [3,181, 7,696]  2,215 -> 1,454 km
  World                             6,353  [nan, nan]  3,649 -> 1,881 km
  Southern & Eastern Europe         8,226  [5,318, 10,387]  1,862 -> 1,197 km
  Southeast Asia                   11,645  [7,972, 17,620]  6,530 -> 2,923 km
  North America                    12,532  [6,690, 17,870]  7,305 -> 3,477 km
  Sub-Saharan Africa               21,342  [16,991, 27,207]  4,973 -> 3,322 km
  Pacific Islands                  27,734  [3,048, 67,871]  13,493 -> 10,998 km
  Australia & New Zealand          74,879  [30,859, 99,058]  14,536 -> 11,689 km

world gain beyond spelling by step: {0: 0.0, 512: 0.4, 1000: 0.5, 2000: 1.1, 4000: 4.0, 8000: 5.6, 16000: 10.8, 23000: 13.5, 32000: 17.2, 45000: 19.8, 64000: 21.2, 90000: 24.0, 143000: 29.2}

final gain beyond spelling by region (95% CI):
  East Asia                      +45.7%  [+42.5, +48.6]  model 1,556 km, spelling 3,008 km
  North America                  +42.0%  [+37.7, +45.5]  model 3,477 km, spelling 6,472 km
  Southeast Asia                 +34.4%  [+30.6, +39.2]  model 2,923 km, spelling 4,601 km
  Northern & Western Europe      +30.7%  [+28.0, +33.2]  model 1,189 km, spelling 1,764 km
  World                          +29.2%  [+27.6, +30.3]  model 1,881 km, spelling 2,681 km
  South America                  +26.7%  [+22.7, +30.4]  model 3,933 km, spelling 5,379 km
  Southern & Eastern Europe      +26.3%  [+23.3, +28.7]  model 1,197 km, spelling 1,740 km
  South Asia                     +24.6%  [+20.7, +29.0]  model 1,546 km, spelling 2,067 km
  Sub-Saharan Africa             +21.7%  [+19.5, +24.6]  model 3,322 km, spelling 4,065 km
  Central America & Caribbean    +19.9%  [+14.7, +24.8]  model 3,010 km, spelling 3,984 km
  Middle East & North Africa     +19.2%  [+15.5, +23.1]  model 1,454 km, spelling 1,777 km
  Australia & New Zealand        +14.1%  [+8.5, +19.7]  model 11,689 km, spelling 13,779 km
  Pacific Islands                +11.3%  [+3.7, +20.6]  model 10,998 km, spelling 11,572 km
  Central Asia                    +9.6%  [-2.4, +20.1]  model 2,339 km, spelling 2,506 km

letters first by region (n 13): early rho 0.88 (p 8e-05), partial 0.88; late rho 0.40 (p 0.18), partial 0.39 (p 0.18)

letters first by iso (n 117): early rho 0.69 (p 5e-18), partial 0.67; late rho 0.09 (p 0.33), partial 0.06 (p 0.52)

country gain vs country mentions: rho 0.25 (p 0.0057, n 117)
lowest: Mongolia -14%, Madagascar -10%, Angola -8%, Iraq -6%, Spain -4%, Portugal +0%
highest: Taiwan +61%, Japan +58%, Vietnam +57%, Thailand +52%, Czechia +47%, United States of America +47%

## 410m: counts at the final checkpoint (log10 km per tenfold, cluster-robust)

  count                        +0.0012  [-0.0123, +0.0146]  p 0.86  n 15,058
  count|own                    +0.0176  [+0.0018, +0.0333]  p 0.029  n 15,058
  own|count                    -0.0266  [-0.0397, -0.0136]  p 5.9e-05  n 15,058
  own|placebo                  -0.0194  [-0.0349, -0.0040]  p 0.014  n 5,716
  placebo|own                  +0.0061  [-0.0208, +0.0330]  p 0.66  n 5,716
  placebo alone                +0.0081  [-0.0191, +0.0354]  p 0.56  n 5,716
  views only                   -0.0454  [-0.0625, -0.0283]  p 1.9e-07  n 15,058
  count|spell                  -0.0063  [-0.0169, +0.0042]  p 0.24  n 15,058
  count|wordlike               -0.0027  [-0.0170, +0.0116]  p 0.71  n 15,058
  count, not word-like         -0.0034  [-0.0181, +0.0112]  p 0.65  n 14,930
  own|count, not word-like     -0.0257  [-0.0388, -0.0126]  p 0.00012  n 14,930
  own|count exact              -0.0464  [-0.0799, -0.0128]  p 0.0068  n 2,864
  own|count,spell              -0.0165  [-0.0272, -0.0058]  p 0.0025  n 15,058
  count|own,spell              +0.0039  [-0.0080, +0.0157]  p 0.52  n 15,058
  learned: own|count,spell     +0.0122  [+0.0018, +0.0227]  p 0.022  n 15,058
  learned: count|own,spell     +0.0147  [+0.0011, +0.0283]  p 0.034  n 15,058
  country level: log km on log mentions -0.136 [-0.229, -0.042], r -0.28

  region gaps (log10 km vs NW Europe): raw -> +spelling -> +counts
  Australia & New Zealand        +0.568 -> +0.348 -> +0.335   by spelling +39%, by counts +2%, counts only +6%
  Central America & Caribbean    +0.138 -> +0.085 -> +0.061   by spelling +39%, by counts +17%, counts only +15%
  Central Asia                   +0.193 -> +0.163 -> +0.121   by spelling +16%, by counts +21%, counts only +19%
  East Asia                      +0.004 -> -0.008 -> +0.000   by spelling +295%, by counts -197%, counts only -358%
  Middle East & North Africa     +0.094 -> +0.084 -> +0.078   by spelling +10%, by counts +6%, counts only -3%
  North America                  +0.252 -> +0.118 -> +0.133   by spelling +53%, by counts -6%, counts only +2%
  Pacific Islands                +0.525 -> +0.328 -> +0.258   by spelling +38%, by counts +13%, counts only +16%
  South America                  +0.207 -> +0.121 -> +0.103   by spelling +41%, by counts +9%, counts only +8%
  South Asia                     -0.058 -> -0.028 -> -0.033   by spelling +51%, by counts -7%, counts only +12%
  Southeast Asia                 +0.158 -> +0.106 -> +0.094   by spelling +33%, by counts +7%, counts only +3%
  Southern & Eastern Europe      +0.161 -> +0.112 -> +0.116   by spelling +30%, by counts -2%, counts only -4%
  Sub-Saharan Africa             +0.279 -> +0.196 -> +0.164   by spelling +30%, by counts +11%, counts only +10%

  exposure alpha: {'count': (0.03, 0.0, 0.03), 'own': (0.3, 0.28, 0.58), 'country_count': (0.6, 0.11, 1.98)}
  country arrival vs mentions: {'rho': 0.15474727974727975, 'p': 0.10486919402734535, 'n': 111}

## 1.4b: counts at the final checkpoint (log10 km per tenfold, cluster-robust)

  count                        +0.0154  [+0.0011, +0.0297]  p 0.034  n 15,058
  count|own                    +0.0449  [+0.0278, +0.0619]  p 2.5e-07  n 15,058
  own|count                    -0.0479  [-0.0633, -0.0325]  p 1e-09  n 15,058
  own|placebo                  -0.0490  [-0.0675, -0.0305]  p 1.9e-07  n 5,716
  placebo|own                  +0.0023  [-0.0211, +0.0257]  p 0.85  n 5,716
  placebo alone                +0.0075  [-0.0167, +0.0318]  p 0.54  n 5,716
  views only                   -0.0880  [-0.1063, -0.0697]  p 4.1e-21  n 15,058
  count|spell                  +0.0080  [-0.0038, +0.0198]  p 0.19  n 15,058
  count|wordlike               +0.0098  [-0.0051, +0.0248]  p 0.2  n 15,058
  count, not word-like         +0.0094  [-0.0061, +0.0249]  p 0.24  n 14,930
  own|count, not word-like     -0.0470  [-0.0626, -0.0314]  p 3.5e-09  n 14,930
  own|count exact              -0.0811  [-0.1113, -0.0510]  p 1.3e-07  n 2,864
  own|count,spell              -0.0379  [-0.0510, -0.0249]  p 1.2e-08  n 15,058
  count|own,spell              +0.0314  [+0.0183, +0.0445]  p 2.8e-06  n 15,058
  learned: own|count,spell     +0.0354  [+0.0228, +0.0481]  p 3.8e-08  n 15,058
  learned: count|own,spell     -0.0157  [-0.0290, -0.0024]  p 0.021  n 15,058
  country level: log km on log mentions -0.153 [-0.244, -0.062], r -0.32

  region gaps (log10 km vs NW Europe): raw -> +spelling -> +counts
  Australia & New Zealand        +0.500 -> +0.274 -> +0.251   by spelling +45%, by counts +5%, counts only +10%
  Central America & Caribbean    +0.163 -> +0.109 -> +0.094   by spelling +33%, by counts +9%, counts only +8%
  Central Asia                   +0.168 -> +0.137 -> +0.109   by spelling +19%, by counts +17%, counts only +15%
  East Asia                      -0.083 -> -0.095 -> -0.080   by spelling -14%, by counts +18%, counts only +25%
  Middle East & North Africa     +0.104 -> +0.094 -> +0.097   by spelling +9%, by counts -2%, counts only -10%
  North America                  +0.223 -> +0.086 -> +0.087   by spelling +62%, by counts -1%, counts only +10%
  Pacific Islands                +0.536 -> +0.334 -> +0.276   by spelling +38%, by counts +11%, counts only +14%
  South America                  +0.227 -> +0.139 -> +0.128   by spelling +39%, by counts +5%, counts only +5%
  South Asia                     -0.056 -> -0.026 -> -0.019   by spelling +54%, by counts +13%, counts only +34%
  Southeast Asia                 +0.107 -> +0.053 -> +0.057   by spelling +50%, by counts -4%, counts only -8%
  Southern & Eastern Europe      +0.137 -> +0.087 -> +0.096   by spelling +36%, by counts -7%, counts only -8%
  Sub-Saharan Africa             +0.244 -> +0.158 -> +0.136   by spelling +35%, by counts +9%, counts only +9%

  exposure alpha: {'count': (0.0, 0.0, 0.03), 'own': (0.33, 0.28, 0.38), 'country_count': (0.58, 0.03, 1.86)}
  country arrival vs mentions: {'rho': 0.04714747889213288, 'p': 0.6168248201979358, 'n': 115}

## 2.8b: counts at the final checkpoint (log10 km per tenfold, cluster-robust)

  count                        +0.0023  [-0.0128, +0.0174]  p 0.77  n 15,058
  count|own                    +0.0436  [+0.0248, +0.0623]  p 5.1e-06  n 15,058
  own|count                    -0.0671  [-0.0862, -0.0480]  p 5.7e-12  n 15,058
  own|placebo                  -0.0695  [-0.0928, -0.0462]  p 4.8e-09  n 5,716
  placebo|own                  +0.0225  [-0.0057, +0.0507]  p 0.12  n 5,716
  placebo alone                +0.0300  [+0.0015, +0.0585]  p 0.039  n 5,716
  views only                   -0.1066  [-0.1303, -0.0830]  p 1e-18  n 15,058
  count|spell                  -0.0051  [-0.0182, +0.0081]  p 0.45  n 15,058
  count|wordlike               -0.0045  [-0.0203, +0.0112]  p 0.57  n 15,058
  count, not word-like         -0.0052  [-0.0215, +0.0110]  p 0.53  n 14,930
  own|count, not word-like     -0.0661  [-0.0852, -0.0470]  p 1.2e-11  n 14,930
  own|count exact              -0.0972  [-0.1334, -0.0609]  p 1.5e-07  n 2,864
  own|count,spell              -0.0574  [-0.0739, -0.0408]  p 1e-11  n 15,058
  count|own,spell              +0.0303  [+0.0151, +0.0456]  p 9.4e-05  n 15,058
  country level: log km on log mentions -0.161 [-0.254, -0.067], r -0.32

  region gaps (log10 km vs NW Europe): raw -> +spelling -> +counts
  Australia & New Zealand        +0.433 -> +0.211 -> +0.184   by spelling +51%, by counts +6%, counts only +13%
  Central America & Caribbean    +0.145 -> +0.091 -> +0.063   by spelling +37%, by counts +19%, counts only +19%
  Central Asia                   +0.172 -> +0.141 -> +0.087   by spelling +18%, by counts +32%, counts only +30%
  East Asia                      -0.102 -> -0.113 -> -0.093   by spelling -11%, by counts +20%, counts only +26%
  Middle East & North Africa     +0.139 -> +0.129 -> +0.123   by spelling +7%, by counts +4%, counts only -1%
  North America                  +0.202 -> +0.067 -> +0.074   by spelling +67%, by counts -4%, counts only +8%
  Pacific Islands                +0.539 -> +0.341 -> +0.246   by spelling +37%, by counts +18%, counts only +21%
  South America                  +0.174 -> +0.088 -> +0.069   by spelling +50%, by counts +11%, counts only +11%
  South Asia                     -0.050 -> -0.021 -> -0.019   by spelling +58%, by counts +3%, counts only +27%
  Southeast Asia                 +0.083 -> +0.030 -> +0.028   by spelling +64%, by counts +2%, counts only -3%
  Southern & Eastern Europe      +0.172 -> +0.123 -> +0.126   by spelling +28%, by counts -2%, counts only -3%
  Sub-Saharan Africa             +0.252 -> +0.167 -> +0.129   by spelling +33%, by counts +15%, counts only +16%
