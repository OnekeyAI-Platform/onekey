## Method

### Data Sets

We compared the clinical characteristics of the patients using an independent sample $t$ test, Mann-Whitney $U$ test, or $\chi^2$ test, where appropriate. Table 1 showed the baseline clinical characteristics of patients in the our cohort respectively. 

【替换成你自己的stats.csv文件】

| feature_name  | train-label=ALL | train-label=0   | train-label=1   | pvalue   | test-label=ALL  | test-label=0    | test-label=1    | pvalue   |
| ------------- | --------------- | --------------- | --------------- | -------- | --------------- | --------------- | --------------- | -------- |
| duration      | 34.8668±21.2199 | 33.9369±21.1188 | 36.8817±21.4766 | 0.375324 | 36.8298±22.0264 | 34.5938±20.2386 | 41.6000±25.5211 | 0.314642 |
| age           | 45.2211±17.3032 | 44.0077±17.5134 | 47.8500±16.6793 | 0.155334 | 43.4894±15.9971 | 41.8125±16.9562 | 47.0667±13.5671 | 0.298955 |
| BMI           | 23.0572±2.4107  | 23.1016±2.3569  | 22.9610±2.5410  | 0.709658 | 23.2806±2.8462  | 23.3506±2.6632  | 23.1313±3.2980  | 0.808578 |
| rad_signature | 0.3192±0.3562   | 0.3147±0.3572   | 0.3289±0.3568   | 0.799681 | 0.3178±0.3697   | 0.3788±0.3981   | 0.1879±0.2675   | 0.099328 |
| chemotherapy  |                 |                 | 0.241697        |          |                 |                 | 0.006059        |          |
| 0             | 75(0.3947)      | 55(0.4231)      | 20(0.3333)      |          | 21(0.4468)      | 10(0.3125)      | 11(0.7333)      |          |
| 1             | 115(0.6053)     | 75(0.5769)      | 40(0.6667)      |          | 26(0.5532)      | 22(0.6875)      | 4(0.2667)       |          |
| gender        |                 |                 |                 | 0.456952 |                 |                 |                 | 0.957727 |
| 0             | 17(0.0895)      | 13(0.1000)      | 4(0.0667)       |          | 3(0.0638)       | 2(0.0625)       | 1(0.0667)       |          |
| 1             | 173(0.9105)     | 117(0.9000)     | 56(0.9333)      |          | 44(0.9362)      | 30(0.9375)      | 14(0.9333)      |          |
| result        |                 |                 |                 | 0.869483 |                 |                 |                 | 0.532249 |
| 0             | 106(0.5579)     | 72(0.5538)      | 34(0.5667)      |          | 22(0.4681)      | 16(0.5000)      | 6(0.4000)       |          |
| 1             | 84(0.4421)      | 58(0.4462)      | 26(0.4333)      |          | 25(0.5319)      | 16(0.5000)      | 9(0.6000)       |          |
| degree        |                 |                 |                 | 0.879561 |                 |                 |                 | 0.750846 |
| 0             | 44(0.2316)      | 29(0.2231)      | 15(0.2500)      |          | 9(0.1915)       | 6(0.1875)       | 3(0.2000)       |          |
| 1             | 145(0.7632)     | 101(0.7769)     | 44(0.7333)      |          | 37(0.7872)      | 25(0.7812)      | 12(0.8000)      |          |
| 2             | 1(0.0053)       | null            | 1(0.0167)       |          | 1(0.0213)       | 1(0.0312)       | null            |          |
| Tstage        |                 |                 |                 | 0.059028 |                 |                 |                 | 0.756437 |
| 0             | 40(0.2105)      | 30(0.2308)      | 10(0.1667)      |          | 12(0.2553)      | 6(0.1875)       | 6(0.4000)       |          |
| 1             | 47(0.2474)      | 35(0.2692)      | 12(0.2000)      |          | 12(0.2553)      | 10(0.3125)      | 2(0.1333)       |          |
| 2             | 50(0.2632)      | 34(0.2615)      | 16(0.2667)      |          | 15(0.3191)      | 12(0.3750)      | 3(0.2000)       |          |
| 3             | 53(0.2789)      | 31(0.2385)      | 22(0.3667)      |          | 8(0.1702)       | 4(0.1250)       | 4(0.2667)       |          |
| smoke         |                 |                 |                 | 0.684509 |                 |                 |                 | 0.010286 |
| 0             | 44(0.2316)      | 29(0.2231)      | 15(0.2500)      |          | 15(0.3191)      | 14(0.4375)      | 1(0.0667)       |          |
| 1             | 146(0.7684)     | 101(0.7769)     | 45(0.7500)      |          | 32(0.6809)      | 18(0.5625)      | 14(0.9333)      |          |
| drink         |                 |                 |                 | 0.042361 |                 |                 |                 | 0.640483 |
| 0             | 60(0.3158)      | 35(0.2692)      | 25(0.4167)      |          | 18(0.3830)      | 13(0.4062)      | 5(0.3333)       |          |
| 1             | 130(0.6842)     | 95(0.7308)      | 35(0.5833)      |          | 29(0.6170)      | 19(0.5938)      | 10(0.6667)      |          |

Table 1 Baseline characteristics of patients in cohorts.

### Feature Extraction

The handcrafted features can be divided into three groups: (I) geometry, (II) intensity and (III) texture. The geometry features describe the three-dimensional shape characteristics of the tumor. The intensity features describe the first-order statistical distribution of the voxel intensities within the tumor. The texture features describe the patterns, or the second- and high-order spatial distributions of the intensities. Here the texture features are extracted using several different methods, including the gray-level co-occurrence matrix (GLCM), gray-level run length matrix (GLRLM), gray level size zone matrix (GLSZM) and neighborhood gray-tone difference matrix (NGTDM) methods. 

### Feature Selection

**Statistics**: We also conducted Mann-Whitney Utest statistical test and feature screening for all radiomic features. Only the pvalue<0.05 of radiomic feature were kept.

**Correlation**: For features with high repeatability, Spearman's rank correlation coefficient was also used to calculate the correlation between features (Fig. 1. spearman correlation of each feature), and one of the features with correlation coefficient greater than 0.9 between any two features is retained. In order to retain the ability to depict features to the greatest extent, we use greedy recursive deletion strategy for feature filtering, that is, the feature with the greatest redundancy in the current set is deleted each time. After this, 23 features were finally kept.

**Lasso**:The least absolute shrinkage and selection operator (LASSO) regression model was used on the discovery data set for signature construction. Depending on the regulation weight *λ*, LASSO shrinks all regression coefficients towards zero and sets the coefficients of many irrelevant features exactly to zero. To find an optimal *λ*, 10-fold cross validation with minimum criteria was employed, where the final value of *λ* yielded minimum cross validation error.  The retained features with nonzero coefficients were used for regression model fitting and combined into a radiomics signature. Subsequently, we obtained a radiomics score for each patient by a linear combination of retained features weighed by their model coefficients. The Python scikit-learn package was used for LASSO regression modeling. 

### Rad Signature

After Lasso feature screening, we input the final features into the machine learning models like lr, svm, random forest, XGBoost and so on for risk model construction. Here, we adopt 5 fold cross verification to obtain the final Rad Signature.

Furthermore, to intuitively and efficiently assess the incremental prognostic value of the radiomics signature to the clinical risk factors, a radiomics nomogram was presented on the validation data set. The nomogram combined the radiomics signature and the clinical risk factors based on the logistic regression analysis. To compare the agreement between the 【任务】 prediction of the nomogram and the actual observation, the calibration curve was calculated.

### Clinical Signature

The building process of clinical signature is almost the same as rad signature. First the features used for building clinical signature were selected by baseline statistic whose pvalue<0.05. We also used the same machine learning model in rad signature building process. 5 fold cross validation and test cohort  was set to be fixed for fair comparation.

### Radiomic Nomogram

Radiomic nomogram was established in combination with radiomic signature and clinical signature. The diagnostic efficacy of radiomic nomogram was tested in test cohort, ROC curves were drawn to evaluate the diagnostic efficacy of nomogram. The calibration efficiency of nomogram was evaluated by drawing calibration curves, and Hosmer-Lemeshow analytical fit was used to evaluate the calibration ability of nomogram. Mapping decision curve analysis (DCA) to evaluate the clinical utility of predictive models.

## Results

### Signature Building

**Features Statistics**: A total of 6 categories, XXX handcrafted features are extracted, including XXX firstorder features, 14 shape features, and the last are texture features. Details of the handcrafted features can be found in Supplementary Table 1. All handcrafted features are extracted with an in-house feature analysis program implemented in Pyradiomics（http://pyradiomics.readthedocs.io）. Fig. 2 shows all features and corresponding pvalue results.

![](img/Rad_feature_ratio.svg)

Fig. 1 Numer and ratio of handcrafted features

![](img/Rad_feature_stats.svg)

Fig. 2 Statistics of radiomic features

Fig.3 shows the correlations between each clinical features, it is indicate that Long Diameter, Short Diameter and Diameter has maximum correlation coefficient. 

![](img/Clinic_feature_corr.svg)

Fig.3 Spearman correlation coefficients of each clinical features 

**Lasso feature selection**: Nonzero coefficients were selected to establish the Rad-score with a least absolute shrinkage and selection operator (LASSO) logistic regression model. Coefficients and MSE(mean standard error) of 10 folds validation is show in Fig.4 and Fig.5

![coeff](img/Rad_feature_lasso.svg)

Fig.4. Coefficients of 10 fold cross validation

![mse](img/Rad_feature_mse.svg)

Fig.5 MSE of 10 fold cross validation

Rad score is show as follow, Fig.6 shows the coefficients value in the final selected none zero features.

【替换成自己的】

```
label = 0.3090601683388831 + +0.127998 * original_firstorder_Kurtosis -0.077429 * original_firstorder_Minimum +0.114108 * original_firstorder_RootMeanSquared -0.047303 * original_glcm_Imc2 -0.063017 * original_glcm_SumEntropy +0.064179 * original_glrlm_LongRunLowGrayLevelEmphasis -0.059840 * original_glrlm_RunEntropy -0.031554 * original_glrlm_ShortRunEmphasis -0.016628 * original_glszm_GrayLevelVariance +0.008884 * original_glszm_ZoneVariance +0.053577 * original_ngtdm_Busyness -0.029125 * original_shape_Sphericity
```

![feat](img/Rad_feature_weights.svg)

Fig.6. The histogram of the Rad-score based on the selected features. 

**Model Comparation**: Table 2. is all model we used to predict 【任务】, XXX model preforms the best performance. So in the building of clinical signature, XXX is selected as base model.

|      |   model_name | Accuracy |      AUC |          95% CI | Sensitivity | Specificity |      PPV |      NPV | Precision |   Recall |       F1 | Threshold | Task        |
| ---: | -----------: | -------: | -------: | --------------: | ----------: | ----------: | -------: | -------: | --------: | -------: | -------: | --------: | ----------- |
|    0 |          SVM | 0.761905 | 0.856072 | 0.7930 - 0.9191 |    0.866667 |    0.713178 | 0.584270 | 0.920000 |  0.584270 | 0.866667 | 0.697987 |  0.207249 | label-train |
|    1 |          SVM | 0.833333 | 0.848485 | 0.7338 - 0.9632 |    0.733333 |    0.878788 | 0.733333 | 0.878788 |  0.733333 | 0.733333 | 0.733333 |  0.291243 | label-test  |
|    2 |          KNN | 0.793651 | 0.893798 | 0.8498 - 0.9378 |    0.850000 |    0.767442 | 0.629630 | 0.916667 |  0.629630 | 0.850000 | 0.723404 |  0.400000 | label-train |
|    3 |          KNN | 0.833333 | 0.820202 | 0.6795 - 0.9609 |    0.666667 |    0.909091 | 0.769231 | 0.857143 |  0.769231 | 0.666667 | 0.714286 |  0.600000 | label-test  |
|    4 | DecisionTree | 1.000000 | 1.000000 |       nan - nan |    1.000000 |    1.000000 | 1.000000 | 1.000000 |  1.000000 | 1.000000 | 1.000000 |  1.000000 | label-train |
|    5 | DecisionTree | 0.708333 | 0.696970 | 0.5514 - 0.8426 |    0.666667 |    1.000000 | 0.526316 | 0.827586 |  0.526316 | 0.666667 | 0.588235 |  1.000000 | label-test  |
|    6 | RandomForest | 0.989418 | 0.999871 | 0.9996 - 1.0000 |    1.000000 |    0.984496 | 0.967742 | 1.000000 |  0.967742 | 1.000000 | 0.983607 |  0.400000 | label-train |
|    7 | RandomForest | 0.770833 | 0.807071 | 0.6746 - 0.9395 |    0.800000 |    0.757576 | 0.600000 | 0.892857 |  0.600000 | 0.800000 | 0.685714 |  0.400000 | label-test  |
|    8 |   ExtraTrees | 1.000000 | 1.000000 |       nan - nan |    1.000000 |    1.000000 | 1.000000 | 1.000000 |  1.000000 | 1.000000 | 1.000000 |  1.000000 | label-train |
|    9 |   ExtraTrees | 0.708333 | 0.796970 | 0.6619 - 0.9321 |    0.866667 |    0.636364 | 0.520000 | 0.913043 |  0.520000 | 0.866667 | 0.650000 |  0.400000 | label-test  |
|   10 |      XGBoost | 1.000000 | 1.000000 |       nan - nan |    1.000000 |    1.000000 | 1.000000 | 1.000000 |  1.000000 | 1.000000 | 1.000000 |  0.506426 | label-train |
|   11 |      XGBoost | 0.833333 | 0.810101 | 0.6663 - 0.9539 |    0.666667 |    0.909091 | 0.769231 | 0.857143 |  0.769231 | 0.666667 | 0.714286 |  0.657828 | label-test  |
|   12 |     LightGBM | 0.830688 | 0.917506 | 0.8799 - 0.9552 |    0.900000 |    0.798450 | 0.675000 | 0.944954 |  0.675000 | 0.900000 | 0.771429 |  0.336961 | label-train |
|   13 |     LightGBM | 0.812500 | 0.783838 | 0.6285 - 0.9391 |    0.600000 |    0.909091 | 0.750000 | 0.833333 |  0.750000 | 0.600000 | 0.666667 |  0.487166 | label-test  |

The optimal model was obtained by using rad features compared with an LR, SVM, KNN, Decision Tree, Random Forest, Extra Trees, XGBoost and LightGBM classifier. XGBoost achieved the best value of auc on the training and test cohort reached yyy and yyy for 【任务】 respectively.  Fig.7 shows each rad signature model's auc on test cohort.

![roc](img/Rad_model_roc.svg)

Fig. 7 ROC analysis of different models on rad signature

Table 3 shows the performance of each model on clinical signature, Fig.8 shows each clinical signature model's auc on test cohort

![roc](img/Clinic_model_roc.svg)

Fig. 8 ROC analysis of different models on rad signature

### Nomgram

**AUC**: In training cohort both clinical signature and rad signature get the prefect fitting. In test cohort clinical signature seems over fitting, but rag signature still fitted good. The Nomogram using Logistic Regression algorithm was preformed to combine clinical signature and rad signature, which shows the best performance. In order to compare the clinical signature and rad signature and nomogram, Delong test was used. 

|      |        Signature | Accuracy |      AUC |          95% CI | Sensitivity | Specificity |      PPV |      NPV | Precision |   Recall |       F1 | Threshold | Cohort |
| ---: | ---------------: | -------: | -------: | --------------: | ----------: | ----------: | -------: | -------: | --------: | -------: | -------: | --------: | -----: |
|    0 | Clinic Signature | 0.631579 | 0.631026 | 0.5467 - 0.7154 |    0.633333 |    0.635659 | 0.441860 | 0.788462 |  0.441860 | 0.633333 | 0.520548 |  0.323198 |  Train |
|    1 |    Rad Signature | 0.773684 | 0.797821 | 0.7266 - 0.8690 |    0.666667 |    0.823077 | 0.634921 | 0.842520 |  0.634921 | 0.666667 | 0.650406 |  0.359420 |  Train |
|    2 |         Nomogram | 0.784211 | 0.812821 | 0.7456 - 0.8801 |    0.683333 |    0.830769 | 0.650794 | 0.850394 |  0.650794 | 0.683333 | 0.666667 |  0.326722 |  Train |
|    3 | Clinic Signature | 0.425532 | 0.426042 | 0.2484 - 0.6037 |    1.000000 |    0.161290 | 0.357143 | 1.000000 |  0.357143 | 1.000000 | 0.526316 |  0.223668 |   Test |
|    4 |    Rad Signature | 0.787234 | 0.779167 | 0.6155 - 0.9429 |    0.733333 |    0.812500 | 0.647059 | 0.866667 |  0.647059 | 0.733333 | 0.687500 |  0.358077 |   Test |
|    5 |         Nomogram | 0.829787 | 0.772917 | 0.6058 - 0.9400 |    0.666667 |    0.906250 | 0.769231 | 0.852941 |  0.769231 | 0.666667 | 0.714286 |  0.347266 |   Test |

|      | Nomogram Vs Clinic | Nomogram Vs Rad | cohort |
| ---: | -----------------: | --------------: | -----: |
|    0 |           0.000399 |        0.066132 |  Train |
|    1 |           0.010361 |        0.544511 |   Test |

Table3. Nomogram indicators.

![](img/train_auc.svg)

![](img/test_auc.svg)

Fig.9 show the AUC in both train and test cohort.

**Calibration Curves**: Nomogram calibration curves show good agreement between predicted and observed 【任务】 training and test cohort. The P values of Hosmer-Lemeshow test inspect of clinical signature, rad signature and nomogram. This shows that Nomogram fits perfectly in both the training and test cohort. Fig.10 shows the calibration curves in train and test cohort.

![](img/train_cali.svg)

![](img/test_cali.svg)

|      | Clinic Signature | Rad Signature | Nomogram |
| ---: | ---------------: | ------------: | -------: |
|    0 |         0.802715 |      0.646688 | 0.318583 |
|    1 |         0.160125 |      0.550619 | 0.124834 |

Fig.10 The calibration curves in train and test cohort, Hosmer-Lemeshow test.

**DCA**: In this study, we also evaluated each model through DCA. The decision curve analysis for the clinical signature, rad signature and radiomic nomogram are presented in Fig. 11. Compared with scenarios in which no prediction model would be used (ie, treat-all or treat-none scheme), radiomic nomogram showed significant benefit for intervention in patients with a prediction probability compared to clinical signature, rad signature. Nomogram is higher than other signatures. Preoperative prediction 【任务】 using radiomic nomogram has been shown to have better clinical benefit.

![](img/test_dca.svg)

Fig. 11 Decision curve of in test cohort

**Interpretation**: Fig.12 shows the nomogram for clinical use.

![](nomogram.png)

