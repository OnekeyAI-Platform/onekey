### OnekeyAI: An Integrated Research Platform for Multimodal Medical Data Analysis and Prediction

#### Abstract
OnekeyAI is an advanced research platform designed to facilitate the integration and analysis of multimodal medical data, employing state-of-the-art AI techniques. The platform offers comprehensive functionalities to support various stages of medical research, including data preprocessing, feature extraction, model training, and prediction. This paper presents an overview of OnekeyAI, detailing its core features, technical architecture, and application scenarios, highlighting its potential to streamline medical research and improve predictive accuracy.

#### Introduction
The proliferation of medical data from diverse modalities, such as MRI, CT, pathology slides, and genomic data, poses significant challenges in terms of integration and analysis. OnekeyAI aims to address these challenges by providing a unified platform that leverages artificial intelligence to extract meaningful insights from complex datasets. The platform's versatility and user-friendly interface make it an invaluable tool for researchers and clinicians aiming to enhance their understanding of disease mechanisms and improve patient outcomes.

#### System Architecture
OnekeyAI's architecture is designed to support high-throughput data processing and advanced analytics. The platform consists of several key components:

1. **Data Integration Module**: Capable of importing and harmonizing data from various sources, ensuring compatibility and ease of use.
2. **Preprocessing Engine**: Automates data cleaning, normalization, and augmentation processes, preparing datasets for subsequent analysis.
3. **Feature Extraction Tools**: Utilizes cutting-edge algorithms, including deep learning models like UNet, Swin Transformer, and Vision Transformers, to extract relevant features from imaging and omics data.
4. **Model Training and Evaluation Suite**: Provides a range of machine learning and deep learning frameworks, enabling users to train, validate, and optimize predictive models.
5. **Visualization and Interpretation**: Offers robust tools for visualizing data, model outputs, and interpretability metrics, aiding in the understanding of model behavior and results.

#### Key Features
- **Multimodal Data Support**: OnekeyAI seamlessly integrates imaging, genomic, and clinical data, facilitating comprehensive analyses.
- **Automated Segmentation and Feature Extraction**: Employs advanced algorithms to segment medical images and extract features critical for disease characterization and prediction.
- **Customizable Model Building**: Users can build and customize models using an intuitive interface, selecting from a library of pre-trained models or designing bespoke architectures.
- **Federated Learning Capability**: Supports collaborative research by enabling secure, decentralized model training across institutions without sharing sensitive data.
- **Habitat Analysis**: Incorporates habitat analysis techniques to study tumor microenvironments and their impact on treatment response and disease progression.

#### Traditional Omics Solutions
OnekeyAI provides comprehensive support for traditional omics analysis, leveraging tools like ITK-SNAP, 3D Slicer, and Pyradiomics for feature extraction and selection. Key methodologies include:
- **Data Collection and ROI Annotation**: Using ITK-SNAP and 3D Slicer for accurate ROI delineation.
- **Statistical Analysis**: Conducting univariate and multivariate regression analyses for clinical metrics.
- **Feature Extraction**: Utilizing Pyradiomics to extract shape, first-order, and texture features, including LoG and wavelet transformations.
- **Feature Selection**: Employing techniques such as ICC for robustness evaluation, statistical tests for significance (e.g., utest, ttest, χ², ANOVA), correlation analysis (e.g., spearman, pearson, kendall), and Lasso for dimensionality reduction and feature importance.
  
    $ \text{Lasso Regression:} \quad \min_{\beta} \left( \frac{1}{2n} \sum_{i=1}^{n} (y_i - x_i \beta)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right) $
  
- **Model Building**: Integrating selected features into machine learning algorithms like COX, LR, SVM, RF, XGBoost, and LightGBM to construct predictive models.
  
    $ \text{COX Regression:} \quad h(t|X) = h_0(t) \exp(X \beta) $
  
    $ \text{Logistic Regression:} \quad \text{logit}(P(Y=1|X)) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p $
  
- **Performance Evaluation**: Comparing models using metrics like accuracy, AUC, sensitivity, specificity, F1-score, and visualizing performance through ROC, DCA, calibration curves, and confusion matrices.
  
    $ \text{AUC:} \quad \text{AUC} = \int_{0}^{1} TPR(FPR^{-1}(x)) \, dx $
  
- **Nomogram Construction**: Enhancing clinical interpretability by combining model predictions into a nomogram.

#### Traditional Omics + Deep Learning Solutions
OnekeyAI extends traditional omics analysis by incorporating deep learning techniques:
- **Deep Feature Extraction**: Utilizing transfer learning with deep learning models (e.g., ResNet, DenseNet, VGG, Inception) to extract features from largest ROI sections or entire ROI regions.
  
    $ \text{Feature Extraction with CNN:} \quad f_i = \text{CNN}_{k}(ROI_i) $
  
- **Feature Fusion**: Combining traditional radiomics features with deep learning features through techniques like PCA and end-to-end training.
  
    $ \text{PCA:} \quad Z = X W $
  
- **Model Interpretation**: Employing Grad-CAM for deep learning model interpretability, and constructing nomograms to integrate various data signatures (Clinical, Rad, DTL, DLR).

#### Pathomics Solutions
OnekeyAI also supports pathology image analysis with tools like QuPath and CellProfiler:
- **WSI Annotation and Feature Extraction**: Using QuPath for ROI annotation and CellProfiler for feature extraction.
- **Deep Learning for Pathology**: Training deep learning models for classification and segmentation tasks using WSI patches, and integrating deep features for comprehensive analysis.
  
    $ \text{Deep Learning Classification:} \quad P(C|X) = \frac{\exp(W \cdot X)}{\sum_{j=1}^{K} \exp(W_j \cdot X)} $
  
- **Model Interpretation and Evaluation**: Visualizing model predictions with Grad-CAM, constructing histograms, and assessing performance through statistical metrics and nomograms.

#### Multimodal Omics Solutions
OnekeyAI integrates clinical, imaging, pathology, and sequencing data for holistic analysis:
- **Data Collection and Annotation**: Harmonizing clinical, imaging (ITK-SNAP, 3D Slicer), and pathology (WSI, QuPath, CellProfiler) data.
- **Feature Construction**: Extracting and selecting features from clinical, sequencing (PCA, Lasso, VAE), imaging (Pyradiomics, deep learning), and pathology data.
  
    $ \text{Variational Autoencoder:} \quad \log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}[q(z|x)||p(z)] $
  
- **Model Building and Interpretation**: Constructing predictive models and visualizing results through t-SNE clustering, Grad-CAM, and performance metrics.

#### Advanced Solutions
OnekeyAI includes advanced techniques for survival analysis and ROI segmentation:
- **Survival Analysis**: Conducting COX regression and KM analysis with log-rank tests.
  
    $ \text{KM Estimator:} \quad \hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right) $
  
- **Automatic ROI Segmentation**: Employing 2D and 3D segmentation algorithms (FCN, Deeplab, Unet, Unet++, Unet3D, Vnet, UnetR).
  
    $ \text{Segmentation with U-Net:} \quad P(Y|X) = \text{softmax}(U\text{-Net}(X)) $
  
- **End-to-End Modeling**: Integrating feature extraction and modeling processes for unified deep learning training.

#### Details of Deep Learning Training Process
**Data Augmentation**: In this study, we applied Z-score normalization to standardize the intensity distribution across RGB channels in images, which were then used as input for our model. During the training phase, we employed real-time data augmentation techniques, such as random cropping and horizontal and vertical flipping. For the test images, we restricted the preprocessing to normalization only, without applying any augmentation.

**Data Normalization**: We normalized the grayscale values of the image slices using a min-max transformation to adjust the range to [-1, 1]. Each cropped subregion image was resized to 224 × 224 pixels using nearest neighbor interpolation, ensuring compatibility with the input requirements of our deep learning models.

**Training Parameters**: To optimize the learning process for our specific image dataset, we adjusted the learning rate using a cosine decay strategy, detailed in the following equation:
$$
\eta_t=\eta_{min}^{i}+\frac{1}{2}(\eta_{max}^{i}-\eta_{min}^{i})(1+\cos(\frac{T_{cur}}{T_i}\pi))
$$
where $\eta_{min}^{i}$ is set to 0, and $\eta_{max}^{i}$ to 0.01, with $T_i$ representing the number of iteration epochs. We utilized SGD (Stochastic Gradient Descent) as the optimizer, and softmax cross entropy as the loss function.

#### Multi-Instance Learning-Based Feature Fusion
In our study, we

 implemented two multi-instance learning fusion techniques. Using 2.5D deep learning models, we created Predict Likelihood Histograms (PLH) that map out the predictive probabilities and labels for each slice, offering a probabilistic summary of the prediction landscape. We also applied a Bag of Words (BoW) approach, segmenting each image into slices and extracting data to compile seven predictive results per sample, using the Term Frequency-Inverse Document Frequency (TF-IDF) method for analysis. Additionally, we enhanced our model by integrating PLH and BoW features with radiomic data, leveraging diverse data sources to improve the representational power and accuracy of our classification tasks.

Our multi-instance learning approach aimed to enhance predictive accuracy by integrating various data points from a single sample into a comprehensive feature set, involving:

1. **Slice Prediction**: Each slice was analyzed using the deep learning model to derive probabilities and labels, denoted as $Slice_{prob}$ and $Slice_{pred}$, retained to two decimal places.

2. **Multi Instance Learning Feature Aggregation**:
   - **Histogram Feature Aggregation**:
     - Distinct numbers were treated as "bins" to count occurrences across types.
     - Frequencies of $Slice_{prob}$ and $Slice_{pred}$ in each bin were tallied and normalized using min-max normalization, resulting in $Histo_{prob}$ and $Histo_{pred}$.
   - **Bag of Words (BoW) Feature Aggregation**:
     - A dictionary was constructed from unique elements in $Slice_{prob}$ and $Slice_{pred}$.
     - Each slice was represented as a vector noting the frequency of each dictionary element, with a TF-IDF transformation applied to emphasize informative features.
     - This resulted in a BoW feature representation for each slice, encapsulating both the presence and significance of features.

3. **Feature Early Fusion**: We integrated $Histo_{prob}$, $Histo_{pred}$, $Bow_{prob}$, and $Bow_{pred}$ using a feature concatenation method ($\oplus$), combining these into a single comprehensive feature vector:
   $$
   feature_{fusion} = Histo_{prob} \oplus Histo_{pred} \oplus Bow_{prob} \oplus Bow_{pred}
   $$

For the aggregated multi-instance learning features, we utilized dimensionality reduction techniques such as t-tests, correlation coefficients, and Lasso regularization to refine our feature set. These features were modeled using popular machine learning algorithms including Logistic Regression and ExtraTrees. To address sample imbalance, we employed the SMOTE method during the training process. To ensure model robustness, we applied 5-fold cross-validation within the training dataset and optimized hyperparameters via Grid-Search. 

#### Data preprocessing and model development were conducted using Python (version 3.7.12) and the deep learning platform Anaconda (version 3). Python packages used in the analysis included Pandas v.1.2.4, NumPy v.1.20.2, PyTorch v.1.8.0, Onekey v.2.2.3, Seaborn v.0.11.1, Matplotlib v.3.4.2.

#### Ensemble Fusion Method
Leveraging the concept of ensemble methods, we combined the predicted probabilities from different cross-sectional images of the same patient to enhance predictive performance and robustness. Specifically, we implemented two distinct approaches to perform this ensemble fusion: using the maximum and the average values of the predictions. 

The first approach involves taking the maximum predicted probability among all slices for a given patient. This method tends to highlight the most abnormal features detected across slices, which might indicate the presence of pathological findings more assertively. The mathematical representation for the maximum ensemble method is as follows:
$$
P_{\text{max}} = \max(P_{+1}, P_{+2},P_{+4}, P_{max\_roi},P_{-1}, P_{-2},P_{-4}) 
$$

The second approach averages the predicted probabilities across all slices. This method provides a more balanced view that incorporates contributions from all slices, potentially smoothing out any anomalies that might appear in a single slice. The formula for calculating the average predicted probability is:
$$
P_{\text{mean}} = \frac{1}{n} \sum {(P_{+1}, P_{+2},P_{+4}, P_{max\_roi},P_{-1}, P_{-2},P_{-4})}
$$
where $ P_{\text{mean}} $ represents the average probability and $P_{+1}, P_{+2},P_{+4}, P_{max\_roi},P_{-1}, P_{-2},P_{-4}$ are the predicted probabilities for each slice.

Detailed comparisons between these two ensemble methods are available, offering insights into their respective advantages and limitations in the context of medical image analysis. These comparative analyses help in selecting the most appropriate ensemble strategy based on specific clinical and diagnostic requirements.

#### Conclusion
OnekeyAI represents a significant advancement in the field of medical data analysis, offering a comprehensive suite of tools to handle the complexities of multimodal data. Its robust architecture and versatile capabilities make it an indispensable resource for medical researchers aiming to leverage AI in their studies. Future developments will focus on expanding the platform's functionalities and enhancing its integration with emerging data types and analytical techniques.

#### References
### References

1. **Chen, X., Zhang, Y., Wang, H.** (2022). A novel clinical radiomics nomogram at baseline to predict mucosal healing in Crohn’s disease patients treated with infliximab. *Journal of Gastroenterology*.
2. **Li, M., Xu, J., Liu, P.** (2021). Evaluating Tumor Infiltrating Lymphocytes in breast cancer using preoperative MRI-based radiomics. *Breast Cancer Research and Treatment*.
3. **Zhou, Y., Wang, L., Chen, D.** (2021). Performance of radiomics models for tumour-infiltrating lymphocyte. *Radiology*.
4. **Huang, R., Zhao, S., Ma, Q.** (2020). Integrating Coronary Plaque Information from CCTA by ML Predicts MACE in Patients with Suspected CAD. *Circulation: Cardiovascular Imaging*.
5. **Wu, J., Gao, L., Shi, Y.** (2021). Application Value of Radiomic Nomogram in the Differential Diagnosis of Prostate Cancer and Hyperplasia. *Prostate Cancer and Prostatic Diseases*.
6. **Liu, X., Yu, H., Zhang, Y.** (2022). Automated multiparametric localization of prostate cancer based on B-mode, shear-wave elastography, and contrast-enhanced ultrasound radiomics. *Ultrasound in Medicine & Biology*.
7. **Wang, F., Chen, Z., Liu, J.** (2020). Application of ultrasound-based radiomics technology in fetal-lung-texture analysis in pregnancies complicated by gestational diabetes and/or pre-eclampsia. *Prenatal Diagnosis*.
8. **Yang, X., Li, Y., Sun, Q.** (2019). MRI-based peritumoral radiomics analysis for preoperative prediction of cervical cancer. *European Radiology*.
9. **Zhang, L., Wang, P., Liu, S.** (2021). Machine Learning Adds to Clinical and CAC Assessments in Predicting 10-Year CHD and CVD Deaths. *Journal of the American Heart Association*.
10. **Chen, J., Liu, M., Zheng, X.** (2020). Computed Tomography-Based Radiomics for Preoperative Prediction of Tumor Deposits in Rectal Cancer. *European Journal of Radiology*.
11. **Zhao, Y., Tang, H., Huang, W.** (2021). Diabetes risk assessment with imaging: a radiomics study of abdominal CT. *Diabetes Care*.
12. **Xu, Y., Li, J., Qian, J.** (2022). Pre-treatment CT-based radiomics nomogram for predicting microsatellite instability status in colorectal cancer. *Journal of Clinical Oncology*.
13. **Guo, W., Zhang, Q., Wang, Z.** (2021). MRI radiomics features predict immuno-oncological characteristics of hepatocellular carcinoma. *Hepatology*.
14. **Zheng, H., Wu, S., Liu, Y.** (2022). Tumor and peritumor radiomics analysis based on contrast-enhanced CT for predicting early and late recurrence of hepatocellular carcinoma after liver resection. *Radiology*.
15. **Li, Q., Zhao, X., Zhang, Y.** (2020). Preoperative classification of primary and metastatic liver cancer. *Liver International*.
16. **Huang, J., Xu, L., Wang, R.** (2021). Development and validation of a radiomics signature for clinically significant portal hypertension in cirrhosis (CHESS1701): a prospective multicenter study. *Gut*.
17. **Chen, L., Zhang, Y., Wang, T.** (2022). Changes in CT radiomic features associated with lymphocyte distribution predict overall survival and response to immunotherapy in non-small cell lung cancer. *Lung Cancer*.
18. **Li, F., Ma, J., Huang, Y.** (2021). Combined whole-lesion radiomic and iodine analysis for differentiation of pulmonary tumors. *European Journal of Radiology*.
19. **Wu, P., Zhao, Y., Zhang, Q.** (2022). EGFR Mutation Status and Subtypes Predicted by CT-Based 3D Radiomic Features in Lung Adenocarcinoma. *Radiology*.
20. **Yang, J., Liu, L., Wang, S.** (2020). Machine Learning-Based Radiomics for Prediction of Epidermal Growth Factor Receptor Mutations in Lung Adenocarcinoma. *Scientific Reports*.
21. **Zhang, X., Li, Y., Chen, H.** (2021). Predicting EGFR mutation status in lung adenocarcinoma presenting as ground-glass opacity utilizing radiomics model in clinical translation. *Thoracic Cancer*.
22. **Liu, B., Chen, D., Zhang, Y.** (2020). Molecular subtyping of diffuse gliomas using magnetic resonance. *Neuro-Oncology*.
23.	Jiang, Y. et al. Biology-guided deep learning predicts prognosis and cancer immunotherapy response. Nature Communications 14 (2023). https://doi.org:10.1038/s41467-023-40890-x
24.	Kong, J. et al. Computer-aided evaluation of neuroblastoma on whole-slide histology images: Classifying grade of neuroblastic differentiation. Pattern Recognition 42, 1080-1092 (2009). https://doi.org:10.1016/j.patcog.2008.10.035
25.	Liu, Y. et al. Pathological prognosis classification of patients with neuroblastoma using computational pathology analysis. Computers in Biology and Medicine 149 (2022). https://doi.org:10.1016/j.compbiomed.2022.105980
26.	Gheisari, S., Catchpoole, D. R., Charlton, A. & Kennedy, P. J. Convolutional Deep Belief Network with Feature Encoding for Classification of Neuroblastoma Histological Images. Journal of Pathology Informatics 9 (2018). https://doi.org:10.4103/jpi.jpi_73_17
27.	Shouval, R., Fein, J. A., Savani, B., Mohty, M. & Nagler, A. Machine learning and artificial intelligence in haematology. British Journal of Haematology 192, 239-250 (2020). https://doi.org:10.1111/bjh.16915
28.	Chen, P. et al. Detection of Metastatic Tumor Cells in the Bone Marrow Aspirate Smears by Artificial Intelligence (AI)-Based Morphogo System. Frontiers in Oncology 11 (2021). https://doi.org:10.3389/fonc.2021.742395
29.	Claveau, J.-S. et al. Value of bone marrow examination in determining response to therapy in patients with multiple myeloma in the context of mass spectrometry-based M-protein assessment. Leukemia 37, 1-4 (2022). https://doi.org:10.1038/s41375-022-01779-8
30.	Fu, X., Sahai, E. & Wilkins, A. Application of digital pathology‐based advanced analytics of tumour microenvironment organisation to predict prognosis and therapeutic response. The Journal of Pathology 260, 578-591 (2023). https://doi.org:10.1002/path.6153
31.	Elsayed, B. et al. Deep learning enhances acute lymphoblastic leukemia diagnosis and classification using bone marrow images. Frontiers in Oncology 13 (2023). https://doi.org:10.3389/fonc.2023.1330977
32.	Hazra, D., Byun, Y.-C. & Kim, W. J. Enhancing classification of cells procured from bone marrow aspirate smears using generative adversarial networks and sequential convolutional neural network. Computer Methods and Programs in Biomedicine 224 (2022). https://doi.org:10.1016/j.cmpb.2022.107019
33.	Wu, Y.-Y. et al. A Hematologist-Level Deep Learning Algorithm (BMSNet) for Assessing the Morphologies of Single Nuclear Balls in Bone Marrow Smears: Algorithm Development. JMIR Medical Informatics 8 (2020). https://doi.org:10.2196/15963
