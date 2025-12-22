# BAB I PENDAHULUAN

## 1.1 Latar Belakang

Depresi adalah salah satu gangguan kesehatan mental yang paling umum, dan prevalensinya terus meningkat di berbagai belahan dunia, termasuk Indonesia. Menurut laporan ((WHO), 2025) pada bulan Agustus, diperkirakan sekitar 332 juta orang di dunia mengalami depresi, dengan prevalensi global sekitar 4% dari populasi. Wanita memiliki angka prevalensi yang lebih tinggi (6,9%) dibandingkan pria (4,6%). Data dari National Health and Nutrition Examination Survey (NHANES) menunjukkan bahwa prevalensi depresi pada remaja dan dewasa usia 12 tahun ke atas meningkat dari 8,2% pada 2013--2014 menjadi 13,1% pada Agustus 2023. Peningkatan ini lebih signifikan pada perempuan, dengan prevalensi mencapai 26,5% pada remaja perempuan usia 12--19 tahun.

Depresi sering kali berkaitan dengan isolasi sosial dan peningkatan risiko ideasi bunuh diri, sehingga menjadikannya masalah kesehatan masyarakat yang sangat mendesak (Kim & Lee, 2022). Dampak depresi terhadap kualitas hidup tercermin dalam temuan (E. D. Lima Filho et al., 2024), yang menyatakan bahwa penderita depresi mengalami kesulitan dalam menjalani aktivitas sehari-hari, termasuk berinteraksi dengan orang lain dan bekerja. Depresi tidak hanya berdampak pada aspek psikologis, tetapi juga fisik dan sosial, yang meningkatkan ketergantungan pada layanan kesehatan, baik dari sisi perawatan medis maupun dukungan psikososial (S. Bilu et al., 2023).

Kesadaran tentang kesehatan mental semakin meningkat, namun stigma dan keterbatasan akses terhadap layanan kesehatan masih menjadi hambatan besar dalam penanganan depresi di Indonesia. Oleh karena itu, deteksi dini depresi sangat penting untuk mencegah dampak jangka panjang terhadap kualitas hidup, baik dalam aspek emosional maupun fisik. Deteksi yang lebih baik dapat membantu merancang intervensi yang lebih tepat sasaran dan dapat meningkatkan kualitas hidup.

Dengan berkembangnya teknologi, penggunaan machine learning untuk mendeteksi depresi semakin mendapatkan perhatian, khususnya melalui analisis data aktivitas motorik dari wearable sensors (aktigrafi). Penelitian menunjukkan bahwa pola aktivitas fisik dan ritme sirkadian dapat menjadi biomarker objektif untuk depresi, menggantikan penilaian subjektif tradisional. Meskipun banyak penelitian yang telah menggunakan teknik pembelajaran mesin untuk mengklasifikasikan depresi, tantangan utama yang dihadapi adalah **ketidakseimbangan data** (imbalanced data) antara kelompok depresi dan kontrol, serta belum ada konsensus yang jelas mengenai model dan strategi penanganan imbalance terbaik.

Meskipun banyak model pembelajaran mesin (machine learning/ML) telah dikembangkan untuk mengklasifikasikan berbagai gangguan mental, termasuk depresi, masih terbatas studi yang secara langsung membandingkan performa algoritma Decision Tree dan Naive Bayes dalam konteks ini. Decision Tree merupakan salah satu algoritma yang sering digunakan karena kemampuannya dalam menangani data dengan struktur non-linear dan kemudahan dalam menginterpretasikan hasilnya. Dalam Decision Tree, model bekerja dengan cara melakukan pemisahan data (splitting) berdasarkan fitur-fitur yang memberikan informasi terbaik untuk membedakan kelas-kelas yang ada. Proses ini membuat Decision Tree sangat efektif pada data yang memiliki interaksi antar fitur dan ketika fitur-fitur memiliki pengaruh non-linier terhadap hasil klasifikasi (E. D. Lima Filho et al., 2024).

Decision Tree adalah algoritma pembelajaran berbasis pohon keputusan yang memecah data dengan menggunakan kriteria yang ditentukan oleh fitur-fitur yang ada untuk membuat keputusan. Proses pembelajaran ini memungkinkan model untuk secara visual menyajikan keputusan dengan cara yang dapat dijelaskan, yang menjadikannya pilihan yang sangat baik dalam konteks medis di mana interpretabilitas sangat penting (E. D. Lima Filho et al., 2024).

Gaussian Naive Bayes adalah varian dari algoritma Naive Bayes yang digunakan untuk klasifikasi data kontinu dengan anggapan bahwa fitur dalam data mengikuti distribusi normal (Gaussian). Dalam metode ini, setiap fitur dihitung berdasarkan rata-rata dan standar deviasi dari data pelatihan, yang kemudian digunakan untuk menghitung probabilitas kelas pada data baru. Algoritma Naive Bayes sendiri berlandaskan pada
teorema Bayes, yang memperkirakan probabilitas suatu kelas berdasarkan fitur yang ada. Meskipun asumsi independensi antar fitur dalam data tidak selalu tepat, Gaussian Naive Bayes tetap efektif berkat kesederhanaannya dan kemampuannya dalam menangani dataset besar dan tidak seimbang. Algoritma ini sangat efisien dalam pelatihan dan pengujian, sehingga cocok untuk aplikasi yang memerlukan model cepat dan mudah diskalakan. Dalam kasus klasifikasi tingkat depresi, Gaussian Naive Bayes dapat digunakan untuk memprediksi tingkat keparahan depresi berdasarkan fitur numerik yang telah diproses, dengan hasil yang seringkali cukup baik meskipun hubungan antara fitur-fitur tersebut bisa sangat kompleks.

Penelitian terbaru oleh (Vyshnavi & Saravanan, 2025) menunjukkan bahwa baik Decision Tree maupun Naive Bayes dapat memberikan hasil yang baik dalam klasifikasi tingkat depresi, namun perbandingan antara keduanya dalam konteks dataset klinis masih jarang dibahas secara mendalam. Penelitian ini menjadikan perbandingan kedua algoritma tersebut sebagai fokus utama, dengan tujuan untuk mengeksplorasi keunggulan dan kelemahan masing-masing dalam klasifikasi tingkat depresi. Mengingat pentingnya akurasi dalam klasifikasi depresi untuk intervensi yang lebih tepat, penelitian ini berupaya untuk mengisi celah yang ada dalam literatur dengan melakukan perbandingan yang lebih terperinci antara Decision Tree dan Naive Bayes, dengan menggunakan dataset klinis yang relevan.

Penelitian ini menggunakan **data aktivitas motorik dari sensor aktigrafi** yang merekam pergerakan tubuh secara kontinyu 24/7. Dataset terdiri dari **55 subjek** (23 depresi, 32 kontrol sehat) dengan ketidakseimbangan rasio 1:1.39. Untuk menangani ketidakseimbangan ini, penelitian ini membandingkan **5 strategi penanganan imbalanced data**: Original (tanpa penanganan), Class Weights, SMOTE (Synthetic Minority Over-sampling Technique), ADASYN (Adaptive Synthetic Sampling), dan kombinasi SMOTE+Weights. 

Pendekatan berbasis **Decision Tree** dan **Gaussian Naive Bayes** memungkinkan identifikasi pola depresi berdasarkan **73 fitur** yang diekstrak dari time series aktivitas, meliputi fitur statistik, temporal (hourly patterns), tidur, circadian rhythm, dan pola aktivitas. Penelitian ini menawarkan kebaruan dalam **perbandingan sistematis strategi imbalance handling** pada data aktigrafi, dengan evaluasi menggunakan multiple metrics: accuracy, precision, recall, F1-score, dan AUC-ROC. 

Hasil penelitian ini berpotensi memberikan kontribusi signifikan dalam: (1) objective depression screening berbasis wearable technology, (2) identifikasi strategi optimal untuk menangani imbalanced clinical data, dan (3) pemahaman biomarker temporal dan sirkadian sebagai indikator depresi.

## 1.2 Research Gap (Kesenjangan Penelitian)

Meskipun terdapat kemajuan signifikan dalam penggunaan machine learning untuk deteksi depresi, beberapa **kesenjangan penelitian** yang mendasari penelitian ini meliputi:

### Gap 1: Limited Systematic Comparison of Imbalance Handling Strategies

**Permasalahan:** Sebagian besar penelitian aktigrafi untuk deteksi depresi menggunakan **single strategy** untuk menangani imbalanced data atau mengabaikan isu imbalance sama sekali (Burton et al., 2013; Jakobsen et al., 2020). Padahal, **class imbalance** (rasio pasien vs kontrol yang tidak seimbang) adalah **characteristic inherent** dari clinical datasets yang dapat menyebabkan **majority class bias** dan **poor minority class detection**.

**Evidence:** Review komprehensif oleh Rashid et al. (2023) pada machine learning approaches untuk imbalanced medical datasets menunjukkan bahwa choice of imbalance strategy dapat mengubah performance hingga 15-25%, namun **belum ada studi sistematis** yang membandingkan multiple strategies (Original, Class Weights, SMOTE, ADASYN, Hybrid) pada **data aktigrafi depresi** secara komprehensif.

**Dampak:** Tanpa perbandingan sistematis, peneliti dan praktisi tidak memiliki guidance tentang **which strategy works best** untuk sensor-based depression data dengan imbalance ratio tertentu.

### Gap 2: Insufficient Algorithm Comparison on Actigraphy Data

**Permasalahan:** Decision Tree dan Gaussian Naive Bayes memiliki **fundamental differences** dalam assumptions dan mechanisms (DT: non-parametric, captures interactions vs GNB: parametric, assumes independence). Namun, **direct head-to-head comparison** pada **actigraphy data** dengan controlled experimental setup masih sangat terbatas.

**Evidence:** Beberapa studi membandingkan algoritma pada medical data umum (Ramadhani & Gultom, 2022) atau pada accelerometer data untuk activity recognition, tetapi **tidak ada studi** yang specifically address: "Which algorithm is superior untuk depression detection dari actigraphy, dan mengapa?" dengan controlled experimental setup pada data depresi klinis.

**Dampak:** Algorithm selection sering arbitrary atau based on "popular choice" tanpa empirical justification untuk specific data characteristics (temporal patterns, circadian features, feature correlations) yang unique pada depression actigraphy.

### Gap 3: Lack of Feature-Level Interpretation for Clinical Translation

**Permasalahan:** Banyak studi ML untuk depression detection mencapai high accuracy tetapi **tidak mengeksplorasi deeply** "which features/patterns actually discriminate depression?" dengan analisis feature-level yang mendalam untuk clinical translation.

**Evidence:** Systematic review oleh Chen et al. (2023) mengidentifikasi behavioral patterns (reduced activity, altered circadian) tetapi **tidak quantify** exact temporal features dan circadian metrics yang most discriminative menggunakan ML feature importance analysis untuk panduan klinis. Studi existing fokus pada overall patterns tanpa granular feature-level analysis yang dapat ditranslasikan ke clinical monitoring protocols.

**Dampak:** Tanpa feature-level insights, **clinical translation** terhambat - clinicians tidak tahu "what specifically to look for" dalam actigraphy data, dan model tetap "black box" meskipun using interpretable algorithms.

### Gap 4: Algorithm-Strategy Interaction Effects Unexplored

**Permasalahan:** Imbalance strategies may interact differently dengan algorithms (e.g., SMOTE dapat beneficial untuk DT tetapi minimal effect untuk NB karena probability estimation nature - Batista et al., 2004). Namun, **interaction effects** antara algorithm choice dan imbalance strategy choice **rarely examined systematically**.

**Evidence:** Batista et al. (2004) hint pada interaction tetapi dengan limited algorithms dan strategies. Fernández et al. (2018) menyatakan effectiveness bergantung pada "dataset characteristics dan algorithm" tetapi **tidak provide empirical mapping** untuk specific scenarios.

**Dampak:** Suboptimal model selection - peneliti mungkin menggunakan kombinasi algoritma-strategi yang suboptimal karena lack of guidance tentang interaction effects, berpotensi menghasilkan performa yang jauh di bawah kombinasi optimal yang mungkin dicapai.

## 1.3 Novelty dan Kontribusi Penelitian

Penelitian ini menawarkan **kebaruan dan kontribusi** yang mengisi
research gaps di atas:

### Novelty 1: Systematic Multi-Strategy Comparison Framework

**Kebaruan:** Penelitian ini adalah **studi komprehensif pertama** yang membandingkan **5 strategi penanganan imbalance** (Original, Class Weights, SMOTE, ADASYN, SMOTE+Weights) secara sistematis pada **2 algoritma** (DT, GNB), menghasilkan **10 varian model** yang dievaluasi dengan **cross-validation yang ketat** dan **pengujian statistik** (Friedman, Wilcoxon).

**Kontribusi:** Menyediakan **framework evaluasi komprehensif** untuk pemilihan strategi pada data depresi berbasis sensor dengan karakteristik imbalance sedang. Framework ini mencakup: (1) protokol eksperimental terkontrol untuk fair comparison, (2) multiple evaluation metrics untuk capture different aspects of performance, (3) statistical validation untuk ensure reliability, dan (4) panduan berbasis bukti empiris yang dapat diterapkan untuk penelitian sensor-based mental health di masa depan.

### Novelty 2: Algorithm-Specific Performance Analysis on Actigraphy

**Kebaruan:** **Perbandingan empiris langsung** antara DT dan GNB pada data aktigrafi depresi dengan **validasi biologis** melalui analisis pola aktivitas. Penelitian ini **menjelaskan secara eksplisit mekanisme** yang mendasari perbedaan performa melalui: (1) analisis pelanggaran asumsi algoritma (independensi, Gaussian distribution), (2) kesesuaian kapabilitas algoritma dengan karakteristik data (pola temporal, feature interactions), (3) validasi terhadap known phenomenology of depression.

**Kontribusi:** **Framework justifikasi algoritmik** yang menggabungkan computational reasoning dan biological plausibility, bukan hanya perbandingan empiris. Framework ini membantu peneliti memahami **trade-offs** antara model complexity, interpretability, dan performance untuk different deployment scenarios: clinical diagnosis (high-stakes), population screening (scalability), atau continuous monitoring (computational efficiency).

### Novelty 3: Feature-Level Clinical Insights

**Kebaruan:** Penelitian ini **melampaui metrik akurasi** dengan menyediakan framework interpretasi komprehensif: (1) **Analisis peringkat kepentingan fitur** untuk identifikasi biomarker objektif, (2) **Visualisasi pola temporal** untuk validasi fenomenologi klinis, (3) **Ekstraksi aturan keputusan** yang dapat diterjemahkan ke panduan klinis, (4) **Validasi biologis** terhadap teori circadian rhythm disruption dalam depresi.

**Kontribusi:** **Framework translasional** untuk praktik klinis:
- **Metodologi identifikasi biomarker**: Pendekatan sistematis untuk mengekstrak temporal dan circadian features yang discriminative
- **Quantifiable decision thresholds**: Pembentukan threshold kuantitatif dari decision rules untuk clinical decision support
- **Clinical monitoring protocol**: Panduan berbasis bukti tentang temporal windows dan circadian metrics yang perlu dimonitor
- Memungkinkan pengembangan **alat pendukung keputusan klinis yang dapat diinterpretasi** dengan biological plausibility

### Novelty 4: Algorithm-Strategy Interaction Mapping

**Kebaruan:** Penelitian ini **secara eksplisit menguji efek interaksi** antara algoritma dan strategi imbalance melalui:
- **Factorial experimental design**: Pengujian sistematis 2 algoritma × 5 strategi dengan controlled setup
- **Statistical validation framework**: Penggunaan non-parametric tests (Friedman, Wilcoxon) untuk memvalidasi signifikansi perbedaan
- **Interaction effect analysis**: Pemeriksaan apakah benefit dari imbalance handling berbeda across algorithms
- **Comparative effectiveness assessment**: Evaluasi hybrid methods vs single methods pada karakteristik data spesifik

**Kontribusi:** **Decision framework** untuk pemilihan model:
- **Algorithm-strategy compatibility matrix**: Kerangka kerja untuk menentukan kombinasi optimal berdasarkan karakteristik data dan requirements aplikasi
- **Trade-off analysis**: Panduan tentang balance antara performance, computational efficiency, dan interpretability
- **Evidence-based selection criteria**: Kriteria pemilihan berdasarkan empirical evidence, bukan arbitrary choice
- **Relative importance quantification**: Metodologi untuk mengukur kontribusi relatif algorithm choice vs strategy choice dalam overall performance

### Novelty 5: Methodological Rigor and Reproducibility

**Kebaruan:** Penelitian ini menerapkan **best practices** yang sering diabaikan:
- **Stratified splitting** untuk preserve class distribution
- **Pipeline feature selection** (variance threshold → correlation filter → SelectKBest) to avoid overfitting
- **Nested evaluation** (5-fold CV + independent test set)
- **Multiple metrics** (accuracy, precision, recall, F1, AUC) untuk comprehensive assessment
- **Statistical testing** untuk validate significance
- **Biological validation** untuk ensure clinical plausibility

**Kontribusi:** **Template metodologi yang dapat direproduksi** untuk penelitian kesehatan mental berbasis sensor di masa depan. Kode dan pipeline dapat diadaptasi untuk kondisi lain (kecemasan, bipolar) atau sensor lain (smartphone, smartwatch).

### Expected Impact

Penelitian ini diharapkan memberikan dampak pada:

1. **Komunitas Ilmiah**: Basis bukti untuk penanganan imbalance dan pemilihan algoritma pada data sensor perilaku
2. **Praktik Klinis**: Biomarker objektif dan aturan keputusan untuk skrining depresi menggunakan teknologi wearable
3. **Pengembangan Teknologi**: Panduan untuk mengembangkan aplikasi monitoring kesehatan mental berbasis ML yang dapat diinterpretasi
4. **Pembuatan Kebijakan**: Bukti pendukung untuk mengintegrasikan penilaian objektif berbasis wearable dalam sistem perawatan kesehatan mental

## 1.4 Rumusan Masalah

Berdasarkan latar belakang tersebut, rumusan masalah yang akan diteliti dalam penelitian ini adalah sebagai berikut:

1. Bagaimana perbandingan performa antara algoritma **Decision Tree** dan **Gaussian Naive Bayes** dalam mengklasifikasikan depresi berdasarkan data aktivitas motorik dari sensor aktigrafi?

2. Bagaimana pengaruh **strategi penanganan data tidak seimbang** (Class Weights, SMOTE, ADASYN, SMOTE+Weights) terhadap performa kedua algoritma dalam mengklasifikasikan depresi?

3. Kombinasi algoritma dan strategi imbalance handling mana yang memberikan performa **optimal** berdasarkan metrik evaluasi (accuracy, precision, recall, F1-score, AUC-ROC)?

4. Fitur temporal dan circadian apa saja yang paling **penting** dalam membedakan individu dengan depresi dari individu sehat?

## 1.5 Batasan Masalah

1. Penelitian ini hanya difokuskan pada perbandingan **dua algoritma klasifikasi**, yaitu **Decision Tree** dan **Gaussian Naive Bayes**, dengan **5 strategi penanganan imbalanced data** (Original, Class Weights, SMOTE, ADASYN, SMOTE+Weights), sehingga total **10 model** yang dibandingkan. Penelitian tidak melibatkan algoritma lain seperti Random Forest, SVM, atau Deep Learning.

2. Dataset yang digunakan adalah **data aktivitas motorik dari sensor aktigrafi** yang terdiri dari **55 subjek** (23 condition/depresi, 32 control/sehat). Dataset bersumber dari penelitian sebelumnya dan telah tersedia dalam format CSV dengan time series aktivitas per menit.

3. Penelitian ini **hanya menggunakan data aktivitas motorik** dari wearable actigraph sensor, **tidak mencakup** data lain seperti:
   - Data teks (wawancara, kuesioner, media sosial)
   - Data fisiologis (EEG, ECG, fMRI)
   - Data smartphone (GPS, screen time, communication patterns)
    - Data voice/speech analysis

4. **Fitur yang diekstrak** terbatas pada **73 fitur** yang mencakup: statistik deskriptif, temporal patterns (hourly), sleep features, circadian rhythm metrics, dan activity patterns. Penelitian tidak mengeksplorasi feature engineering lanjutan seperti deep features atau wavelet transforms.

5. Evaluasi model menggunakan **stratified 80-20 split** (44 training, 11 test) dengan **5-fold stratified cross-validation** pada training set. Penelitian tidak menggunakan teknik validasi lain seperti leave-one-out cross-validation atau nested cross-validation.

6. **Metrik evaluasi** mencakup: accuracy, precision, recall, F1-score (macro), AUC-ROC, confusion matrix, dan statistical tests (Friedman, Wilcoxon). Penelitian tidak mengevaluasi computational cost, training time, atau model complexity secara mendalam.

7. Model **Decision Tree** menggunakan **default hyperparameters** dengan batasan: max_depth=5, min_samples_split=10, min_samples_leaf=5 untuk mencegah overfitting. **Gaussian Naive Bayes** menggunakan default scikit-learn tanpa hyperparameter tuning ekstensif.

8. Penelitian fokus pada **klasifikasi binary** (depresi vs tidak depresi), **tidak mencakup**:
   - Multi-class classification (tingkat severitas depresi: mild, moderate, severe)
    - Regression (prediksi skor MADRS/HDRS kontinyu)
    - Time series forecasting (prediksi trajectory depresi)

9. **Generalizability** penelitian terbatas pada karakteristik dataset spesifik ini (kemungkinan populasi Eropa/Skandinavia, setting klinis tertentu, periode waktu tertentu). Validasi eksternal pada populasi, device, atau setting berbeda belum dilakukan.

10. Penelitian ini merupakan **retrospective analysis** pada data yang sudah ada, bukan **prospective clinical trial**. Implementasi real-world dan clinical utility belum diuji.

## 1.6 Tujuan Penelitian

1. **Membandingkan performa** algoritma **Decision Tree** dan **Gaussian Naive Bayes** dalam mengklasifikasikan depresi berdasarkan data aktivitas motorik dari sensor aktigrafi, dengan evaluasi menggunakan multiple metrics (accuracy, precision, recall, F1-score, AUC-ROC).

2. **Menganalisis pengaruh** 5 strategi penanganan data tidak seimbang (**Class Weights**, **SMOTE**, **ADASYN**, **SMOTE+Weights**, dan **Original**) terhadap performa klasifikasi pada kedua algoritma.

3. **Mengidentifikasi kombinasi optimal** antara algoritma dan strategi imbalance handling yang memberikan performa terbaik dalam mengklasifikasikan depresi, dengan validasi menggunakan cross-validation dan statistical tests.

4. **Mengekstrak dan mengidentifikasi** fitur-fitur penting dari data aktivitas motorik (statistik, temporal, sleep, circadian, activity patterns) yang paling berkontribusi dalam membedakan individu dengan depresi dari individu sehat.

5. **Memvalidasi temuan** melalui analisis feature importance, activity pattern visualization, dan interpretasi decision rules untuk memastikan biological plausibility dan clinical relevance dari model.

# BAB II TINJAUAN PUSTAKA

## 2.1 Kajian Pustaka

### 2.1.1 Aktivitas Motorik sebagai Biomarker Depresi

Depresi adalah gangguan kesehatan mental yang mempengaruhi kualitas hidup individu secara global, dengan kriteria diagnostik yang diatur dalam DSM-V, yang mencakup gejala seperti perasaan sedih mendalam, kehilangan minat, gangguan tidur, dan perasaan tidak berharga (Haroz et al., 2022). Salah satu manifestasi fisik yang signifikan dari depresi adalah **perubahan pola aktivitas motorik** dan **gangguan ritme sirkadian** yang dapat diukur secara objektif menggunakan sensor aktigrafi.

Penelitian oleh **Xu et al. (2024)** menunjukkan bahwa data aktivitas motorik dari wearable sensors dapat digunakan untuk mendeteksi depresi dengan akurasi yang tinggi menggunakan pendekatan machine learning. Mereka menggunakan wearable sensor untuk merekam behavioral markers dan berhasil memprediksi severitas depresi dengan performa tinggi. Studi ini menekankan pentingnya **temporal features** dan **circadian patterns** sebagai indikator depresi.

**Chen et al. (2023)** dalam systematic review dan meta-analysis mereka yang berjudul "Actigraphy-based Depression Detection Using Deep Learning" menganalisis berbagai studi yang menggunakan aktigrafi untuk monitoring depresi. Mereka menemukan bahwa pasien depresi menunjukkan: (1) **reduced total activity**, (2) **altered circadian rhythms**, (3) **increased nocturnal activity** (sleep fragmentation), dan (4) **blunted morning activity** (delayed activation). Temuan ini menunjukkan bahwa aktigrafi merupakan tool objektif yang valid untuk assessing motor activity dalam depresi.

### 2.1.2 Machine Learning untuk Klasifikasi Depresi pada Data Aktigrafi

Menurut penelitian **Tazawa et al. (2022)**, yang menggunakan data dari multimodal wristband-type wearable device pada pasien depresi mayor, algoritma machine learning dapat mencapai akurasi tinggi dalam membedakan pasien depresi dari kontrol sehat dan menilai severitas pasien. Penelitian ini mengekstrak features seperti mean activity, variance, circadian amplitude, dan intradaily variability, yang semuanya menunjukkan perbedaan signifikan antara kelompok depresi dan kontrol.

Penelitian oleh (Shin et al., 2022), mengembangkan model berbasis teks menggunakan Naive Bayes untuk mendeteksi depresi dan risiko bunuh diri melalui wawancara klinis. Model ini menunjukkan hasil yang signifikan dengan AUC sebesar 0.905, sensitivitas 0.699, dan spesifisitas 0.964, yang menunjukkan kemampuannya dalam mendeteksi depresi secara akurat berdasarkan kata-kata yang diucapkan oleh peserta selama wawancara. Pendekatan berbasis teks ini menunjukkan potensi besar sebagai penanda diagnostik objektif untuk depresi dan risiko bunuh diri.

(Saqib et al., 2021) dalam *"Machine Learning Methods for Predicting Postpartum Depression: Scoping Review"* mengkaji penggunaan berbagai metode machine learning untuk memprediksi depresi postpartum, dengan algoritma yang digunakan mencakup Naive Bayes, Random Forest, Decision Trees, dan lainnya. Studi ini mencatat bahwa nilai area under the curve (AUC) untuk algoritma yang diterapkan berkisar antara 0.78 hingga 0.93, yang menunjukkan variasi dalam performa model yang digunakan dalam memprediksi depresi postpartum (Saqib et al., 2021).

(Asma et al., 2024) membandingkan *Decision Tree* dan *Naive Bayes* untuk mendeteksi depresi menggunakan data teks multibahasa. Hasilnya, *Naive Bayes* lebih efisien dalam hal waktu dan memori, sementara *Decision Tree* lebih unggul dalam *interpretabilitas*, memudahkan pemahaman keputusan klasifikasi. Kedua algoritma ini memiliki kelebihan masing-masing, sesuai dengan kebutuhan analisis yang berbeda (Asma et al., 2024).

### 2.1.3 Imbalanced Data dalam Klasifikasi Medis

Ketidakseimbangan data (imbalanced data) merupakan tantangan umum dalam klasifikasi medis, termasuk deteksi depresi, di mana jumlah sampel kelas minority (pasien) seringkali lebih sedikit dibanding kelas majority (kontrol sehat). **He & Garcia (2009)** dalam "Learning from Imbalanced Data" menjelaskan bahwa standard machine learning algorithms cenderung bias terhadap majority class, menghasilkan high accuracy tetapi poor sensitivity untuk minority class detection.

**Chawla et al. (2002)** memperkenalkan **SMOTE (Synthetic Minority
Over-sampling Technique)**, teknik oversampling yang membuat synthetic samples di antara existing minority samples. Penelitian mereka menunjukkan SMOTE meningkatkan F1-score 15-20% dibanding random oversampling pada berbagai datasets. Namun, SMOTE menghasilkan synthetic samples secara uniform tanpa mempertimbangkan kesulitan klasifikasi di region tertentu.

**He et al. (2008)** mengembangkan **ADASYN (Adaptive Synthetic Sampling)**, improvement dari SMOTE yang secara adaptif men-generate lebih banyak synthetic samples di regions yang "difficult to learn" (berdekatan dengan decision boundary). Penelitian mereka pada 15 imbalanced datasets menunjukkan ADASYN outperform SMOTE dengan rata-rata peningkatan 3-8% pada G-mean dan AUC.

**Fernández et al. (2018)** dalam "Learning from Imbalanced Data Sets" comprehensive book membahas berbagai strategi: (1) **Data-level methods** (SMOTE, ADASYN, undersampling), (2) **Algorithm-level methods** (cost-sensitive learning, class weights), dan (3) **Hybrid methods** (kombinasi keduanya). Mereka menyimpulkan bahwa tidak ada "best strategy" universal - effectiveness bergantung pada characteristics dataset dan algorithm yang digunakan.

### 2.1.4 Aplikasi Imbalanced Data Handling pada Data Medis

**Rashid et al. (2023)** dalam comprehensive review "Machine Learning Approaches for Imbalanced Medical Datasets" mengevaluasi berbagai strategi imbalance pada medical datasets. Mereka menemukan SMOTE variants (borderline-SMOTE, ADASYN) consistently outperform basic SMOTE, terutama pada high imbalance ratios (>1:5). Untuk medical datasets, **class weights** kombinasi dengan **oversampling** menunjukkan hasil terbaik.

**Batista et al. (2004)** membandingkan various balancing techniques dengan Decision Tree dan Naive Bayes. Hasil menunjukkan: (1) SMOTE + Decision Tree meningkatkan sensitivity 25% vs baseline, (2) Naive Bayes benefit minimal dari oversampling (< 5% improvement) karena probability estimation nature, dan (3) Hybrid methods (SMOTE + class weights) memberikan best trade-off antara sensitivity dan specificity.

Penelitian perbandingan algoritma pada sensor data menunjukkan bahwa **Decision Tree** umumnya mengungguli **Naive Bayes** dalam menangkap **temporal patterns** dan **feature interactions** dalam sensor data, karena Naive Bayes violated **independence assumption** ketika sensor readings berkorelasi tinggi across time dan axes (Ramadhani & Gultom, 2022).

**Zhang et al. (2022)** dalam "Predicting Depressive Symptom Severity Through Individuals' Nearby Bluetooth Device Count Data" menggunakan smartphone sensor data untuk memprediksi depression symptoms. Mereka menemukan bahwa **behavioral patterns** yang terekam dari device interactions berkorelasi dengan depression severity. **Wang et al. (2023)** dalam CrossCheck study juga validate bahwa **temporal and behavioral patterns** dari passive sensing merupakan valid markers untuk mental health changes.

### 2.1.5 Circadian Rhythm Disruption dalam Depresi

**De Crescenzo et al. (2022)** dalam systematic review dan meta-analysis "Actigraphic Features of Bipolar Disorder" menjelaskan bahwa **circadian rhythm disruption** adalah core feature dari mood disorders termasuk major depression. Manifestasi meliputi: (1) **phase delay** (shifted sleep-wake timing), (2) **reduced amplitude** (blunted day-night difference), (3) **increased fragmentation** (irregular patterns), dan (4) **altered acrophase** (peak activity timing). Intervensi yang target circadian system (light therapy, sleep deprivation) menunjukkan antidepressant effects yang rapid dan substantial.

Penelitian terbaru menguatkan bahwa pasien dengan depresi menunjukkan: (1) **phase delay** dalam sleep onset, (2) **reduced amplitude** dalam ritme aktivitas, (3) **irregular patterns** dengan higher intradaily variability, dan (4) **blunted morning activity**. Temuan ini memperkuat pentingnya **circadian features** dalam machine learning models untuk depression detection (Chen et al., 2023; Tazawa et al., 2022).

### Tabel 2.1: Ringkasan Kajian Pustaka Relevan

| **Penulis** | **Tahun** | **Tema/Judul** | **Data/Metode** | **Hasil Utama** |
|-------------|-----------|----------------|-----------------|-----------------|
| **Burton et al.** | 2013 | Activity Monitoring in Patients with Depression: A Systematic Review | Systematic review 22 studi menggunakan aktigrafi untuk monitoring depresi | Pasien depresi menunjukkan: (1) reduced total activity, (2) altered circadian rhythms, (3) increased nocturnal activity, (4) blunted morning activity. Aktigrafi valid tool untuk objective assessment. |
| **Garcia-Ceja et al.** | 2018 | Depression Detection Using Smartphone Accelerometer Data | Smartphone accelerometer, Random Forest, temporal & circadian features | Akurasi 91% dalam klasifikasi depresi. Temporal features dan circadian patterns adalah indikator kunci. |
| **Jakobsen et al.** | 2020 | Detecting Depression Using Actigraphy and Machine Learning | Data aktigrafi pasien depresi mayor, Decision Tree, SVM | Akurasi 85-90%. Features: mean activity, variance, circadian amplitude, intradaily variability menunjukkan perbedaan signifikan antara depresi dan kontrol. |
| **Canzian & Musolesi** | 2015 | Trajectories of Depression: Unobtrusive Monitoring via Smartphone Mobility Traces | GPS & accelerometer, location entropy, circadian movement patterns | Location entropy dan circadian movement patterns berkorelasi kuat dengan severitas depresi (r=-0.68). Objective behavioral data valid proxy untuk mental health assessment. |
| **Hickie et al.** | 2013 | Manipulating Sleep-Wake Cycle and Circadian Rhythms in Major Depression | Review circadian rhythm disruption sebagai core feature depresi | Manifestasi: phase delay, reduced amplitude, increased fragmentation, altered acrophase. Circadian-targeted interventions (light therapy) menunjukkan rapid antidepressant effects. |
| **Robillard et al.** | 2018 | Sleep-Wake Cycle and Melatonin Rhythms in Adolescents with Mood Disorders | Aktigrafi & melatonin assays pada remaja dengan depresi | Remaja depresi: 2-3 hours phase delay, 40% reduced melatonin amplitude, irregular sleep-wake patterns, blunted cortisol awakening response. |
| **Chawla et al.** | 2002 | SMOTE: Synthetic Minority Over-sampling Technique | SMOTE untuk imbalanced datasets, berbagai klasifikasi tasks | SMOTE meningkatkan F1-score 15-20% vs random oversampling. Synthetic interpolation antara minority samples mengatasi class imbalance. |
| **He et al.** | 2008 | ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning | ADASYN pada 15 imbalanced datasets | ADASYN outperform SMOTE dengan 3-8% peningkatan pada G-mean dan AUC. Adaptive generation lebih banyak samples di "difficult to learn" regions. |
| **He & Garcia** | 2009 | Learning from Imbalanced Data | Comprehensive review imbalanced learning methods | Standard ML algorithms bias terhadap majority class. Data-level (sampling) dan algorithm-level (cost-sensitive) approaches keduanya effective. |
| **López et al.** | 2013 | Classification with Imbalanced Data: Empirical Results and Current Trends | Evaluasi 44 strategi imbalance pada 66 datasets | SMOTE variants (borderline-SMOTE, ADASYN) consistently outperform basic SMOTE pada high imbalance ratios (>1:5). |
| **Batista et al.** | 2004 | Balancing Techniques for Medical Data Classification | SMOTE, undersampling, hybrid methods dengan DT dan NB | SMOTE + Decision Tree meningkatkan sensitivity 25% vs baseline. Naive Bayes benefit minimal (<5%) dari oversampling. Hybrid methods best trade-off. |
| **Ramadhani & Gultom** | 2022 | Comparison of C4.5 and Naive Bayes for Medical Data Classification | Decision Tree (C4.5) vs Naive Bayes pada data medis | Decision Tree unggul dalam feature interactions dan non-linear relationships (89.2%) vs Naive Bayes (82.1%). NB independence assumption violated pada medical data. |
| **Salekin et al.** | 2018 | Passive Detection of Mental Health Status Through Smartphone-Based Digital Phenotyping | Smartphone sensor data, multiple ML algorithms | Tree-based models (DT, RF) outperform probabilistic models (NB, LR) dengan margin 8-12%. Superior dalam menangkap temporal patterns dan feature interactions. |
| **Maswadi et al.** | 2021 | Human Activity Classification Using Decision Tree and Naïve Bayes Classifiers | Accelerometer data, DT vs NB untuk activity recognition | Decision Tree 99.9% akurasi vs Naive Bayes 89.5%. DT lebih baik menangkap temporal patterns dan feature interactions dalam sensor data. |
| **Saeb et al.** | 2015 | Mobile Phone Sensor Correlates of Depressive Symptom Severity | Smartphone GPS & accelerometer untuk prediksi PHQ-9 scores | Circadian movement patterns (r=0.51), location entropy (r=-0.43), normalized entropy strongest predictors. Temporal dan spatial behavioral patterns valid depression markers. |
| **Fernández et al.** | 2018 | Learning from Imbalanced Data Sets (Book) | Comprehensive coverage: data-level, algorithm-level, hybrid methods | Tidak ada "best strategy" universal. Effectiveness bergantung pada dataset characteristics dan algorithm. Hybrid approaches often optimal untuk medical data. |

**Catatan Tabel:**
- **Warna Hijau** (baris 1-6): Penelitian aktigrafi dan sensor-based depression detection
- **Warna Biru** (baris 7-11): Penelitian imbalanced data handling methods  
- **Warna Oranye** (baris 12-16): Penelitian comparison DT vs NB dan behavioral data classification

## 2.2 Tinjauan Teoritis

### 2.2.1 Depresi dan Manifestasi Psikomotor

Depresi merupakan salah satu gangguan mental yang sangat umum terjadi di seluruh dunia dan memberikan dampak besar terhadap kualitas hidup individu serta fungsi sosial dan pekerjaan. Gangguan ini tidak sekadar perasaan sedih sesaat, melainkan kondisi suasana hati yang berkepanjangan yang memengaruhi aspek emosional, perilaku, dan kognitif seseorang, termasuk kemampuan untuk berkonsentrasi, menjalankan aktivitas sehari-hari, serta menjaga hubungan interpersonal. World Health Organization (WHO) menjelaskan bahwa depresi merupakan hasil interaksi kompleks antara faktor sosial, psikologis, dan biologis, dan kontribusi dari faktor-faktor ini dapat memperburuk kondisi seseorang setelah mengalami peristiwa hidup yang merugikan seperti kehilangan, stres berkepanjangan, atau trauma berat (WHO, 2022).

Salah satu gejala kunci depresi mayor yang sering terabaikan adalah **psychomotor changes** - perubahan pada aktivitas motorik dan gerakan tubuh. DSM-5 mengidentifikasi dua jenis: (1) **psychomotor retardation** (perlambatan gerakan, speech, dan reaksi), dan (2) **psychomotor agitation** (restlessness, pacing, hand-wringing). Penelitian menunjukkan 60-70% pasien depresi mayor mengalami psychomotor symptoms yang dapat diukur secara objektif melalui aktigrafi (Buyukdura et al., 2011).

Depresi sering ditandai oleh perasaan rendah diri, kehilangan minat terhadap aktivitas yang sebelumnya menyenangkan, gangguan tidur, perubahan nafsu makan, kelelahan, dan kesulitan dalam berkonsentrasi atau membuat keputusan, yang semuanya dapat mengurangi kemampuan fungsional individu secara signifikan. Kondisi ini tidak hanya mengganggu kesejahteraan psikologis, tetapi juga berdampak pada produktivitas dan peran sosial seseorang dalam masyarakat (R. P. Lima Filho & Oliveira, 2024).

Secara sosial, depresi dapat memperburuk hubungan interpersonal dan mengurangi keterlibatan dalam aktivitas komunitas, sehingga individu dengan depresi cenderung mengalami penurunan dukungan sosial yang justru memperkuat gejala mereka. Penelitian menunjukkan bahwa depresi berhubungan erat dengan disfungsi fungsional dalam berbagai domain kehidupan, termasuk keterbatasan dalam pekerjaan, aktivitas sosial, serta kehidupan keluarga, yang pada akhirnya mengurangi kualitas hidup secara keseluruhan (S. L. Bilu & al., 2023).

Kualitas hidup (*quality of life*) yang buruk bukan hanya mencerminkan penderitaan psikologis, tetapi juga dampak riil dalam fungsi sehari-hari---seperti kemampuan untuk bekerja, berpartisipasi dalam hubungan sosial, serta memenuhi tuntutan keluarga dan pekerjaan---yang semuanya dapat terhambat secara signifikan oleh gejala depresi yang tidak ditangani. Penelitian klinis juga melaporkan bahwa gejala inti depresi, seperti suasana hati yang rendah, tidak hanya berdampak pada kesejahteraan psikologis, tetapi juga memicu gangguan pada aktivitas fisik dan sosial yang akhirnya memperburuk tingkat fungsi individu secara umum (Assessment, 2023).

2.2.1.1 Faktor faktor yang mempengaruhi depresi

Menurut penelitian Singh dan Kaur (2024) menunjukkan bahwa depresi tidak hanya dipengaruhi oleh faktor biologis seperti ketidakseimbangan neurotransmiter di otak atau faktor genetik, tetapi juga dipengaruhi oleh tekanan psikologis dan faktor sosial. Stres kronis, pengalaman traumatis, dan isolasi sosial dapat memicu atau memperburuk depresi. Dalam konteks biologis, faktor-faktor seperti penurunan kadar serotonin dan dopamin telah lama dianggap sebagai penyebab utama depresi. Model ini juga menekankan pentingnya pengaruh lingkungan sosial, seperti keluarga, pekerjaan, dan hubungan sosial, yang dapat memperburuk atau memperbaiki kondisi depresi (Singh & Kaur, 2024).

Menurut penelitian Sigmund Freud, dalam teori psikoanalisisnya, menyatakan bahwa depresi sering kali merupakan hasil dari ketegangan emosional yang tidak terselesaikan yang berasal dari pengalaman masa kecil, seperti konflik dengan orang tua atau trauma masa kecil yang terinternalisasi. Freud berpendapat bahwa "anger turned inward" (kemarahan yang diarahkan ke dalam diri) adalah salah satu mekanisme yang mendasari depresi, di mana individu memproyeksikan perasaan marah terhadap diri mereka sendiri sebagai akibat dari pengalaman emosional yang terpendam. Meskipun teori ini lebih berfokus pada ketidaksadaran dan pengalaman masa kecil, juga mengakui bahwa pengaruh faktor lingkungan dan pola hubungan interpersonal dapat memperburuk kecenderungan untuk mengalami depresi. Teori ini, meskipun kontroversial dan kurang digunakan dalam terapi modern, tetap memberikan wawasan mengenai peran pengalaman masa kecil dan konflik emosional dalam perkembangan gangguan depresi (Freud, 2025).

Teori stres dan coping, yang dikembangkan oleh Lazarus dan Folkman (2024), menjelaskan bahwa depresi sering kali dipicu oleh stres kronis yang tidak dapat diatasi dengan cara-cara yang adaptif. Individu yang tidak memiliki strategi coping yang efektif untuk menangani stres atau masalah emosional berisiko lebih tinggi mengalami depresi. Stres yang datang dari berbagai sumber, seperti pekerjaan, masalah keluarga, atau kehilangan orang yang tercinta, dapat mengganggu keseimbangan emosional dan mempengaruhi kesehatan mental. Teori ini menekankan pentingnya mengembangkan keterampilan coping yang sehat, seperti mencari dukungan sosial, berfokus pada solusi, atau mengubah pola pikir, untuk mencegah dan mengatasi depresi (Lazarus & Folkman, 2024).

### 2.2.2 Aktigrafi sebagai Tool Objektif untuk Assessing Aktivitas Motorik

**Aktigrafi** adalah metode pengukuran aktivitas motorik menggunakan accelerometer-based devices yang dikenakan di pergelangan tangan atau pergelangan kaki. Device ini merekam accelerations dalam multiple axes, yang kemudian ditransformasikan menjadi **activity counts** - numerical representation dari movement intensity dan frequency. Aktigrafi telah divalidasi extensively sebagai objective measure untuk: (1) **sleep-wake patterns**, (2) **circadian rhythms**, (3) **daily activity levels**, dan (4) **physical activity patterns** (Ancoli-Israel et al., 2003).

Dalam konteks depresi, aktigrafi menawarkan beberapa keuntungan dibanding self-report measures: (1) **Objectivity** - tidak terpengaruh reporting bias atau recall bias, (2) **Continuity** - monitoring 24/7 dalam natural environment, (3) **Quantifiability** - data numerik yang dapat dianalisis secara statistik dan computational, dan (4) **Non-invasiveness** - tidak mengganggu aktivitas sehari-hari. Penelitian terbaru menunjukkan aktigrafi measures berkorelasi dengan depression severity dan dapat digunakan untuk screening dan assessing patient severity (Tazawa et al., 2022; De Crescenzo et al., 2022).

### 2.2.3 Machine Learning untuk Objective Depression Assessment

**Machine learning (ML)** dalam bidang psikologi, khususnya dalam deteksi depresi, telah berkembang pesat dalam beberapa tahun terakhir, seiring dengan kemajuan teknologi dan peningkatan ketersediaan data behavioral dari wearable sensors dan smartphones. Sebelumnya, deteksi depresi bergantung pada evaluasi klinis yang dilakukan oleh tenaga medis melalui structured interviews dan questionnaires, yang sering kali membutuhkan waktu, sumber daya, dan expertise yang besar. Machine learning menawarkan alternatif yang menjanjikan untuk **scalable**, **objective**, dan **continuous** depression monitoring (Mohr et al., 2022).

Penerapan *machine learning* dalam deteksi depresi menawarkan sejumlah keunggulan dibandingkan dengan metode konvensional. Salah satu keuntungan utamanya adalah *kecepatan* dalam menganalisis data dan memberikan hasil yang lebih konsisten dan objektif. Algoritma ini dapat mengidentifikasi pola-pola tersembunyi dalam data yang mungkin tidak terdeteksi oleh analisis manual atau evaluasi klinis tradisional. Selain itu, dengan kemampuan untuk menangani dataset yang besar dan beragam, machine learning memungkinkan deteksi dini depresi yang lebih efektif, bahkan sebelum gejala menjadi jelas atau parah. Hal ini sangat penting karena depresi yang didiagnosis lebih awal cenderung lebih mudah diobati dan dapat mencegah perkembangan kondisi yang lebih serius (Sabouri et al., 2023).

Dalam teori Health Informatics dan Computational Intelligence, machine learning (ML) dianggap sebagai pendekatan kognitif berbasis data yang mampu meningkatkan efisiensi, akurasi, dan kecepatan pengambilan keputusan dalam bidang kesehatan. Secara teoretis, ML beroperasi dengan membangun model prediktif yang belajar dari data historis untuk mengenali pola, menganalisis hubungan antar variabel klinis, dan memberikan keputusan berbasis bukti tanpa intervensi manusia secara langsung (Topol, 2019).

Dalam domain Clinical Decision Support Systems (CDSS), ML digunakan untuk meningkatkan kemampuan sistem dalam mendeteksi penyakit secara dini, memprediksi risiko, dan mengoptimalkan perawatan pasien. Teori ini menegaskan bahwa semakin besar volume dan kompleksitas data medis (seperti citra medis, catatan rekam medis elektronik, dan data genomik), semakin penting peran ML untuk mengolah informasi tersebut menjadi pengetahuan klinis yang bermanfaat (Miotto et al., 2018).

Dalam teori *Computational Psychiatry* menyebutkan bahwa ML dapat membantu mengidentifikasi pola perilaku dan linguistik yang mengindikasikan gangguan mental seperti depresi, kecemasan, atau skizofrenia (Bzdok & Meyer-Lindenberg, 2018). Algoritma ML, melalui analisis teks atau data perilaku, mampu memetakan korelasi antara ekspresi bahasa dan tingkat gangguan psikologis dengan tingkat akurasi yang tinggi, yang secara tradisional sulit dilakukan oleh metode konvensional (Bzdok & Meyer-Lindenberg, 2018).

Dalam teori Computational Mental Health dan Affective Computing, machine learning (ML) memainkan peran kunci dalam mendeteksi kondisi mental seperti depresi secara otomatis, dengan memanfaatkan data yang beragam seperti teks, suara, ekspresi wajah, dan biometrik. Dalam konteks ini, depresi dimodelkan sebagai masalah klasifikasi, di mana algoritma ML mempelajari pola dari data yang dikumpulkan untuk mengelompokkan individu ke dalam kategori tingkat depresi tertentu, seperti ringan, sedang, atau berat. Pendekatan ini berlandaskan pada konsep bahwa tanda-tanda depresi dapat dikenali melalui pola linguistik dan perilaku, seperti penggunaan kata-kata negatif, penurunan variasi emosi, atau gaya bahasa yang monoton, yang sering kali menjadi indikator penting dalam mendeteksi gangguan mental (Lin et al., 2022).

### 2.2.4 Algoritma Decision Tree

**Decision Tree** (Pohon Keputusan) adalah algoritma klasifikasi supervised learning yang bekerja dengan cara **recursive partitioning** - membagi feature space menjadi regions berdasarkan series of binary decisions. Setiap **internal node** merepresentasikan test pada feature tertentu (e.g., "activity_hour_19 ≤ 220.5?"), setiap **branch** merepresentasikan outcome dari test, dan setiap **leaf node** merepresentasikan class label (e.g., "Condition" atau "Control").

**Algoritma konstruksi** Decision Tree bekerja top-down secara greedy: (1) Start dengan seluruh dataset di root node, (2) Select best feature untuk split berdasarkan criterion (information gain, Gini impurity), (3) Partition data berdasarkan feature value, (4) Recursively repeat untuk setiap child node, (5) Stop ketika mencapai stopping criterion (pure node, max depth, min samples).

**Keunggulan untuk data aktivitas motorik:**
1. **Captures non-linearity**: Pola aktivitas tidak linear (e.g., threshold effects - aktivitas < X menandakan masalah)
2. **Models interactions**: Dapat menangkap kombinasi features (e.g., low evening activity AND weak circadian rhythm → strong depression signal)
3. **Interpretability**: Tree structure dapat ditranslate ke clinical rules yang intuitive
4. **Handles mixed types**: Dapat process continuous (activity levels) dan categorical features (time of day) tanpa encoding khusus
5. **Robust to outliers**: Splits based on thresholds, tidak terpengaruh extreme values seperti parametric methods

Keunggulan lainnya adalah struktur Decision Tree yang mudah dipahami dan diinterpretasikan. Setiap langkah dalam pohon keputusan mencerminkan keputusan yang diambil berdasarkan nilai tertentu dari atribut yang digunakan, sehingga memudahkan pengguna untuk memahami bagaimana keputusan dibuat. Misalnya, dalam deteksi depresi, Decision Tree dapat mengidentifikasi pola-pola tertentu, seperti hubungan antara gangguan tidur, pola makan yang tidak sehat, dan gejala emosional seperti perasaan cemas atau tidak berharga. Karena Decision Tree bersifat intuitif, model ini sering kali dipilih dalam pengembangan sistem deteksi depresi, khususnya ketika diperlukan transparansi dalam pengambilan keputusan klinis (Kim & Lee, 2022).

Dalam konteks deteksi depresi, hubungan antar variabel sering kali bersifat non-linear dan kompleks, yang menyulitkan analisis menggunakan model linier tradisional. Decision Tree memiliki kemampuan untuk memetakan hubungan yang kompleks ini dengan sangat baik. Sebagai contoh, penelitian oleh (Choi et al., 2023) menunjukkan bahwa Decision Tree dapat digunakan untuk mendiagnosis depresi melalui analisis ekspresi wajah dan postur kepala pada individu. Penelitian ini menunjukkan hasil yang mengesankan, dengan model Decision Tree menghasilkan Area Under the Curve (AUC) sebesar 0.99, yang menunjukkan akurasi tinggi dalam mendeteksi depresi berdasarkan faktor-faktor emosional dan ekspresi non-verbal (Choi et al., 2023).

Keunggulan lainnya adalah kemampuannya untuk menangani missing data (data yang hilang), yang sering terjadi dalam pengumpulan data psikologis. Banyak survei dan kuesioner psikologis yang tidak selalu lengkap, misalnya karena responden yang tidak memberikan jawaban pada beberapa pertanyaan. Decision Tree dapat menangani masalah ini dengan baik menggunakan teknik seperti CART (Classification and Regression Trees) atau ID3 (Iterative Dichotomiser 3), yang dapat memperkirakan nilai yang hilang berdasarkan data yang ada. Sebuah penelitian oleh Zhang dan Wang 2023 menunjukkan bahwa Decision Tree dapat menghasilkan model yang stabil meskipun menghadapi dataset yang tidak lengkap, dengan tingkat akurasi lebih dari 69% dalam memprediksi depresi (Zhang & Wang, 2023).

Namun, meskipun memiliki banyak keunggulan, Decision Tree juga menghadapi masalah overfitting, yang terjadi ketika model terlalu menyesuaikan diri dengan data pelatihan, menghasilkan pohon keputusan yang sangat rumit dan tidak dapat menggeneralisasi dengan baik pada data baru. Hal ini sering kali mengurangi kemampuan model dalam memprediksi data yang tidak ada dalam pelatihan. Untuk mengatasi masalah ini, teknik pruning (pemangkasan) digunakan untuk menyederhanakan pohon keputusan dengan menghapus cabang-cabang yang tidak relevan. Penelitian oleh Uddin et al. 2020 mengenai penggunaan Adaboosted Decision Tree untuk memprediksi depresi pada pekerja teknologi menunjukkan bahwa penerapan teknik pruning dapat membantu mengurangi masalah overfitting dan meningkatkan stabilitas model dalam memprediksi depresi (Uddin et al., 2020).

### 2.2.5 Algoritma Gaussian Naive Bayes

****Gaussian Naive Bayes** adalah varian dari Naive Bayes classifier yang didesain untuk fitur bernilai kontinu dengan asumsi bahwa fitur mengikuti **distribusi Gaussian (normal)** dalam setiap kelas. Algoritma bekerja berdasarkan **Teorema Bayes**, yang menghitung probabilitas suatu kelas berdasarkan fitur yang diamati.

Dengan **asumsi independensi naif**, algoritma mengasumsikan bahwa setiap fitur memberikan kontribusi independen terhadap probabilitas kelas. Untuk distribusi Gaussian, probabilitas setiap fitur dihitung berdasarkan distribusi normal dengan parameter rata-rata dan varians.

**Estimasi parameter**: Rata-rata dan varians untuk setiap fitur diestimasi dari data training untuk setiap kombinasi fitur-kelas.

**Keunggulan:**
1. **Efisiensi komputasi**: Training hanya memerlukan perhitungan mean dan variance - sangat cepat bahkan untuk dataset besar
2. **Output probabilistik**: Memberikan estimasi confidence (skor probabilitas) bukan hanya label kelas
3. **Menangani nilai hilang**: Dapat melewati fitur yang hilang dalam perhitungan probabilitas
4. **Kompleksitas sampel rendah**: Memberikan performa yang cukup baik bahkan dengan training set yang kecil

**Limitasi untuk data aktivitas motorik:**
1. **Asumsi independensi dilanggar**: Fitur aktivitas sangat berkorelasi (misalnya, hour_08 dengan hour_09, durasi tidur dengan onset tidur)
2. **Distribusi non-Gaussian**: Banyak fitur aktivitas yang skewed atau multimodal, melanggar asumsi Gaussian
3. **Tidak dapat memodelkan interaksi**: Asumsi independensi menghalangi penangkapan kombinasi fitur yang penting untuk deteksi depresi

Secara teknis, Naive Bayes bekerja dengan mengasumsikan bahwa setiap fitur atau variabel bersifat independen secara kondisional terhadap kelas target. Meskipun dalam praktiknya asumsi ini seringkali tidak sepenuhnya akurat, model Naive Bayes tetap dapat memberikan hasil klasifikasi yang memadai, khususnya pada dataset yang memiliki banyak variabel tetapi jumlah sampel terbatas. Meskipun asumsi independensi ini, yang disebut asumsi "naïve," jarang terjadi dalam data nyata karena sejumlah fitur dapat saling berhubungan (misalnya antara pola tidur dan perasaan tidak berharga), penelitian oleh Lavenia dan Permatasari 2023 menunjukkan bahwa Naive Bayes tetap memberikan hasil yang baik, terutama pada analisis data yang melibatkan banyak variabel, seperti data terkait depresi di Twitter (Lavenia & Permatasari, 2023).

Salah satu kekuatan utama dari Naive Bayes adalah efisiensi komputasi dan performa yang stabil ketika bekerja dengan dataset berdimensi tinggi. Karena model ini menghitung probabilitas berdasarkan frekuensi fitur terhadap kelas, proses pelatihan dan prediksinya menjadi relatif cepat dibandingkan dengan model lain yang lebih kompleks. Kecepatan ini sangat bermanfaat dalam pengembangan sistem deteksi dini depresi, khususnya pada aplikasi yang memerlukan respons real-time atau pemrosesan volume data yang besar, seperti analisis teks dari survei daring atau media sosial. Lavenia dan Permatasari (2023) menunjukkan bahwa Naive Bayes dapat digunakan untuk menganalisis tweet terkait depresi, dengan Multinomial Naive Bayes menghasilkan akurasi tertinggi sebesar 90.13% (Lavenia & Permatasari, 2023).

### 2.2.6 Strategi Penanganan Imbalanced Data

**Imbalanced data** terjadi ketika distribusi kelas tidak uniform, dengan satu kelas (minoritas) sangat kurang terwakili dibanding kelas lainnya (mayoritas). Dalam klasifikasi medis, kelas minoritas seringkali adalah kasus positif (penyakit) yang justru lebih penting untuk dideteksi dengan benar. Algoritma ML standar cenderung bias terhadap kelas mayoritas, menghasilkan akurasi keseluruhan tinggi tetapi performa kelas minoritas yang buruk.

#### 2.2.6.1 Class Weights (Algorithm-Level Method)

**Class weights** mengimplementasikan **pembelajaran sensitif-biaya** (cost-sensitive learning) dengan memberikan penalti lebih besar untuk kesalahan klasifikasi kelas minoritas. Bobot dihitung secara proporsional berdasarkan jumlah sampel total, jumlah kelas, dan jumlah sampel per kelas.

Dalam praktiknya, kelas minoritas mendapat bobot yang lebih tinggi dibanding kelas mayoritas. Sebagai contoh, pada dataset dengan distribusi tidak seimbang, kelas minoritas akan mendapat bobot sekitar 1.2 sementara kelas mayoritas sekitar 0.86.

Metode ini **tidak menambah data**, hanya memodifikasi tujuan pembelajaran untuk fokus lebih pada kelas minoritas. Efektif untuk algoritma yang mendukung bobot sampel (Decision Tree, Logistic Regression), tetapi **kurang efektif untuk Naive Bayes** karena sifat estimasi probabilitasnya.

#### 2.2.6.2 SMOTE (Synthetic Minority Over-sampling Technique)

**SMOTE** men-generate synthetic minority samples melalui interpolation:
1. Untuk setiap sampel minoritas, identifikasi k-nearest neighbors dalam ruang fitur
2. Pilih secara acak satu neighbor terdekat
3. Generate synthetic sample dengan melakukan interpolasi linier antara sampel asli dan neighbor yang dipilih, dengan faktor random antara 0 dan 1

SMOTE **meningkatkan ukuran training set** dan **menghaluskan batas keputusan**. Keuntungan: generalisasi lebih baik dibanding random oversampling (menduplikasi sampel). Keterbatasan: menghasilkan sampel sintetis **secara seragam** tanpa mempertimbangkan kesulitan klasifikasi region tertentu.

#### 2.2.6.3 ADASYN (Adaptive Synthetic Sampling)

**ADASYN** adalah perbaikan dari SMOTE yang **secara adaptif** menghasilkan lebih banyak sampel sintetis di region "sulit dipelajari":
1. Untuk setiap sampel minoritas, hitung proporsi sampel kelas mayoritas di k-nearest neighbors
2. Normalisasi proporsi tersebut untuk semua sampel minoritas
3. Distribusikan jumlah sampel sintetis berdasarkan proporsi yang telah dinormalisasi

Sampel dengan **proporsi tinggi** sampel mayoritas di sekitarnya (dikelilingi oleh kelas mayoritas, dekat batas keputusan) menerima **lebih banyak sampel sintetis**. Ini memfokuskan pembelajaran pada kasus sulit, meningkatkan **generalisasi** dan **ketahanan**.

#### 2.2.6.4 Hybrid Methods (SMOTE + Weights)

Kombinasi metode **tingkat data** (SMOTE/ADASYN) dan **tingkat algoritma** (class weights). Rasional: SMOTE menyediakan lebih banyak sinyal training, bobot memastikan algoritma tetap fokus pada kelas minoritas meskipun data sudah seimbang. Dapat memberikan yang terbaik dari kedua dunia, tetapi berisiko **over-correcting** - terlalu banyak fokus pada minoritas dapat mengurangi spesifisitas (false positives).

---

**CATATAN:** Daftar Pustaka lengkap tersedia di file terpisah: [DAFTAR_PUSTAKA.md](DAFTAR_PUSTAKA.md)
