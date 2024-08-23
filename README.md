**Goal** 
To explore the correlation between speech features (e.g. MFCC)  and the corresponding confidence measure of the same audio sample.

Methodology 
Tools
To cross validate the correlation, the experiements are conducted using both pyannote and VBx to extract the speech features of the same audio.
1.	Pyannote (https://github.com/pyannote/pyannote-audio )
2.	VBx (https://github.com/BUTSpeechFIT/VBx/tree/master)

Steps
Pyannote
1.	Perform speaker diarization tasks using pyannote from the sample audio files, store the output, which is a list of segments , including starting time and the ending time during which there is only one speaker in this segment, into a separate file
2.	Randomly  select 100 pairs of segments, extract their mfcc features and compute their confidence measure
3.	Conduct correlation analysis between the mfcc features and the corresponding confidence measures

Samples
20 mandarin audio files from the dataset Aishell-4 (https://www.openslr.org/111/)

Computation

Terminology
**Confidence measure**
According to the cited artical, confidence measure is a speaker separability indicator, which is distance metric between two speaker models built from the diarization hypothesis.

Meaning 
It is high for better diairzation hypotheses.

Use
Propose it as a measure to access the quality of a diarization hypothesis.

Computation
## Bayesian Information Criterion (BIC) for Speaker Diarization

The **Bayesian Information Criterion (BIC)** is used to compare two sequences of acoustic feature vectors \(X_1, X_2\) segregated by the segmentation system. The BIC is computed for two hypotheses:

1. **Different Speaker Hypothesis:** Each sequence belongs to a different speaker.
2. **Joint Hypothesis:** Both sequences belong to the same speaker.

The **confidence measure** 
$\(C_{\text{BIC}}\) is the difference between the BIC values for both hypotheses. To avoid adjusting BIC penalty parameters, we force the models for both hypotheses to have the same complexity. This way, the confidence measure is defined as:

\[
C_{\text{BIC}} = \Delta \text{BIC} = \log\left(\frac{\mathcal{L}(X_1 | \theta_1) \mathcal{L}(X_2 | \theta_2)}{\mathcal{L}(X_{1,2} | \theta_{1,2})}\right)
\]$

where:

- \(\mathcal{L}\) denotes likelihood.
- \(\theta_s\) is the model obtained from every vector sequence \(X_s\).

In this context:

- We use 12 MFCCs, including C0, as feature vectors.
- Every speaker model is a 32-component GMM, while the global model is a 64-component GMM.

This measure is presented as a confidence measure for speaker segmentation, showing good performance. **We expect \(C_{\text{BIC}}\) to be higher for better diarization hypotheses**.




**Bayesian Information Criterion (BIC)**
The models can be tested using corresponding BIC values. Lower BIC value indicates lower penalty terms hence a better model.

calculation
The BIC is defined as:
BIC=−2ln(L)+kln(n)

where:
L is the likelihood of the model given the data.
k is the number of parameters in the model.
n is the number of data points.

Source:
https://www.geeksforgeeks.org/bayesian-information-criterion-bic/
https://medium.com/@analyttica/what-is-bayesian-information-criterion-bic-b3396a894be6



PCA

Computation

Input of Bayesian Information Criterion

Mel-Frequency Cepstral Coefficients (MFCC)

Input of MFCC

The pathways to explore the correlaiton 

Likelihood 
Computation

GMM as a Probability Distribution: 
The GMM represents a probability distribution over the space of the input data. It's a mixture of multiple Gaussian distributions, each capturing a different "cluster" or pattern within the data.

Likelihood of a Data Point: 
For each data point, the GMM assigns a likelihood value, which indicates how likely it is that the data point belongs to the distribution represented by the GMM.

Total Log-Likelihood: 
To obtain the overall fit of the GMM, calculate the likelihood of all data points together. Since likelihoods are often small numbers, multiplying them can lead to two small values, therefore, the log-likelihood is a summation of the logarithms of the individual likelihoods.

Higher Log-Likelihood:
a higher log-likelihood suggests that the GMM is a good fit for the data, meaning the data points are likely to have been generated from the distribution represented by the GMM.

Lower Log-Likelihood: 
a lower log-likelihood indicates a poorer fit, suggesting the GMM might not capture the underlying patterns in the data well.


MFCC

Input 

Confidence measure


Inputs 
1.Input of confidence measure
BICs
2.Input of BIC
mfcc features
3.Input of MFCC
segments obtained from speaker diarization task

#Citation 
C. Vaquero, A. Ortega, A. Miguel and E. Lleida, "Quality Assessment for Speaker Diarization and Its Application in Speaker Characterization," in IEEE Transactions on Audio, Speech, and Language Processing, vol. 21, no. 4, pp. 816-827, April 2013, doi: 10.1109/TASL.2012.2236317. keywords: {Reliability;Speech;Accuracy;Density estimation robust algorithm;Quality assessment;NIST;Materials;Speaker diarization;confidence measures;speaker characterization;telephone conversations} https://ieeexplore.ieee.org/document/6392899


Comparison between this confidence measure and other ones
KullBack-Leibler (KL) divergence between the Gaussian speaker models in the speaker factor space and the number of iterations that the 
