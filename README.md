**Table of contents**

1. [Project Goal](#project-goal)
2. [Methodology](#methodology) 
3. [Data](#data)
4. [Experiment Process](#experiment-process)
5. [Conclusions](#conclusions)
6. [Terminologies](#terminologies)
7. [Inputs](#inputs)
8. [Parameters For Analysis](#parameters-for-analysis)
9. [Additional Reference](#additional-reference)
10. [Acknowledgment](#acknowledgment)

# Project Goal

Explore whether certain speech features, including MFCCs, x-vectors and embeddings, are correlated with confidence measures which evaluate the quality of the speaker diarization outcome.

## Approaches to achieve the goal

### Approach 1

Explore the correlation between speech features (MFCC and x-vector) and the ∆BIC (delta Bayesian Information Criterion) proposed by [(Serafini et al. (2023)](https://arxiv.org/abs/2305.18074) and [Vaquero et al.(2013)](https://ieeexplore.ieee.org/document/6392899), which evaluates the probability that two feature vectors belong to different speakers or the same speaker.

#### Additional assumption

Since a higher ∆BIC value means better diarization result, if the speech features are correlated with ∆BIC, it is possible to identify the diarization performance by looking at the patterns in the speech features.

### Approach 2

Explore the correlation between speech features (MFCC, x-vector and embeddings) and the Cosine Similarity Score (also called mean cosine similarities in this analysis) and Silhouette Score proposed by [Chowdhury et al.(2024)](https://arxiv.org/abs/2406.17124), which evaluates the accuracy of the diarization system in identifying a segment within a speaker model.

Note:
Since the mean cosine similarities can be calculated based on different units (per segment or per speaker), it is only called cosine similarity score when it is calculated bassed on speaker level, this analysis use "mean cosine similarities" for the calculation based on segment level, and call it "cosine similarity score" for the calculation based on speaker level.

#### Additional assumption

Since for both the Cosine Similarity Score and Silhouette Score, the values range from -1 to 1, 1 means higher accuracy that the current segments is classified to the correct speaker model, -1 means lower accuracy.

If the speech features are correlated with Cosine Similarity Score or Silhouette Score, there is a feasibility to identify the perforamnce of the diarization system by simply looking at the mfcc or x-vectors or embedding patterns of the sample audio.

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

The provided equation describes the confidence measure for speaker diarization based on the Bayesian Information Criterion (BIC). Here's an explanation of the parameters in the equation:

\begin{equation}
C_{\text{BIC}} = \Delta \text{BIC} = \log\left(\frac{\mathcal{L}(X_1 \mid \theta_1) \mathcal{L}(X_2 \mid \theta_2)}{\mathcal{L}(X_{1,2} \mid \theta_{1,2})}\right)
\end{equation}

\textbf{Parameters:}

\begin{itemize}
    \item \(\mathcal{L}(X_s \mid \theta_s)\):
    \begin{itemize}
        \item \(\mathcal{L}\): Represents the likelihood function.
        \item \(X_s\): The sequence of acoustic feature vectors for the \(s\)-th segment, where \(s\) can be 1, 2, or 1,2.
        \item \(\theta_s\): The parameters of the model for the \(s\)-th sequence, estimated from the data in that segment.
    \end{itemize}
    
    \item \(X_1\) and \(X_2\):
    \begin{itemize}
        \item These represent two different sequences of acoustic feature vectors that have been segregated by the segmentation system. 
        \item \(X_1\) corresponds to the first sequence, while \(X_2\) corresponds to the second sequence.
    \end{itemize}
    
    \item \(X_{1,2}\):
    \begin{itemize}
        \item This represents the combined sequence of \(X_1\) and \(X_2\), i.e., both sequences considered together as if they were from the same speaker.
    \end{itemize}
    
    \item \(\theta_1\) and \(\theta_2\):
    \begin{itemize}
        \item These are the model parameters estimated from \(X_1\) and \(X_2\) individually.
        \item \(\theta_{1,2}\) is the model parameter estimated from the combined sequence \(X_{1,2}\).
    \end{itemize}
    
    \item \(\log\):
    \begin{itemize}
        \item The natural logarithm function used to compute the difference in BIC values between the two hypotheses.
    \end{itemize}
    
\end{itemize}

\textbf{Hypotheses:}

\begin{itemize}
    \item \textbf{Different Speaker Hypothesis}: Assumes that \(X_1\) and \(X_2\) belong to different speakers. This hypothesis is represented by the product of the likelihoods \(\mathcal{L}(X_1 \mid \theta_1)\) and \(\mathcal{L}(X_2 \mid \theta_2)\).
  
    \item \textbf{Same Speaker Hypothesis}: Assumes that \(X_1\) and \(X_2\) belong to the same speaker. This is represented by the likelihood \(\mathcal{L}(X_{1,2} \mid \theta_{1,2})\).
\end{itemize}

\textbf{Interpretation:}

\begin{itemize}
    \item \(C_{\text{BIC}}\): This is the confidence measure that indicates the strength of the evidence favoring the different speaker hypothesis over the same speaker hypothesis.
    \item A higher \(C_{\text{BIC}}\) value suggests that the sequences \(X_1\) and \(X_2\) are more likely to belong to different speakers, leading to a better speaker diarization hypothesis.
\end{itemize}

The method ensures that both hypotheses (same speaker and different speaker) are compared with the same model complexity, making the comparison fair and avoiding the need for tuning BIC penalty parameters.


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
