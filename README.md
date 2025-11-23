
**Table of contents**

1. [Project Goal](#project-goal)
2. [Methodology](#methodology) 
3. [Data](#data)
4. [Conclusions](#conclusions)
5. [Experiment Process](#experiment-process)
7. [Terminologies](#terminologies)
8. [Inputs](#inputs)
9. [Parameters For Analysis](#parameters-for-analysis)
10.[Additional Reference](#additional-reference)
12. [Acknowledgment](#acknowledgment)


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


# Methodology

The experiements are conducted using the latest popular speaker diarization tool [pyannote](https://github.com/pyannote/pyannote-audio) to extract the speech features necessary for correlation analysis. 


# Data

The experiments use 30 audios from the mandarin multi-channel meeting speech corpus [AISHELL-4](https://www.openslr.org/111/) in desired path for processing.


# Conclusions

## Conclusion for approach 1

1. There is no apparrent correlation between MFCC and the ∆BIC
2. There is no apparrent correlation between x-vector and the ∆BIC

## Conclusion for approach 2

### Correlation analysis based on **mean cosine similarities**

- Overall
    1. There is no correlation between x-vector and the mean cosine similarities
    2. There is no correlation between embedding and the mean cosine similarities
    3. There are correlations between the **2nd MFCC coefficient** and the mean cosine similarities 
    4. Based on the additional assumption above, the correlation described above provides **a possibility of identifying the range of the cosine similarity score by looking at the pattern of the 2nd MFCC coefficient**

- The correlation in detail
    1. There are apparent both linear and non-linear correlations between the **2nd MFCC coefficient** and the mean cosine    similarities
    2. The linear correlations are apparent in analysis based on MFCC based mean cosine similarities across different granular levels (segment level and speaker level)
    3. The non-linear correlations are apparent across Cosine Similarity Scores (speaker level mean cosine similarities) obtained based on different features (MFCC and X-vector)


### Highlighted figure

|      Category       |MFCC cos sim Linear|MFCC cos sim Non Linear|X-vector cos sim Linear|X-vector cos sim Non Linear|   Level   |
|:-------------------:|:-----------------:|:---------------------:|:---------------------:|:-------------------------:|:---------:|
|2nd MFCC coefficients| **0.82**          |    -                  | **0.64**              |   -                       | segment   |
|2nd MFCC coefficients| **0.62**          |    **0.73**           | **0.61**              |   **0.72**                | speaker   |
|6th MFCC coefficients| 0.65              |    -                  | -                     |   -                       | segment   |
|  mean MFCC value    | 0.73              |    -                  | -                     |   -                       | segment   |
|  mean MFCC value    | -                 |    -                  | -                     |   0.63                    | speaker   |

*Note:
- "cos sim" means mean cosine smilarity

### Correlation analysis based on **Silhouette Score**

There is no apparrent correlation between the target speech features (MFCCs，x-vectors and embeddings) and the corresponding Silhouette Scores.


# Experiment Process

## Notebooks with actual data and outcome

- [Diarization]([https://gitlab.uzh.ch/mi.zhou/speaker-diarization/-/blob/main/pyannote/Final%20versions/diarization.ipynb?ref_type=heads](https://github.com/ni-byte/Speaker-diarization-correlation-analysis/blob/main/Pyannote/Final%20version/diarization.ipynb)
- [Feature extraction and sample selection](https://gitlab.uzh.ch/mi.zhou/speaker-diarization/-/blob/main/pyannote/Final%20versions/sample_selection_and_features_extraction.ipynb?ref_type=heads)
- [Correlation analysis based on Delta BIC](https://gitlab.uzh.ch/mi.zhou/speaker-diarization/-/blob/main/pyannote/Final%20versions/correlation_with_delta_BIC.ipynb?ref_type=heads)
- [Correlation analysis based on Cosine Similarities](https://gitlab.uzh.ch/mi.zhou/speaker-diarization/-/blob/main/pyannote/Final%20versions/correlation_with_cos_sim.ipynb?ref_type=heads)
- [Correlation analysis based on Silhouette Score](https://gitlab.uzh.ch/mi.zhou/speaker-diarization/-/blob/main/pyannote/Final%20versions/correlation_with_sil_score.ipynb?ref_type=heads)


## Clean versions that can be used as templates

- [Diarization](https://gitlab.uzh.ch/mi.zhou/speaker-diarization/-/blob/main/pyannote/Clean%20versions/diarization.ipynb?ref_type=heads)
- [Feature extraction and sample selection](https://gitlab.uzh.ch/mi.zhou/speaker-diarization/-/blob/main/pyannote/Clean%20versions/sample_selection_and_features_extraction.ipynb?ref_type=heads)
- [Correlation analysis based on Delta BIC](https://gitlab.uzh.ch/mi.zhou/speaker-diarization/-/blob/main/pyannote/Clean%20versions/correlation_with_delta_BIC.ipynb?ref_type=heads)
- [Correlation analysis based on Cosine Similarities](https://gitlab.uzh.ch/mi.zhou/speaker-diarization/-/blob/main/pyannote/Clean%20versions/correlation_with_cos_sim.ipynb?ref_type=heads)
- [Correlation analysis based on Silhouette Score](https://gitlab.uzh.ch/mi.zhou/speaker-diarization/-/blob/main/pyannote/Clean%20versions/correlation_with_sil_score.ipynb?ref_type=heads)

# Terminologies

## Terminologies for approach 1

1. **∆BIC (Delta Bayesian Information Criterion)**

  - Concept

     - According to [Serafini et al. (2023)](https://arxiv.org/abs/2305.18074), ∆BIC is a sort of penalized likelihood ratio expressing the    difference between two assumptions, one of which is that two sequences of acoustic feature vectors are split into two Gaussian populations, another of which is that the two sequences belong to the same Gaussian population.

  - Meaning of the value

     - According to [Serafini et al. (2023)](https://arxiv.org/abs/2305.18074) and [Vaquero et al.(2013)](https://ieeexplore.ieee.org/document/6392899), higher values indicate a greater probability that the two segments are from different speakers, lower values suggest they are more likely from the same speaker. 

  - Computation

     - ∆BIC = (bic_1 + bic_2) - bic_combined
        - bic_1 or bic_2 is the likelihood of the segment belonging to a specific speaker
        - bic_combined is the likelihood of the two segments belonging to the same speaker

2. **Bayesian Information Criterion (BIC)**

  - Concept

     - BIC is used to compare two sequences of acoustic feature vectors \(X_1, X_2\) segregated by the segmentation system. It is computed for two hypotheses:
        - a. Different speaker hypothesis: each sequence belongs to a different speaker.
        - b. Joint hypothesis: both sequences belong to the same speaker.

  - Meaning of the value

      - Lower BIC value indicates that the model can explain the data well without being overly complex.
      - Higher BIC value suggests that the model is either too complex or doesn't fit the data well.

  - Computation

    - BIC=−2ln(L)+kln(n)
        - L is the likelihood (see terminology 3) of the model given the data. 
        - k is the number of parameters in the model. 
        - n is the number of data points.


3. **Likelihood and GMM**

  - Concept
    - GMM
       GMM is a probability distribution over the space of the input data. 
       It is a mixture of multiple Gaussian distributions, each capturing a different "cluster" or pattern within the data.

  - Likelihood of a data point

    - For each data point, the GMM assigns a likelihood value, which indicates how likely it is that the data point belongs to the  distribution represented by the GMM.

  - Total log-likelihood

      - To obtain the overall fit of the GMM, it is required to calculate the likelihood of all data points together. Since likelihoods are often small numbers, multiplying them can lead to two small values, therefore, the log-likelihood is a summation of the logarithms of the individual likelihoods.

  - Meaning of the value

    - Higher log-likelihood
       A higher log-likelihood suggests that the GMM is a good fit for the data, meaning the data points have high probability to have been generated from the distribution represented by the GMM.

    -  Lower log-likelihood
       A lower log-likelihood indicates the GMM might not capture the underlying patterns in the data well


4. **Euclidean distance**

- Concept

   - Euclidean distance gives the distance between any two points in a Euclidean space (an n_dimensional plane).
   It is the most common and familiar distance metric, often referred to as the “ordinary” distance.

- Meaning of the value

   - Euclidean distance between two points in the an n_dimensional plane is defined as the length of the line segment joining the two points.

- Computation

   - Assume two points (x1, y1) and (x2, y2) in a 2-dimensional space, the euclidean distance between them is given by:

    - \[ d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} \]
       - d is Euclidean Distance
       - (x1, y1) is Coordinate of the first point
       - (x2, y2) is Coordinate of the second point


## Terminologies for both approach 1 and 2

1. **Cosine similarity**

- Concept

   - Cosine similarity is a metric that is helpful in determining how similar the data objects are irrespective of their size

- Meaning of the value

   - The cosine similarity always belongs to the interval [−1, 1], 1 means the two vectors have the highest similarity, -1 means the two vectors have the lowest similarity


- Computation

   - Assume  A and B are two vectors,

      - cosine_similarity(x, y) = (x . y) / (||x|| \times ||y||)
         - x . y = product (dot) of the vectors ‘x’ and ‘y’
         - ||x|| and ||y|| = length (magnitude) of the two vectors ‘x’ and ‘y’
         - ||x|| \times ||y|| = regular product of the two vectors ‘x’ and ‘y’


- Application in approach 1

    - In the pyannote repository, the default clustering technique is [AgglomerativeClustering](https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/speaker_diarization.py), and [cosine similarity is set default](https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/clustering.py) to determine the similarity of embeddings for clustering. It is therefore can be inferenced that there might be correlation between the cosine similarity of speech features (mfccs, x-vectors) and the delta BIC scores, which evaluate the similarity of differenct segments. Therefore, cosine similarity of speech features are used to explore the correlation in approach 1.


- Application in approach 2

    - Cosine similarity are mainly used for the calculation of the mean cosine similarity of each segment, and the cosine similarity score of each speaker in approach 2.


## Terminology for approach 2

1. **Cosine similarity score**

-  Concept
    - Cosine similarity score is the confidence scores as the mean cosine similarity between a segment’s
    speaker embeddings and the speaker’s centroid.

-  Meaning of the value
    - 1 means the highest similarity, -1 means the lowest similarity


- Computation

    - Obtain the speaker embeddings of the target segments
    -  Obtain the speaker centroid by averaging the embeddings extracted in the 1st step
    -  Calculate the cosine similarity between each embedding and the speaker controid
    - Calculate the mean cosine similarity obtained in the 3rd step


2. **Speaker centroids**

  - Speaker centroids are the average of the embeddings associated with all of the segments assigned to a speaker. (Chowdhury et al. (2024))


3. **Silhouette score**

-  Concept

    - The silhouette score is a metric used to evaluate how good clustering results are in data clustering. 


-  Meaning of the value

    - It ranges from -1 to +1, a higher score indicates better clustering results.
    - Positive values indicate that data points belong to the correct clusters, indicating good clustering results.
    - A score of zero suggests overlapping clusters or data points equally close to multiple clusters.
    - Negative values indicate that data points are assigned to incorrect clusters, indicating poor clustering results.


-   Computation

    - This score is calculated by measuring each data point’s similarity to the cluster it belongs to and how different it is from other clusters.

    - The formula: 
        - Silhouette score=(b - a) / max(a, b). 
             - a is the average distance between current data point and other data points within the same cluster.
             - b is the distance between a sample and the nearest cluster that the sample is not a part of.


# Inputs

1. Input of delta BIC:  BICs

2. Input of BIC:  MFCC features

3. Input of MFCC: segments obtained from speaker diarization task

4. Input of x-vector: segments obtained from speaker diarization task

5. Input of euclidean distance: coordinates/values of each dimension of the mfcc features of the target segments

6. Input of cosine similarity: embeddings, or MFCCs, or x-vectors.

7. Input of cosine similarity score: cosine similarity

8. Input of speaker’s centroid: embeddings, or MFCCs, or x-vectors.

9. Input of silhouette Score: embeddings, or MFCCs, or x-vectors.


# Parameters For Analysis 

## Parameters for approach 1

### Parameters for analysis based on Delta BIC

**Note:**
- **Y** means a correlation analysis between the two parameters was conducted
- **"MFCC Euc Dis"** means Euclidean Distance between the MFCCs of each pair of selected segments (similar for X-vector Euc Dis)
- **"MFCC Cos Sim"** means Cosine Similarities between the MFCCs of each pair of selected segments (similar for X-vector Cos Sim)
- **"Delta BIC"** means Delta BIC value of each pair of selected segments

| Parameter |    MFCC Euc Dis  |   MFCC Cos Sim   | X-vector Euc Dis | X-vector Cos Sim |
|:---------:| :--------------: | :--------------: | :--------------: | :--------------: |
| Delta BIC |      Y           |         Y        |      Y           |        Y         |

## Parameters for approach 2

### Parameters for analysis based on Cosine Similarities

**Note:**
- **"Embedding Cos Seg"** is the mean cosine similarities of each embedding, taking the mean based on each segment (similar for "MFCC  Cos  Seg" and "X-vector Cos Seg")
- **"Embedding Cos Spe"** is the mean cosine similarities of each embeddings, taking the mean based on each speaker (similar for "MFCC  Cos  Spe" and "X-vector Cos Spe")
- **"Segment Mean MFCC Vector"** is the mean MFCC vector of each segment, the use of vector is to analyse the correlation between each MFCC coefficient and the mean cosine similarity
- **"Speaker Mean MFCC Vector"** is the mean MFCC vector of each speaker
- **"Segment Mean MFCC Value"** is the mean MFCC value of each segment
- **"Speaker Mean MFCC Value"** is the mean MFCC value of each speaker

|      Parameter    | Segment Mean MFCC Vector | Speaker Mean MFCC Vector | Segment Mean MFCC Value | Speaker Mean MFCC Value |
|:-----------------:| :----------------------: | :----------------------: | :---------------------: | :---------------------: |
| Embedding Cos Seg |              Y           |             -            |             Y           |               -         |
| Embedding Cos Spe |              -           |             Y            |             -           |               Y         |
|  MFCC  Cos  Seg   |              Y           |             -            |             Y           |               Y         |
|  MFCC  Cos  Spe   |              -           |             Y            |             -           |               -         |
| X-vector Cos Seg  |              Y           |             -            |             Y           |               Y         |
| X-vector Cos Spe  |              -           |             Y            |             -           |               -         |
 

**Note:**
- **"Segment Mean X-vector Value"** is the mean X-vector value of each segment
- **"Speaker Mean X-vector Value"** is the mean X-vector value of each speaker

|      Parameter    | Segment Mean X-vector Value  | Speaker Mean X-vector Vector | 
|:-----------------:| :--------------------------: | :--------------------------: |
| Embedding Cos Seg |              Y               |             -                |
| Embedding Cos Spe |              -               |             Y                |
|  MFCC  Cos  Seg   |              Y               |             -                |
|  MFCC  Cos  Spe   |              -               |             Y                |
| X-vector Cos Seg  |              Y               |             -                |
| X-vector Cos Spe  |              -               |             Y                |


**Note:**
- **"Segment Mean Embedding Value"** is the mean Embedding value of each segment
- **"Speaker Mean Embedding Value"** is the mean Embedding value of each speaker

|      Parameter    | Segment Mean Embedding Value | Speaker Mean Embedding Vector| 
|:-----------------:| :--------------------------: | :--------------------------: |
| Embedding Cos Seg |              Y               |             -                |
| Embedding Cos Spe |              -               |             Y                |
|  MFCC  Cos  Seg   |              Y               |             -                |
|  MFCC  Cos  Spe   |              -               |             Y                |
| X-vector Cos Seg  |              Y               |             -                |
| X-vector Cos Spe  |              -               |             Y                |

### Parameters for analysis based on Silhouette Scores

**Note:**
- **"Sil Sam"** means sample level Silhouette score 
- **"Sil Spe"** means speaker level Silhouette score

|      Parameter    |  Sample Mean MFCC Vector | Speaker Mean MFCC Vector |  Sample Mean MFCC Value | Speaker Mean MFCC Value |
|:-----------------:| :----------------------: | :----------------------: | :---------------------: | :---------------------: |
| MFCC Sil Sam      |              Y           |             -            |             Y           |               -         |
| MFCC Sil Spe      |              -           |             Y            |             -           |               Y         |


|      Parameter    | Sample Mean X-vector Value |Speaker Mean X-vector Value|
|:-----------------:| :------------------------: | :-----------------------: |
| X-vector Sil Sam  |              Y             |              -            |
| X-vector Sil Spe  |              -             |              Y            |


|      Parameter    | Sample Mean Embedding Value |Speaker Mean Embedding Value|
|:-----------------:| :-------------------------: | :------------------------: |
| Embedding Sil Sam |              Y              |              -             |
| Embedding Sil Spe |              -              |              Y             |


# Additional Reference

Bayesian Information Criterion:

https://www.geeksforgeeks.org/bayesian-information-criterion-bic/ https://medium.com/@analyttica/what-is-bayesian-information-criterion-bic-b3396a894be6


Euclidean space:

https://en.wikipedia.org/wiki/Euclidean_space

https://www.geeksforgeeks.org/euclidean-distance/


Cosine similarity: 

https://www.geeksforgeeks.org/cosine-similarity/

https://towardsdatascience.com/cosine-similarity-how-does-it-measure-the-similarity-maths-behind-and-usage-in-python-50ad30aad7db

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#sklearn.metrics.pairwise.cosine_similarity


Silhouette score:

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

https://en.wikipedia.org/wiki/Silhouette_(clustering)

https://medium.com/@hazallgultekin/what-is-silhouette-score-f428fb39bf9a


# Acknowledgement

- Part of the code snippets are generated using chatgpt, but it is acrosschecked with the source and validated or corrected by checking as the codes run.
- The diarization and feature extraction tasks are performed by the code from the pyannote repository, see following citation:

@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}

@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}

