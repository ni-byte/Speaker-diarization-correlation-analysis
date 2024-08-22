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
Confidence measure
According to “Quot”, confidence measure is a speaker separability indicator, which is distance metric between two speaker models built from the diarization hypothesis.

Meaning 
It is high for better diairzation hypotheses.

Use
Propose it as a measure to access the quality of a diarization hypothesis.

Computation

Input of confidence measures

Bayesian Information Criterion (BIC)
 
 
The models can be tested using corresponding BIC values. Lower BIC value indicates lower penalty terms hence a better model.

Source:
https://medium.com/@analyttica/what-is-bayesian-information-criterion-bic-b3396a894be6

PCA

Computation

Input of Bayesian Information Criterion

Mel-Frequency Cepstral Coefficients (MFCC)

Input of MFCC

The pathways to explore the correlaiton 

Likelihood 
Computation


MFCC

Input 

Confidence measure


Inputs 
1.Input of confidence measure
2.Input of BIC
3.Input of MFCC
#Citation 
C. Vaquero, A. Ortega, A. Miguel and E. Lleida, "Quality Assessment for Speaker Diarization and Its Application in Speaker Characterization," in IEEE Transactions on Audio, Speech, and Language Processing, vol. 21, no. 4, pp. 816-827, April 2013, doi: 10.1109/TASL.2012.2236317. keywords: {Reliability;Speech;Accuracy;Density estimation robust algorithm;Quality assessment;NIST;Materials;Speaker diarization;confidence measures;speaker characterization;telephone conversations} https://ieeexplore.ieee.org/document/6392899


Comparison between this confidence measure and other ones
KullBack-Leibler (KL) divergence between the Gaussian speaker models in the speaker factor space and the number of iterations that the 
![image](https://github.com/user-attachments/assets/4d46945e-06ba-4349-b167-112ce32a258f)
