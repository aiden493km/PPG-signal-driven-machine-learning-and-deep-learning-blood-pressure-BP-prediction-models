# PPG-signal-driven-machine-learning-and-deep-learning-blood-pressure-BP-prediction-models
#### This is the 2024 SURF project of Xi'an Jiaotong-Liverpool University: 
- Incorporating Biophysical Modeling and Deep Learning for Wearable Blood Pressure Monitoring.

We proposed a solution: extracting BP features, using Random Forest algorithm to classify BP, and innovatively using Deep Learning model **SEM-ResNet** to predict BP. 

<img width="2480" height="3508" alt="海报水印" src="https://github.com/user-attachments/assets/3cf0f383-b804-4902-9a19-863b9c306939" />

# Dataset
- The MWPPG sensor is sampled at lkHz
- The raw PPG signal is analyzed andaggregated using the pyPPG toolboxThe data is combined with itscorresponding SP and DP
- Data processing: extract and normalize feature by the pyPPG toolbox

# Method
### BP classification:

**Random Forest** model is used to classify BP into three ranges:
- Hypo & Normal & Hyper

### BP prediction:

- SEM-ResNet (Deep Learning model) is used to predict BP in this stage

# References
Ma, C., Sun, Y., Zhang, P., Song, F., Feng, Y., He, Y., & Zhang, G. (2024).

SMART-BP: SEM-ResNet and Auto-Regressor Based on a Two-Stage Framework for Noninvasive Blood Pressure Measurement.

IEEE Transactions on Instrumentation and Measurement, 73, 2503718.

DOI: 10.1109/TIM.2023.3342856

Project Relevance: The multi-stage BP prediction framework of this project draws on the two-stage classification-regression strategy (CCP+FRP) proposed in the paper, especially the multi-scale feature fusion of SEM-ResNet, which is used to optimize the BP estimation of PPG signals.
