"# Noisy-Fair-Classification" 
This project is based on AIF360 (https://github.com/IBM/AIF360).
Environment: python 3.7

The codes locate in AIF360/noisyfair/
The results can be found in results/
The main codes can be found in AIF360/noisyfair

########################################################
# tau_adult.py and tau_compas.py

Command: 
cd AIF360/noisyfair
python tau_adult.py/tau_compas.py protected_attr eta0 eta1 repeat_times

Ex: python tau_adult.py race 0.3 0.1 10
Output: Adult_race_0.3_0.1_10.xlsx

########################################################
# eta_adult.py and eta_compas.py

Command: python eta_adult.py/eta_compas.py protected_attr repeat_times

Ex: python eta_adult.py race 10
Output: Adult_race_tau10.xlsx

#########################################################