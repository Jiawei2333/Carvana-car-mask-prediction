# Carvana-car-mask-prediction
Files:

Carvana_main.ipynb: The main function used for this task, including data preprocessing, model implementation, prediction, metric evaluation results analyses and visualization.

DataGenerator.py:  A customized class inheriting features from keras.utils.Sequence. It provides one batch data on the fly and offers data augmentation option.

model.py: Defines a U-net architecture and its metric funtion "dice_coef".
