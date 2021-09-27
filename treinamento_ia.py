import pandas as pd
from ia import *
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators= 1000, random_state= 42)
rf.fit(modelo_tunado)

