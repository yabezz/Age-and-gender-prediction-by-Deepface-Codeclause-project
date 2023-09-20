import os
import cv2
import pandas as pd
from deepface import DeepFace

data = {
    "Name": [],
    "Age": [],
    "Gender": [],    
}

for file in os.listdir("Dataset"):
    result = DeepFace.analyze(cv2.imread(f"Dataset/{file}"), actions=("age", "gender", "race"))
    data["Name"].append(file.split("."[0]))
    data["Age"].append(result[0]["age"])
    data["Gender"].append(result[0]["dominant_gender"])

     
predicted_data = pd.DataFrame(data)
print(predicted_data)

predicted_data.to_csv("results.csv")