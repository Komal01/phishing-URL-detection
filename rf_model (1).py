import pickle
import numpy,sklearn,pandas

"""# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
"""

def predictor(splitted_data):
    print("/n script rf_model")
    # load the model from disk
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    print("model loaded")
    print(splitted_data.shape)
    print(list(splitted_data))
    x = splitted_data.columns[3:9]
    preds = loaded_model.predict(splitted_data[x])
    print("pridction complete")
    print(preds)
    if preds == 0:
        str1 = "Spoofed webpage: Yes"
    else: str1 = "Spoofed webpage: NO"
    
    score = loaded_model.predict_proba(splitted_data[x])
    str2 = "Confidence score: "+ str(score[0][1])

    return str1,str2







