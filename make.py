import json

file_name = "/Users/vaisakh/programs/Notebooks/lab/raw/SentimentalAnalysis.py"


def init_cell(file_name):
    file = open(file_name, "r").read()
    file = file.split("# In[")
    cells = []
    for i in range(len(file)):
        cells.append((file[i][4:]).strip())
    return cells


json.dump({"sentiment": init_cell(file_name)}, open("tmp.json", 'w'), indent=4)
