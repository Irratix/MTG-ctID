import urllib.request
import zipfile
import os

GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.2024.wikigiga.100d.zip"
GLOVE_DIR = "data/"
GLOVE_PATH = GLOVE_DIR + "glove.2024.wikigiga.100d.txt"

def download_glove():
    if os.path.exists(GLOVE_PATH):
        print("GloVe embeddings already downloaded")
        return
    
    os.makedirs(GLOVE_DIR, exist_ok=True)
    print("Downloading GloVe embeddings...")
    urllib.request.urlretrieve(GLOVE_URL, GLOVE_DIR + "glove.zip")
    
    print("Extracting...")
    with zipfile.ZipFile(GLOVE_DIR + "glove.zip", 'r') as f:
        f.extractall(GLOVE_DIR)
    os.remove(GLOVE_DIR + "glove.zip")
    print("Done")

if __name__ == "__main__":
    download_glove()