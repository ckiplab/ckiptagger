import os
import sys
import zipfile

import gdown

def download_data(path):
    file_id = "16jCrS27EVCOFYWWx9Z0ByFG_Y-Df7CiJ"
    url = f"https://drive.google.com/uc?id={file_id}"
    data_zip = os.path.join(path, "data.zip")
    gdown.download(url, data_zip, quiet=False)
    
    with zipfile.ZipFile(data_zip, "r") as zip_ref:
        zip_ref.extractall(path)
    return
    
def main():
    download_data("./")
    return
    
if __name__ == "__main__":
    main()    
    sys.exit()
    