import os
import sys
import zipfile

def download_data_gdown(path):
    import gdown
    
    file_id = "1efHsY16pxK0lBD2gYCgCTnv1Swstq771"
    url = f"https://drive.google.com/uc?id={file_id}"
    data_zip = os.path.join(path, "data.zip")
    gdown.download(url, data_zip, quiet=False)
    
    with zipfile.ZipFile(data_zip, "r") as zip_ref:
        zip_ref.extractall(path)
    return
    
downlaod_data_gdown = download_data_gdown
download_data = download_data_gdown

def download_data_url(path):
    import urllib.request
    
    url = "https://ckip.iis.sinica.edu.tw/data/ckiptagger/data.zip"
    data_zip = os.path.join(path, "data.zip")
    urllib.request.urlretrieve(url, data_zip)
    
    with zipfile.ZipFile(data_zip, "r") as zip_ref:
        zip_ref.extractall(path)
    return
    
def main():
    download_data("./m1")
    download_data_gdown("./m2")
    download_data_url("./m3")
    return
    
if __name__ == "__main__":
    main()    
    sys.exit()
    
