from pathlib import Path
import pandas as pd
import tarfile
import urllib.request


def download_and_extract_ridership_data ():
    tarball_path = Path("datasets/ridership.tgz")

    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/dTmC0945/L-MCI-BSc-Data-Science-II/raw/main/data/ridership.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")

download_and_extract_ridership_data()
