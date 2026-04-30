from huggingface_hub import hf_hub_download

# Download the zip file
path = hf_hub_download(
    repo_id="princeton-nlp/less_data",
    filename="less-data.zip",
    repo_type="dataset"
)

# Unzip it
import zipfile
with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall("runs/less/extracted_data")


breakpoint()