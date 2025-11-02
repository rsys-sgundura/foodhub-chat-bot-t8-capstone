from huggingface_hub import HfApi
import os

# Authenticate using your Hugging Face token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the deployment folder to your Hugging Face Space
api.upload_folder(
    folder_path="deployment",     # Local folder to upload
    repo_id="SudeendraMG/foodhub-chat-bot-t8-capstone",      # Replace with your Hugging Face Space ID
    repo_type="space",                            # Type of repo: space, model, or dataset
    path_in_repo="",                              # Optional: subfolder path inside the repo
)
