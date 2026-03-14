from huggingface_hub import upload_folder

upload_folder(
    folder_path="/mnt/align4_drive/arunas/sae-filters/SAEScoping/experiments/outputs_scoping/recover",
    repo_id="arunasank/sae-filters"
)
