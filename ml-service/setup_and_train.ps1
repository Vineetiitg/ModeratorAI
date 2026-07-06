$ErrorActionPreference = 'Stop'

$VenvPath = "$env:USERPROFILE\.safechat_venv"

Write-Host "Reusing existing venv $VenvPath..."
if (-not (Test-Path $VenvPath)) {
    venv\Scripts\python.exe -m venv $VenvPath
    & "$VenvPath\Scripts\python.exe" -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    & "$VenvPath\Scripts\python.exe" -m pip install transformers datasets sentencepiece scikit-learn pandas loguru openpyxl
}

Write-Host "Starting fine-tuning process!!!"
$env:TRANSFORMERS_BYPASS_TORCH_LOAD_VULN_CHECK = "1"
& "$VenvPath\Scripts\python.exe" training/train_classifier.py --dataset_path training/final_training_data_v4.csv --batch_size 2 --grad_accum 8 --epochs 3
