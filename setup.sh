#!/bin/bash

ENV_NAME="venv_NLP"
if [ ! -d "$ENV_NAME" ]; then 
    python3 -m venv $ENV_NAME
    echo "---- Creating virtual environment '$ENV_NAME'. ----"
else
    echo "---- Virtual environment '$ENV_NAME' already exists. ----"
fi
source $ENV_NAME/Scripts/activate
pip3 install --upgrade pip
pip install -r requirements.txt

echo "---- Downloading NLTK data (punkt_tab) ----"
python -c "import nltk; nltk.download('punkt_tab', download_dir='./nltk_data')"

chmod 755 setup.sh #rwx->u and rx->g/o

echo "---- Setup complete, type 'source $ENV_NAME/Scripts/activate' to activate the virtual environment. ----"