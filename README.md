# semantic_similarity
semantic similarity web application



Setup without docker:

1) git clone https://github.com/andreyverbovskiy/semantic_similarity.git
2) cd semantic_similarity
3) (Optional but recommended):
Setup virtual environment:
python -m venv env
.\env\Scripts\activate
4) pip install -r requirements.txt
5) (Pull the CharecterBERT repo to run the model):
git clone https://github.com/helboukkouri/character-bert.git
cd character-bert
pip install transformers==4.34.0 scikit-learn==1.3.1 gdown==4.7.1
python download.py --model='general_character_bert'
If it did not download itself, go to the web address the error gives you and download manually, the put the file inside character-bert\pretrained-models\general_character_bert   and then run the command again
(For some reason some files in the chartbert library could not locate each other, so a few extra steps need to be done)
copy the content in character-bert\utils\character_cnn.py and create new file called utils_character_cnn.py in folder character-bert\modeling\
7) streamlit run nlp_app.py
8) go to http://localhost:8501


With docker (requires docker installed):

1) (Pull the CharecterBERT repo to run the model):
git clone https://github.com/andreyverbovskiy/semantic_similarity.git
cd semantic_similarity
git clone https://github.com/helboukkouri/character-bert.git
cd character-bert
pip install transformers==4.34.0 scikit-learn==1.3.1 gdown==4.7.1
python download.py --model='general_character_bert'
If it did not download itself, go to the web address the error gives you and download manually, the put the file inside character-bert\pretrained-models\general_character_bert   and then run the command again
(For some reason some files in the chartbert library could not locate each other, so a few extra steps need to be done)
copy the content in character-bert\utils\character_cnn.py and create new file called utils_character_cnn.py in folder character-bert\modeling\
cd ..
2) docker-compose up --build
3) go to http://localhost:8501

