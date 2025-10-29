'''
Author: zy
Date: 2025-10-24 21:28:13
LastEditTime: 2025-10-24 21:28:20
LastEditors: zy
Description: 
FilePath: \haitianbei\scripts\print_versions.py

'''
import importlib.metadata as im
names = [
  'numpy','openai','pandas','pillow','scikit-learn','scipy','sentence-transformers',
  'spacy','spacy-entity-linker','tokenizers','torch','torchaudio','torchvision',
  'tqdm','transformers','rdflib','networkx','matplotlib','neo4j'
]
for n in names:
    try:
        print(f"{n}=={im.version(n)}")
    except Exception as e:
        print(f"{n}==UNKNOWN  # {e}")
