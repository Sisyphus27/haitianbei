r'''
Author: zy
Date: 2025-10-24 21:28:13
LastEditTime: 2025-10-24 21:28:20
LastEditors: zy
Description: 
FilePath: \haitianbei\scripts\print_versions.py

'''
import importlib.metadata as im
import logging as _logging
names = [
  'numpy','openai','pandas','pillow','scikit-learn','scipy','sentence-transformers',
  'spacy','spacy-entity-linker','tokenizers','torch','torchaudio','torchvision',
  'tqdm','transformers','rdflib','networkx','matplotlib','neo4j'
]
if not _logging.getLogger().handlers:
  _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for n in names:
  try:
    _logging.info(f"{n}=={im.version(n)}")
  except Exception as e:
    _logging.info(f"{n}==UNKNOWN  # {e}")
