from modelscope import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('AI-ModelScope/distilbert-base-cased')
model = DistilBertModel.from_pretrained("AI-ModelScope/distilbert-base-cased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)