finetuned_model_path = '../../Fine-Tuned_Models/Terminase_Task_Models/Secondary_Structure_Task/Random_Split/model.pt' # set to desired model path.

# Test on an example input

input_text = 'CCCCCCCHHHHHHHHHHHHHHHCCCCCCCCCCHHHHCCCCCCCHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCEEEEECCCCCHHHHHHHHHHHHHHHCCCCEEEEEECCHHHHHHHHHHHHHHHHHHHHHCCCCCEEEECCHHHHHHHCCCCCEEEEECHHHHHCCCCCCCHHHHCCCCCCCCCHHHHHHHHCCCCEEEEECCCCCCCHHHHHHHHHCCCCEEEEEECCCCCCCCCEEEECCHHHHHHCCCECCEEEEEECCCCCEEEEEECCCCCCCEEEEEEECCCCCEEEEEECCCCCHHHHCCCCHHHCHEEEECCCCEEEEECCCEEECCCCCCCCCCCCCHHHHHHHHHHHHHHHHHHHHHHCCCCCEEEEEECCHHHHCCCCCCHHHHHHHHHHHHHHHHHHHHCCCCCHHHHHHHHHCCCCCEEEECCCCCCCCCCCHHHHHHHHHHHHHHHHHCCCCCCEEEEEECHHHCCCCCCCCEEEEEECCCCCCCHHHHHHCCCCCCCECCCCCEECCCCCCCCCCCEEEEECCHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCHHHHHHCCCCCCCCHCCCCCCCCCCCHHHHHHHHHHCCCHHCCCCCCCCCCCCHHHHHHHHHHHHHHHHHCCCCCEECCCCCCCEEECCHHHHHHHHHHHHHHCCCEEEEEECCCHHHHHHHHHHHHHHCCCEEEEEEEEEEEEEEEEECCEEEEEEEEEEECEECHHHHHHHHHHHCCCCHHHHHHHHHCCCCCHHHHHHHCHHHHHHHHHHHHHHHHHHHHHHCCEEEECCCCCCCCCCCEEECCCCCEEEEECCCCCCCCCCCCCCCHHCCCCCCCEEECCCHHHHHHHHHCCCCCEEEEEECCCCCCCCCCCCCCCCCEEEEEECCCEEEEEEEEECCCCCCCCCCHHHHHHHHHHHHHHHHHHHCCCCEEEEEECCHHHHHHHHHHHHCCCC'

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, PreTrainedTokenizerFast

pretrained_model_path = '../../Pre-Trained_Models/Secondary_Structure_Model'

config = RobertaConfig.from_pretrained(pretrained_model_path)
config.num_labels = 2

model = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, config=config)
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../../Pre-Trained_Models/byte-level-BPE.tokenizer.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

state_dict = torch.load(finetuned_model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

input_ids = tokenizer.encode(input_text, add_special_tokens=True, max_length=128)
input_ids = torch.tensor(input_ids).unsqueeze(0)

outputs = model(input_ids)
predicted_class_index = torch.argmax(outputs.logits, dim=1).item()

label_dict = {0: 'Non-Terminase', 1: 'Terminase'}
print("Predicted Label:", label_dict[predicted_class_index])
