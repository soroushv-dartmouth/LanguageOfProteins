finetuned_model_path = '../../Fine-Tuned_Models/Terminase_Task_Models/Primary_Structure_Task/Random_Split/model.pt' # set to desired model path.

# Test on an example input

input_text = 'MQSPLNVDQLVMARDRHRLNRLQKGKDANQKQYQELFEKSHAKVMQRIERIPNIKLNQDLPVTQYADKLIEAIQQHQVIIVAGETGSGKTTQLPQIA' \
             'MLAGRGLTGLIGHTQPRRLAARSVSQRIAEEVGEKLGESISFKVRFNEQGSQDSIVRLMTDGILLAELANDRFLTKYDTIIIDEAHERSLNIDFIMG' \
             'YLKQLLPRRKDLKVIITSATLDVNRFSNYFNGAPIYEVEGRSYPVEVRYRPISEMSIVGSDDDEFDDFEENLPRAVVAAVEECFQDAQEKGHPEHAD' \
             'ILIFASTEQEIRELQETLIKHGPRHTEVLPLYARLALAEQQKIFNPSGGGRRIIIATNVAETALTVPNIRYVIDSGFARISRYNYRSRVQRLPIEAV' \
             'SQAAANQRKGRCGRIAPGVCIRLYSEEDFLSRPEFTEPEIKRTNLASVILQMQSLGLGSLEDFDFIEPPDHRLVNDGRKLLIELGAMVEKSKAPLSE' \
             'RGVGGDSTNPPKSSFTRGGLQGKGDSLTKIGQQMAKMPIDPRLARMILGGAHFGALNEALVIVAALAVQDPRERPADKQMQADQKHALFRETDSDFL' \
             'FYIKLWDTLHNNRESMSENKRRTFARNHFLSWLRLREWKKTHEQLVDLAKGLKLSFNEKKASYENLHRALLTGLLSFIANKTDERNVFMAVRQQKAR' \
             'IFPASTLHKTNTPWVMAFEMVETSQVYLRTLAKIEPEWILLAARDLLKHHYFEPHWSKKAGVVNAYDQISLFGLIIEPKRPVNYEKVDQPAAHEIFL' \
             'RDALTTGNLGINPPFLKHNLLKLEEVERVEDKLRRRDLVVDEETIYQFYASKVPPEVASRRSFEDWRATVEPHDPRYLFIDDDALWLNDRPTTQQFP' \
             'DYLRNGELRLATSYRFDPSHDEDGATVKIPLQALPQVDENIWSWGIPGWRLELIEALLKSLPKDKRRSLVPIPDTAKKLAARIDAVNLREHIFSFLA' \
             'FQLRGEQITEKDFSLDRVEQYLLPLIKVIDEKGRVIEQGRDLAELKARCRTETHSPVKQLKGEFKTFPENFVFEASQKVTGVVVKQYQALVPTKDFA' \
             'ALEQKDESGVVIQTFNDQAEAVKQHREGIIRLVHMQLGDLVRQLKKQISKPLALAYSPLGDKAQLEQMLVYATLHLSINELPVNMQEFQKLVEDVKK' \
             'SFLTHGQTALKEITDIYIQWQEIRRKLLVLDPSIFGKNIDDIEDQLDLMSLSDFVYRKPSDVWSEFPRYLKALILRLERLPNNLQRDDSAIDQIDPW' \
             'MEKLFQFKNDARLKELYFMVEELRISLFSQPMKTKTPVSPTRLQKVWERLGIS'

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, AutoTokenizer

pretrained_model_path = '../../Pre-Trained_Models/Primary_Structure_Model'

config = RobertaConfig.from_pretrained(pretrained_model_path)
config.num_labels = 2

model = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

state_dict = torch.load(finetuned_model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

input_ids = tokenizer.encode(input_text, add_special_tokens=True, max_length=128)
input_ids = torch.tensor(input_ids).unsqueeze(0)

outputs = model(input_ids)
predicted_class_index = torch.argmax(outputs.logits, dim=1).item()

label_dict = {0: 'Non-Terminase', 1: 'Terminase'}
print("Predicted Label:", label_dict[predicted_class_index])
