import sys
import torch
from torch.utils.data import DataLoader
import pickle
import json
from lstm_encoder import EncoderLSTM
from lstm_decoder import DecoderLSTM
from train_function import MODELS, TestingData, test, Attention
from bleu_eval import BLEU

# Load the trained seq2seq model with attention
seq2seq_model = torch.load('SavedModel/model0.h5', map_location=torch.device('cpu'))
print('Seq2Seq Model loaded: ', seq2seq_model)

# Prepare the testing datase
testing_dataset = TestingData(sys.argv[1])
test_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=True)

# Retrieve the index-to-word mapping
with open('i2w.pickle', 'rb') as file_handle:
    index_to_word_map = pickle.load(file_handle)

# Generate captions using the testing dataloader and seq2seq model
generated_captions = test(test_dataloader, seq2seq_model, index_to_word_map)

# Save the generated captions to the specified file
output_filepath = sys.argv[2]
with open(output_filepath, 'w') as output_file:
    for video_id, caption_text in generated_captions:
        output_file.write(f'{video_id},{caption_text}\n')

# Load testing labels to evaluate BLEU score
test_labels_data = json.load(open("data/testing_label.json"))

# Initialize a dictionary to store results for BLEU score evaluation
captions_result = {}

# Read the generated captions from the output file
with open(output_filepath, 'r') as result_file:
    for line in result_file:
        line = line.strip()
        comma_idx = line.index(',')
        vid_id = line[:comma_idx]
        cap_text = line[comma_idx + 1:]
        captions_result[vid_id] = cap_text

# Compute BLEU scores for each generated caption against the references
bleu_scores_list = []
for item in test_labels_data:
    refs_captions = [cap.rstrip('.') for cap in item['caption']]
    vid_bleu_score = BLEU(captions_result[item['id']], refs_captions, True)
    bleu_scores_list.append(vid_bleu_score)

# Calculate the average of the BLEU scores
avg_bleu_score = sum(bleu_scores_list) / len(bleu_scores_list)
print("Average BLEU score is:", avg_bleu_score)
