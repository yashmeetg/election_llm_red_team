import anthropic
import os
from tqdm import tqdm
import random
import json
import pandas as pd

client = anthropic.Anthropic()

class Claude:
    def __init__(self, model_name):
        self.model_name = model_name
        
    def predict(self, prompt):
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.0,
            system="",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content

    
def get_all_responses_haiku(sample_val=None):
    claude_model = Claude('haiku')

    responses_df = pd.DataFrame()
    rows = []

    prompts_df = pd.read_csv('all_prompts.csv')        
    
    sample_df = prompts_df
    if sample_val != None:
        sample_df = prompts_df.sample(sample_val)
    outputs = []
    for i, row in tqdm(sample_df.iterrows()):
        response = claude_model.predict(row['prompt'])

        outputs.append(response)

    sample_df['outputs'] = outputs
    sample_df.to_csv('output_haiku.csv')

    return sample_df

if __name__ == '__main__':
    output_df = get_all_responses_haiku()