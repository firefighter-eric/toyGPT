import re

import pdfplumber
from tqdm import tqdm

# %% config
exp_dir = 'data/three-body'
pdf_path = f'{exp_dir}/p0/三体-全三部-刘慈欣.pdf'
stop_words_path = f'{exp_dir}/p0/stop_words.txt'

output_char_path = f'{exp_dir}/p0/char.txt'
output_text_path = f'{exp_dir}/p0/three_body.txt'


# %%

def read_pdf(path: str) -> list[str]:
    outputs = []
    with pdfplumber.open(path) as f:
        for page in tqdm(f.pages, desc='Reading PDF'):
            outputs.append(page.extract_text())
    return outputs


pages = read_pdf(pdf_path)
stop_words = open(stop_words_path, 'r').read().split('\n')

for stop_word in stop_words:
    pages = [_.replace(stop_word, '') for _ in pages]

# remove next page
doc = ''.join(pages)
char = sorted(set(doc))

# remove new line
doc = re.sub('([^！？。”’])\n', r'\1', doc)

# %% save
with open(output_char_path, 'w') as f:
    f.write('\n'.join(char))

with open(output_text_path, 'w') as f:
    f.write(doc)
print('done')
