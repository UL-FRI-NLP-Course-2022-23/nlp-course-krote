import nemo.collections.nlp as nemo_nlp
import torch.cpu
from tqdm import tqdm
from googletrans import Translator
import re
from deep_translator import GoogleTranslator

def backtranslation_nemo(input_path, output_path, slen_model_path, ensl_model_path):
    slen_model = nemo_nlp.models.MTEncDecModel.restore_from('../models/v1.2.6/slen/aayn_base.nemo', map_location=torch.device('cpu'))
    ensl_model = nemo_nlp.models.MTEncDecModel.restore_from('../models/v1.2.6/ensl/aayn_base.nemo', map_location=torch.device('cpu'))
    torch.cuda.empty_cache()


    with open(input_path) as corpus:
        lines = [line.rstrip() for line in corpus]

    translated = ""

    for orig_sentence in tqdm(lines):
        english = slen_model.translate([orig_sentence], source_lang="sl", target_lang="en")
        sl_translation = ensl_model.translate(english, source_lang="en", target_lang="sl")[0]
        if sl_translation.lower() != orig_sentence.lower():
            # output_file.write(sl_translation+"\n")
            translated += sl_translation+"\n"
        else:
            translated += "---\n"
            # output_file.write("---\n")

    with open(output_path, 'w') as output_file:
        output_file.write(translated)


def get_text(text):
    matches = re.search(r"text=([^=]+),", text)
    return matches.group(1) + "\n"


def backtranslation_google(input_path, output_path):
    translator_slen = GoogleTranslator(source='sl', target='en')
    translator_ensl = GoogleTranslator(source='en', target='sl')

    with open(output_path, 'a+') as output_file:
        with open(input_path) as corpus:
            lines = [line.rstrip() for line in corpus]
            for i, source in tqdm(enumerate(lines)):
                if i > 4634:
                    english = translator_slen.translate(source)
                    sl_translation = translator_ensl.translate(english)
                    output_file.write(sl_translation+"\n")


if __name__ == "__main__":
    corpus_path = '../data/sentences.txt'
    output_path = '../data/sentences_backtranslated_cpu.txt'
    slen_model_path = '../models/v1.2.6/slen/aayn_base.nemo'
    ensl_model_path = '../models/v1.2.6/ensl/aayn_base.nemo'
    # backtranslation_nemo(input_path=corpus_path,
    #                 output_path=output_path,
    #                 slen_model_path=slen_model_path,
    #                 ensl_model_path=ensl_model_path)
    backtranslation_google(corpus_path, '../data/sentences_google.txt')
