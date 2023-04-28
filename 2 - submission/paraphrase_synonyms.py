import xml.etree.ElementTree as ET
from lxml import etree as ET
import stanza
from typing import Optional, Sequence, Dict
from tqdm import tqdm


class SynonymModel:
    def __init__(self, thesaurus_path: str, synonym_threshold: float = 0.05, output_path: str = "data/outputs/synonyms.txt"):
        self.thesaurus = ET.parse(thesaurus_path)
        self.threshold = synonym_threshold
        self.allowed_pos = ["ADJ", "NOUN", "VERB"]
        self.output_path = output_path
        self.doc = None

    def get_replaceable_words(self, sentence) -> Sequence[Dict]:
        items = []
        for word in sentence.words:
            items.append({"text": word.text, "POS": word.upos, "replace": word.upos in self.allowed_pos})
        return items

    def get_synonym(self, word) -> Optional[str]:
        record = self.thesaurus.xpath(f"./entry[headword[text() = '{word}']]")
        if record is not None:
            try:
                synonym_record = record[0].xpath("./groups_core/group[1]/candidate[1]")
                if synonym_record is not None:
                    score = synonym_record[0].attrib['score']
                    if float(score) < self.threshold:
                        return word  # don't change the word if the synonym has too low score
                    else:
                        synonym = synonym_record[0].xpath("./s/text()")[0]
                        return synonym
            except IndexError:
                return word

    def replace_with_synonyms(self, corpus, nlp):
        doc = nlp(corpus)
        for sentence in tqdm(doc.sentences):
            new_sentence = ""
            replaceable_words = self.get_replaceable_words(sentence)
            for i, w in enumerate(replaceable_words):
                text = w["text"]
                if w["replace"]:
                    orig_text = text
                    text = self.get_synonym(text)

                if i == 0:
                    new_sentence += text.capitalize()
                else:
                    if w["POS"] == "PUNCT":
                        if w["text"] == "(":
                            new_sentence += " "
                        new_sentence += text
                    else:
                        if new_sentence[-1] == "(":
                            new_sentence += f"{text}"
                        else:
                            new_sentence += f" {text}"
            with open(self.output_path, "a+") as f:
                f.write(new_sentence+"\n")


if __name__ == "__main__":
    thesaurus_path = '/data/thesaurus/CJVT_Thesaurus-v1.0.xml'
    corpus_path = "/data/sentences_google.txt"
    # corpus = open(corpus_path).read()

    model = SynonymModel(thesaurus_path)
    model.get_synonym("videti")
