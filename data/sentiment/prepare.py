
import numpy as np

class TokenizerWrapper:
    def __init__(self, fname, tokenizer, threshold_select=0., sep="\t"):
        self.fname = fname
        self.data = self._read_data(f"original/{fname}_data.txt", sep)
        self.tokenizer = tokenizer
        self.threshold_select = threshold_select
        
        self._process()
    
    def _read_data(self, fname, sep):
        data = {
            "text":[],
            "selected_text":[],
            "tokens":[],
            "tags":[]
        }
        with open(fname) as fin:
            for line in fin:
                #print(line)
                _, text, selected_text = line.split(sep)
                data["text"].append(text)
                data["selected_text"].append(selected_text.strip())
        return data

    def _process(self):
        self.objects_ = self.tokenizer.encode_batch(self.data["text"])
        self.data["tokens"] = [t.tokens for t in self.objects_]
        for i, obj in enumerate(self.objects_):
            text = self.data["text"][i]
            selected_text = self.data["selected_text"][i]
            if len(selected_text):
                self.data["tags"].append(self._create_target(obj, text, selected_text))
            else:
                self.data["tags"].append([])

    def _create_target(self, obj, text, selected_text):
        offsets = obj.offsets

        # find selected text index
        index_from = text.find(selected_text)
        assert index_from>=0, f"Text not found! text:/*{text}*/ selected_text:/*{selected_text}*/"
        
        index_to = index_from + len(selected_text)
        selected_text_mask = np.zeros(len(text))
        selected_text_mask[index_from:index_to-1] = 1
        
        target = []
        for start, end in offsets:
            target.append("I-" if np.mean(selected_text_mask[start:end])>self.threshold_select else "O")
        
        counts = sum([k=="I-" for k in target])
        if counts>0:
            target[target.index("I-")] = "B-"
        if counts>1:
            target[len(target) - 1 - target[::-1].index("I-")] = "E-"

        return target
    
    def save(self):
        with open(f"{self.fname}.txt","w") as fout:
            for tokens, tags in zip(self.data["tokens"],self.data["tags"]):
                if len(tags) == 0:
                    lines = "\n".join(tokens[1:-1])
                else:
                    lines = "\n".join([f"{text_one} {tags_one}" for (text_one,tags_one) in zip(tokens[1:-1], tags[1:-1])])
                fout.write(lines+"\n\n")





if __name__ == "__main__":
    from tokenizers import BertWordPieceTokenizer
    import transformers
    import os

    DISTILBERT_PATH = 'distilbert-base-uncased'

    # Save Tokenizer locally
    if not os.path.isdir(DISTILBERT_PATH):
        tmp_tok = transformers.DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)
        os.mkdir(DISTILBERT_PATH)
        tmp_tok.save_pretrained(DISTILBERT_PATH)
        del tmp_tok

    # Testing the Tokenizer
    tokenizer = BertWordPieceTokenizer(f'{DISTILBERT_PATH}/vocab.txt', lowercase=True)

    train_wrap = TokenizerWrapper("train", tokenizer)
    train_wrap.save()
    dev_wrap = TokenizerWrapper("dev", tokenizer)
    dev_wrap.save()
    test_wrap = TokenizerWrapper("test", tokenizer)
    test_wrap.save()
