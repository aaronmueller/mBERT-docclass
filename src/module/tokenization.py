import logging
import sys
import unicodedata

from pytorch_transformers import XLMTokenizer

logger = logging.getLogger(__name__)


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def lowercase_and_remove_accent(text):
    """
    Lowercase and strips accents from a piece of text based on
    https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py
    """
    text = ' '.join(text)
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output).lower().split(' ')


def romanian_preprocessing(text):
    '''Sennrich's WMT16 scripts for Romanian preprocessing, used by model `xlm-mlm-enro-1024`'''
    # https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/normalise-romanian.py
    text = text.replace("\u015e", "\u0218").replace("\u015f", "\u0219")
    text = text.replace("\u0162", "\u021a").replace("\u0163", "\u021b")
    # https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/remove-diacritics.py
    text = text.replace("\u0218", "S").replace("\u0219", "s")  #s-comma
    text = text.replace("\u021a", "T").replace("\u021b", "t")  #t-comma
    text = text.replace("\u0102", "A").replace("\u0103", "a")
    text = text.replace("\u00C2", "A").replace("\u00E2", "a")
    text = text.replace("\u00CE", "I").replace("\u00EE", "i")
    return text


class NewXLMTokenizer(XLMTokenizer):
    def __init__(self,
                 vocab_file,
                 merges_file,
                 unk_token="<unk>",
                 bos_token="<s>",
                 sep_token="</s>",
                 pad_token="<pad>",
                 cls_token="</s>",
                 mask_token="<special1>",
                 additional_special_tokens=[
                     "<special0>", "<special1>", "<special2>", "<special3>",
                     "<special4>", "<special5>", "<special6>", "<special7>",
                     "<special8>", "<special9>"
                 ],
                 lang2id=None,
                 id2lang=None,
                 do_lowercase_and_remove_accent=True,
                 **kwargs):
        try:
            super(NewXLMTokenizer, self).__init__(
                vocab_file,
                merges_file,
                unk_token=unk_token,
                bos_token=bos_token,
                sep_token=sep_token,
                pad_token=pad_token,
                cls_token=cls_token,
                mask_token=mask_token,
                additional_special_tokens=additional_special_tokens,
                lang2id=lang2id,
                id2lang=id2lang,
                do_lowercase_and_remove_accent=do_lowercase_and_remove_accent,
                **kwargs)
        except:
            pass

        self.merges_file = vocab_file.replace('vocab.json', 'merges.txt')
        self.cache = {}
        self.bpe_ranks = dict()

    def load_bpe(self, lang):
        merges_file = self.merges_file.replace('.txt', f'.{lang}.txt')
        merges = open(merges_file, encoding='utf-8').read().split('\n')[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks[lang] = dict(zip(merges, range(len(merges))))

    def bpe(self, token, lang):
        word = tuple(token[:-1]) + (token[-1] + '</w>', )
        if token in self.cache:
            return self.cache[(lang, token)]
        if lang not in self.bpe_ranks:
            self.load_bpe(lang)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks[lang].get(pair, float('inf')))
            if bigram not in self.bpe_ranks[lang]:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[
                        i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[(lang, token)] = word
        return word

    def _tokenize(self, text, lang='en', bypass_tokenizer=False):
        if lang and self.lang2id and lang not in self.lang2id:
            logger.error(
                "Supplied language code not found in lang2id mapping. Please check that your language is supported by the loaded pretrained model."
            )
        if bypass_tokenizer:
            text = text.split()
        elif lang not in self.lang_with_custom_tokenizer:
            text = self.moses_pipeline(text, lang=lang)
            # TODO: make sure we are using `xlm-mlm-enro-1024`, since XLM-100 doesn't have this step
            if lang == 'ro':
                text = romanian_preprocessing(text)
            text = self.moses_tokenize(text, lang=lang)
        elif lang == 'th':
            text = self.moses_pipeline(text, lang=lang)
            try:
                if 'pythainlp' not in sys.modules:
                    from pythainlp.tokenize import word_tokenize as th_word_tokenize
                else:
                    th_word_tokenize = sys.modules['pythainlp'].word_tokenize
            except (AttributeError, ImportError) as e:
                logger.error(
                    "Make sure you install PyThaiNLP (https://github.com/PyThaiNLP/pythainlp) with the following steps"
                )
                logger.error("1. pip install pythainlp")
                raise e
            text = th_word_tokenize(text)
        elif lang == 'zh':
            try:
                if 'jieba' not in sys.modules:
                    import jieba
                else:
                    jieba = sys.modules['jieba']
            except (AttributeError, ImportError) as e:
                logger.error(
                    "Make sure you install Jieba (https://github.com/fxsjy/jieba) with the following steps"
                )
                logger.error("1. pip install jieba")
                raise e
            text = ' '.join(jieba.cut(text))
            text = self.moses_pipeline(text, lang=lang)
            text = text.split()
        elif lang == 'ja':
            text = self.moses_pipeline(text, lang=lang)
            text = self.ja_tokenize(text)
        else:
            raise ValueError('It should not reach here')

        if self.do_lowercase_and_remove_accent and not bypass_tokenizer:
            text = lowercase_and_remove_accent(text)

        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend([t for t in self.bpe(token, lang).split(' ')])

        return split_tokens


class LangPrefixXLMTokenizer(NewXLMTokenizer):
    def convert_tokens_to_ids(self, tokens, lang):
        """ Converts a single token, or a sequence of tokens, (str/unicode) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        def _convert_token_to_id(token, lang):
            unk_id = self.encoder.get(self.unk_token)
            raw_id = self.encoder.get(token, unk_id)
            pfx_id = self.encoder.get(f'{lang}_{token}', unk_id)
            assert (pfx_id == unk_id) or (raw_id == unk_id)
            if pfx_id == unk_id and raw_id == unk_id:
                return unk_id
            elif pfx_id == unk_id:
                return raw_id
            else:
                return pfx_id

        if isinstance(tokens, str):
            return _convert_token_to_id(tokens, lang)

        ids = []
        for token in tokens:
            ids.append(_convert_token_to_id(token, lang))
        return ids
