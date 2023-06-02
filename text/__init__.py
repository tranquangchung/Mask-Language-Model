""" from https://github.com/keithito/tacotron """
import re
import os
from text import cleaners
from text.symbols import _silences, symbols
import pdb

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
print(_symbol_to_id)
print(len(_symbol_to_id))

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

###############
def load_lexicon_phone(filename):
    lexicon_phone = dict()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            lexicon, phone = line.strip().split(" ", 1)
            lexicon_phone[lexicon] = phone
    return lexicon_phone

def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)
  return sequence

def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')

# def load_dictionary_llm():
#   path = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/model/vocab/vietnamese_mt5.txt"
#   dictionary = {}
#   with open(path, "r") as fread:
#     data = fread.readlines()
#     for item in data:
#       item = item.strip().split('\t')
#       dictionary[item[0]] = item[1].strip()
#   return dictionary
# dictionary_llm = load_dictionary_llm()
# def text_to_sequence_mt5(sequence, tokenizer):
#   token_subwords = []
#   subwords = []
#   for word in sequence.split():
#     tokenized = tokenizer.batch_encode_plus([word.strip()], padding='longest', max_length=512, truncation=True,
#                                             return_tensors="pt")
#     token_subword = tokenized.input_ids[0][:-1].tolist()
#     print(word, "-", tokenizer.convert_ids_to_tokens(token_subword), "-", token_subword)
#     token_subwords += token_subword
#     subwords += tokenizer.convert_ids_to_tokens(token_subword)
#   pdb.set_trace()
#   token_subwords = [number for number in token_subwords if number != 259]
#   return token_subwords

def text_to_sequence_mt5(text, cleaner_names, tokenizer):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence_mt5(m.group(2), tokenizer)
    text = m.group(3)
  return sequence

def chunk(text, length=15):
   words = text.split()
   for i in range(0, len(words), length):
       yield " ".join(words[i: i + length])

def split_text(text, length=15):
    output = list(chunk(text, length))
    return output

def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  # for s in symbols:
  #   if _should_keep_symbol(s):
  #       pass
  #   else:
  #       print("$"*20)
  #       print(s)
  #       print("$"*20)
  # print(symbols)
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]

# def _phone_to_sequence(text):
#     sequence = []
#     words = text.split(" ")
#     for word in words:
#         for phone in lexicon_phone[word].split():
#             sequence.append(phone_to_id[phone])
#     return sequence

def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])

def _arpabet_to_sequence_mt5(text, tokenizer):
  id_subwords = []
  for s in text.split():
    alphabet = s.split("_")[-1]
    id_subword = tokenizer.convert_tokens_to_ids(alphabet)
    if id_subword == 2 and alphabet == "sp":
      id_subword = 290 # tokenizer.convert_tokens_to_ids('_') thay the token sp bang 1 token _
    id_subwords.append(id_subword)
  return id_subwords

def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'

def sil_phonemes_ids():
    return [_symbol_to_id[sil] for sil in _silences]

def _arpabet_to_sequence_mt5(text, tokenizer):
  id_subwords = []
  for s in text.split():
    alphabet = s.split("_")[-1]
    id_subword = tokenizer.convert_tokens_to_ids(alphabet)
    if id_subword == 2 and alphabet == "sp":
      id_subword = 290 # tokenizer.convert_tokens_to_ids('_') thay the token sp bang 1 token _
    id_subwords.append(id_subword)
  return id_subwords