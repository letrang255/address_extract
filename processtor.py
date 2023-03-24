#author hieu.tran5 (nlp_team)
#user trang.le4

import copy
import math
import unicodedata
from symspellpy import SymSpell, Verbosity
from symspellpy.editdistance import EditDistance, DistanceAlgorithm
import os
import re
import json
import string
from momo.sciences.nlp.address_extract.utils import remove_accents, remove_all_special_tokens

EMPTY_STRING = ''
SPACE_SYMBOL = ' '
HYPHEN_SYMBOL = '-'
SEPARATE_SYMBOL = '/'


def load_txt_file(cur_dir_path, filename, separator=None):
  with open(os.path.join(cur_dir_path, filename), 'r') as f:
    return [line.strip().split(separator) if separator else line.strip() for line in f]


class IDCardProcessor:
  def __init__(self, max_distance=2):
    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    self.max_distance = max_distance
    self.sym_spell = SymSpell(max_dictionary_edit_distance=5, prefix_length=7)
    # term_index is the column of the term and count_index is the
    # column of the term frequency
    self.sym_spell.load_dictionary(os.path.join(cur_dir_path, 'resources/single_word_dict.txt'),
                                   term_index=0,
                                   count_index=1)
    self.sym_spell.load_bigram_dictionary(os.path.join(cur_dir_path, 'resources/bigram_word_dict.txt'),
                                          term_index=0,
                                          count_index=2)
    self.distance = EditDistance(DistanceAlgorithm.LEVENSHTEIN)
    self.address_dict = json.load(open(os.path.join(cur_dir_path, 'resources/address_by_level_dict.json'), 'r'))
    self.letter2num_dict = json.load(open(os.path.join(cur_dir_path, 'resources/letter2num.json'), 'r'))
    self.place_list = json.load(open(os.path.join(cur_dir_path, 'resources/issue_places.json'), 'r'))
    self.accent2name = json.load(open(os.path.join(cur_dir_path, 'resources/accent2name.json'), 'r'))
    self.gender_list = ['Nam', 'Nữ']
    self.unit_prefixes = ['TP', 'TX', 'T', 'H', 'Q', 'P', 'X']
    self.ethnic_list = load_txt_file(cur_dir_path, 'resources/ethnics.txt')
    self.religion_list = load_txt_file(cur_dir_path, 'resources/religions.txt')
    self.country_list = load_txt_file(cur_dir_path, 'resources/countries.txt')
    self.first_name_list = json.load(open(os.path.join(cur_dir_path, 'resources/first_name.json'), 'r'))
    self.popular_first_name = ['nguyễn', 'phạm', 'đặng', 'thạch']
    self.exception_units = load_txt_file(cur_dir_path, 'resources/address_exception_units.txt', separator='|')
    self.best_supported_country = 'Việt Nam|Vietnam|country'

  def compute_distance(self, str_1, str_2, max_distance):
    norm_1, norm_2 = remove_accents(str_1), remove_accents(str_2)
    dist_1, dist_2 = self.distance.compare(str_1, str_2, max_distance), self.distance.compare(norm_1, norm_2,
                                                                                              max_distance)
    return (dist_1 + dist_2) / 2

  @staticmethod
  def common_process(text):
    dict_map = {
      "òa": "oà", "Òa": "Oà", "ÒA": "OÀ",
      "óa": "oá", "Óa": "Oá", "ÓA": "OÁ",
      "ỏa": "oả", "Ỏa": "Oả", "ỎA": "OẢ",
      "õa": "oã", "Õa": "Oã", "ÕA": "OÃ",
      "ọa": "oạ", "Ọa": "Oạ", "ỌA": "OẠ",
      "òe": "oè", "Òe": "Oè", "ÒE": "OÈ",
      "óe": "oé", "Óe": "Oé", "ÓE": "OÉ",
      "ỏe": "oẻ", "Ỏe": "Oẻ", "ỎE": "OẺ",
      "õe": "oẽ", "Õe": "Oẽ", "ÕE": "OẼ",
      "ọe": "oẹ", "Ọe": "Oẹ", "ỌE": "OẸ",
      "ùy": "uỳ", "Ùy": "Uỳ", "ÙY": "UỲ",
      "úy": "uý", "Úy": "Uý", "ÚY": "UÝ",
      "ủy": "uỷ", "Ủy": "Uỷ", "ỦY": "UỶ",
      "ũy": "uỹ", "Ũy": "Uỹ", "ŨY": "UỸ",
      "ụy": "uỵ", "Ụy": "Uỵ", "ỤY": "UỴ",
    }
    exception_pairs = [('úy', 'uý'), ('ủy', 'uỷ'), ('ũy', 'uỹ'), ('ùy', 'uỳ'), ('ụy', 'uỵ'),
                       ('úa', 'uá'), ('ủa', 'uả'), ('ũa', 'uã'), ('ùa', 'uà'), ('ụa', 'uạ')]
    text = unicodedata.normalize('NFC', text.strip().strip('.')).replace('_', ' ')
    for i, j in dict_map.items():
      text = re.sub(j + r'\b', i, text)
      text = re.sub(i + r'\B', j, text)
    for pair in exception_pairs:
      text = re.sub(r'([qQ])' + pair[0] + r'\b', r'\1' + pair[1], text)

    return text

  @staticmethod
  def filter_by_length(raw_text, corrected_text):
    if not corrected_text or len(corrected_text) < 0.5 * len(raw_text):
      corrected_text = raw_text
    return corrected_text

  def _measure_distance(self, span_unit_str: str, unit: str, max_distance: int = -1, use_acronym: bool = False,
      edit_ratio: float = 0.2):
    def gen_acronym(unit):
      def check_tok_condition(tok):
        return tok not in self.unit_prefixes and not tok.isnumeric()

      toks = unit.split()
      n_tokens = len(toks) - 2 if len(toks[-1]) == 1 else len(toks) - 1
      acr1, acr2, acr3 = [], [], []
      for i, tok in enumerate(toks):
        acr1.append(tok[0] if check_tok_condition(tok) else tok)
        acr2.append(tok[0] if i < n_tokens and check_tok_condition(tok) else tok)
        acr3.append(tok[0] if i < n_tokens and check_tok_condition(tok) else f' {tok}')
      acr1, acr2, acr3, acr4 = ''.join(acr1), ''.join(acr2), ''.join(acr3), ' '.join(acr2)
      if re.search(r'\d', unit) is not None:
        space_acr = EMPTY_STRING
        for i in range(len(acr1)):
          if len(space_acr) > 0 and space_acr[-1].isalpha() and acr1[i].isnumeric():
            space_acr += SPACE_SYMBOL
          space_acr += acr1[i]
        return [acr1, acr2, acr3, acr4, space_acr]
      return [acr1, acr2, acr3, acr4]

    min_score = math.inf
    best_unit = None
    is_acronym = False
    for alias in unit.split('|'):
      if max_distance == -1:
        max_distance = min(round(edit_ratio * len(alias)), 3)
      full_unit = self.distance.compare(alias.lower(), span_unit_str.lower(), max_distance)
      if min_score > full_unit >= 0:
        min_score = full_unit
        best_unit = alias
      if use_acronym and unit.count(' ') > 0:
        for acr_unit in gen_acronym(alias.lower()):
          acronym_unit = self.distance.compare(acr_unit.lower(), span_unit_str.lower(), max_distance)
          if acronym_unit == 0:
            min_score = acronym_unit
            best_unit = alias
            is_acronym = True

    return -1 if min_score == math.inf else min_score, best_unit, is_acronym

  def _process_digit(self, c):
    if '0' <= c <= '9':
      new_c = c
    elif c in self.letter2num_dict:
      new_c = self.letter2num_dict[c]
    else:
      new_c = EMPTY_STRING
    return new_c

  def _get_best(self, v, ref, best_score, best_v, max_distance):
    cur_score, cur_name, _ = self._measure_distance(v,
                                                    ref,
                                                    max_distance=max_distance,
                                                    use_acronym=False)
    if best_score > cur_score >= 0:
      best_score = cur_score
      best_v = cur_name
    return best_score, best_v

  def _find_best_matching_from_spans(self, current_index: int,
      best_score: float,
      tokens: list,
      addr_dict: dict,
      n_gram: int = 5,
      edit_ratio: float = 0.2):
    best_unit = None
    span_unit = []
    best_idx = -1
    for sub_i in range(current_index, max(0, current_index - n_gram), -1):
      span_unit.append(tokens[sub_i])
      if span_unit:
        span_unit_str = ' '.join(span_unit[::-1])
        for unit in addr_dict:
          unit, level = unit.rsplit('|', 1)
          cur_score, cur_unit, is_acronym = self._measure_distance(span_unit_str,
                                                                   unit,
                                                                   use_acronym=True,
                                                                   edit_ratio=edit_ratio)
          if best_score >= cur_score >= 0:
            if best_score > cur_score:
              best_unit = [(cur_unit, unit, span_unit_str, level), ]
              best_idx = [sub_i, ]
            else:
              best_unit.append((cur_unit, unit, span_unit_str, level))
              best_idx.append(sub_i)
            best_score = cur_score
    return best_unit, best_score, best_idx

  def parse_address(self, idx, current_addr_dict, num_attempts, segmented_tokens, edit_ratio):
    prefixes = [(f'{pref} ', f'{pref}.') for pref in self.unit_prefixes]
    parsed_units = []
    unit_scores = []
    best_score = math.inf
    MAX_TRIES = 3
    best_units, best_score, best_idxes = self._find_best_matching_from_spans(idx,
                                                                             best_score,
                                                                             segmented_tokens,
                                                                             current_addr_dict,
                                                                             edit_ratio=edit_ratio)
    if best_units:
      best_longest_units = {}
      for best_unit, best_idx in zip(best_units, best_idxes):
        if best_unit[1] in best_longest_units and best_idx < best_longest_units[best_unit[1]][1]:
          best_longest_units[best_unit[1]] = (best_unit, best_idx)
        elif best_unit[1] not in best_longest_units:
          best_longest_units[best_unit[1]] = (best_unit, best_idx)

      for best_unit, best_idx in sorted(best_longest_units.values(), key=lambda x: x[1]):
        cur_parsed_units = []
        cur_unit_scores = []
        for pf in prefixes:
          if best_unit[0].startswith(pf[0]):
            best_unit = (
              ' '.join([pf[1], ] + best_unit[0].split()[1:]), best_unit[1], best_unit[2], best_unit[3])
            break
        for exp_unit in self.exception_units:
          if best_unit[0] == exp_unit[0]:
            best_unit = (exp_unit[1], best_unit[1], best_unit[2], best_unit[3])
        cur_parsed_units.append(best_unit)
        cur_unit_scores.append(best_score)
        idx = best_idx
        num_attempts = 0
        if type(current_addr_dict) != list:
          lower_parsed_units, lower_unit_scores = self.parse_address(
              idx - 1, current_addr_dict[f"{cur_parsed_units[-1][1]}|{cur_parsed_units[-1][-1]}"],
              0, segmented_tokens, edit_ratio
          )
          cur_parsed_units = cur_parsed_units + lower_parsed_units
          cur_unit_scores = cur_unit_scores + lower_unit_scores
        if cur_parsed_units:
          is_changed = False
          if not parsed_units or len(cur_parsed_units) > len(parsed_units):
            is_changed = True
          elif len(cur_parsed_units) == len(parsed_units):
            for score_idx in range(len(cur_unit_scores)):
              if cur_unit_scores[score_idx] < unit_scores[score_idx]:
                is_changed = True
              else:
                is_changed = False
                if cur_unit_scores[score_idx] > unit_scores[score_idx]:
                  break
          if is_changed:
            parsed_units = cur_parsed_units
            unit_scores = cur_unit_scores

      if parsed_units:
        return parsed_units, unit_scores
    else:
      num_attempts += 1

    if idx <= 0 or ((num_attempts == 0 or num_attempts >= MAX_TRIES) and not best_units):
      return parsed_units, unit_scores

    return self.parse_address(idx - 1, current_addr_dict, num_attempts, segmented_tokens, edit_ratio)

  @staticmethod
  def preprocess_address(address):
    address = address.strip().split('<br>', 1)[0]
    address = re.sub(r'([a-z0-9]+-[a-z0-9]+){4,}$', '', address)
    address = re.sub(r'(hotline|SĐT(:| )).+$', '', address, flags=re.IGNORECASE)
    return address

  def correct_normalize_address(self, text, option=1):
    if text is None or text.lower() == 'null' or text.lower() == 'n/a' or len(text) < 2:
      return text, 0.5, True, False

    address = self.common_process(text)
    address = self.preprocess_address(address)
    try:
      result = self.sym_spell.word_segmentation(address, max_edit_distance=0)
    except:
      print(address)
      return address, 0.5, True
    if option == 0:
      address = self.common_process(result.corrected_string)
      segmented_tokens = [EMPTY_STRING, ] + re.sub(r"([!\"#$%&'()*+,-./:;<=>?@\[\]^_`{|}~])", SPACE_SYMBOL,
                                                   address).split()
    else:
      segmented_tokens = [EMPTY_STRING, ] + re.sub(r"([!\"#$%&'()*+,-./:;<=>?@\[\]^_`{|}~])", SPACE_SYMBOL,
                                                   result.corrected_string).split()
    tmp_addr_dict = copy.deepcopy(self.address_dict)
    tmp_addr_dict[self.best_supported_country] = self.address_dict
    idx = len(segmented_tokens) - 1
    num_attempts = -1
    parsed_units, unit_scores = self.parse_address(idx, tmp_addr_dict, num_attempts, segmented_tokens, edit_ratio=0.3)
    total_score = sum([(1 - score / len(unit[0])) if score != math.inf else 0.5 for unit, score in
                       zip(parsed_units, unit_scores)])

    # Look for district level when can not detect city level.
    if not parsed_units:
      unit_scores = []
      for city in self.address_dict:
        tmp_addr_dict = self.address_dict[city]
        idx = len(segmented_tokens) - 1
        num_attempts = 0
        cur_parsed_units, cur_unit_scores = self.parse_address(idx, tmp_addr_dict, num_attempts,
                                                               segmented_tokens, edit_ratio=0.2)
        if cur_parsed_units:
          cur_parsed_units.insert(0, (city.split('|', 1)[0], city, "", city.rsplit('|', 1)[-1]))
          cur_unit_scores.insert(0, 0)
          if (
              not parsed_units
              or len(cur_parsed_units) > len(parsed_units)
              or cur_unit_scores[1] < unit_scores[1]
              or (cur_unit_scores[1] == unit_scores[1]
                  and len(cur_parsed_units) == 3 and cur_unit_scores[2] < unit_scores[2])
          ):
            parsed_units = cur_parsed_units
            unit_scores = cur_unit_scores
            total_score = sum([(1 - score/len(unit[0])) if score != math.inf else 0.5
                               for unit, score in zip(parsed_units, unit_scores)])

    # Copy street address
    addr_tokens = re.sub(r'([.,])([^0-9])', r'\1 \2', address).split()
    if parsed_units:
      tmp_addr = ' '.join(addr_tokens)
      pattern_string = re.compile('\W*'.join(parsed_units[-1][2].lower().replace('\\', '').split()))
      best_k = None
      min_diff = math.inf
      for grp in re.finditer(pattern_string, tmp_addr.lower()):
        street_start = tmp_addr[:grp.start()]
        larger_n_tok = len(tmp_addr)
        tmp_correct_addr = ', '.join([street_start, ] + [unit[0] for unit in parsed_units[::-1]])
        diff = self.distance.compare(tmp_correct_addr, tmp_addr, max_distance=larger_n_tok)
        if diff < min_diff:
          min_diff = diff
          best_k = grp.start()
      if best_k is not None:
        street_start = tmp_addr[:best_k]
      else:
        max_score = -1
        for k, token in enumerate(addr_tokens):
          idx = 0
          token = token.lower()
          n_token = len(token)
          for c in parsed_units[-1][2].lower():
            if idx >= n_token:
              break
            if c == token[idx] or (c in string.punctuation and token[idx] in string.punctuation):
              idx += 1
            else:
              break
          if max_score < idx:
            best_k = k
            max_score = idx
        street_start = ' '.join(addr_tokens[:best_k])
      street_start = re.sub(r'([^0-9])\s*-\s*([^0-9])?', r'\1 \2', street_start.strip().strip(','))
      total_score /= len(parsed_units)
    else:
      street_start = address

    address_as_dict = self._format_output_as_dict(parsed_units, street_start)
    return address_as_dict

  @staticmethod
  def _format_output_as_dict(parsed_units, street_start):
    outputs = {
      'country': '',
      'city': '',
      'district': '',
      'ward': '',
      'street': ' '.join([u.capitalize() for u in street_start.split()]) if street_start else ''
    }
    for idx, unit in enumerate(parsed_units):
      outputs[unit[3]] = unit[0]
    return outputs

  def correct_name(self, text):
    name = self.common_process(text.lower().replace(HYPHEN_SYMBOL, SPACE_SYMBOL))
    if name == EMPTY_STRING:
      return name, 0.5, False
    no_accent_name = remove_accents(name)
    ho = no_accent_name.split()[0]
    best_score = math.inf
    best_name = name
    if ho in self.accent2name and no_accent_name in self.accent2name[ho]:
      for cname in self.accent2name[ho][no_accent_name]:
        best_score, best_name = self._get_best(name, cname, best_score, best_name, self.max_distance)
        if best_score == 0:
          break
      norm_name = best_name
      total_score = 1 - best_score / (len(norm_name) + 1e-9)
    else:
      name_tok = name.split()
      n_name_tok = len(name_tok)
      if n_name_tok == 1:
        first_name = name_tok[0]
        middle_name = []
        last_name = EMPTY_STRING
      else:
        first_name = name_tok[0]
        middle_name = name_tok[1: n_name_tok - 1]
        last_name = name_tok[n_name_tok - 1]

      name_list_by_type = {
        1: (self.first_name_list, self.popular_first_name),
      }
      norm_name = []
      total_score = 0
      for in_name, tp in [(first_name, 1), *zip(middle_name, [2] * len(middle_name)), (last_name, 3)]:
        if in_name.isnumeric():
          continue
        if tp == 1:
          no_accent_name = remove_accents(in_name)
          best_name = in_name
          best_score = math.inf
          if no_accent_name in name_list_by_type[tp][0]:
            for n in name_list_by_type[tp][0][no_accent_name]:
              best_score, best_name = self._get_best(in_name, n, best_score, best_name, self.max_distance)
              if best_score == 0:
                break
            norm_name.append(best_name)
            total_score += (1 - best_score / (len(best_name) + 1e-9))
          else:
            best_score = math.inf
            best_name = in_name
            for n in name_list_by_type[tp][1]:
              best_score, best_name = self._get_best(in_name, n, best_score, best_name, 1)
              if best_score == 0:
                break
            norm_name.append(best_name)
            total_score += (1 - best_score / (len(best_name) + 1e-9))
        else:
          if in_name != EMPTY_STRING:
            norm_name.append(in_name)
            total_score += 0.5
      if norm_name:
        total_score /= len(norm_name)
      norm_name = ' '.join(norm_name)

    norm_name = self.filter_by_length(text, norm_name)
    is_modified = remove_all_special_tokens(norm_name) != remove_all_special_tokens(text)
    return norm_name.upper(), total_score if abs(total_score) != math.inf else 0.5, is_modified

  def correct_idcard(self, text):
    norm_text = []
    score = 0
    for c in text:
      norm_c = self._process_digit(c)
      norm_text.append(norm_c)
      if norm_c != c:
        score += 1
    norm_text = self.filter_by_length(text, ''.join(norm_text))
    is_modified = remove_all_special_tokens(norm_text) != remove_all_special_tokens(text)
    return norm_text, (1 - score / (len(text) + 1e-9)), is_modified

  def correct_passport(self, text):
    norm_text = []
    score = 0
    num2letter = {"3": "B", "8": "B"}
    if text:
      if text[0].isalnum():
        if len(text) == 8 and text[0] in num2letter:
          norm_c = num2letter[text[0]]
          score += 1
        else:
          norm_c = text[0]
        norm_text.append(norm_c)
      for c in text[1:]:
        norm_c = self._process_digit(c)
        norm_text.append(norm_c)
        if norm_c != c:
          score += 1
    norm_text = self.filter_by_length(text, ''.join(norm_text))
    is_modified = remove_all_special_tokens(norm_text) != remove_all_special_tokens(text)
    return norm_text, (1 - score / (len(text) + 1e-9)), is_modified

  def correct_date(self, text, include_text=False):
    if include_text:
      no_limit = 'Không thời hạn'
      cur_score, _, _ = self._measure_distance(text,
                                               no_limit,
                                               max_distance=int(0.5 * len(no_limit)),
                                               use_acronym=False)
      if 0 <= cur_score != math.inf or len(re.findall('[^0-9\W]', text)) / len(text) > 0.5:
        is_modified = remove_all_special_tokens(no_limit) != remove_all_special_tokens(text)
        return no_limit, (1 - cur_score / len(no_limit)), is_modified

    score = 0
    norm_text = []
    separate_flag = False
    num_date_element = 0
    text = SEPARATE_SYMBOL + text
    for c in text[::-1]:
      if c in '-/':
        if not separate_flag and 0 < num_date_element < 4:
          if norm_text[-1] == '9':  # 999 -> 1999
            norm_text.append('1')
          elif norm_text[-1] == '0':  # 021 -> 2021
            norm_text.append('2')
        elif separate_flag and 0 < num_date_element < 2:  # 12/3/1999 -> 12/03/1999
          norm_text.append('0')
        norm_text.append(SEPARATE_SYMBOL)
        separate_flag = True
        num_date_element = 0
      elif c in '0123456789' + ''.join(self.letter2num_dict.keys()):
        if (num_date_element == 2 and separate_flag) or (num_date_element == 4 and not separate_flag):
          norm_text.append(SEPARATE_SYMBOL)
          num_date_element = 0
          separate_flag = True
        num_date_element += 1
        norm_c = self._process_digit(c)
        norm_text.append(norm_c)
        score += 1 if norm_c != c else 0
    norm_text = ''.join(norm_text[::-1]).strip(SEPARATE_SYMBOL)
    norm_text = self.filter_by_length(text, norm_text)
    is_modified = remove_all_special_tokens(norm_text) != remove_all_special_tokens(text)
    return norm_text, (1 - score / len(text)), is_modified

  def correct_ethnic(self, text):
    ethnic = self.common_process(text)
    best_score = math.inf
    best_ethnic = ethnic
    for eth in self.ethnic_list:
      best_score, best_ethnic = self._get_best(ethnic, eth, best_score, best_ethnic, self.max_distance)
      if best_score == 0:
        break
    best_ethnic = self.filter_by_length(text, best_ethnic)
    is_modified = remove_all_special_tokens(best_ethnic) != remove_all_special_tokens(text)
    return best_ethnic, (1 - best_score / len(best_ethnic)) if best_score != math.inf else 0.5, is_modified

  def correct_religion(self, text):
    religion = self.common_process(text)
    best_score = math.inf
    best_religion = religion
    for rel in self.religion_list:
      best_score, best_religion = self._get_best(religion, rel, best_score, best_religion, self.max_distance)
      if best_score == 0:
        break
    best_religion = self.filter_by_length(text, best_religion)
    is_modified = remove_all_special_tokens(best_religion) != remove_all_special_tokens(text)
    return best_religion, (1 - best_score / len(best_religion)) if best_score != math.inf else 0.5, is_modified

  def correct_gender(self, text):
    gender = self.common_process(text)
    best_score = math.inf
    best_gender = gender
    for gen in self.gender_list:
      best_score, best_gender = self._get_best(gender, gen, best_score, best_gender, self.max_distance)
      if best_score == 0:
        break
    best_gender = self.filter_by_length(text, best_gender)
    is_modified = remove_all_special_tokens(best_gender) != remove_all_special_tokens(text)
    return best_gender, (1 - best_score / len(best_gender)) if best_score != math.inf else 0.5, is_modified

  def correct_nationality(self, text):
    country = self.common_process(text)
    best_score = math.inf
    best_country = country
    for ct in self.country_list:
      best_score, best_country = self._get_best(country, ct, best_score, best_country, self.max_distance)
      if best_score == 0:
        break
    best_country = self.filter_by_length(text, best_country)
    is_modified = remove_all_special_tokens(best_country) != remove_all_special_tokens(text)
    return best_country, (1 - best_score / len(best_country)) if best_score != math.inf else 0.5, is_modified

  def correct_issue_place(self, text):
    place = self.common_process(text)
    best_score = math.inf
    best_place = place
    check_contain = False
    for prefix, places in self.place_list.items():
      if prefix == EMPTY_STRING:
        check_contain = True
      for iplace in places:
        if check_contain and place in iplace:
          is_modified = remove_all_special_tokens(iplace) != remove_all_special_tokens(text)
          return iplace, 1.0, is_modified
        for p in prefix.split('|'):
          best_score, best_place = self._get_best(place,
                                                  f"{p} {iplace}".strip(),
                                                  best_score,
                                                  best_place,
                                                  self.max_distance)
          if best_score == 0:
            best_place = self.filter_by_length(text, best_place.strip())
            is_modified = remove_all_special_tokens(best_place) != remove_all_special_tokens(text)
            return best_place, (1 - best_score / len(best_place)), is_modified

    best_place = self.filter_by_length(text, best_place.strip())
    is_modified = remove_all_special_tokens(best_place) != remove_all_special_tokens(text)
    return best_place, (1 - best_score / len(best_place)) if best_score != math.inf else 0.5, is_modified
