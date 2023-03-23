import os
import json
import unicodedata
from collections import Counter
import re
import string


def create_address_dictionary():
  def get_bigram(s):
    return [' '.join(s[i:i+2]) for i in range(len(s)-1)]
  single_dict = Counter()
  bigram_dict = Counter()
  adminstrative_dict_by_level = {}
  adminstrative_data = json.load(open('resources/raw/sorted.json', 'r'))
  for province_data in adminstrative_data:
    if province_data[1] not in adminstrative_dict_by_level:
      adminstrative_dict_by_level[province_data[1]] = {}
    single_dict.update(province_data[1].split())
    single_dict.update(province_data[2].split())
    if province_data[1].count(' ') > 0:
      bigram_dict.update(get_bigram(province_data[1].split()))
      bigram_dict.update(get_bigram(province_data[2].split()))

    for district_data in province_data[-1]:
      if district_data[1].isnumeric():
        distric_name = f"{district_data[2]} {district_data[1]}"
      else:
        distric_name = district_data[1]
      if distric_name not in adminstrative_dict_by_level[province_data[1]]:
        adminstrative_dict_by_level[province_data[1]][distric_name] = []
      single_dict.update(district_data[1].split())
      single_dict.update(district_data[2].split())
      if district_data[1].count(' ') > 0:
        bigram_dict.update(get_bigram(district_data[1].split()))
        bigram_dict.update(get_bigram(district_data[2].split()))
      for ward_data in district_data[-1]:
        if ward_data[1].isnumeric():
          ward_name = f"{ward_data[2]} {ward_data[1]}"
        else:
          ward_name = ward_data[1]
        adminstrative_dict_by_level[province_data[1]][distric_name].append(ward_name)
        single_dict.update(ward_data[1].split())
        single_dict.update(ward_data[2].split())
        if ward_data[1].count(' ') > 0:
          bigram_dict.update(get_bigram(ward_data[1].split()))
          bigram_dict.update(get_bigram(ward_data[2].split()))



  with open('resources/single_word_dict.txt', 'w') as f:
    for k, v in single_dict.items():
      f.write(f'{k.lower()} {v}\n')
  with open('resources/bigram_word_dict.txt', 'w') as f:
    for k, v in bigram_dict.items():
      k = k.lower().split()
      if len(k) < 2:
        print(k, v)
      f.write(f'{k[0]} {k[1]} {v}\n')
  with open('resources/address_by_level_dict.json', 'w') as f:
    json.dump(adminstrative_dict_by_level, f, indent=4, ensure_ascii=False)


def create_address_dictionary_with_tree():
  def get_bigram(s):
    return [' '.join(s[i:i + 2]) for i in range(len(s) - 1)]
  def remove_prefix(alias):
    with open('resources/unit_label.txt', 'r') as f: unit_labels = [line.strip().split('|') for line in f]
    for units in unit_labels:
      for unit in units:
        if alias.startswith(f"{unit} "):
          replaced_alias = alias.replace(unit, '', 1).strip()
          return [alias.replace(unit, '', 1).strip(),] + list(map(lambda u: f'{u} {replaced_alias}', units))
    return [alias,]

  single_dict = Counter()
  bigram_dict = Counter()
  adminstrative_dict_by_level = {}
  adminstrative_data = json.load(open('resources/raw/tree.json', 'r'))
  for province_data in adminstrative_data.values():
    aliases = []
    for alias in province_data[0][::-1]:
      if re.search(r'\d', alias) is None:
        alias = remove_prefix(alias)
        for a in alias:
          if a not in aliases:
            aliases.append(a)
      else:
        if alias not in aliases:
          aliases.append(alias)
    province_name = '|'.join(aliases)
    if province_name not in adminstrative_dict_by_level:
      adminstrative_dict_by_level[province_name] = {}
    for alias in province_data[0]:
      alias = alias.split()
      single_dict.update(alias)
      if len(alias) > 1:
        bigram_dict.update(get_bigram(alias))

    for district_data in province_data[1].values():
      aliases = []
      for alias in district_data[0][::-1]:
        if re.search(r'\d', alias) is None:
          alias = remove_prefix(alias)
          for a in alias:
            if a not in aliases:
              aliases.append(a)
        else:
          if alias not in aliases:
            aliases.append(alias)
          num = re.search(r'\d+', alias).group()
          if len(num) == 1:
            aliases.append(alias.replace(num, '0' + num))
          elif num[0] == '0':
            aliases.append(alias.replace(num, num[1:]))
      distric_name = '|'.join(aliases)
      if distric_name not in adminstrative_dict_by_level[province_name]:
        adminstrative_dict_by_level[province_name][distric_name] = []
      for alias in district_data[0]:
        alias = alias.split()
        single_dict.update(alias)
        if len(alias) > 1:
          bigram_dict.update(get_bigram(alias))
      if district_data[1]:
        for ward_data in district_data[1].values():
          aliases = []
          for alias in ward_data[0][::-1]:
            if re.search(r'\d', alias) is None:
              alias = remove_prefix(alias)
              for a in alias:
                if a not in aliases:
                  aliases.append(a)
            else:
              if alias not in aliases:
                aliases.append(alias)
              num = re.search(r'\d+', alias).group()
              if len(num) == 1:
                aliases.append(alias.replace(num, '0' + num))
              elif num[0] == '0':
                aliases.append(alias.replace(num, num[1:]))
          ward_name = '|'.join(aliases)
          adminstrative_dict_by_level[province_name][distric_name].append(ward_name)
          for alias in ward_data[0]:
            alias = alias.split()
            single_dict.update(alias)
            if len(alias) > 1:
              bigram_dict.update(get_bigram(alias))

  with open('resources/single_word_dict.txt', 'w') as f:
    for k, v in single_dict.items():
      f.write(f'{k.lower()} {v}\n')
  with open('resources/bigram_word_dict.txt', 'w') as f:
    for k, v in bigram_dict.items():
      k = k.lower().split()
      if len(k) < 2:
        print(k, v)
      f.write(f'{k[0]} {k[1]} {v}\n')
  with open('resources/address_by_level_dict.json', 'w') as f:
    json.dump(adminstrative_dict_by_level, f, indent=4, ensure_ascii=False)


def create_name_dictionary():
  def get_bigram(s):
    return [' '.join(s[i:i + 2]) for i in range(len(s) - 1)]

  single_dict = Counter()
  bigram_dict = Counter()
  first_name = set()
  middle_name = set()
  last_name = set()
  accent2name = {}
  with open('resources/tmp.txt', 'r') as f:
    for line in f:
      line = line.replace('-', ' ').replace('.', '').replace(',', '').strip().lower()
      line = ' '.join(line.split())
      no_accent_line = remove_accents(line)
      if no_accent_line not in accent2name:
        accent2name[no_accent_line] = set()
      accent2name[no_accent_line].add(line)
      tokens = line.split()
      single_dict.update(tokens)
      bigram_dict.update(get_bigram(tokens))
      if len(tokens) > 1:
        first_name.add(tokens[0])
      if len(tokens) > 2:
        middle_name.update(tokens[1:-1])
      last_name.add(tokens[-1])

  with open('resources/accent2name.json', 'w') as f:
    for key, value in accent2name.items():
      accent2name[key] = list(value)
    json.dump(accent2name, f, indent=4, ensure_ascii=False)
  with open('resources/last_name.txt', 'w') as out_file:
    out_file.write('\n'.join(last_name))
  with open('resources/middle_name.txt', 'w') as out_file:
    out_file.write('\n'.join(middle_name))
  with open('resources/first_name.txt', 'w') as out_file:
    out_file.write('\n'.join(first_name))
  with open('resources/single_name.txt', 'w') as out_file:
    for k, v in single_dict.items():
      out_file.write(f'{k.lower()} {v}\n')
  with open('resources/bigram_name.txt', 'w') as out_file:
    for k, v in bigram_dict.items():
      k = k.lower().split()
      if len(k) < 2:
        print(k, v)
      out_file.write(f'{k[0]} {k[1]} {v}\n')

def update_name_dictionary():
  def get_bigram(s):
    return [' '.join(s[i:i + 2]) for i in range(len(s) - 1)]

  filename = 'eval/uit_member.json'
  single_dict = Counter()
  bigram_dict = Counter()
  first_name = set()
  middle_name = set()
  last_name = set()
  with open('resources/last_name.txt', 'r') as f:
    for line in f:
      last_name.add(line.strip())
  with open('resources/middle_name.txt', 'r') as f:
    for line in f:
      middle_name.add(line.strip())
  with open('resources/first_name.txt', 'r') as f:
    for line in f:
      first_name.add(line.strip())
  with open('resources/accent2name.json', 'r') as f:
    accent2name = json.load(f)

  file_type = filename[filename.rfind('.')+1:]
  with open(f'resources/{filename}', 'r', encoding='utf-8') as f:
    if file_type == 'json':
      data = json.load(f)
      for row in data:
        line = ' '.join(row['full_name'].split()).lower()
        no_accent_line = remove_accents(line)
        if no_accent_line not in accent2name:
          accent2name[no_accent_line] = []
        if line not in accent2name[no_accent_line]:
          accent2name[no_accent_line].append(line)
        tokens = line.split()
        single_dict.update(tokens)
        bigram_dict.update(get_bigram(tokens))
        if len(tokens) > 1:
          first_name.add(tokens[0])
        if len(tokens) > 2:
          middle_name.update(tokens[1:-1])
        last_name.add(tokens[-1])
    elif file_type != 'json':
      for idx, line in enumerate(f):
        if idx == 0:
          continue
        if file_type == 'csv':
          line = line.split(',')[0]
        if re.search(r'\d', line) is not None or not line.isupper():
          print(line)
          continue
        line = line.replace('-', ' ').replace('.', '').replace(',', '').strip().lower()
        line = ' '.join(line.split())
        # tmp = line[0]
        # for c in line[1:]:
        #     if c == ' ':
        #         tmp += c
        #     if c.isupper():
        #         tmp += ' ' + c
        #     else:
        #         if tmp[-1] == ' ':
        #             tmp = tmp[:-1] + c
        #         else:
        #             tmp += c
        # line = ' '.join(tmp.split()).lower()
        if line.count(' ') <= 1:
          print(line)
          continue
        no_accent_line = remove_accents(line)
        if no_accent_line not in accent2name:
          accent2name[no_accent_line] = []
        if line not in accent2name[no_accent_line]:
          accent2name[no_accent_line].append(line)
        tokens = line.split()
        single_dict.update(tokens)
        bigram_dict.update(get_bigram(tokens))
        if len(tokens) > 1:
          first_name.add(tokens[0])
        if len(tokens) > 2:
          middle_name.update(tokens[1:-1])
        last_name.add(tokens[-1])

  with open('resources/accent2name.json', 'w') as f:
    for key, value in accent2name.items():
      accent2name[key] = list(value)
    json.dump(accent2name, f, indent=4, ensure_ascii=False)
  with open('resources/last_name.txt', 'w') as out_file:
    out_file.write('\n'.join(last_name))
  with open('resources/middle_name.txt', 'w') as out_file:
    out_file.write('\n'.join(middle_name))
  with open('resources/first_name.txt', 'w') as out_file:
    out_file.write('\n'.join(first_name))
  # with open('resources/single_name.txt', 'w') as out_file:
  #     for k, v in single_dict.items():
  #         out_file.write(f'{k.lower()} {v}\n')
  # with open('resources/bigram_name.txt', 'w') as out_file:
  #     for k, v in bigram_dict.items():
  #         k = k.lower().split()
  #         if len(k) < 2:
  #             print(k, v)
  #         out_file.write(f'{k[0]} {k[1]} {v}\n')


def remove_accents(input_str):
  s1 = u'àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'
  s0 = u'aaaaeeeiioooouuyadiuouaaaaaaaaaaaaeeeeeeeeiioooooooooooouuuuuuuyyyy'
  return ''.join([s0[s1.index(c)] if c in s1 else c for c in input_str])

def remove_all_special_tokens(text):
  text = text.lower()
  text = re.sub(f"[{string.punctuation}\s\t]", '', text)
  text = unicodedata.normalize('NFC', text)
  return text


def evaluate(pred: dict, label: dict):
  def compare_address(pred_addr: list, label_addr: str):
    flag = True
    for unit in pred_addr:
      if label_addr.rfind(unit[0].lower()) == -1:
        flag = False
        break
      else:
        label_addr = label_addr.rstrip(unit[0].lower())

    return flag

  def compare_name(pred_name: list, label_name: str):
    pass

  def compare_birthday(pred_birth: list, label_birth: str):
    pass

  def compare_idcard(pred_id: list, label_id: str):
    pass

  def compare_issue_place(pred_iplace: list, label_iplace: str):
    pass

  n_correct = n_error = 0
  n_sample = 0
  for pred_addr, label_addr in zip(pred['current_address'], label['current_address']):
    n_sample += 1
    if compare_address(pred_addr, label_addr):
      n_correct += 1

  return n_correct/n_sample


if __name__ == '__main__':
  # create_address_dictionary()
  # create_address_dictionary_with_tree()
  update_name_dictionary()
  # create_name_dictionary()
