#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

import os
import subprocess
import locale
locale.getpreferredencoding = lambda: "UTF-8"

def download(lang, tgt_dir="./"):
  lang_fn, lang_dir = os.path.join(tgt_dir, lang+'.tar.gz'), os.path.join(tgt_dir, lang)
  os.makedirs(lang_dir, exist_ok=True)

  cmd = ";".join([
        f"wget https://dl.fbaipublicfiles.com/mms/tts/{lang}.tar.gz -O {lang_fn}",
        f"tar zxvf {lang_fn} --directory {tgt_dir}"
  ])
  print(f"Download model for language: {lang}")
  subprocess.check_output(cmd, shell=True)
  print(f"Model checkpoints in {lang_dir}: {os.listdir(lang_dir)}")
  return lang_dir


LANGS = ["eng", "deu"]

for LANG in LANGS:
    ckpt_dir = download(LANG, 'mms_ckpt/')