import os
import re
import glob
import json
import tempfile
import math
import torch
from torchaudio import save as save_audio
#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

import argparse
import subprocess
import pkg_resources
from natural_voices.data_utils import TextMapper
from natural_voices.models import SynthesizerTrn
from natural_voices.text import _clean_text
from natural_voices.utils import get_hparams_from_file, load_checkpoint


class VITS():
    def __init__(self, model_path, device) -> None:

        self.device = device
        self.model_path = model_path
        self.config_path = os.path.join(model_path, 'config.json')

        # Load config
        assert os.path.isfile(self.config_path), f"{self.config_path} doesn't exist"
        self.hps = get_hparams_from_file(self.config_path)

        self.lang = self.hps.data.lang
        vocab_file = pkg_resources.resource_filename(__name__, f"vocabs/{self.lang}.txt")

        g_pth = os.path.join(self.model_path, f'G.pth')
        assert os.path.isfile(g_pth), f"{g_pth} doesn't exist"

        # Make text mapper
        self.text_mapper = TextMapper(vocab_file, self.lang)

        # Make actual VITS model
        net_g = SynthesizerTrn(
            len(self.text_mapper.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)
        
        net_g.to(device)
        _ = net_g.eval()

        # Load checkpoint
        print(f"load {g_pth}")
        _ = load_checkpoint(g_pth, net_g, None)

        self.model = net_g 

    def __call__(self, txt):
        cleaned_text = _clean_text(txt, ['basic_cleaners'])
        print('Cleaned text: ', cleaned_text)
        stn_tst = self.text_mapper(cleaned_text)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            hyp = self.model.infer(
                            x_tst, x_tst_lengths, noise_scale=.667,
                            noise_scale_w=0.8, length_scale=1.0
                        )[0][0].cpu().float()
        return hyp


def main(args):

    # Get device 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Run inference with {device}")

    # Make model
    vits_model = VITS(args.model_path, device)
    
    # Get input text and generate audio
    # txt = input(f"\nWrite some text in {vits_model.lang}: ")
    if args.txt:
        txt = args.txt
    
    else:
        txt = "Die Schweizer Medien arbeiten zusammen mit der Eidgenössischen Technischen HochschuleZürich, um eine künstlichen Intelligenz zu entwickeln, die Deutsche Sätze schweizerisch auszusprechen kann ."
    
    for i in range(10):
        audio = vits_model(txt)

        # save
        os.makedirs("results", exist_ok=True)
        sample_path = os.path.join("results", f"{args.model_path.replace('/', '_')}_{i}_{args.epoch}.wav")
        print(f"File saved at {sample_path}")
        save_audio(sample_path, audio, vits_model.hps.data.sampling_rate, encoding="PCM_S")


if __name__ == "__main__":
    # parse arguments for rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", '-m', type=str, help="Path to the model folder.")
    parser.add_argument("--epoch", '-e', type=int, default=10000)
    parser.add_argument("--txt", '-t', type=str)
    args = parser.parse_args()
    main(args)