#!/usr/bin/env python3

import os, sys
import argparse
import numpy as np
import hist
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import array
import ROOT
import mplhep as hep
hep.style.use("CMS")

from dqm_plot import DQMPlotter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make Tau validation plots.')
    parser.add_argument('-f', '--file', type=str, required=True,                                   help='Paths to the DQM ROOT file.')
    parser.add_argument('-s', '--step', type=str, default='HLT',                                   help='Validation step ("HLT" or "Offline")')
    parser.add_argument('-o', '--odir', type=str, default="HLTTauValidationPlots", required=False, help='Path to the output directory.')
    parser.add_argument('-l', '--label', type=str, default="TenTau (200 PU)", required=False,      help='Sample label for plotting.')
    args = parser.parse_args()

    plotter = DQMPlotter(figsize=(10, 10))

    file_path = args.file
    root_file = ROOT.TFile.Open(file_path, "READ")
    if not root_file or root_file.IsZombie():
        print(f"Error: Could not open {file_path}")
        return [], {}

