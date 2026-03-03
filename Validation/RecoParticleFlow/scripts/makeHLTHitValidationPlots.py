#!/usr/bin/env python3

import os
import argparse
import numpy as np
import utils
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

import ROOT
import mplhep as hep
hep.style.use("CMS")

from makeHLTPFValidationPlots import plot2D, debug

if __name__ == '__main__':

    def check_list_length(value):
        if len(value) <= 1:
            raise argparse.ArgumentTypeError("List must have more than one item")
        return value

    class DependencyAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if option_string == "--compare_files_labels" and not namespace.compare_files:
                parser.error("`--compare_files_labels` requires `--compare_files` to be set")
            if option_string == "--compare_files_labels" and not namespace.met:
                parser.error("`--compare_files_labels` requires `--met` to be set")
            if len(namespace.met) > 1:
                parser.error("When comparing files only one MET collection can be used.")
            if len(values) != len(namespace.compare_files):
                parser.error("`--compare_files_labels` must have the same size as `--compare_files`")
            setattr(namespace, self.dest, values)
        
    full_command = 'python3 Validation/RecoParticleFlow/scripts/makeHLTPFValidationPlots.py --odir <your_dir> -l TTbar -f <your_ROOT_file>'
    parser = argparse.ArgumentParser(description='Make HLT PF validation plots. \nExample command:\n' + full_command)
    parser.add_argument('-o', '--odir', default="HLTPFValidationPlots", help='Path to the output directory.')
    parser.add_argument('-l', '--sample_label', default="", help='Sample label for plotting.')
    parser.add_argument('-e', '--era', default="Phase2", help="Chose between ['Phase2', 'Run3'].")
    parser.add_argument('--EnFracCut', default=0.01, help='Cut on the sim cluster energy fraction.')
    parser.add_argument('--PtCut', default=0.1, help='Cut on the sim cluster energy fraction.')
    parser.add_argument('--digis', action='store_true', help='Plot digis, otherwise plot hits.')

    mutual_excl2 = parser.add_mutually_exclusive_group(required=True)
    mutual_excl2.add_argument('-f', '--file', help='Paths to the DQM ROOT file.')
    mutual_excl2.add_argument('-x', '--compare_files', nargs='+', type=check_list_length,
                              help='Compare the same collection in different DQM files.', )
    parser.add_argument('-y', '--compare_files_labels', nargs='+',
                        action=DependencyAction, help='Compare the same collection in different DQM files.',)
    
    args = parser.parse_args()

    utils.createDir(args.odir)
    parentDir = os.path.dirname(args.odir)
    if args.odir[-1] == '/':
        parentDir = os.path.dirname(parentDir)
    utils.createIndexPHP(src=parentDir, dest=args.odir)
    
    fontsize = 16    
    colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    markers = ('o', 's', 'd')
    errorbar_kwargs = dict(capsize=3, elinewidth=0.8, capthick=2, linewidth=2, linestyle='')

    nSimClustersLabel = '# SimClusters'
    nPFClustersLabel = '# PFClusters'

    titles = {'responsePt': r"$p_{T}^{Reco}/p_{T}^{Sim}$", 
              'response': r"$E^{Reco}/E^{Sim}$", 
              'av_responsePt': r"$<p_{T}^{Reco}/p_{T}^{Sim}>$", 
              'av_response': r"$<E^{Reco}/E^{Sim}>$", 
              'resolutionPt': r"$\sigma(p_{T}^{Reco}/p_{T}^{Sim}) / <p_{T}^{Reco}/p_{T}^{Sim}>$", 
              'resolution': r"$\sigma(E^{Reco}/E^{Sim}) / <E^{Reco}/E^{Sim}>$", 
              'eff': 'Efficiency',
              'fake': 'Fake Rate',
              'split': 'Split Rate',
              'merge': 'Merge Rate'}

    dqm_dir = f'DQMData/Run 1/HLT/Run summary/ParticleFlow/' + ('Digis' if args.digis else 'Hits')
    afile = ROOT.TFile.Open(args.file)
    utils.checkRootDir(afile, dqm_dir)

    debug('Start caching Hits histograms...')
    subdirs = []
    cached_histos = {}
    directory = afile.GetDirectory(dqm_dir)
    for key in directory.GetListOfKeys():
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TDirectory):
            subdirs.append(obj.GetName())
            subdir = afile.GetDirectory(f"{dqm_dir}/{obj.GetName()}")
            for subkey in subdir.GetListOfKeys():
                name = subkey.GetName()
                cached_histos[f"{obj.GetName()}/{name}"] = subkey.ReadObj()
        else:
            name = key.GetName()
            cached_histos[name] = key.ReadObj()
    debug(' ...done.')

    if args.digis:
        vars2D = {
            'EcalEBDigisADC_Eta': dict(xtitle=r'$\max(ADC Counts)$', ytitle=r'$\eta$', var='# EB Digis'),
            'EcalEBDigisADC_Phi': dict(xtitle=r'$\max(ADC Counts)$', ytitle=r'$\phi$', var='# EB Digis'),
            'EcalEBDigisEta_Phi': dict(xtitle=r'$\eta$', ytitle=r'$\phi$', var='# EB Digis'),
            'EcalEEDigisADC_Eta': dict(xtitle=r'$\max(ADC Counts)$', ytitle=r'$\eta$', var='# EE Digis'),
            'EcalEEDigisADC_Phi': dict(xtitle=r'$\max(ADC Counts)$', ytitle=r'$\phi$', var='# EE Digis'),
            'EcalEEDigisEta_Phi': dict(xtitle=r'$\eta$', ytitle=r'$\phi$', var='# EE Digis'),
        }
    else:
        vars2D = {
            **{f'{det}UncalibRecHitsEn_Eta': dict(ytitle=r'$\eta$', var=f'# Uncalibrated {det} Reconstructed Hits', xtitle='Amplitude [ADC Counts]', logz=True)
               for det in ('EE', 'EB')},
            **{f'{det}UncalibRecHitsEn_Phi': dict(ytitle=r'$\phi$', var=f'# Uncalibrated {det} Reconstructed Hits', xtitle='Amplitude [ADC Counts]', logz=True)
               for det in ('EE', 'EB')},
            **{f'{det}UncalibRecHitsEta_Phi': dict(ytitle=r'$\phi$', var=f'# Uncalibrated {det} Reconstructed Hits', xtitle=r'$\eta$', logz=True)
           for det in ('EE', 'EB')},
            **{f'{det}UncalibRecHitsEta_Phi': dict(ytitle=r'$\phi$', var=f'# Uncalibrated {det} Reconstructed Hits', xtitle=r'$\eta$', logz=True)
           for det in ('EE', 'EB')},
            **{f'{det}RecHitsEn_Eta': dict(ytitle=r'$\eta$', var=f'# Reconstructed {det} Hits', xtitle='Energy [GeV]', logz=True)
           for det in ('EE', 'EB')},
            **{f'{det}RecHitsEn_Phi': dict(ytitle=r'$\phi$', var=f'# Reconstructed {det} Hits', xtitle='Energy [GeV]', logz=True)
           for det in ('EE', 'EB')},
            **{f'{det}RecHitsEta_Phi': dict(ytitle=r'$\phi$', var=f'# Reconstructed {det} Hits', xtitle=r'$\eta$', logz=True)
           for det in ('EE', 'EB')},
            **{f'{det}SimHitsEn_Eta': dict(ytitle=r'$\eta$', var=f'# Simulated {det} Hits', xtitle='Energy [GeV]', logz=True)
           for det in ('EE', 'EB')},
            **{f'{det}SimHitsEn_Phi': dict(ytitle=r'$\phi$', var=f'# Simulated {det} Hits', xtitle='Energy [GeV]', logz=True)
           for det in ('EE', 'EB')},
            **{f'{det}SimHitsEta_Phi': dict(ytitle=r'$\phi$', var=f'# Simulated {det} Hits', xtitle=r'$\eta$', logz=True)
           for det in ('EE', 'EB')},
            'PFRecHitsEn_Eta': dict(ytitle=r'$\eta$', var='# PF Reconstructed Hits', xtitle='Energy [GeV]', logz=False),
            'PFRecHitsEn_Phi': dict(ytitle=r'$\phi$', var='# PF Reconstructed Hits', xtitle='Energy [GeV]', logz=False),
            'PFRecHitsEta_Phi': dict(ytitle=r'$\phi$', var='# PF Reconstructed Hits', xtitle=r'$\eta$', logz=False),
        }

    for name, props in vars2D.items():
        root_hist = cached_histos[f"{name}"]
        plot2D(root_hist, args.sample_label, args.era, props, outname=os.path.join(args.odir, name))
