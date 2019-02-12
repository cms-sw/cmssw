#!/usr/bin/env python
import sys
import os
import ROOT
import argparse

# This is an example of plotting the standard tracking validation
# plots from an explicit set of DQM root files.

from Validation.RecoTrack.plotting.validation import SimpleValidation, SimpleSample

from Validation.RecoTrack.plotting.plotting import Subtract, FakeDuplicate, CutEfficiency, Transform, AggregateBins, ROC, Plot, PlotEmpty, PlotGroup, PlotOnSideGroup, PlotFolder, Plotter
from Validation.RecoTrack.plotting.html import PlotPurpose

from Validation.RecoParticleFlow.defaults_cfi import ptbins, etabins, response_distribution_name

def parse_sample_string(ss):
    spl = ss.split(":")
    if not (len(spl) >= 3):
        raise Exception("Sample must be in the format name:DQMfile1.root:DQMfile2.root:...")
    
    name = spl[0]
    files = spl[1:]

    #check that all supplied files are actually ROOT files
    for fi in files:
        print "Trying to open DQM file {0} for sample {1}".format(fi, name)
        if not os.path.isfile(fi):
            raise Exception("Could not read DQM file {0}, it does not exist".format(fi))
        tf = ROOT.TFile(fi)
        if not tf:
            raise Exception("Could not read DQM file {0}, it's not a ROOT file".format(fi))
        d = tf.Get("DQMData/Run 1/Physics/Run summary")
        if not d:
            raise Exception("Could not read DQM file {0}, it's does not seem to be a harvested DQM file".format(fi))
    return name, files

def parse_plot_string(ss):
    spl = ss.split(":")
    if not (len(spl) >= 3):
        raise Exception("Plot must be in the format folder:name:hist1:hist2:...")
    
    folder = spl[0]
    name = spl[1]
    histograms = spl[2:]

    return folder, name, histograms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sample", type=str, action='append',
        required=True,
        help="DQM files to compare for a single sample, in the format 'name:file1.root:file2.root:...:fileN.root'",
        #default=[
        #    "QCD:DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root:DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root"
        #]
    )
    parser.add_argument("-p", "--plots",
        type=str, action='append',
        required=False,
        help="Plots to put on a single canvas, in the format 'folder:name:plot1:plot2:...:plotN'",
        default = [],
        #default=[
        #    "JetResponse:reso_dist_10_24:reso_dist_10_24_eta05:reso_dist_10_24_eta13"
        #]
    )
    parser.add_argument("--doResponsePlots",
        action='store_true',
        required=False,
        help="If enabled, do all jet response plots"
    )
    args = parser.parse_args()

    #collect all the SimpleSample objects    
    samples = []
    plots = []
 
    sample_strings = args.sample
    for ss in sample_strings:
        name, files = parse_sample_string(ss)
        samp = SimpleSample(name, name, [(fn, "Option {0}".format(i)) for fn, i in zip(files, range(len(files)))])
        samples += [samp]
    
    for ss in args.plots:
        folder, name, histograms = parse_plot_string(ss)
        plots += [(folder, name, histograms)]
    
    if args.doResponsePlots:
        plots += [("JetResponse", "reso_pt", ["preso_eta05", "preso_eta13","preso_eta21","preso_eta25","preso_eta30"])]
        plots += [("JetResponse", "response_pt", ["presponse_eta05", "presponse_eta13", "presponse_eta21", "presponse_eta25", "presponse_eta30"])]
        for iptbin in range(len(ptbins)-1):
            pthistograms = []
            for ietabin in range(len(etabins)-1):
                pthistograms += [response_distribution_name(iptbin, ietabin)]
            plots += [("JetResponse", "response_{0:.0f}_{1:.0f}".format(ptbins[iptbin], ptbins[iptbin+1]), pthistograms)]

    return samples, plots

def addPlots(plotter, folder, name, section, histograms, opts):
    folders = [folder]
    plots = [PlotGroup(name, [Plot(h, **opts) for h in histograms])]
    plotter.append("ParticleFlow", folders, PlotFolder(*plots, loopSubFolders=False, page="pf", section=section))


def main():

    #plot-dependent style options
    plot_opts = {
        "reso_pt": {"xlog": True},
        "response_pt": {"xlog": True},
    }
    for iptbin in range(len(ptbins)-1):
        plot_opts["response_{0:.0f}_{1:.0f}".format(ptbins[iptbin], ptbins[iptbin+1])] = {"stat": True}

    samples, plots = parse_args()

    plotter = Plotter()

    for folder, name, histograms in plots:
        opts = plot_opts.get(name, {})
        fullfolder =  "DQMData/Run 1/Physics/Run summary/{0}".format(folder)
        print "Booking histogram group {0}={1} from folder {2}".format(name, histograms, folder)
        addPlots(plotter, fullfolder, name, folder, histograms, opts)

    outputDir = "plots" # Plot output directory
    description = "Simple ParticleFlow comparison"

    plotterDrawArgs = dict(
        separate=False, # Set to true if you want each plot in it's own canvas
    #    ratio=False,   # Uncomment to disable ratio pad
    )

    val = SimpleValidation(samples, outputDir)
    report = val.createHtmlReport(validationName=description)
    val.doPlots([plotter], plotterDrawArgs=plotterDrawArgs)
    report.write()

if __name__ == "__main__":
    main()
