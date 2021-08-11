#!/usr/bin/env python3
import sys
import os
import ROOT
import argparse

# This is an example of plotting the standard tracking validation
# plots from an explicit set of DQM root files.

from Validation.RecoTrack.plotting.validation import SimpleValidation, SimpleSample

from Validation.RecoTrack.plotting.plotting import Subtract, FakeDuplicate, CutEfficiency, Transform, AggregateBins, ROC, Plot, PlotEmpty, PlotGroup, PlotOnSideGroup, PlotFolder, Plotter
from Validation.RecoTrack.plotting.html import PlotPurpose

from Validation.RecoParticleFlow.defaults_cfi import ptbins, etabins, response_distribution_name, muLowOffset, muHighOffset, npvLowOffset, npvHighOffset, candidateType, offset_name
from offsetStack import offsetStack

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
        #KH d = tf.Get("DQMData/Run 1/Physics/Run summary")
        d = tf.Get("DQMData/Run 1/ParticleFlow/Run summary")
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
    parser.add_argument("--doOffsetPlots",
        action='store_true',
        required=False,
        help="If enabled, do all offset plots"
    )
    parser.add_argument("--doMETPlots",
        action='store_true',
        required=False,
        help="If enabled, do all JetMET plots"
    )
    parser.add_argument("--doPFCandPlots",
        action='store_true',
        required=False,
        help="If enabled, do all PFCandidate plots"
    )

    parser.add_argument( "--offsetVar", type=str,   action='store', default="npv", help="variable to bin offset eT" )
    parser.add_argument( "--offsetDR",  type=float, action='store', default=0.4,   help="offset deltaR value" )
    args = parser.parse_args()

    #collect all the SimpleSample objects
    samples = []
    plots = []

    sample_strings = args.sample
    for ss in sample_strings:
        name, files = parse_sample_string(ss)
        samp = SimpleSample(name, name, [(fn, fn.split('/')[-2]) for fn in files])
        samples += [samp]

    for ss in args.plots:
        folder, name, histograms = parse_plot_string(ss)
        plots += [(folder, name, histograms)]


    # This needs to be also changed whenever changing binning
    if args.doResponsePlots:
        # Needs to add extra folders here if the DQM files have other folders of histograms
        JetFolderDirs = ["JetResponse/slimmedJets/JEC", "JetResponse/slimmedJets/noJEC", "JetResponse/slimmedJetsPuppi/JEC", "JetResponse/slimmedJetsPuppi/noJEC"]

        for JetFolderDir in JetFolderDirs:
            plots += [(JetFolderDir, "reso_pt", ["preso_eta05", "preso_eta13",
                                                  "preso_eta21","preso_eta25","preso_eta30","preso_eta50"])]
            plots += [(JetFolderDir, "reso_pt_rms", ["preso_eta05_rms",
                                                      "preso_eta13_rms","preso_eta21_rms","preso_eta25_rms","preso_eta30_rms",
                                                      "preso_eta50_rms"])]
            plots += [(JetFolderDir, "response_pt", ["presponse_eta05",
                                                      "presponse_eta13", "presponse_eta21", "presponse_eta25", "presponse_eta30",
                                                      "presponse_eta50"])]
            for iptbin in range(len(ptbins)-1):
                pthistograms = []
                for ietabin in range(len(etabins)-1):
                    pthistograms += [response_distribution_name(iptbin, ietabin)]
                plots += [(JetFolderDir, "response_{0:.0f}_{1:.0f}".format(ptbins[iptbin], ptbins[iptbin+1]), pthistograms)]

    if args.doOffsetPlots:
        if args.offsetVar == "npv" :
            varHigh, varLow = npvHighOffset, npvLowOffset
        else :
            varHigh, varLow = muHighOffset, muLowOffset
        for ivar in range( varLow, varHigh ) :
            offsetHists = []
            for itype in candidateType :
                offsetHists += [ offset_name( args.offsetVar, ivar, itype ) ]
            plots += [("Offset/{0}Plots/{0}{1}".format(args.offsetVar, ivar), "{0}{1}".format(args.offsetVar, ivar), offsetHists)]

    if args.doMETPlots:
        doMETPlots(files, plots)


    if args.doPFCandPlots:
        doPFCandPlots(files, plots)

    return samples, plots, args.doOffsetPlots, args.offsetVar, args.offsetDR, args.doPFCandPlots

# function that does METValidation from JetMET
def doMETPlots(files, plots):
    #get the names of the histograms
    #MetValidation
    METHistograms = []
    f = ROOT.TFile(files[0])
    d = f.Get("DQMData/Run 1/JetMET/Run summary/METValidation/slimmedMETsPuppi")
    for i in d.GetListOfKeys():
        METHistograms.append([i.GetName()])
    # append plots
    METFolderDirs = ["METValidation/slimmedMETs","METValidation/slimmedMETsPuppi"]
    for METFolderDir in METFolderDirs:
        for  METHistogram in  METHistograms:
            plots += [(METFolderDir, "", METHistogram)]

    #JetValidation
    JetHistograms = []
    d = f.Get("DQMData/Run 1/JetMET/Run summary/JetValidation/slimmedJets")
    for i in d.GetListOfKeys():
        JetHistograms.append([i.GetName()])
    JetValFolderDirs = ["JetValidation/slimmedJets", "JetValidation/slimmedJetsAK8", "JetValidation/slimmedJetsPuppi"]
    for JetValFolderDir in JetValFolderDirs:
        for JetHistogram in JetHistograms:
            plots += [(JetValFolderDir, "", JetHistogram)]

# does PFCandidate Plots
def doPFCandPlots(files, plots):
    #we are going to hard code the end part of the histogram names because there's only 4
    hist_list = ["Charge", "Eta", "Phi", "Log10Pt", "PtLow","PtMid", "PtHigh"]
    f = ROOT.TFile(files[0])
    d = f.Get("DQMData/Run 1/ParticleFlow/Run summary/PackedCandidates")
    #get the name of the folders, which can use to complete plot name as well probably
    PFFolderNames = []

    for i in d.GetListOfKeys():
        PFFolderNames.append(i.GetName())

    for PFFolderName in PFFolderNames:
        for hist in hist_list:
            plots += [(PFFolderName, "", [PFFolderName + hist])]


def addPlots(plotter, folder, name, section, histograms, opts, Offset=False):
    folders = [folder]
    #plots = [PlotGroup(name, [Plot(h, **opts) for h in histograms])]
    #KH print plots
    if Offset :
        plots = [PlotGroup(name, [Plot(h, **opts) for h in histograms])]
        plotter.append("Offset", folders, PlotFolder(*plots, loopSubFolders=False, page="offset", section=section))
    elif "JetResponse" in folder :
        plots = [PlotGroup(name, [Plot(h, **opts) for h in histograms])]
        plotter.append("ParticleFlow/" + section, folders, PlotFolder(*plots, loopSubFolders=False, page="pf", section=section))
        for plot in plots:
            plot.setProperties(ncols=3)
	    plot.setProperties(legendDw=-0.68)
	    plot.setProperties(legendDh=0.005)
	    plot.setProperties(legendDy=0.24)
	    plot.setProperties(legendDx=0.05)
    elif "JetMET" in folder:
        for h in histograms:
            plots = [PlotGroup(h, [Plot(h, **opts)])]
        for plot in plots:
            plot.setProperties(legendDw=-0.5)
            plot.setProperties(legendDh=0.01)
            plot.setProperties(legendDy=0.24)
            plot.setProperties(legendDx=0.05)
        plotter.append("JetMET" + section, folders, PlotFolder(*plots, loopSubFolders=False, page="JetMET", section=section))
    if "PackedCandidates" in folder:
        for h in histograms:
            if ("PtMid" in h or "PtHigh" in h):
                plots = [PlotGroup(h, [Plot(h, ymin = pow(10,-1), ylog = True)])]
            else:
                plots = [PlotGroup(h, [Plot(h, **opts)])]

        for plot in plots:
            plot.setProperties(legendDw=-0.5)
            plot.setProperties(legendDh=0.01)
            plot.setProperties(legendDy=0.24)
            plot.setProperties(legendDx=0.05)
        plotter.append("ParticleFlow/PackedCandidates/" + section, folders, PlotFolder(*plots, loopSubFolders=False, page="PackedCandidates", section= section))


def main():

    # plot-dependent style options
    # style options can be found from Validation/RecoTrack/python/plotting/plotting.py
    styledict_resolution = {"xlog": True, "xgrid":False, "ygrid":False,
        "xtitle":"GenJet pT (GeV)", "ytitle":"Jet pT resolution",
        "xtitleoffset":7.7,"ytitleoffset":3.8,"adjustMarginLeft":0.00}

    styledict_response = {"xlog": True, "xgrid":False, "ygrid":False,
        "xtitle":"GenJet pT (GeV)", "ytitle":"Jet response",
        "xtitleoffset":7.7,"ytitleoffset":3.8,"adjustMarginLeft":0.00}
    plot_opts = {
        "reso_pt": styledict_resolution,
        "reso_pt_rms": styledict_resolution,
        "response_pt": styledict_response
    }
    for iptbin in range(len(ptbins)-1):
        plot_opts["response_{0:.0f}_{1:.0f}".format(ptbins[iptbin], ptbins[iptbin+1])] = {"stat": True}

    samples, plots, doOffsetPlots, offsetVar, offsetDR, doPFCandPlots = parse_args()

    plotter = Plotter()


    for folder, name, histograms in plots:
        opts = plot_opts.get(name, {})

        #fullfolder =  "DQMData/Run 1/Physics/Run summary/{0}".format(folder)
        #fullfolder =  "DQMData/Run 1/ParticleFlow/Run summary/{0}".format(folder)
        fullJetFolder = "DQMData/Run 1/ParticleFlow/Run summary/{0}".format(folder)
        fullMETFolder = "DQMData/Run 1/JetMET/Run summary/{0}".format(folder)
        fullPFCandFolder = "DQMData/Run 1/ParticleFlow/Run summary/PackedCandidates/{0}".format(folder)
        print "Booking histogram group {0}={1} from folder {2}".format(name, histograms, folder)
        if "Offset/" in folder:
            opts = {'xtitle':'Default', 'ytitle':'Default'}
            addPlots(plotter, fullJetFolder, name, folder, histograms, opts, True)
        if "JetResponse" in folder:
            addPlots(plotter, fullJetFolder, name, folder, histograms, opts)
        if "METValidation" in folder or "JetValidation" in folder:
            addPlots(plotter, fullMETFolder, name, folder, histograms, opts)
        if doPFCandPlots:
            addPlots(plotter, fullPFCandFolder, name, folder, histograms, opts)

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

    #add tdr-style stack plots to offset html file
    if doOffsetPlots :
        offsetDir = "OffsetStacks"
        fullOffsetDir = os.path.join( outputDir, offsetDir )
        os.makedirs( fullOffsetDir )

        for s in samples :
            offFile = open( outputDir + "/" + s.label() + "_offset.html", "r")
            lines = offFile.readlines()
            offFile.close()
            for f in s.files() :
                fname = f.split('/')[-2]
                outName = offsetStack( [(fname,f)], offsetVar, offsetDR, fullOffsetDir )
                outName = outName.replace("plots/", "") #KH: This "plots" look redundant and causes trouble for .html. Stripping it off.
                addLine( outName, lines )

                for f2 in s.files() :
                    if f == f2 : continue
                    fname2 = f2.split('/')[-2]
                    outName = offsetStack( [(fname,f), (fname2,f2)], offsetVar, offsetDR, fullOffsetDir )
                    outName = outName.replace("plots/", "") #KH: This "plots" look redundant and causes trouble for .html. Stripping it off.
                    addLine( outName, lines )

            offFile = open( outputDir + "/" + s.label() + "_offset.html", "w")
            lines = "".join(lines)
            offFile.write(lines)
            offFile.close()

def addLine(name, oldLines) :
    newLines = [
        '   <td><a href="{0}">{0}</a></td>\n'.format(name),
        '  <br/>\n',
        '  <br/>\n'
    ]
    oldLines[8:len(newLines)] = newLines

if __name__ == "__main__":
    main()
