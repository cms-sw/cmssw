#! /usr/bin/env python

# Pure trick to start ROOT in batch mode, pass this only option to it
# and the rest of the command line options to this code.
import sys
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
from ROOT import TCanvas, TLegend, TPaveText, THStack, TFile, TProfile, TH1D, kGray, kAzure, kMagenta, kOrange, kWhite, kRed, kBlue, kGreen, kPink, kYellow, gROOT
gROOT.SetBatch(True)
sys.argv = oldargv


from plot_utils import setTDRStyle
from collections import namedtuple, OrderedDict
import sys, os
import argparse

def setColorIfExists(histos, h, color):
    if h in histos.keys():
        histos[h].SetFillColor(color)

def assignOrAddIfExists(h, p):
    if not h:
        h = p.ProjectionX()
    else:
        h.Add(p.ProjectionX("B"), +1.000)
    return h

def createPlots(plot):
    DETECTORS = ["TIB", "TIDF", "TIDB",
                 "InnerServices", "TOB",
                 "TEC", "PixBar",
                 "PixFwdPlus", "PixFwdMinus",
                 "Phase1PixelBarrel", "Phase2OTBarrel",
                 "Phase2OTForward", "Phase2PixelEndcap",
                 "BeamPipe"]
    IBs = ["TIB", "TIDF", "TIDB", "InnerServices", "Phase1PixelBarrel"]
    theDirName = "Figures"

    hist_x0_detectors = {}
    Plot_params = namedtuple('Params',
                             ['plotNumber', 'abscissa', 'ordinate', 'ymin', 'ymax', 'xmin', 'xmax'])
    plots = {}
    plots.setdefault("x_vs_eta", Plot_params(10, "#eta", "t/X_{0}", 0.0, 2.575, -4.0, 4.0))
    plots.setdefault("x_vs_phi", Plot_params(20, "#varphi [rad]", "t/X_{0}", 0.0, 6.2, -4.0, 4.0))
    plots.setdefault("x_vs_R",   Plot_params(40, "R [cm]", "t/X_{0}", 0.0, 70.0, 0.0, 1200.0))
    plots.setdefault("x_vs_eta", Plot_params(1010, "#eta", "t/#lambda_{I}", 0.0, 0.73, -4.0, 4.0))
    plots.setdefault("x_vs_phi", Plot_params(1020, "#varphi [rad]", "t/#lambda_{I}", 0.0, 1.2, -4.0, 4.0))
    plots.setdefault("x_vs_R",   Plot_params(1040, "#R [cm]", "t/#lambda_{I}", 0.0, 7.5, 0.0, 1200.0))
    if plot not in plots.keys():
        print("Error: chosen plot name not known %s" % plot)
        return

    hist_x0_IB = None
    # We need to keep the file content alive for the lifetime of the
    # full function....
    subDetectorFiles = []
    for subDetector in DETECTORS:
        subDetectorFilename = "matbdg_%s.root" % subDetector
        if not os.path.exists(subDetectorFilename):
            print("Error opening file: %s" % subDetectorFilename)
            continue

        subDetectorFiles.append(TFile(subDetectorFilename))
        subDetectorFile = subDetectorFiles[-1]
        print ("Opening file: %s" % subDetectorFilename)
        prof_x0_XXX = subDetectorFile.Get("%d" % plots[plot].plotNumber)

        # Merge together the "inner barrel detectors".
        if subDetector in IBs:
            hist_x0_IB = assignOrAddIfExists(hist_x0_IB, prof_x0_XXX)

        hist_x0_detectors[subDetector] = prof_x0_XXX.ProjectionX()

    cumulative_matbdg = TH1D("CumulativeSimulMatBdg",
                             "CumulativeSimulMatBdg",
                             hist_x0_IB.GetNbinsX(),
                             hist_x0_IB.GetXaxis().GetXmin(),
                             hist_x0_IB.GetXaxis().GetXmax())
    cumulative_matbdg.SetDirectory(0)

    # category profiles
    prof_x0_SUP   = subDetectorFile.Get("%d" % (100 + plots[plot].plotNumber))
    prof_x0_SEN   = subDetectorFile.Get("%d" % (200 + plots[plot].plotNumber))
    prof_x0_CAB   = subDetectorFile.Get("%d" % (300 + plots[plot].plotNumber))
    prof_x0_COL   = subDetectorFile.Get("%d" % (400 + plots[plot].plotNumber))
    prof_x0_ELE   = subDetectorFile.Get("%d" % (500 + plots[plot].plotNumber))
    prof_x0_OTH   = subDetectorFile.Get("%d" % (600 + plots[plot].plotNumber))
    prof_x0_AIR   = subDetectorFile.Get("%d" % (700 + plots[plot].plotNumber))

    hist_x0_SUP = None
    hist_x0_SEN = None
    hist_x0_CAB = None
    hist_x0_COL = None
    hist_x0_ELE = None
    hist_x0_OTH = None
    hist_x0_OTH = None

    # add to summary histogram
    hist_x0_SUP = assignOrAddIfExists( hist_x0_SUP, prof_x0_SUP )
    hist_x0_SEN = assignOrAddIfExists( hist_x0_SEN, prof_x0_SEN )
    hist_x0_CAB = assignOrAddIfExists( hist_x0_CAB, prof_x0_CAB )
    hist_x0_COL = assignOrAddIfExists( hist_x0_COL, prof_x0_COL )
    hist_x0_ELE = assignOrAddIfExists( hist_x0_ELE, prof_x0_ELE )
    hist_x0_OTH = assignOrAddIfExists( hist_x0_OTH, prof_x0_OTH )
    hist_x0_OTH = assignOrAddIfExists( hist_x0_OTH, prof_x0_AIR )

    # colors
    kpipe  = kGray+2
    kpixel = kAzure-5
    ktib   = kMagenta-2
    ktob   = kOrange+10
    ktec   = kOrange-2
    kout   = kGray
    ksen   = 27
    kele   = 46
    kcab   = kOrange-8
    kcol   = 30
    ksup   = 38
    koth   = kOrange-2

    setColorIfExists(hist_x0_detectors, "BeamPipe", kpipe) #   Beam Pipe	 = dark gray
    setColorIfExists(hist_x0_detectors, "Pixel", kpixel) #     Pixel 	 = dark blue
    setColorIfExists(hist_x0_detectors, "Phase1PixelBarrel", kpixel) #
    setColorIfExists(hist_x0_detectors, "Phase2OTBarrel", ktib) #
    setColorIfExists(hist_x0_detectors, "Phase2OTForward", ktec) #
    setColorIfExists(hist_x0_detectors, "Phase2PixelEndcap", ktib) #
    setColorIfExists(hist_x0_detectors, "TIB", ktib) # 	  TIB and TID  = violet
    setColorIfExists(hist_x0_detectors, "TID", ktib) # 	  TIB and TID  = violet
    setColorIfExists(hist_x0_detectors, "TOB", ktob) #         TOB          = red
    setColorIfExists(hist_x0_detectors, "TEC", ktec) #         TEC          = yellow gold
    setColorIfExists(hist_x0_detectors, "TkStrct", kout) #     Support tube = light gray

    hist_x0_SEN.SetFillColor(ksen) # Sensitive   = brown
    hist_x0_ELE.SetFillColor(kele) # Electronics = red
    hist_x0_CAB.SetFillColor(kcab) # Cabling     = dark orange
    hist_x0_COL.SetFillColor(kcol) # Cooling     = green
    hist_x0_SUP.SetFillColor(ksup) # Support     = light blue
    hist_x0_OTH.SetFillColor(koth) # Other+Air   = light orange



    # First Plot: BeamPipe + Pixel + TIB/TID + TOB + TEC + Outside
    # stack
    stackTitle_SubDetectors = "Tracker Material Budget;%s;%s" % (
        plots[plot].abscissa,plots[plot].ordinate)
    stack_x0_SubDetectors = THStack("stack_x0",stackTitle_SubDetectors)
    for det, histo in hist_x0_detectors.iteritems():
        stack_x0_SubDetectors.Add(histo)
        cumulative_matbdg.Add(histo, 1)

    # canvas
    can_SubDetectors = TCanvas("can_SubDetectors","can_SubDetectors",800,800)
    can_SubDetectors.Range(0,0,25,25)
    can_SubDetectors.SetFillColor(kWhite)

    # Draw
    stack_x0_SubDetectors.SetMinimum(plots[plot].ymin)
    stack_x0_SubDetectors.SetMaximum(plots[plot].ymax)
    stack_x0_SubDetectors.Draw("HIST")
    stack_x0_SubDetectors.GetXaxis().SetLimits(plots[plot].xmin, plots[plot].xmax)


    # Legenda
    theLegend_SubDetectors = TLegend(0.180,0.8,0.98,0.92)
    theLegend_SubDetectors.SetNColumns(3)
    theLegend_SubDetectors.SetFillColor(0)
    theLegend_SubDetectors.SetFillStyle(0)
    theLegend_SubDetectors.SetBorderSize(0)

    for det, histo in hist_x0_detectors.iteritems():
        theLegend_SubDetectors.AddEntry(histo, det,  "f")

    theLegend_SubDetectors.Draw()

    # text
    text_SubDetectors = TPaveText(0.180,0.727,0.402,0.787,"NDC")
    text_SubDetectors.SetFillColor(0)
    text_SubDetectors.SetBorderSize(0)
    text_SubDetectors.AddText("CMS Simulation")
    text_SubDetectors.SetTextAlign(11)
    text_SubDetectors.Draw()

    # Store
    can_SubDetectors.Update()
    can_SubDetectors.SaveAs("%s/Tracker_SubDetectors_%s.pdf" % (theDirName, plot))
    can_SubDetectors.SaveAs("%s/Tracker_SubDetectors_%s.root" % (theDirName, plot))


    # Second Plot: BeamPipe + SEN + ELE + CAB + COL + SUP + OTH/AIR +
    # Outside stack
    stackTitle_Materials = "Tracker Material Budget;%s;%s" % (plots[plot].abscissa,
                                                              plots[plot].ordinate)
    stack_x0_Materials = THStack("stack_x0",stackTitle_Materials)
    stack_x0_Materials.Add(hist_x0_detectors["BeamPipe"])
    stack_x0_Materials.Add(hist_x0_SEN)
    stack_x0_Materials.Add(hist_x0_ELE)
    stack_x0_Materials.Add(hist_x0_CAB)
    stack_x0_Materials.Add(hist_x0_COL)
    stack_x0_Materials.Add(hist_x0_SUP)
    stack_x0_Materials.Add(hist_x0_OTH)
#    stack_x0_Materials.Add(hist_x0_detectors["TkStrct"])

    # canvas
    can_Materials = TCanvas("can_Materials","can_Materials",800,800)
    can_Materials.Range(0,0,25,25)
    can_Materials.SetFillColor(kWhite)

    # Draw
    stack_x0_Materials.SetMinimum(plots[plot].ymin)
    stack_x0_Materials.SetMaximum(plots[plot].ymax)
    stack_x0_Materials.Draw("HIST")
    stack_x0_Materials.GetXaxis().SetLimits(plots[plot].xmin, plots[plot].xmax)

    # Legenda
    theLegend_Materials = TLegend(0.180,0.8,0.95,0.92)
    theLegend_Materials.SetNColumns(3)
    theLegend_Materials.SetFillColor(0)
    theLegend_Materials.SetBorderSize(0)

#    theLegend_Materials.AddEntry(hist_x0_detectors["TkStrct"],   "Support and Thermal Screen",  "f")
    theLegend_Materials.AddEntry(hist_x0_detectors["BeamPipe"],  "Beam Pipe",                   "f")
    theLegend_Materials.AddEntry(hist_x0_OTH,       "Other",                       "f")
    theLegend_Materials.AddEntry(hist_x0_SUP,       "Mechanical Structures",       "f")
    theLegend_Materials.AddEntry(hist_x0_COL,       "Cooling",                     "f")
    theLegend_Materials.AddEntry(hist_x0_CAB,       "Cables",                      "f")
    theLegend_Materials.AddEntry(hist_x0_ELE,       "Electronics",                 "f")
    theLegend_Materials.AddEntry(hist_x0_SEN,       "Sensitive",                   "f")
    theLegend_Materials.Draw()

    # text
    text_Materials = TPaveText(0.180,0.727,0.402,0.787,"NDC")
    text_Materials.SetFillColor(0)
    text_Materials.SetBorderSize(0)
    text_Materials.AddText("CMS Simulation")
    text_Materials.SetTextAlign(11)
    text_Materials.Draw()

    # Store
    can_Materials.Update()
    can_Materials.SaveAs("%s/Tracker_Materials_%s.pdf" % (theDirName, plot))
    can_Materials.SaveAs("%s/Tracker_Materials_%s.root" % (theDirName, plot))

    return cumulative_matbdg

def createPlotsReco(reco_file, label):
    cumulative_matbdg = None
    sDETS = OrderedDict()
    sDETS["PXB"] = kRed
    sDETS["PXF"] = kBlue
    sDETS["TIB"] = kGreen
    sDETS["TID"] = kYellow
    sDETS["TOB"] = kOrange
    sDETS["TEC"] = kPink
    sPREF = ["Original_RadLen_vs_Eta_", "RadLen_vs_Eta_"]

    c = TCanvas("c", "c", 1024, 1024);
    diffs = []
    if not os.path.exists(reco_file):
        print("Error: missing file %s" % reco_file)
        raise RuntimeError
    file = TFile(reco_file)
    prefix = "/DQMData/Run 1/Tracking/Run summary/RecoMaterial/"
    for s in sPREF:
        hs = THStack("hs","");
        histos = []
        for det, color in sDETS.iteritems():
            layer_number = 0
            while True:
                layer_number += 1
                name = "%s%s%s%d" % (prefix, s, det, layer_number)
                prof = file.Get(name)
                # If we miss an object, since we are incrementally
                # searching for consecutive layers, we may safely
                # assume that there are no additional layers and skip
                # to the next detector.
                if not prof:
                    print("Missing profile %s" % name)
                    break
                else:
                    histos.append(prof.ProjectionX("_px", "hist"));
                    diffs.append(histos[-1]);
                    histos[-1].SetFillColor(color + layer_number);
                    histos[-1].SetLineColor(color + layer_number + 1);

        name = "CumulativeRecoMatBdg_%s" % s
        if s == "RadLen_vs_Eta_":
            cumulative_matbdg = TH1D(name, name,
                                     histos[0].GetNbinsX(),
                                     histos[0].GetXaxis().GetXmin(),
                                     histos[0].GetXaxis().GetXmax())
            cumulative_matbdg.SetDirectory(0)
        for h in histos:
            print "Adding: ", h.GetName(), len(histos)
            hs.Add(h)
            if cumulative_matbdg:
                cumulative_matbdg.Add(h, 1.)
        hs.Draw()
        hs.GetYaxis().SetTitle("RadLen")
        c.Update()
        c.Modified()
        c.SaveAs("%sstacked_%s.png" % (s, label))
    hs = THStack("diff","")
    for d in range(0,len(diffs)/2):
        diffs[d+len(diffs)/2].Add(diffs[d], -1.)
        hs.Add(diffs[d+len(diffs)/2]);
    hs.Draw()
    hs.GetYaxis().SetTitle("RadLen")
    c.Update()
    c.Modified()
    c.SaveAs("RadLen_difference_%s.png" % label)
    return cumulative_matbdg

def materialBudget_Simul_vs_Reco(reco_file, label):
    setTDRStyle()

    # plots

    cumulative_matbdg_sim = createPlots("x_vs_eta")
    cumulative_matbdg_rec = createPlotsReco(reco_file, label)

    cc = TCanvas("cc", "cc", 1024, 1024)
    cumulative_matbdg_sim.SetMinimum(0.)
    cumulative_matbdg_sim.SetMaximum(3.5)
    cumulative_matbdg_sim.GetXaxis().SetRangeUser(-3.0, 3.0)
    cumulative_matbdg_sim.SetLineColor(kOrange)
    cumulative_matbdg_rec.SetMinimum(0.)
    cumulative_matbdg_rec.SetMaximum(3.)
    cumulative_matbdg_rec.SetLineColor(kAzure+1)
    l = TLegend(0.18, 0.8, 0.95, 0.92)
    l.AddEntry(cumulative_matbdg_sim, "Sim Material", "f")
    l.AddEntry(cumulative_matbdg_rec, "Reco Material", "f")
    cumulative_matbdg_sim.Draw("HIST")
    cumulative_matbdg_rec.Draw("HIST SAME")
    l.Draw()
    filename = "MaterialBdg_Reco_vs_Simul_%s.png" % label
    cc.SaveAs(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic Material Plotter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--reco',
                        help='Input Reconstruction Material file, DQM format',
                        required=True)
    parser.add_argument('-l', '--label',
                        help='Label to use in naming the plots',
                        required=True)
    args = parser.parse_args()
    materialBudget_Simul_vs_Reco(args.reco, args.label)
