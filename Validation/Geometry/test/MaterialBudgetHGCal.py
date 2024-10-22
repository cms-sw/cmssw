#! /usr/bin/env python3

# Pure trick to start ROOT in batch mode, pass this only option to it
# and the rest of the command line options to this code.
from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from array import array
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
from ROOT import TCanvas, TLegend, TPaveText, THStack, TFile, TLatex, TPaveLabel
from ROOT import TProfile, TProfile2D, TH1D, TH2D, TH2F, TPaletteAxis, TH1, TH1F, TColor, TExec, TLine
from ROOT import kBlack, kWhite, kOrange, kAzure, kBlue
from ROOT import gROOT, gStyle
gROOT.SetBatch(True)
sys.argv = oldargv

from Validation.Geometry.plot_utils import setTDRStyle, Plot_params, drawEtaValues
 
from Validation.Geometry.plot_hgcal_utils import plots, COMPOUNDS, DETECTORS, sDETS, hist_label_to_num, TwikiPrintout, acustompalette, dEdx, MatXo, drawHalfEtaValues
from collections import namedtuple, OrderedDict
import sys, os
import argparse

def paramsGood_(detector, plot):
    """Check the validity of the arguments.

       Common function to check the validity of the parameters passed
       in. It returns a tuple composed by a bool and a string. The
       bool indicates if all checks are ok, the string the name of the
       appropriate ROOT file to open (empty string in case the any
       check failed)

    """

    if plot not in plots.keys():
        print("Error, unknown plot %s" % plot)
        return (False, '')

    if detector not in DETECTORS and detector not in COMPOUNDS.keys():
        print('Error, unknown detector: %s' % detector)
        return (False, '')

    theDetectorFilename = ''
    if detector in DETECTORS:
        theDetectorFilename = 'matbdg_%s.root' % detector
    else:
        theDetectorFilename = 'matbdg_%s.root' % COMPOUNDS[detector][0]

    if not checkFile_(theDetectorFilename):
        print("Error, missing file %s" % theDetectorFilename)
        raise RuntimeError
    return (True, theDetectorFilename)

def checkFile_(filename):
    return os.path.exists(filename)

def setColorIfExists_(histos, h, color):
    if h in histos.keys():
        histos[h].SetFillColor(color)

def assignOrAddIfExists_(h, p):
    """Assign the projection of p to h.

       Function to assign the projection of p to h, in the case in
       which h is None, otherwise add the projection to the already
       valid h object

    """

    if not h:
        h = p.ProjectionX()
    else:
        h.Add(p.ProjectionX("B_%s" % h.GetName()), +1.000)
    return h

def createPlots_(plot, compounddetectorname):
    """Cumulative material budget from simulation.
    
       Internal function that will produce a cumulative profile of the
       material budget inferred from the simulation starting from the
       single detectors that compose the tracker. It will iterate over
       all existing detectors contained in the DETECTORS
       dictionary. The function will automatically skip non-existent
       detectors.

    """

    theDirname = "Figures"

    hist_X0_detectors = OrderedDict()
    if plot not in plots.keys():
        print("Error: chosen plot name not known %s" % plot)
        return

    # We need to keep the file content alive for the lifetime of the
    # full function....
    subDetectorFiles = []

    hist_X0_elements = OrderedDict()
    prof_X0_elements = OrderedDict()
    for subDetector,color in DETECTORS.items():
        subDetectorFilename = "matbdg_%s.root" % subDetector
        if not checkFile_(subDetectorFilename):
            print("Error opening file: %s" % subDetectorFilename)
            continue

        subDetectorFiles.append(TFile(subDetectorFilename))
        subDetectorFile = subDetectorFiles[-1]
        print ("Opening file: %s" % subDetectorFilename)
        prof_X0_XXX = subDetectorFile.Get("%d" % plots[plot].plotNumber)

        hist_X0_detectors[subDetector] = prof_X0_XXX.ProjectionX()

        # category profiles
        for label, [num, color, leg] in hist_label_to_num.items():
            prof_X0_elements[label] = subDetectorFile.Get("%d" % (num + plots[plot].plotNumber))
            hist_X0_elements[label] = assignOrAddIfExists_(hist_X0_elements.setdefault(label, None),
                                                          prof_X0_elements[label])

    cumulative_matbdg = TH1D("CumulativeSimulMatBdg",
                             "CumulativeSimulMatBdg",
                             hist_X0_detectors["BeamPipe"].GetNbinsX(),
                             hist_X0_detectors["BeamPipe"].GetXaxis().GetXmin(),
                             hist_X0_detectors["BeamPipe"].GetXaxis().GetXmax())
    cumulative_matbdg.SetDirectory(0)

    # colors
    for det, color in DETECTORS.items():
        setColorIfExists_(hist_X0_detectors, det, color)

    for label, [num, color, leg] in hist_label_to_num.items():
        hist_X0_elements[label].SetFillColor(color)

    # First Plot: BeamPipe + Tracker + ECAL + HCal + HGCal + MB + MGNT
    # stack
    stackTitle_SubDetectors = "Material Budget;%s;%s" % (
        plots[plot].abscissa,plots[plot].ordinate)
    stack_X0_SubDetectors = THStack("stack_X0",stackTitle_SubDetectors)
    for det, histo in hist_X0_detectors.items():
        stack_X0_SubDetectors.Add(histo)
        cumulative_matbdg.Add(histo, 1)

    # canvas
    can_SubDetectors = TCanvas("can_SubDetectors","can_SubDetectors",800,800)
    #can_SubDetectors.Range(0,0,25,25)
    can_SubDetectors.SetFillColor(kWhite)

    # Draw
    stack_X0_SubDetectors.SetMinimum(plots[plot].ymin)
    stack_X0_SubDetectors.SetMaximum(plots[plot].ymax)
    stack_X0_SubDetectors.Draw("HIST")
    #stack_X0_SubDetectors.GetXaxis().SetLimits(plots[plot].xmin, plots[plot].xmax)


    # Legenda
    theLegend_SubDetectors = TLegend(0.130,0.7,0.93,0.90) #(0.180,0.8,0.98,0.90)
    theLegend_SubDetectors.SetNColumns(2)
    theLegend_SubDetectors.SetFillColor(0)
    theLegend_SubDetectors.SetFillStyle(0)
    theLegend_SubDetectors.SetBorderSize(0)

    for det, histo in hist_X0_detectors.items():
        theLegend_SubDetectors.AddEntry(histo, det,  "f")

    theLegend_SubDetectors.Draw()

    # text
    text_SubDetectors = TPaveText(0.130,0.627,0.352,0.687,"NDC") #(0.180,0.727,0.402,0.787,"NDC")
    text_SubDetectors.SetFillColor(0)
    text_SubDetectors.SetBorderSize(0)
    text_SubDetectors.AddText("CMS Simulation")
    text_SubDetectors.SetTextAlign(11)
    text_SubDetectors.Draw()

    # Store
    can_SubDetectors.Update()
    if not checkFile_(theDirname):
        os.mkdir(theDirname)
    can_SubDetectors.SaveAs("%s/MaterialBdg_%s_%s.pdf" % (theDirname, compounddetectorname,plot))
    can_SubDetectors.SaveAs("%s/MaterialBdg_%s_%s.png" % (theDirname, compounddetectorname,plot))
    can_SubDetectors.SaveAs("%s/MaterialBdg_%s_%s.root" % (theDirname, compounddetectorname,plot))

    if plot == "x_vs_eta" or plot == "l_vs_eta":
        canname = "MBCan_1D_%s_%s_total"  % (compounddetectorname, plot)
        can2 = TCanvas(canname, canname, 800, 800)
        can2.Range(0,0,25,25)
        can2.SetFillColor(kWhite)
        gStyle.SetOptStat(0)
        gStyle.SetOptTitle(0);
        #title = TPaveLabel(.11,.95,.35,.99,"Total accumulated material budget","brndc")
        stack_X0_SubDetectors.GetStack().Last().SetMarkerStyle(34)
        stack_X0_SubDetectors.GetStack().Last().GetXaxis().SetRangeUser( 1.0, 3.5)
        stack_X0_SubDetectors.GetStack().Last().Draw();
        stack_X0_SubDetectors.GetYaxis().SetTitleOffset(1.15);
        can2.Update()
        can2.Modified()
        can2.SaveAs("%s/%s_%s_total_Zplus.pdf" % (theDirname, compounddetectorname, plot))
        can2.SaveAs("%s/%s_%s_total_Zplus.png" % (theDirname, compounddetectorname, plot))
        stack_X0_SubDetectors.GetStack().Last().GetXaxis().SetRangeUser( -3.5, -1.0)
        stack_X0_SubDetectors.GetStack().Last().Draw();
        stack_X0_SubDetectors.GetYaxis().SetTitleOffset(1.15);
        can2.Update()
        can2.Modified()
        can2.SaveAs("%s/%s_%s_total_Zminus.pdf" % (theDirname, compounddetectorname, plot))
        can2.SaveAs("%s/%s_%s_total_Zminus.png" % (theDirname, compounddetectorname, plot))

        #Also print them to give them exact numbers
        etavalues = []
        matbudginX0 = []
        matbudginIntLen = []
        for binx in range(0,stack_X0_SubDetectors.GetStack().Last().GetXaxis().GetNbins()):
            bincontent = stack_X0_SubDetectors.GetStack().Last().GetBinContent(binx) 
            if bincontent == 0: continue
            etavalues.append( stack_X0_SubDetectors.GetStack().Last().GetBinCenter(binx)  )
            if plot == "x_vs_eta": 
                matbudginX0.append( bincontent  )
                d1 = {'Eta': etavalues, 'MatBudInX0': matbudginX0}
                df1 = pd.DataFrame(data=d1).round(2)
                df1.to_csv(r'/afs/cern.ch/work/a/apsallid/CMS/PFCalStudies/CMS-HGCAL/matbudV10fromVertexToBackofHGCal/CMSSW_11_0_X_2019-06-04-2300/src/Validation/Geometry/test/EtavsMatBudinXo.txt',sep=' ', index=False, header=False)
                #print df1
            if plot == "l_vs_eta": 
                matbudginIntLen.append( bincontent )
                d2 = {'Eta': etavalues, 'MatBudInIntLen': matbudginIntLen}
                df2 = pd.DataFrame(data=d2).round(2)
                df2.to_csv(r'/afs/cern.ch/work/a/apsallid/CMS/PFCalStudies/CMS-HGCAL/matbudV10fromVertexToBackofHGCal/CMSSW_11_0_X_2019-06-04-2300/src/Validation/Geometry/test/EtavsMatBudInIntLen.txt',sep=' ', index=False, header=False)
                #print df2

    return cumulative_matbdg

def createPlots2D_(plot, compounddetectorname):
    """2D material budget map to know exactly what we are adding.
    """

    #IBs = ["InnerServices", "Phase2PixelBarrel", "TIB", "TIDF", "TIDB"]
    theDirname = "Figures"

    hist_X0_detectors = OrderedDict()
    if plot not in plots.keys():
        print("Error: chosen plot name not known %s" % plot)
        return

    # We need to keep the file content alive for the lifetime of the
    # full function....
    subDetectorFiles = []

    hist_X0_elements = OrderedDict()
    prof_X0_elements = OrderedDict()

    for subDetector,color in DETECTORS.items():
        subDetectorFilename = "matbdg_%s.root" % subDetector
        if not checkFile_(subDetectorFilename):
            print("Error opening file: %s" % subDetectorFilename)
            continue

        subDetectorFiles.append(TFile(subDetectorFilename))
        subDetectorFile = subDetectorFiles[-1]
        print ("Opening file: %s" % subDetectorFilename)
        prof_X0_XXX = subDetectorFile.Get("%d" % plots[plot].plotNumber)

        #hist_X0_detectors[subDetector] = prof_X0_XXX
        hist_X0_detectors[subDetector] = prof_X0_XXX.ProjectionXY("_pxy","B")
        print(subDetector)

    # First Plot: BeamPipe + Tracker + ECAL + HCal + HGCal + MB + MGNT
 
    # Create "null" histo
    minX = 1.03*hist_X0_detectors["BeamPipe"].GetXaxis().GetXmin()
    maxX = 1.03*hist_X0_detectors["BeamPipe"].GetXaxis().GetXmax()
    minY = 1.03*hist_X0_detectors["BeamPipe"].GetYaxis().GetXmin()
    maxY = 1.03*hist_X0_detectors["BeamPipe"].GetYaxis().GetXmax()

    frame = TH2F("frame", "", 10, minX, maxX, 10, minY, maxY);
    frame.SetMinimum(0.1)
    frame.SetMaximum(10.)
    frame.GetXaxis().SetTickLength(frame.GetXaxis().GetTickLength()*0.50)
    frame.GetYaxis().SetTickLength(frame.GetXaxis().GetTickLength()/4.)

    hist2d_X0_total = hist_X0_detectors["BeamPipe"]

    # stack
    hist2dTitle = ('%s %s;%s;%s;%s' % (plots[plot].quotaName,
                                       "All detectors",
                                       plots[plot].abscissa,
                                       plots[plot].ordinate,
                                       plots[plot].quotaName))

    hist2d_X0_total.SetTitle(hist2dTitle)
    frame.SetTitle(hist2dTitle)
    frame.SetTitleOffset(0.5,"Y")

    #If here you put different histomin,histomaxin plot_utils you won't see anything 
    #for the material plots. 
    if plots[plot].histoMin != -1.:
        hist2d_X0_total.SetMinimum(plots[plot].histoMin)
    if plots[plot].histoMax != -1.:
        hist2d_X0_total.SetMaximum(plots[plot].histoMax)

    #
    # canvas
    can_SubDetectors = TCanvas("can_SubDetectors","can_SubDetectors",2480+248, 580+58+58)
    can_SubDetectors.SetTopMargin(0.1)
    can_SubDetectors.SetBottomMargin(0.1)
    can_SubDetectors.SetLeftMargin(0.04)
    can_SubDetectors.SetRightMargin(0.06)
    can_SubDetectors.SetFillColor(kWhite)
    gStyle.SetOptStat(0)
    gStyle.SetTitleFillColor(0)
    gStyle.SetTitleBorderSize(0)
    gStyle.SetOptTitle(0)

    hist2d_X0_total.GetYaxis().SetTickLength(hist2d_X0_total.GetXaxis().GetTickLength()/4.)
    hist2d_X0_total.GetYaxis().SetTickLength(hist2d_X0_total.GetXaxis().GetTickLength()/4.)
    hist2d_X0_total.SetTitleOffset(0.5,"Y")
    hist2d_X0_total.GetYaxis().SetTitleOffset(0.50);
    #hist2d_X0_total.GetXaxis().SetTitleOffset(1.15);
    #hist2d_X0_total.GetXaxis().SetNoExponent(True)
    #hist2d_X0_total.GetYaxis().SetNoExponent(True)


    # colors
    for det, color in DETECTORS.items():
        hist_X0_detectors[det].SetMarkerColor(color)
        hist_X0_detectors[det].SetFillColor(color)

    for det, histo in hist_X0_detectors.items():
        print(det)
        histo.Draw("same")

    # Legenda
    theLegend_SubDetectors = TLegend(0.100,0.7,0.90,0.90)#(0.180,0.8,0.98,0.90)
    theLegend_SubDetectors.SetNColumns(3)
    theLegend_SubDetectors.SetFillColor(0)
    theLegend_SubDetectors.SetFillStyle(0)
    theLegend_SubDetectors.SetBorderSize(0)

    for det, histo in hist_X0_detectors.items():
        theLegend_SubDetectors.AddEntry(histo, det,  "f")
    #theLegend_SubDetectors.AddEntry(hgbound1, "HGCal Eta Boundaries [1.3, 3.0]",  "l") 

    theLegend_SubDetectors.Draw()

    
    # text
    text_SubDetectors = TPaveText(0.100,0.627,0.322,0.687,"NDC")#(0.180,0.727,0.402,0.787,"NDC")
    text_SubDetectors.SetFillColor(0)
    text_SubDetectors.SetBorderSize(0)
    text_SubDetectors.AddText("CMS Simulation")
    text_SubDetectors.SetTextAlign(11)
    text_SubDetectors.Draw()
    

    #Add eta labels
    keep_alive = []
    if plots[plot].iDrawEta:
        keep_alive.extend(drawEtaValues())
    
    # Store
    can_SubDetectors.Update()
    if not checkFile_(theDirname):
        os.mkdir(theDirname)
    can_SubDetectors.SaveAs("%s/MaterialBdg_%s_%s.png" % (theDirname, compounddetectorname, plot))
    #It seems that it is too heavy to create .pdf and .root
    #can_SubDetectors.SaveAs("%s/MaterialBdg_FromVertexToEndofHGCal_%s.pdf" % (theDirname, plot))
    #can_SubDetectors.SaveAs("%s/MaterialBdg_FromVertexToEndofHGCal_%s.root" % (theDirname, plot))

    hist2d_X0_total.GetXaxis().SetRangeUser( 0., 7000.)
    #Draw eta values in the zoom case
    keep_alive = []
    keep_alive.extend(drawHalfEtaValues())
    #hist2d_X0_total.Draw("COLZ") 
    can_SubDetectors.Update()
    can_SubDetectors.Modified()
    can_SubDetectors.SaveAs("%s/MaterialBdg_%s_%s_Zpluszoom.png" % (theDirname, compounddetectorname, plot))

def createPlotsReco_(reco_file, label, debug=False):
    """Cumulative material budget from reconstruction.


       Internal function that will produce a cumulative profile of the
       material budget in the reconstruction starting from the single
       detectors that compose the tracker. It will iterate over all
       existing detectors contained in the sDETS dictionary. The
       function will automatically stop everytime it encounters a
       non-existent detector, until no more detectors are left to
       try. For this reason the keys in the sDETS dictionary can be as
       inclusive as possible.

    """

    cumulative_matbdg = None
    sPREF = ["Original_RadLen_vs_Eta_", "RadLen_vs_Eta_"]

    c = TCanvas("c", "c", 1024, 1024);
    diffs = []
    if not checkFile_(reco_file):
        print("Error: missing file %s" % reco_file)
        raise RuntimeError
    file = TFile(reco_file)
    prefix = "/DQMData/Run 1/Tracking/Run summary/RecoMaterial/"
    for s in sPREF:
        hs = THStack("hs","");
        histos = []
        for det, color in sDETS.items():
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
                    if debug:
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

def materialBudget_Simul_vs_Reco(reco_file, label, debug=False):
    """Plot reco vs simulation material budget.
    
       Function are produces a direct comparison of the material
       budget as extracted from the reconstruction geometry and
       inferred from the simulation one.

    """

    setTDRStyle()

    # plots
    cumulative_matbdg_sim = createPlots_("x_vs_eta")
    cumulative_matbdg_rec = createPlotsReco_(reco_file, label, debug=False)

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

def createCompoundPlots(detector, plot):
    """Produce the requested plot for the specified detector.

       Function that will plot the requested @plot for the specified
       @detector. The specified detector could either be a real
       detector or a compound one. The list of available plots are the
       keys of plots dictionary (imported from plot_utils.

    """

    theDirname = 'Images'
    if not checkFile_(theDirname):
        os.mkdir(theDirname)

    goodToGo, theDetectorFilename = paramsGood_(detector, plot)
    if not goodToGo:
        return

    theDetectorFile = TFile(theDetectorFilename)
    #

    # get TProfiles
    prof_X0_elements = OrderedDict()
    hist_X0_elements = OrderedDict()
    for label, [num, color, leg] in hist_label_to_num.items():
        #print label, num, color, leg
        prof_X0_elements[label] = theDetectorFile.Get("%d" % (num + plots[plot].plotNumber))
        hist_X0_elements[label] = prof_X0_elements[label].ProjectionX()
        hist_X0_elements[label].SetFillColor(color)
        hist_X0_elements[label].SetLineColor(kBlack)

    files = []
    if detector in COMPOUNDS.keys():
        for subDetector in COMPOUNDS[detector][1:]:
            subDetectorFilename = "matbdg_%s.root" % subDetector

            # open file
            if not checkFile_(subDetectorFilename):
                continue

            subDetectorFile = TFile(subDetectorFilename)
            files.append(subDetectorFile)
            print("*** Open file... %s" %  subDetectorFilename)

            # subdetector profiles
            for label, [num, color, leg] in hist_label_to_num.items():
                prof_X0_elements[label] = subDetectorFile.Get("%d" % (num + plots[plot].plotNumber))
                hist_X0_elements[label].Add(prof_X0_elements[label].ProjectionX("B_%s" % prof_X0_elements[label].GetName())
                                            , +1.000)

    # stack
    stackTitle = "Material Budget %s;%s;%s" % (detector,
                                               plots[plot].abscissa,
                                               plots[plot].ordinate)
    stack_X0 = THStack("stack_X0", stackTitle);
    for label, [num, color, leg] in hist_label_to_num.items():
        stack_X0.Add(hist_X0_elements[label])

    # canvas
    canname = "MBCan_1D_%s_%s"  % (detector, plot)
    can = TCanvas(canname, canname, 800, 800)
    can.Range(0,0,25,25)
    can.SetFillColor(kWhite)
    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(1);

    # Draw
    stack_X0.Draw("HIST");
    stack_X0.GetYaxis().SetTitleOffset(1.15);

    # Legenda
    theLegend = TLegend(0.40, 0.65, 0.60, 0.89)
    if plot == "x_vs_phi" or plot == "l_vs_phi": theLegend = TLegend(0.65, 0.30, 0.89, 0.70)
    if plot == "x_vs_R" or plot == "l_vs_R": theLegend = TLegend(0.75, 0.60, 0.95, 0.90)

    for label, [num, color, leg] in hist_label_to_num.items():
        theLegend.AddEntry(hist_X0_elements[label], leg, "f")
    theLegend.Draw();

    # Store
    can.Update();
    can.SaveAs( "%s/%s_%s.pdf" % (theDirname, detector, plot))
    can.SaveAs( "%s/%s_%s.png" % (theDirname, detector, plot))

    #Let's also save the total accumulated budget vs eta since muon id relies 
    #on adequate calorimeter thickness
    if plot == "x_vs_eta" or plot == "l_vs_eta":
        canname = "MBCan_1D_%s_%s_total"  % (detector, plot)
        can2 = TCanvas(canname, canname, 800, 800)
        can2.Range(0,0,25,25)
        can2.SetFillColor(kWhite)
        gStyle.SetOptStat(0)
        gStyle.SetOptTitle(0);
        #title = TPaveLabel(.11,.95,.35,.99,"Total accumulated material budget","brndc")
        stack_X0.GetStack().Last().GetXaxis().SetRangeUser( 0., 3.)
        stack_X0.GetStack().Last().Draw();
        stack_X0.GetYaxis().SetTitleOffset(1.15);
        can2.Update()
        can2.Modified()
        can2.SaveAs("%s/%s_%s_total_Zplus.pdf" % (theDirname, detector, plot))
        can2.SaveAs("%s/%s_%s_total_Zplus.png" % (theDirname, detector, plot))
        stack_X0.GetStack().Last().GetXaxis().SetRangeUser( -3., 0.)
        stack_X0.GetStack().Last().Draw();
        stack_X0.GetYaxis().SetTitleOffset(1.15);
        can2.Update()
        can2.Modified()
        can2.SaveAs("%s/%s_%s_total_Zminus.pdf" % (theDirname, detector, plot))
        can2.SaveAs("%s/%s_%s_total_Zminus.png" % (theDirname, detector, plot))
 

def create2DPlots(detector, plot, plotnum, plotmat, dosingledetector = True):
    """Produce the requested plot for the specified detector.

       Function that will plot the requested 2D-@plot for the
       specified @detector. The specified detector could either be a
       real detector or a compound one. The list of available plots
       are the keys of plots dictionary imported from plot_utils.
    """

    #gStyle.Reset()
    #Better to use an underscore. 
    plotmat = plotmat.replace(" ", "_")


    if plotmat != "": 
        theDirname = ('Images/%s' % plotmat).replace(" ", "")
    else: 
        theDirname = 'Images'

    if not checkFile_(theDirname):
        os.mkdir(theDirname)
    if not os.path.isdir(('Images/%s/ZPlusZoom' % plotmat).replace(" ", "")):
        os.mkdir( ('Images/%s/ZPlusZoom' % plotmat).replace(" ", "") )
    if not os.path.isdir(('Images/%s/ZMinusZoom' % plotmat).replace(" ", "")):
        os.mkdir( ('Images/%s/ZMinusZoom' % plotmat).replace(" ", "") )

    goodToGo, theDetectorFilename = paramsGood_(detector, plot)
    if not goodToGo:
        return

    theDetectorFile = TFile(theDetectorFilename)

    prof2d_X0_det_total = TProfile2D()
    prof2d_X0_det_total.Reset()

    # get TProfiles
    #prof2d_X0_det_total = theDetectorFile.Get('%s' % plots[plot].plotNumber)
    prof2d_X0_det_total = theDetectorFile.Get('%s' % plotnum)
    print("==================================================================") 
    print(plotnum)

    # histos
    prof2d_X0_det_total.__class__ = TProfile2D
    hist_X0_total = prof2d_X0_det_total.ProjectionXY()

    # keep files live forever
    files = []
    if detector in COMPOUNDS.keys() and not dosingledetector:
        #When the loop was:
        #for subDetector in COMPOUNDS[detector][1:]:
        #and the detector was single it never went in the loop and read the single file 
        #from above. I alter this to COMPOUNDS[detector] to do the multi material budget plot.
        #This won't effect the single detector due to the alter in the if above
        for subDetector in COMPOUNDS[detector]:
            # filenames of single components
            subDetectorFilename = "matbdg_%s.root" % subDetector
 
            # open file
            if not checkFile_(subDetectorFilename):
                print("Error, missing file %s" % subDetectorFilename)
                continue
    
            subDetectorFile = TFile(subDetectorFilename)
            files.append(subDetectorFile)
            print("*** Open file... %s" %  subDetectorFilename)

            # subdetector profiles
            prof2d_X0_det_total = subDetectorFile.Get('%s' % plots[plot].plotNumber)
            prof2d_X0_det_total.__class__ = TProfile2D

            # add to summary histogram
            hist_X0_total.Add(prof2d_X0_det_total.ProjectionXY("B_%s" % prof2d_X0_det_total.GetName()), +1.000 )

    # # properties
    #gStyle.SetPalette(1)
    gStyle.SetStripDecimals(False)
    # #

    # Create "null" histo
    minX = 1.03*prof2d_X0_det_total.GetXaxis().GetXmin()
    maxX = 1.03*prof2d_X0_det_total.GetXaxis().GetXmax()
    minY = 1.03*prof2d_X0_det_total.GetYaxis().GetXmin()
    maxY = 1.03*prof2d_X0_det_total.GetYaxis().GetXmax()

    frame = TH2F("frame", "", 10, minX, maxX, 10, minY, maxY);
    frame.SetMinimum(0.1)
    frame.SetMaximum(10.)
    frame.GetXaxis().SetTickLength(frame.GetXaxis().GetTickLength()*0.50)
    frame.GetYaxis().SetTickLength(frame.GetXaxis().GetTickLength()/4.)

    # Ratio
    if plots[plot].iRebin:
        prof2d_X0_det_total.Rebin2D()

    # stack
    hist2dTitle = ('%s %s;%s;%s;%s' % (plots[plot].quotaName,
                                       detector,
                                       plots[plot].abscissa,
                                       plots[plot].ordinate,
                                       plots[plot].quotaName))

    if dosingledetector: 
        hist2d_X0_total = prof2d_X0_det_total
    else : 
        hist2d_X0_total = hist_X0_total
    hist2d_X0_total.SetTitle(hist2dTitle)
    frame.SetTitle(hist2dTitle)
    frame.SetTitleOffset(0.5,"Y")

    #If here you put different histomin,histomaxin plot_utils you won't see anything 
    #for the material plots. 
    if plots[plot].histoMin != -1.:
        hist2d_X0_total.SetMinimum(plots[plot].histoMin)
    if plots[plot].histoMax != -1.:
        hist2d_X0_total.SetMaximum(plots[plot].histoMax)

    #
    can2name = "MBCan_2D_%s_%s_%s" % (detector, plot, plotmat)
    can2 = TCanvas(can2name, can2name, 2480+248, 580+58+58)
    can2.SetTopMargin(0.1)
    can2.SetBottomMargin(0.1)
    can2.SetLeftMargin(0.04)
    can2.SetRightMargin(0.06)
    can2.SetFillColor(kWhite)
    gStyle.SetOptStat(0)
    gStyle.SetTitleFillColor(0)
    gStyle.SetTitleBorderSize(0)

    #hist2d_X0_total.SetMaximum(hist2d_X0_total.GetMaximum())
    # Color palette
    # gStyle.SetPalette()#1
    acustompalette()
    ex1 = TExec("ex1","acustompalette();");
    ex1.Draw();

    #for i in range(100): MyPaletteArray.append(i+1)

    #gStyle.SetPalette(first_color_number);

    # Log?
    can2.SetLogz(plots[plot].zLog)

    # Draw in colors
    #frame.Draw()
    #hist2d_X0_total.Draw("COLZsame") #Dummy draw to create the palette object
    hist2d_X0_total.Draw("COLZ") #Dummy draw to create the palette object

    # Store
    can2.Update()

    #Aesthetic
    palette = hist2d_X0_total.GetListOfFunctions().FindObject("palette")
    if palette:
        palette.__class__ = TPaletteAxis
        palette.SetX1NDC(0.945)
        palette.SetX2NDC(0.96)
        palette.SetY1NDC(0.1)
        palette.SetY2NDC(0.9)
        palette.GetAxis().SetTickSize(.01)
        palette.GetAxis().SetTitle("")
        if plots[plot].zLog:
            palette.GetAxis().SetLabelOffset(-0.01)
    paletteTitle = TLatex(1.12*maxX, maxY, plots[plot].quotaName)
    paletteTitle.SetTextAngle(90.)
    paletteTitle.SetTextSize(0.05)
    paletteTitle.SetTextAlign(31)
    paletteTitle.Draw()
    hist2d_X0_total.GetYaxis().SetTickLength(hist2d_X0_total.GetXaxis().GetTickLength()/4.)
    hist2d_X0_total.GetYaxis().SetTickLength(hist2d_X0_total.GetXaxis().GetTickLength()/4.)
    hist2d_X0_total.SetTitleOffset(0.5,"Y")
    hist2d_X0_total.GetYaxis().SetTitleOffset(0.45);
    #hist2d_X0_total.GetXaxis().SetTitleOffset(1.15);
    #hist2d_X0_total.GetXaxis().SetNoExponent(True)
    #hist2d_X0_total.GetYaxis().SetNoExponent(True)

    #Add eta labels
    keep_alive = []
    if plots[plot].iDrawEta:
        keep_alive.extend(drawEtaValues())

    can2.Modified()
    hist2d_X0_total.SetContour(255)

    # Store
    can2.Update()
    can2.Modified()

    can2.SaveAs( "%s/%s_%s%s.pdf" % (theDirname, detector, plot, plotmat))
    can2.SaveAs( "%s/%s_%s%s.png" % (theDirname, detector, plot, plotmat))
    #can2.SaveAs( "%s/%s_%s%s.root" % (theDirname, detector, plot, plotmat))

    #Zoom in a little bit
    if plot == "x_vs_z_vs_Rsum" or plot == "l_vs_z_vs_Rsum" or plot == "x_vs_z_vs_Rsumcos" or plot == "l_vs_z_vs_Rsumcos" or plot == "x_vs_z_vs_Rloc" or plot == "l_vs_z_vs_Rloc" or  plot == "x_vs_z_vs_Rloccos" or plot == "l_vs_z_vs_Rloccos":
        #Z+
        #hist2d_X0_total.GetXaxis().SetLimits( 3100., 5200.)
        if dosingledetector: hist2d_X0_total.GetXaxis().SetRangeUser( 3100., 5400.)
        else : hist2d_X0_total.GetXaxis().SetRangeUser( 0., 7000.)
        #Do not draw eta values in the zoom case
        keep_alive = []
        #hist2d_X0_total.Draw("COLZ") 
        can2.Update()
        can2.Modified()
        can2.SaveAs( "%s/%s/%s_%s%s_ZplusZoom.pdf" % (theDirname, "ZPlusZoom", detector, plot, plotmat))
        can2.SaveAs( "%s/%s/%s_%s%s_ZplusZoom.png" % (theDirname, "ZPlusZoom", detector, plot, plotmat))
        #Z-
        #hist2d_X0_total.GetXaxis().SetLimits( 3100., 5200.)
        if dosingledetector: hist2d_X0_total.GetXaxis().SetRangeUser( -5400., -3100.)
        else : hist2d_X0_total.GetXaxis().SetRangeUser( 0., -7000.)
        #Do not draw eta values in the zoom case
        keep_alive = []
        #hist2d_X0_total.Draw("COLZ") 
        can2.Update()
        can2.Modified()
        can2.SaveAs( "%s/%s/%s_%s%s_ZminusZoom.pdf" % (theDirname, "ZMinusZoom", detector, plot, plotmat))
        can2.SaveAs( "%s/%s/%s_%s%s_ZminusZoom.png" % (theDirname, "ZMinusZoom", detector, plot, plotmat))


    gStyle.SetStripDecimals(True)

def createRatioPlots(detector, plot):
    """Create ratio plots.

       Function that will make the ratio between the radiation length
       and interaction length, for the specified detector. The
       specified detector could either be a real detector or a
       compound one.

    """

    goodToGo, theDetectorFilename = paramsGood_(detector, plot)
    if not goodToGo:
        return

    theDirname = 'Images'
    if not os.path.exists(theDirname):
        os.mkdir(theDirname)

    theDetectorFile = TFile(theDetectorFilename)
    # get TProfiles
    prof_x0_det_total = theDetectorFile.Get('%d' % plots[plot].plotNumber)
    prof_l0_det_total = theDetectorFile.Get('%d' % (10000+plots[plot].plotNumber))

    # histos
    hist_x0_total = prof_x0_det_total.ProjectionX()
    hist_l0_total = prof_l0_det_total.ProjectionX()

    if detector in COMPOUNDS.keys():
        for subDetector in COMPOUNDS[detector][1:]:

            # file name
            subDetectorFilename = "matbdg_%s.root" % subDetector

            # open file
            if not checkFile_(subDetectorFilename):
                print("Error, missing file %s" % subDetectorFilename)
                continue

            subDetectorFile = TFile(subDetectorFilename)

            # subdetector profiles
            prof_x0_det_total = subDetectorFile.Get('%d' % plots[plot].plotNumber)
            prof_l0_det_total = subDetectorFile.Get('%d' % (10000+plots[plot].plotNumber))
            # add to summary histogram
            hist_x0_total.Add(prof_x0_det_total.ProjectionX("B_%s" % prof_x0_det_total.GetName()), +1.000 )
            hist_l0_total.Add(prof_l0_det_total.ProjectionX("B_%s" % prof_l0_det_total.GetName()), +1.000 )
    #
    hist_x0_over_l0_total = hist_x0_total
    hist_x0_over_l0_total.Divide(hist_l0_total)
    histTitle = "Material Budget %s;%s;%s" % (detector,
                                              plots[plot].abscissa,
                                              plots[plot].ordinate)
    hist_x0_over_l0_total.SetTitle(histTitle)
    # properties
    hist_x0_over_l0_total.SetMarkerStyle(1)
    hist_x0_over_l0_total.SetMarkerSize(3)
    hist_x0_over_l0_total.SetMarkerColor(kBlue)

    # canvas
    canRname = "MBRatio_%s_%s" % (detector, plot)
    canR = TCanvas(canRname,canRname,800,800)
    canR.Range(0,0,25,25)
    canR.SetFillColor(kWhite)
    gStyle.SetOptStat(0)

    # Draw
    hist_x0_over_l0_total.Draw("E1")

    # Store
    canR.Update()
    canR.SaveAs("%s/%s_%s.pdf" % (theDirname, detector, plot))
    canR.SaveAs("%s/%s_%s.png" % (theDirname, detector, plot))

    
def GetSiliconZValuesFromXML(): 
    """Someone can use the xml files of the geometry. Just do 
    cd $CMSSW_BASE/src/Geometry/HGCalCommonData/test
    cmsRun testHGCalParameters_cfg.py
    The scintillator position in the HEB is very close to the silicon but they have a 
    1.95 mm difference. So, we take the silicon position. 

    This returns a np.array of the z position of the silicon values in mm. 
    """
    
    #First as they come out of the testHGCalParameters_cfg.py as cm 
    #plus either the thermal screen front (297.0 cm)
    #or the EE front (319.05 cm) 
    layersmaxZ = ( 319.05, 319.815, 320.725, 322.255, 323.165, 324.695, 325.605, 327.135, 328.045, 329.575, 330.485, 332.015, 332.925, 334.455, 335.365, 336.895, 337.805, 339.335, 340.245, 341.775, 342.685, 344.215, 345.125, 346.655, 347.565, 349.095, 350.005, 351.535, 352.445, 357.715, 362.615, 367.515, 372.415, 377.315, 382.215, 387.115, 392.015, 397.105, 402.195, 407.285, 412.375, 420.765, 429.155, 437.545, 445.935, 454.325, 462.715, 471.105, 479.495, 487.885, 496.275, 504.665, 513.055)

    layersmaxZ = np.asarray(layersmaxZ)
    layersmaxZ = 10. * layersmaxZ # in mm

    print("Total number of layers from XML " , layersmaxZ.size - 1) # Minus 1 for the EE front input by hand. 

    return layersmaxZ


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic Material Plotter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--reco',
                        help='Input Reconstruction Material file, DQM format')
    parser.add_argument('-l', '--label',
                        help='Label to use in naming the plots')
    parser.add_argument('-c', '--compare',
                        help='Compare simulation and reco materials',
                        action='store_true',
                        default=False)
    parser.add_argument('-m', '--multi',
                        help='Combining multiple detectors',
                        action='store_true',
                        default=False)
    parser.add_argument('-s', '--single',
                        help='Material budget for single detector from simulation',
                        action='store_true',
                        default=False)
    parser.add_argument('-d', '--detector',
                        help='Detector for which you want to compute the material budget',
                        type=str,)
    args = parser.parse_args()

    if args.compare and args.single:
        print("Error, too many actions required")
        raise RuntimeError

    if args.compare:
        if args.reco is None:
            print("Error, missing inpur reco file")
            raise RuntimeError
        if args.label is None:
            print("Error, missing label")
            raise RuntimeError
        materialBudget_Simul_vs_Reco(args.reco, args.label, debug=False)

    if args.multi: 
        #Material Budget plot from Vertex to end of HGCal
        cumulative_matbdg_sim = createPlots_("x_vs_eta",args.detector)
        cumulative_matbdg_sim = createPlots_("l_vs_eta",args.detector)
        cumulative_matbdg_sim = createPlots2D_("x_vs_z_vs_Rsum",args.detector)
        cumulative_matbdg_sim = createPlots2D_("l_vs_z_vs_Rsum",args.detector)

        required_2DplotsForalldetectors = ["x_vs_z_vs_Rsum", "l_vs_z_vs_Rsum"]

        for p in required_2DplotsForalldetectors:
            #First the total
            create2DPlots(args.detector, p, plots[p].plotNumber, "", not args.multi)
            #Then, the rest
            #for label, [num, color, leg] in hist_label_to_num.iteritems():
                #print label, num, color, leg
                #create2DPlots(args.detector, p, num + plots[p].plotNumber, leg)

    if args.single:
        if args.detector is None:
            print("Error, missing detector")
            raise RuntimeError
        required_2Dplots = ["x_vs_eta_vs_phi", "l_vs_eta_vs_phi", "x_vs_z_vs_Rsum", "l_vs_z_vs_Rsum", "x_vs_z_vs_Rsumcos", "l_vs_z_vs_Rsumcos", "x_vs_z_vs_Rloc", "l_vs_z_vs_Rloc", "x_vs_z_vs_Rloccos", "l_vs_z_vs_Rloccos"]
        # The plots below require z bin 1 cm to have more statistics for eta, phi 
        # while the plots above are with 200000 events for z bin 1 mm. 
        #required_2Dplots = ["x_vs_eta_vs_phi", "l_vs_eta_vs_phi", "x_vs_z_vs_Rsum", "l_vs_z_vs_Rsum", "x_vs_z_vs_Rsumcos"]

        required_plots = ["x_vs_eta", "x_vs_phi", "x_vs_R", "l_vs_eta", "l_vs_phi", "l_vs_R"]

        required_ratio_plots = ["x_over_l_vs_eta", "x_over_l_vs_phi"]

        #Function to help filling the twiki with all these plots
        #First I loop through labels to put the hide button in twiki
        #All HGCal
        print("---+++ Results: Plots for individual material in all HGCal")
        for label, [num, color, leg] in hist_label_to_num.items():
            for p in ["x_vs_z_vs_Rsum", "l_vs_z_vs_Rsum", "x_vs_z_vs_Rsumcos", "l_vs_z_vs_Rsumcos", "x_vs_z_vs_Rloc", "l_vs_z_vs_Rloc"]:
                TwikiPrintout(p, leg, "all")
        #Z+
        print("---+++ Results: Plots for individual material in Z+ Endcap")
        for label, [num, color, leg] in hist_label_to_num.items():
            for p in ["x_vs_z_vs_Rsum", "l_vs_z_vs_Rsum", "x_vs_z_vs_Rsumcos", "l_vs_z_vs_Rsumcos", "x_vs_z_vs_Rloc", "l_vs_z_vs_Rloc"]:
                TwikiPrintout(p, leg, "zplus")
        #Z-
        print("---+++ Results: Plots for individual material in Z- Endcap")
        for label, [num, color, leg] in hist_label_to_num.items():
            for p in ["x_vs_z_vs_Rsum", "l_vs_z_vs_Rsum", "x_vs_z_vs_Rsumcos", "l_vs_z_vs_Rsumcos", "x_vs_z_vs_Rloc", "l_vs_z_vs_Rloc"]:
                TwikiPrintout(p, leg, "zminus")

        #Below is the silicon position from the xml geometry file
        #Should we put them on top of plots like the eta values?
        print(GetSiliconZValuesFromXML())

        for p in required_2Dplots:
            #First the total
            create2DPlots(args.detector, p, plots[p].plotNumber, "")
            #Then, the rest
            for label, [num, color, leg] in hist_label_to_num.items():
                #print label, num, color, leg
                create2DPlots(args.detector, p, num + plots[p].plotNumber, leg)


        for p in required_plots:
            createCompoundPlots(args.detector, p)
        for p in required_ratio_plots:
            createRatioPlots(args.detector, p)
