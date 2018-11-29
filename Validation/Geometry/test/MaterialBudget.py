#! /usr/bin/env python

# Pure trick to start ROOT in batch mode, pass this only option to it
# and the rest of the command line options to this code.
import six
import sys
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
from ROOT import TCanvas, TPad, TGaxis, TLegend, TPaveText, THStack, TFile, TLatex
from ROOT import TProfile, TProfile2D, TH1D, TH2F, TPaletteAxis, TStyle, TColor
from ROOT import kBlack, kWhite, kOrange, kAzure, kBlue, kRed, kGreen
from ROOT import kGreyScale, kTemperatureMap
from ROOT import kTRUE, kFALSE
from ROOT import gROOT, gStyle, gPad
gROOT.SetBatch(True)
sys.argv = oldargv

from Validation.Geometry.plot_utils import setTDRStyle, Plot_params, plots, COMPOUNDS, DETECTORS, sDETS, hist_label_to_num, drawEtaValues
from collections import namedtuple, OrderedDict
import sys, os, copy
import argparse

def paramsGood_(detector, plot, geometryOld = '', geometryNew = ''):
    """Check the validity of the arguments.

       Common function to check the validity of the parameters passed
       in. It returns a tuple composed by a bool and a string. The
       bool indicates if all checks are ok, the string the appropriate
       ROOT filename to open (empty string in case any check failed)
       If geometry comparison is being made, a list of strings is
       returned instead.

    """

    theFiles = []

    if plot not in plots.keys():
        print("Error, unknown plot %s" % plot)
        return (False, '')

    if detector not in DETECTORS and detector not in COMPOUNDS.keys():
        print('Error, unknown detector: %s' % detector)
        return (False, '')

    if detector not in DETECTORS:
        detector = COMPOUNDS[detector][0]

    if geometryNew:
        oldgeoFilename = 'matbdg_%s_%s.root' % (detector,geometryOld)
        theFiles.append(oldgeoFilename)
        newgeoFilename = 'matbdg_%s_%s.root' % (detector,geometryNew)
        theFiles.append(newgeoFilename)
    else:
        theFiles.append('matbdg_%s_%s.root' % (detector,geometryOld))

    for thisFile in theFiles:
        if not checkFile_(thisFile):
            print("Error, missing file %s" % thisFile)
            raise RuntimeError

    if len(theFiles) >  1:
        return (True, theFiles)
    else:
        return (True, theFiles[0])

def checkFile_(filename):
    return os.path.exists(filename)

def setColorIfExists_(histos, h, color):
    if h in histos.keys():
        histos[h].SetFillColor(color)

def assignOrAddIfExists_(h1, h2):
    """Assign the projection of h2 to h1.

       Function to assign the h2 to h1 in the case in
       which h1 is None, otherwise add h2 to the already
       valid h1 object

    """

    if not h1:
        h1 = h2
    else:
        h1.Add(h2, +1.000)
    return h1

def get1DHisto_(detector,plotNumber,geometry):
    """
     This function opens the appropiate ROOT file, 
     extracts the TProfile and turns it into a Histogram,
     if it is a compound detector, this function
     takes care of the subdetectors' addition unless the
     detector's ROOT file is present in which case no addition
     is performed and the detector ROOT file is used.
    """
    histo = None
    rootFile = TFile()

    detectorFilename = 'matbdg_%s_%s.root'%(detector,geometry)
    if detector not in COMPOUNDS.keys() or checkFile_(detectorFilename):
        if not checkFile_(detectorFilename):
            print('Warning: %s not found' % detectorFilename)
            return 0
        print('Reading from: %s File' % detectorFilename)
        rootFile = TFile.Open(detectorFilename,'READ')
        prof = rootFile.Get("%d" % plotNumber)
        if not prof: return 0
        # Prevent memory leaking by specifing a unique name
        prof.SetName('%u_%s_%s' %(plotNumber,detector,geometry))
        histo = prof.ProjectionX()
    else:
        theFiles = []
        histos = OrderedDict()
        for subDetector in COMPOUNDS[detector]:
            subDetectorFilename = 'matbdg_%s_%s.root' % (subDetector,geometry)
            if not checkFile_(subDetectorFilename):
                print('Warning: %s not found'%subDetectorFilename)
                continue
            print('Reading from: %s File' % subDetectorFilename)
            subDetectorFile = TFile.Open(subDetectorFilename,'READ')
            theFiles.append(subDetectorFile)
            prof = subDetectorFile.Get('%d'%(plotNumber)) 
            if not prof: return 0
            prof.__class__ = TProfile
            histo = assignOrAddIfExists_(histo,prof.ProjectionX())

    return copy.deepcopy(histo)

def get2DHisto_(detector,plotNumber,geometry):
    """
     This function opens the appropiate ROOT file, 
     extracts the TProfile2D and turns it into a Histogram,
     if it is a compound detector, this function
     takes care of the subdetectors' addition.

     Note that it takes plotNumber as opposed to plot
    """
    histo = None
    rootFile = TFile()

    detectorFilename = 'matbdg_%s_%s.root'%(detector,geometry)
    if detector not in COMPOUNDS.keys() or checkFile_(detectorFilename):
        if not checkFile_(detectorFilename):
            print('Warning: %s not found' % detectorFilename)
            return 0
        rootFile = TFile.Open(detectorFilename,'READ')
        prof = rootFile.Get("%d" % plotNumber)
        if not prof: return 0
        # Prevent memory leaking by specifing a unique name
        prof.SetName('%u_%s_%s' %(plotNumber,detector,geometry))
        prof.__class__ = TProfile2D
        histo = prof.ProjectionXY()
    else:
        histos = OrderedDict()
        theFiles = []
        for subDetector in COMPOUNDS[detector]:
            subDetectorFilename = 'matbdg_%s_%s.root' % (subDetector,geometry)
            if not checkFile_(subDetectorFilename):
                print('Warning: %s not found'%subDetectorFilename)
                continue
            subDetectorFile = TFile.Open(subDetectorFilename,'READ')
            theFiles.append(subDetectorFile)
            print('*** Open file... %s' % subDetectorFilename)
            prof = subDetectorFile.Get('%d'%plotNumber)
            if not prof: return 0
            prof.__class__ = TProfile2D
            if not histo:
                histo = prof.ProjectionXY('B_%s' % prof.GetName())
            else:
                histo.Add(prof.ProjectionXY('B_%s' % prof.GetName()))

    return copy.deepcopy(histo)

def createCompoundPlotsGeometryComparison(detector, plot, geometryOld,
                                          geometryNew):

    setTDRStyle()

    goodToGo, theFiles = paramsGood_(detector,plot,
                                     geometryOld,geometryNew)

    if not goodToGo:
        return

    oldHistos = OrderedDict()
    newHistos = OrderedDict()
    ratioHistos = OrderedDict()
    diffHistos = OrderedDict()

    def setUpCanvas(canvas):

        gStyle.SetOptStat(False)
    
        mainPadTop = [        
            TPad("mainPadTop"+str(i)+'_'+canvas.GetName(),
                 "mainPad"+str(i),
                 i*0.25, 0.60, (i+1)*0.25, 1.0)
            for i in range(4)
            ]
        
        subPadTop = [
            TPad("subPadTop"+str(i)+'_'+canvas.GetName(),
                "subPad"+str(i),
                 i*0.25, 0.50, (i+1)*0.25, 0.6)
            for i in range(4)
            ]
        
        mainPadBottom = [
            TPad("mainPadBottom"+str(i)+'_'+canvas.GetName(),
                 "subPad"+str(i),
                 i*0.25, 0.10, (i+1)*0.25, 0.5)
            for i in range(4)
            ]
        
        subPadBottom = [
            TPad("subPadBottom"+str(i)+'_'+canvas.GetName(),
                 "subPad"+str(i),
                 i*0.25, 0.00, (i+1)*0.25, 0.1)
            for i in range(4)
            ]
        
        mainPad = mainPadTop + mainPadBottom
        subPad = subPadTop + subPadBottom    
        
        leftMargin = 0.12
        rightMargin = 0.12
        topMargin = 0.12
        bottomMargin = 0.3
        for i in range(8):
            mainPad[i].SetLeftMargin(leftMargin)
            mainPad[i].SetRightMargin(rightMargin)
            mainPad[i].SetTopMargin(topMargin)
            mainPad[i].SetBottomMargin(1e-3)
            mainPad[i].Draw()
            subPad[i].SetLeftMargin(leftMargin)
            subPad[i].SetRightMargin(rightMargin)
            subPad[i].SetTopMargin(1e-3)
            subPad[i].SetBottomMargin(bottomMargin)
            subPad[i].Draw()

        return mainPad, subPad

    canComparison = TCanvas("canComparison","canComparison",2400,1200)
    mainPad, subPad = setUpCanvas(canComparison)


    def setStyleHistoSubPad(histo):
        histo.SetTitle('')
        histo.SetMarkerColor(kBlack)
        histo.SetMarkerStyle(20) # Circles
        histo.SetMarkerSize(.5)
        histo.SetLineWidth(1)

        histo.GetYaxis().SetTitleSize(14)
        histo.GetYaxis().SetTitleFont(43)
        histo.GetYaxis().SetLabelSize(0.17)
        histo.GetYaxis().SetTitleOffset(5.0)
        histo.GetYaxis().SetNdivisions(6,3,0)

        histo.GetXaxis().SetTitleSize(25)
        histo.GetXaxis().SetTitleFont(43)
        histo.GetXaxis().SetTitleOffset(6.0)
        histo.GetXaxis().SetLabelSize(0.17)

        return histo
        
    def makeRatio(histoX,histoY):
        # return stylized ratio histoX/histoY
        histoXOverY = copy.deepcopy(histoX)
        histoXOverY.Divide(histoY)
        histoXOverY.GetYaxis().SetTitle('#frac{%s}{%s}' % (geometryNew,geometryOld))

        return histoXOverY

    def makeDiff(histoNew,histoOld):
        # Return stylized histoNew - histoOld
        diff = copy.deepcopy(histoNew)
        diff.Add(histoOld,-1.0)
        diff.GetYaxis().SetTitle(geometryNew 
                                 + " - "
                                 + geometryOld)
        diff.GetYaxis().SetNdivisions(6,3,0)

        diff.GetXaxis().SetTitleSize(25)
        diff.GetXaxis().SetTitleFont(43)
        diff.GetXaxis().SetTitleOffset(3.5)
        diff.GetXaxis().SetLabelSize(0.17)
        
        return diff


    # Plotting the different categories

    def setUpTitle(detector,label,plot):
        title = 'Material Budget %s [%s];%s;%s' % (detector,label,
                                                   plots[plot].abscissa,
                                                   plots[plot].ordinate)
        return title

    def setUpLegend(gOld,gNew,label):
        legend = TLegend(0.4,0.7,0.7,0.85)
        legend.AddEntry(gOld,"%s %s [%s]"%(detector,geometryOld,label),"F") #(F)illed Box
        legend.AddEntry(gNew,"%s %s [%s]"%(detector,geometryNew,label),"P") #(P)olymarker
        legend.SetTextFont(42)
        legend.SetTextSize(0.03)
        return legend

    def setRanges(h):
        legendSpace = 1. + 0.3 # 30%
        minX = h.GetXaxis().GetXmin()
        maxX = h.GetXaxis().GetXmax()
        minY = h.GetYaxis().GetXmin()
        maxY = h.GetBinContent(h.GetMaximumBin()) * legendSpace
        h.GetYaxis().SetRangeUser(minY, maxY)
        h.GetXaxis().SetRangeUser(minX, maxX)


    ########### Ratio ###########

    counter = 0
    legends = OrderedDict() #KeepAlive
    for label, [num, color, leg] in six.iteritems(hist_label_to_num):

        mainPad[counter].cd()
        oldHistos[label] = get1DHisto_(detector,
                                       num+plots[plot].plotNumber
                                       ,geometryOld)
        oldHistos[label].SetTitle(setUpTitle(detector,leg,plot))
        oldHistos[label].SetFillColor(color)
        oldHistos[label].SetLineColor(kBlack)
        oldHistos[label].SetLineWidth(1)
        setRanges(oldHistos[label])
        oldHistos[label].Draw("HIST")

        newHistos[label] = get1DHisto_(detector,
                                       num+plots[plot].plotNumber
                                       ,geometryNew)
        newHistos[label].SetMarkerSize(.5)
        newHistos[label].SetMarkerStyle(20)
        newHistos[label].Draw('SAME P')

        legends[label]= setUpLegend(oldHistos[label],newHistos[label],
                                    leg);
        legends[label].Draw()

        # Ratio
        subPad[counter].cd()
        ratioHistos[label] = makeRatio( newHistos[label],oldHistos[label] )
        ratioHistos[label] = setStyleHistoSubPad(ratioHistos[label])
        ratioHistos[label].Draw("HIST P")

        counter += 1

    theDirname = "Images"

    if not checkFile_(theDirname):
        os.mkdir(theDirname)
        
    canComparison.SaveAs( "%s/%s_ComparisonRatio_%s_%s_vs_%s.png"
                          % (theDirname,detector,plot,geometryOld,geometryNew) )

    ######## Difference ########

    canDiff = TCanvas("canDiff","canDiff",2400,1200)

    mainPadDiff, subPadDiff = setUpCanvas(canDiff)
 
    counter = 0
    for label, [num, color, leg] in six.iteritems(hist_label_to_num):
        mainPadDiff[counter].cd()
        oldHistos[label].SetTitle(setUpTitle(detector,leg,plot))
        oldHistos[label].Draw("HIST")
        newHistos[label].Draw('SAME P')

        legends[label].Draw()

        # Difference
        subPadDiff[counter].cd()
        diffHistos[label] = makeDiff( newHistos[label],oldHistos[label] )
        diffHistos[label] = setStyleHistoSubPad(diffHistos[label])
        diffHistos[label].SetTitle('')
        diffHistos[label].SetFillColor(color+1)
        diffHistos[label].Draw("HIST")
        counter +=1

    canDiff.SaveAs( "%s/%s_ComparisonDifference_%s_%s_vs_%s.png"
                          % (theDirname,detector,plot,geometryOld,geometryNew) )
        

def setUpPalette(histo2D, plot) :

    # Configure Palette for 2D Histos

    minX = 1.03*histo2D.GetXaxis().GetXmin();
    maxX = 1.03*histo2D.GetXaxis().GetXmax();
    minY = 1.03*histo2D.GetYaxis().GetXmin();
    maxY = 1.03*histo2D.GetYaxis().GetXmax();

    palette = histo2D.GetListOfFunctions().FindObject("palette")
    if palette:
        palette.__class__ = TPaletteAxis
        palette.SetX1NDC(0.945)
        palette.SetY1NDC(gPad.GetBottomMargin())
        palette.SetX2NDC(0.96)
        palette.SetY2NDC(1-gPad.GetTopMargin())
        palette.GetAxis().SetTickSize(.01)
        palette.GetAxis().SetTitle("")
        if plots[plot].zLog:
            palette.GetAxis().SetLabelOffset(-0.01)
            if histo2D.GetMaximum()/histo2D.GetMinimum() < 1e3 :
                palette.GetAxis().SetMoreLogLabels(True)
                palette.GetAxis().SetNoExponent(True)

    paletteTitle = TLatex(1.12*maxX, maxY, plots[plot].quotaName)
    paletteTitle.SetTextAngle(90.)
    paletteTitle.SetTextSize(0.05)
    paletteTitle.SetTextAlign(31)
    paletteTitle.Draw()

    histo2D.GetXaxis().SetTickLength(histo2D.GetXaxis().GetTickLength()/4.)
    histo2D.GetYaxis().SetTickLength(histo2D.GetYaxis().GetTickLength()/4.)
    histo2D.SetTitleOffset(0.5,'Y')
    histo2D.GetXaxis().SetNoExponent(True)
    histo2D.GetYaxis().SetNoExponent(True)

def create2DPlotsGeometryComparison(detector, plot, 
                                    geometryOld, geometryNew):

    setTDRStyle()

    print('Extracting plot: %s.'%(plot))
    goodToGo, theFiles = paramsGood_(detector,plot,
                                     geometryOld,geometryNew)

    if not goodToGo:
        return
    
    gStyle.SetOptStat(False)

    old2DHisto = get2DHisto_(detector,plots[plot].plotNumber,geometryOld)
    new2DHisto = get2DHisto_(detector,plots[plot].plotNumber,geometryNew)

    if plots[plot].iRebin:
        old2DHisto.Rebin2D()
        new2DHisto.Rebin2D()

    def setRanges(h):
        h.GetXaxis().SetRangeUser(plots[plot].xmin, plots[plot].xmax)
        h.GetYaxis().SetRangeUser(plots[plot].ymin, plots[plot].ymax)
        if plots[plot].histoMin != -1.:
            h.SetMinimum(plots[plot].histoMin)
        if plots[plot].histoMax != -1.:
            h.SetMaximum(plots[plot].histoMax)

    ratio2DHisto = copy.deepcopy(new2DHisto)
    ratio2DHisto.Divide(old2DHisto)
    # Ratio and Difference have the same call
    # But different 'Palette' range so we are
    # setting the range only for the Ratio
    ratio2DHisto.SetMinimum(0.2)
    ratio2DHisto.SetMaximum(1.8)
    setRanges(ratio2DHisto)

    diff2DHisto = copy.deepcopy(new2DHisto)
    diff2DHisto.Add(old2DHisto,-1.0)
    setRanges(diff2DHisto)


    def setPadStyle():
        gPad.SetLeftMargin(0.05)
        gPad.SetRightMargin(0.08)
        gPad.SetTopMargin(0.10)
        gPad.SetBottomMargin(0.10)
        gPad.SetLogz(plots[plot].zLog)
        gPad.SetFillColor(kWhite)
        gPad.SetBorderMode(0)

    can = TCanvas('can','can',
                  2724,1336)
    can.Divide(1,2)
    can.cd(1)
    setPadStyle()
    gPad.SetLogz(plots[plot].zLog)
    
    gStyle.SetOptStat(0)
    gStyle.SetFillColor(kWhite)
    gStyle.SetPalette(kTemperatureMap)

    ratio2DHisto.SetTitle("%s, Ratio: %s/%s;%s;%s"
                          %(plots[plot].quotaName,
                            geometryOld, geometryNew,
                            plots[plot].abscissa,
                            plots[plot].ordinate))
    ratio2DHisto.Draw('COLZ')

    can.Update()

    setUpPalette(ratio2DHisto,plot)

    etasTop = []
    if plots[plot].iDrawEta:
        etasTop.extend(drawEtaValues())

    can.cd(2)

    diff2DHisto.SetTitle('%s, Difference: %s - %s %s;%s;%s'
                         %(plots[plot].quotaName,geometryNew,geometryOld,detector,
                           plots[plot].abscissa,plots[plot].ordinate))
    setPadStyle()
    diff2DHisto.Draw("COLZ")
    can.Update()
    setUpPalette(diff2DHisto,plot)

    etasBottom = []
    if plots[plot].iDrawEta:
        etasBottom.extend(drawEtaValues())

    can.Modified()

    theDirname = "Images"

    if not checkFile_(theDirname):
        os.mkdir(theDirname)
        
    can.SaveAs( "%s/%s_Comparison_%s_%s_vs_%s.png"
                % (theDirname,detector,plot,geometryOld,geometryNew) )
    gStyle.SetStripDecimals(True)

def createPlots_(plot, geometry):
    """Cumulative material budget from simulation.
    
       Internal function that will produce a cumulative profile of the
       material budget inferred from the simulation starting from the
       single detectors that compose the tracker. It will iterate over
       all existing detectors contained in the DETECTORS
       dictionary. The function will automatically skip non-existent
       detectors.

    """

    IBs = ["InnerServices", "Phase2PixelBarrel", "TIB", "TIDF", "TIDB"]
    theDirname = "Figures"

    if plot not in plots.keys():
        print("Error: chosen plot name not known %s" % plot)
        return

    hist_X0_detectors = OrderedDict()
    hist_X0_IB = None
    hist_X0_elements = OrderedDict()

    for subDetector,color in six.iteritems(DETECTORS):
        h = get1DHisto_(subDetector,plots[plot].plotNumber,geometry)
        if not h: 
            print('Warning: Skipping %s'%subDetector)
            continue
        hist_X0_detectors[subDetector] = h


        # Merge together the "inner barrel detectors".
        if subDetector in IBs:
            hist_X0_IB = assignOrAddIfExists_(
                hist_X0_IB,
                hist_X0_detectors[subDetector]
                )

        # category profiles
        for label, [num, color, leg] in six.iteritems(hist_label_to_num):
            if label is 'SUM': continue
            hist_label = get1DHisto_(subDetector, num + plots[plot].plotNumber, geometry)
            hist_X0_elements[label] = assignOrAddIfExists_(
                hist_X0_elements.setdefault(label,None),
                hist_label,
                )
            hist_X0_elements[label].SetFillColor(color)


    cumulative_matbdg = TH1D("CumulativeSimulMatBdg",
                             "CumulativeSimulMatBdg",
                             hist_X0_IB.GetNbinsX(),
                             hist_X0_IB.GetXaxis().GetXmin(),
                             hist_X0_IB.GetXaxis().GetXmax())
    cumulative_matbdg.SetDirectory(0)

    # colors
    for det, color in six.iteritems(DETECTORS):
        setColorIfExists_(hist_X0_detectors, det,  color)

    # First Plot: BeamPipe + Pixel + TIB/TID + TOB + TEC + Outside
    # stack
    stackTitle_SubDetectors = "Tracker Material Budget;%s;%s" % (
        plots[plot].abscissa,plots[plot].ordinate)
    stack_X0_SubDetectors = THStack("stack_X0",stackTitle_SubDetectors)
    for det, histo in six.iteritems(hist_X0_detectors):
        stack_X0_SubDetectors.Add(histo)
        cumulative_matbdg.Add(histo, 1)

    # canvas
    can_SubDetectors = TCanvas("can_SubDetectors","can_SubDetectors",800,800)
    can_SubDetectors.Range(0,0,25,25)
    can_SubDetectors.SetFillColor(kWhite)

    # Draw
    stack_X0_SubDetectors.SetMinimum(plots[plot].ymin)
    stack_X0_SubDetectors.SetMaximum(plots[plot].ymax)
    stack_X0_SubDetectors.Draw("HIST")
    stack_X0_SubDetectors.GetXaxis().SetLimits(plots[plot].xmin, plots[plot].xmax)


    # Legenda
    theLegend_SubDetectors = TLegend(0.180,0.8,0.98,0.92)
    theLegend_SubDetectors.SetNColumns(3)
    theLegend_SubDetectors.SetFillColor(0)
    theLegend_SubDetectors.SetFillStyle(0)
    theLegend_SubDetectors.SetBorderSize(0)

    for det, histo in six.iteritems(hist_X0_detectors):
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
    if not checkFile_(theDirname):
        os.mkdir(theDirname)
    can_SubDetectors.SaveAs("%s/Tracker_SubDetectors_%s.pdf" % (theDirname, plot))
    can_SubDetectors.SaveAs("%s/Tracker_SubDetectors_%s.root" % (theDirname, plot))


    # Second Plot: BeamPipe + SEN + ELE + CAB + COL + SUP + OTH/AIR +
    # Outside stack
    stackTitle_Materials = "Tracker Material Budget;%s;%s" % (plots[plot].abscissa,
                                                              plots[plot].ordinate)
    stack_X0_Materials = THStack("stack_X0",stackTitle_Materials)
    stack_X0_Materials.Add(hist_X0_detectors["BeamPipe"])
    for label, [num, color, leg] in six.iteritems(hist_label_to_num):
        if label is 'SUM':
            continue
        stack_X0_Materials.Add(hist_X0_elements[label])

    # canvas
    can_Materials = TCanvas("can_Materials","can_Materials",800,800)
    can_Materials.Range(0,0,25,25)
    can_Materials.SetFillColor(kWhite)

    # Draw
    stack_X0_Materials.SetMinimum(plots[plot].ymin)
    stack_X0_Materials.SetMaximum(plots[plot].ymax)
    stack_X0_Materials.Draw("HIST")
    stack_X0_Materials.GetXaxis().SetLimits(plots[plot].xmin, plots[plot].xmax)

    # Legenda
    theLegend_Materials = TLegend(0.180,0.8,0.95,0.92)
    theLegend_Materials.SetNColumns(3)
    theLegend_Materials.SetFillColor(0)
    theLegend_Materials.SetBorderSize(0)

    theLegend_Materials.AddEntry(hist_X0_detectors["BeamPipe"],  "Beam Pipe", "f")
    for label, [num, color, leg] in six.iteritems(hist_label_to_num):
        if label is 'SUM':
            continue
        theLegend_Materials.AddEntry(hist_X0_elements[label], leg, "f")
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
    can_Materials.SaveAs("%s/Tracker_Materials_%s.pdf" % (theDirname, plot))
    can_Materials.SaveAs("%s/Tracker_Materials_%s.root" % (theDirname, plot))

    return cumulative_matbdg

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
        for det, color in six.iteritems(sDETS):
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

def materialBudget_Simul_vs_Reco(reco_file, label, geometry, debug=False):
    """Plot reco vs simulation material budget.
    
       Function are produces a direct comparison of the material
       budget as extracted from the reconstruction geometry and
       inferred from the simulation one.

    """

    setTDRStyle()

    # plots
    cumulative_matbdg_sim = createPlots_("x_vs_eta", geometry)
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

def createCompoundPlots(detector, plot, geometry):
    """Produce the requested plot for the specified detector.

       Function that will plot the requested @plot for the specified
       @detector. The specified detector could either be a real
       detector or a compound one. The list of available plots are the
       keys of plots dictionary (imported from plot_utils.

    """

    theDirname = 'Images'
    if not checkFile_(theDirname):
        os.mkdir(theDirname)

    goodToGo, theDetectorFilename = paramsGood_(detector, plot, geometry)
    if not goodToGo:
        return

    hist_X0_elements = OrderedDict()

    # stack
    stackTitle = "%s;%s;%s" % (detector,
                                               plots[plot].abscissa,
                                               plots[plot].ordinate)
    stack_X0 = THStack("stack_X0", stackTitle);
    theLegend = TLegend(0.70, 0.70, 0.89, 0.89);

    def setRanges(h):
        legendSpace = 1. + 0.3 # 30%
        minY = h.GetYaxis().GetXmin()
        maxY = h.GetBinContent(h.GetMaximumBin()) * legendSpace
        h.GetYaxis().SetRangeUser(minY, maxY)

    for label, [num, color, leg] in six.iteritems(hist_label_to_num):
        # We don't want the sum to be added as part of the stack
        if label is 'SUM':
            continue
        hist_X0_elements[label] = get1DHisto_(detector,
                                              num + plots[plot].plotNumber,
                                              geometry)
        hist_X0_elements[label].SetFillColor(color)
        hist_X0_elements[label].SetLineColor(kBlack)
        stack_X0.Add(hist_X0_elements[label])
        theLegend.AddEntry(hist_X0_elements[label], leg, "f")

    # canvas
    canname = "MBCan_1D_%s_%s"  % (detector, plot)
    can = TCanvas(canname, canname, 800, 800)
    can.Range(0,0,25,25)
    can.SetFillColor(kWhite)
    gStyle.SetOptStat(0)

    setTDRStyle()

    # Draw
    setRanges(stack_X0.GetStack().Last())
    stack_X0.Draw("HIST");
    theLegend.Draw();

    cmsMark = TLatex()
    cmsMark.SetNDC();
    cmsMark.SetTextAngle(0);
    cmsMark.SetTextColor(kBlack);    
    cmsMark.SetTextFont(61)
    cmsMark.SetTextSize(7e-2)
    cmsMark.SetTextAlign(11)
    cmsMark.DrawLatex(0.1,0.91,"CMS")

    simuMark = TLatex()
    simuMark.SetNDC();
    simuMark.SetTextAngle(0);
    simuMark.SetTextColor(kBlack);    
    simuMark.SetTextSize(3e-2)
    simuMark.SetTextAlign(11)
    simuMark.DrawLatex(0.26,0.91,"#font[52]{Preliminary Simulation}")
 
    # Store
    can.Update();
    can.SaveAs( "%s/%s_%s_%s.pdf" 
                % (theDirname, detector, plot, geometry))
    can.SaveAs( "%s/%s_%s_%s.png" 
                % (theDirname, detector, plot, geometry))


def create2DPlots(detector, plot, geometry):
    """Produce the requested plot for the specified detector.

       Function that will plot the requested 2D-@plot for the
       specified @detector. The specified detector could either be a
       real detector or a compound one. The list of available plots
       are the keys of plots dictionary (imported from plot_utils).

    """

    theDirname = 'Images'
    if not checkFile_(theDirname):
        os.mkdir(theDirname)

    goodToGo, theDetectorFilename = paramsGood_(detector, plot, geometry)
    if not goodToGo:
        return

    theDetectorFile = TFile(theDetectorFilename)

    hist_X0_total = get2DHisto_(detector,plots[plot].plotNumber,geometry)

    # # properties
    gStyle.SetStripDecimals(False)

    # Ratio
    if plots[plot].iRebin:
        hist_X0_total.Rebin2D()

    # stack
    hist2dTitle = ('%s %s;%s;%s;%s' % (plots[plot].quotaName,
                                       detector,
                                       plots[plot].abscissa,
                                       plots[plot].ordinate,
                                       plots[plot].quotaName))

    hist_X0_total.SetTitle(hist2dTitle)
    hist_X0_total.SetTitleOffset(0.5,"Y")

    if plots[plot].histoMin != -1.:
        hist_X0_total.SetMinimum(plots[plot].histoMin)
    if plots[plot].histoMax != -1.:
        hist_X0_total.SetMaximum(plots[plot].histoMax)

    can2name = "MBCan_2D_%s_%s" % (detector, plot)
    can2 = TCanvas(can2name, can2name, 2480+248, 580+58+58)
    can2.SetTopMargin(0.1)
    can2.SetBottomMargin(0.1)
    can2.SetLeftMargin(0.04)
    can2.SetRightMargin(0.06)
    can2.SetFillColor(kWhite)
    gStyle.SetOptStat(0)
    gStyle.SetTitleFillColor(0)
    gStyle.SetTitleBorderSize(0)

    # Color palette
    gStyle.SetPalette(kGreyScale)

    # Log?
    can2.SetLogz(plots[plot].zLog)

    # Draw in colors
    hist_X0_total.Draw("COLZ")

    # Store
    can2.Update()

    #Aesthetic
    setUpPalette(hist_X0_total,plot)

    #Add eta labels
    keep_alive = []
    if plots[plot].iDrawEta:
        keep_alive.extend(drawEtaValues())

    can2.Modified()
    hist_X0_total.SetContour(255)

    # Store
    can2.Update()
    can2.Modified()

    can2.SaveAs( "%s/%s_%s_%s_bw.pdf" 
                 % (theDirname, detector, plot, geometry))
    can2.SaveAs( "%s/%s_%s_%s_bw.png" 
                 % (theDirname, detector, plot, geometry))
    gStyle.SetStripDecimals(True)

def createRatioPlots(detector, plot, geometry):
    """Create ratio plots.

       Function that will make the ratio between the radiation length
       and interaction length, for the specified detector. The
       specified detector could either be a real detector or a
       compound one.

    """

    goodToGo, theDetectorFilename = paramsGood_(detector, plot, geometry)
    if not goodToGo:
        return

    theDirname = 'Images'
    if not os.path.exists(theDirname):
        os.mkdir(theDirname)

    theDetectorFile = TFile(theDetectorFilename)
    # get TProfiles
    prof_x0_det_total = theDetectorFile.Get('%d' % plots[plot].plotNumber)
    prof_l0_det_total = theDetectorFile.Get('%d' % (1000+plots[plot].plotNumber))

    # histos
    hist_x0_total = get1DHisto_(detector,plots[plot].plotNumber,geometry)
    hist_l0_total = get1DHisto_(detector,1000+plots[plot].plotNumber,geometry)

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
    canR.SaveAs("%s/%s_%s_%s.pdf" 
                % (theDirname, detector, plot, geometry))
    canR.SaveAs("%s/%s_%s_%s.png" 
                % (theDirname, detector, plot, geometry))

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
    parser.add_argument('-s', '--single',
                        help='Material budget for single detector from simulation',
                        action='store_true',
                        default=False)
    parser.add_argument('-d', '--detector',
                        help='Detector for which you want to compute the material budget',
                        type=str,)
    parser.add_argument('-g', '--geometry',
                        help='Geometry, used to determine filenames',
                        type=str)
    parser.add_argument('-gc', '--geometry-comparison',
                        help='Compare the material budget for two different geometries'
                        +'-g should be specied',
                        type=str)
    args = parser.parse_args()


    if args.geometry is None:
        print("Error, missing geometry")
        raise RuntimeError

    if args.geometry_comparison and args.geometry is None:
        print("Error, geometry comparison requires two geometries")
        raise RuntimeError
    
    if args.geometry_comparison and args.geometry:

        # For the definition of the properties of these graphs
        # check plot_utils.py

        required_plots = ["x_vs_eta","x_vs_phi","x_vs_R",
                          "l_vs_eta","l_vs_phi","l_vs_R"]
        required_2Dplots = ["x_vs_eta_vs_phi",
                            "l_vs_eta_vs_phi",
                            "x_vs_z_vs_R",
                            "l_vs_z_vs_R_geocomp",
                            "x_vs_z_vs_Rsum",
                            "l_vs_z_vs_Rsum"]

        for p in required_plots:
            createCompoundPlotsGeometryComparison(args.detector, p, args.geometry,
                                                  args.geometry_comparison)
        for p in required_2Dplots:
            create2DPlotsGeometryComparison(args.detector, p, args.geometry,
                                                    args.geometry_comparison)

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
        materialBudget_Simul_vs_Reco(args.reco, args.label, args.geometry, debug=False)

    if args.single and not args.geometry_comparison:
        if args.detector is None:
            print("Error, missing detector")
            raise RuntimeError
        required_2Dplots = ["x_vs_eta_vs_phi", "l_vs_eta_vs_phi", "x_vs_z_vs_R",
                            "l_vs_z_vs_R", "x_vs_z_vs_Rsum", "l_vs_z_vs_Rsum"]
        required_plots = ["x_vs_eta", "x_vs_phi", "l_vs_eta", "l_vs_phi"]

        required_ratio_plots = ["x_over_l_vs_eta", "x_over_l_vs_phi"]

        for p in required_2Dplots:
            create2DPlots(args.detector, p, args.geometry)
        for p in required_plots:
            createCompoundPlots(args.detector, p, args.geometry)
        for p in required_ratio_plots:
            createRatioPlots(args.detector, p, args.geometry)
