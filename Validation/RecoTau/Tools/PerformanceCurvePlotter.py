"""
        PerformanceCurvePlotter

        Author: Evan K. Friis, UC Davis (friis@physics.ucdavis.edu)

        Plot the efficiency for a given fake rate 
        for a series of cuts in a set pt range.

        With the default cuts (20 < pt < 50), this is designed
        to reproduce (independently) the TancBenchmark plots produced in
        RecoTauTag/TauTagTools/test/MVABenchmarks.py


            Instructions:

            Add the releases to as shown.  You must specify and signal 
            and backround file for each release, and a descriptive label.
            To choose the chain of discriminators to plot, select the appropriate
            EDProducer defined in Validation.RecoTau.TauTagValidation_cfi

            PreFourTanc = TauValidationInfo("Signal.root", 
                                            "Background.root",
                                            "310pre5 TaNC",
                                            Validation.RecoTau.RecoTauValidation_cfi.RunTancValidation)

            PreFourIso = TauValidationInfo("Signal.root", 
                                            "Background.root",
                                            "310pre4 Iso",
                                            Validation.RecoTau.RecoTauValidation_cfi.PFTausHighEfficiencyLeadingPionBothProngs)


            ReleasesToBuildFrom = []
            ReleasesToBuildFrom.append(PreFourTanc)
            ReleasesToBuildFrom.append(PreFourIso)

            PlotPerformanceCurves(ReleasesToBuildFrom, "output.png")

"""

from ROOT import TH1F, TGraph, TCanvas, TPad, EColor, TFile, TGraphAsymmErrors, Double, TLegend, gPad, TPaveText, gROOT, gStyle
import FWCore.ParameterSet.Config as cms
import copy
import Validation.RecoTau.RecoTauValidation_cfi 

"""
        Kinematic cuts applied to the sample.  The Validation histograms
        will be integrated over this range.  You can only select one variable.
"""

KinematicVar    = 'pt'
LatexVar        = "True p_{T}"
KinematicVarMin = 20
KinematicVarMax = 50 
VarUnit         = "GeV/c"

# Uncomment me to display electron and muon rejection performance
HideMuonAndElectronRej = True


gROOT.SetBatch(True)
gROOT.SetStyle("Plain")
gStyle.SetOptStat(0)
gStyle.SetPalette(1)
gStyle.SetTitleBorderSize(0)

class TauValidationInfo:
   """ Atomic object to hold information about where files are located, which discriminants to plot """
   DiscriminatorToMarkerStyleDict = {} 
   StaticMarkerStyleIterator      = 20
   StaticSummaryLineColorIterator = 2 
   StupidRootStupidNameCounter    = 0  # to give each a unique name
   DiscriminatorLegend = TLegend(0.15, 0.6, 0.5, 0.92)
   DiscriminatorLegend.SetFillColor(0)
   DiscriminatorBackground = 0
   SummaryLegend = TLegend(0.6, 0.3, 0.9, 0.4)
   SummaryLegend.SetBorderSize(0)
   SummaryLegend.SetFillColor(0)
   SummaryBackground = 0
   def __init__(self, SignalFile, BackgroundFile, Label, ValidationProducer):
      self.Label = Label
      self.SignalFile = TFile.Open(SignalFile)
      self.BackgroundFile = TFile.Open(BackgroundFile)
      self.ValidationProducer = ValidationProducer
      self.InfoDictionary      = { 'producer' : self.ValidationProducer.TauProducer.value(),
                                   'label'    : self.ValidationProducer.ExtensionName.value(),
                                   'var'      : KinematicVar }

      # Store each discriminator as a point.  Initally, retrieve the names. Each discriminator will eventual be a tuple holding the effs/fake rates etc
      self.DiscriminatorPoints = [ {'name' : DiscInfo.discriminator.value().replace(self.InfoDictionary['producer'].replace('Producer', ''), '').replace('Discrimination', ''), 
                                    'loc'  : self.BuildFullPathForDiscriminator(DiscInfo.discriminator.value()) } for DiscInfo in ValidationProducer.discriminators.value() ]
      if HideMuonAndElectronRej:
         NewPoints = []
         for aPoint in self.DiscriminatorPoints:
            if aPoint['name'].find("Electron") == -1 and aPoint['name'].find("Muon") == -1:
               NewPoints.append(aPoint)
         self.DiscriminatorPoints = NewPoints
               
      # Add denominator info
      self.Denominator         =  {'loc'  : "DQMData/RecoTauV/%(producer)s%(label)s_ReferenceCollection/nRef_Taus_vs_%(var)sTauVisible" % self.InfoDictionary } 

   def BuildFullPathForDiscriminator(self, DiscName):
      self.InfoDictionary['disc'] = DiscName
      output = "DQMData/RecoTauV/%(producer)s%(label)s_%(disc)s/%(disc)s_vs_%(var)sTauVisible" % self.InfoDictionary
      del self.InfoDictionary['disc']
      return output
   def RebinHistogram(self, Histo, MinBinValue, MaxBinValue):
      """ Rebin a range of an input histogram into a new histogram w/ 1 bin. """
      OutputHisto = TH1F("temp_%i" % TauValidationInfo.StupidRootStupidNameCounter, "temp", 1, MinBinValue, MaxBinValue)
      TauValidationInfo.StupidRootStupidNameCounter += 1
      MinBinNumber = Histo.FindBin(MinBinValue)
      MaxBinNumber = Histo.FindBin(MaxBinValue)
      #print "Integrating histogram between values (%f, %f)" % (Histo.GetBinLowEdge(MinBinNumber), Histo.GetBinLowEdge(MaxBinNumber)+Histo.GetBinWidth(MaxBinNumber))
      OutputHisto.SetBinContent(1,Histo.Integral(MinBinNumber, MaxBinNumber))
      return OutputHisto
   def LoadHistograms(self):
      # Load original histograms, and rebin them if necessary
      # Get denominator
      print "Loading histograms for %s" % self.Label
      SignalDenominatorHisto                   = self.SignalFile.Get(self.Denominator['loc'])
      self.Denominator['SignalHisto']          = SignalDenominatorHisto
      self.Denominator['SignalHistoRebin']     = self.RebinHistogram(SignalDenominatorHisto, KinematicVarMin, KinematicVarMax)
      BackgroundDenominatorHisto               = self.BackgroundFile.Get(self.Denominator['loc'])
      self.Denominator['BackgroundHisto']      = BackgroundDenominatorHisto
      self.Denominator['BackgroundHistoRebin'] = self.RebinHistogram(BackgroundDenominatorHisto, KinematicVarMin, KinematicVarMax)
      # Get numerators
      for aPoint in self.DiscriminatorPoints:
         SignalPointHisto               = self.SignalFile.Get(aPoint['loc'])
         aPoint['SignalHisto']          = SignalPointHisto
         aPoint['SignalHistoRebin']     = self.RebinHistogram(SignalPointHisto, KinematicVarMin, KinematicVarMax)
         BackgroundPointHisto           = self.BackgroundFile.Get(aPoint['loc'])
         aPoint['BackgroundHisto']      = BackgroundPointHisto
         aPoint['BackgroundHistoRebin'] = self.RebinHistogram(BackgroundPointHisto, KinematicVarMin, KinematicVarMax)
   def ComputeEfficiencies(self):
      print "Computing efficiencies for %s" % self.Label
      print "-----------------------------------------------------"
      print "%-40s %10s %10s" % ("Discriminator", "SignalEff", "FakeRate") 
      print "-----------------------------------------------------"
      # Get denominators
      SignalDenominatorHisto     = self.Denominator['SignalHistoRebin']
      BackgroundDenominatorHisto = self.Denominator['BackgroundHistoRebin']
      for aPoint in self.DiscriminatorPoints:
         SignalPointHisto = aPoint['SignalHistoRebin']
         BackgroundPointHisto = aPoint['BackgroundHistoRebin']
         # compute length 1 TGraphAsymmErrors, to compute the errors correctly
         TempSignalEffPoint = TGraphAsymmErrors(SignalPointHisto, SignalDenominatorHisto)
         TempBackgroundEffPoint = TGraphAsymmErrors(BackgroundPointHisto, BackgroundDenominatorHisto)
         # Create a new TGraphAsymmErrors, where the x and y coordinates are the Signal/Background efficiencies, respectively
         PerformancePoint = TGraphAsymmErrors(1) #only one point
         xValueSignal = Double(0) #stupid root pass by reference crap
         yValueSignal = Double(0) 
         xValueBackground = Double(0)
         yValueBackground = Double(0) 
         TempSignalEffPoint.GetPoint(0, xValueSignal, yValueSignal)
         TempBackgroundEffPoint.GetPoint(0, xValueBackground, yValueBackground)
         aPoint['SignalEff']     = yValueSignal
         aPoint['BackgroundEff'] = yValueBackground
         print "%-40s %10.3f %10.4f" % (aPoint['name'], aPoint['SignalEff'], aPoint['BackgroundEff'])
         PerformancePoint.SetPoint(0, yValueSignal, yValueBackground)
         PerformancePoint.SetPointError(0, 
                                        TempSignalEffPoint.GetErrorYlow(0),             #ex low
                                        TempSignalEffPoint.GetErrorYhigh(0),            #ex high
                                        TempBackgroundEffPoint.GetErrorYlow(0),         #ey low
                                        TempBackgroundEffPoint.GetErrorYhigh(0) )       #ey high
         PerformancePoint.SetMarkerSize(2)
         try:
            PerformancePoint.SetMarkerStyle(TauValidationInfo.DiscriminatorToMarkerStyleDict[aPoint['name']])
         except KeyError:
            PerformancePoint.SetMarkerStyle(TauValidationInfo.StaticMarkerStyleIterator)
            TauValidationInfo.DiscriminatorToMarkerStyleDict[aPoint['name']] = TauValidationInfo.StaticMarkerStyleIterator
            self.DiscriminatorLegend.AddEntry( PerformancePoint, aPoint['name'],"P")
            TauValidationInfo.StaticMarkerStyleIterator += 1
         aPoint['PerformancePoint'] = PerformancePoint
   def BuildTGraphSummary(self):
      print "Building summary"
      self.SummaryTGraph = TGraph(len(self.DiscriminatorPoints))
      for index, aPoint in enumerate(self.DiscriminatorPoints):
         self.SummaryTGraph.SetPoint(index, aPoint['SignalEff'], aPoint['BackgroundEff'])
      self.SummaryTGraph.SetLineColor(TauValidationInfo.StaticSummaryLineColorIterator)
      self.SummaryTGraph.SetLineWidth(2)
      self.SummaryLegend.AddEntry( self.SummaryTGraph, self.Label,"L")
      TauValidationInfo.StaticSummaryLineColorIterator += 1


def PlotPerformanceCurves(ReleasesToBuildFrom, OutputFile):
   # Setup
   myCanvas = TCanvas("Validation", "Validation", 800, 800)
   # Cut label on the plots
   CutLabel = TPaveText(0.6, 0.1, 0.9, 0.27, "brNDC")
   CutLabel.AddText("%0.1f%s < %s < %0.1f%s" % (KinematicVarMin, VarUnit, LatexVar, KinematicVarMax, VarUnit))
   CutLabel.SetFillStyle(0)
   CutLabel.SetBorderSize(0)

   # Build the TGraphs of sigEff versus bkgFakeRate for each release
   for aRelease in ReleasesToBuildFrom:
      aRelease.LoadHistograms()
      aRelease.ComputeEfficiencies()
      aRelease.BuildTGraphSummary()

   CurrentHistogram = 0

   for aRelease in ReleasesToBuildFrom:
      if CurrentHistogram == 0:
         aRelease.SummaryTGraph.Draw("ALP")
         CurrentHistogram = aRelease.SummaryTGraph.GetHistogram()
      else:
         aRelease.SummaryTGraph.Draw("LP")


   CurrentHistogram.SetAxisRange(0, 1)
   CurrentHistogram.GetYaxis().SetRangeUser(0.0001,1)
   CurrentHistogram.GetXaxis().SetTitle("Efficiency")
   CurrentHistogram.GetYaxis().SetTitle("Fake Rate")
   CurrentHistogram.SetTitle("Performance Points")

   gPad.SetLogy(True)

   for aRelease in ReleasesToBuildFrom:
      for aPoint in aRelease.DiscriminatorPoints:
         aPoint['PerformancePoint'].Draw("P")

   TauValidationInfo.SummaryLegend.Draw()
   TauValidationInfo.DiscriminatorLegend.Draw()
   CutLabel.Draw()

   myCanvas.SaveAs(OutputFile)
