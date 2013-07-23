from cuts import *

from ROOT import TFile,TStyle,TKey,TTree,TH1F,TH2D
from ROOT import TMath,TCanvas,TCut,TEfficiency
from ROOT import gStyle,gROOT,gPad,gDirectory
from ROOT import kBlue

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT
ROOT.gROOT.SetBatch(1)


def plotDphiValues():
    
    inputFiles=['../../../gem_csc_delta_pt5_pad4.root',
                '../../../gem_csc_delta_pt10_pad4.root',  
                '../../../gem_csc_delta_pt20_pad4.root',
                '../../../gem_csc_delta_pt40_pad4.root',
                '../../../gem_csc_delta_pt15_pad4.root',
                '../../../gem_csc_delta_pt30_pad4.root']

    analyzer = "GEMCSCAnalyzer"
    ## Trees
    trk_delta = "trk_delta"
    trk_eff = "trk_eff"

    c = TCanvas("c","c",600,600)
    gStyle.SetStatStyle(0)
    gStyle.SetOptStat(1110)
            

    for inputFile in inputFiles:
    
        file = TFile.Open(inputFile)
        if not file:
            sys.exit('Input ROOT file %s is missing.' %(inputFile))

        dirAna = file.Get(analyzer)
        if not dirAna:
            sys.exit('Directory %s does not exist.' %(dirAna))

        treeDelta = dirAna.Get(trk_eff)
        if not treeDelta:
            sys.exit('Tree %s does not exist.' %(treeDelta))

        treeDelta.Draw("TMath::Abs(dphi_pad_odd)>>h_name", TCut("%s && %s" %(ok_pad1.GetTitle(), ok_lct1.GetTitle())));
        h = TH1F(gDirectory.Get("h_name").Clone("h_name"))
        if not h:
            sys.exit('h does not exist')
        h.SetTitle("Bending angle")
        h.SetLineWidth(2)
        h.SetLineColor(kBlue)
        h.SetMinimum(0.)
        h.Draw("same")
    c.SaveAs("test.png")
##         print h.GetMean()
##         print h.GetRMS()
##         plot = TH1F(gDirectory.Get("num_" + h_name))
##           if not num:
##                   sys.exit('num does not exist')
##                     num = TH1F(num.Clone("eff_" + h_name))

