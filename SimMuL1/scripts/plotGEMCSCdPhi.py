##
## This script overlays the histograms of bending angles for even and odd chambers
##

from cuts import *
from tdrStyle import *

## ROOT modules
from ROOT import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT
ROOT.gROOT.SetBatch(1)

def getTree(fileName):
    """Get tree for given filename"""

    analyzer = "GEMCSCAnalyzer"
    trk_eff = "trk_eff"

    file = TFile.Open(fileName)
    if not file:
        sys.exit('Input ROOT file %s is missing.' %(fileName))

    dir = file.Get(analyzer)
    if not dir:
        sys.exit('Directory %s does not exist.' %(dir))
        
    tree = dir.Get(trk_eff)
    if not tree:
        sys.exit('Tree %s does not exist.' %(tree))

    return tree

def plotGEMCSCdPhi(filesDir, plotDir, oddEven = "even", ext = ".png", useReverseOrdering = False):
    """Plot the GEM-CSC bending angles"""
    
    t = getTree("%sgem_csc_delta_pt5_pad4.root"%(filesDir))
    t1 = getTree("%sgem_csc_delta_pt20_pad4.root"%(filesDir))
    
    dphi_pt5 = TH1F("dphi_pt5","",600,0.0,0.03)
    dphi_pt20 = TH1F("dphi_pt20","",600,0.0,0.03)
    
    c = TCanvas("c","c",700,450)
    c.Clear()
    ##    c.SetGridx(1)
    ##    c.SetGridy(1)

    gStyle.SetTitleStyle(0)
    gStyle.SetTitleAlign(13) ##// coord in top left
    gStyle.SetTitleX(0.)
    gStyle.SetTitleY(1.)
    gStyle.SetTitleW(1)
    gStyle.SetTitleH(0.058)
    gStyle.SetTitleBorderSize(0)
    
    gStyle.SetPadLeftMargin(0.126)
    gStyle.SetPadRightMargin(0.04)
    gStyle.SetPadTopMargin(0.06)
    gStyle.SetPadBottomMargin(0.13)
    gStyle.SetOptStat(0)
    gStyle.SetMarkerStyle(1)

    #setTDRStyle()
    
    if oddEven == "even":
        ok_pad_lct = ok_pad2_lct2       
        var = "dphi_pad_even"
        if useReverseOrdering:
            closeFar = 'Even ("close")'
        else:
            closeFar = '"Close" (even)'
    else:
        ok_pad_lct = ok_pad1_lct1
        var = "dphi_pad_odd"
        if useReverseOrdering:
            closeFar = 'Odd ("far")'
        else:
            closeFar = '"Far" (odd)'
            
    t.Draw("TMath::Abs(%s)>>dphi_pt5"%(var) , ok_pad_lct)
    t1.Draw("TMath::Abs(%s)>>dphi_pt20"%(var) , ok_pad_lct)
    
    dphi_pt5.Scale(1/dphi_pt5.Integral())
    dphi_pt20.Scale(1/dphi_pt20.Integral())
    dphi_pt5.SetLineColor(kRed)
    dphi_pt20.SetLineColor(kBlue)
    dphi_pt5.SetLineWidth(2)
    dphi_pt20.SetLineWidth(2)
    dphi_pt20.GetXaxis().SetTitle("#Delta#phi(GEM,CSC) [rad]")
    dphi_pt20.GetYaxis().SetTitle("Arbitray units")
    dphi_pt20.SetTitle("           GEM-CSC Bending Angle                CMS Phase-2 Simulation Preliminary")
    dphi_pt20.GetXaxis().SetLabelSize(0.05)
    dphi_pt20.GetYaxis().SetLabelSize(0.05)
    dphi_pt20.Draw()
    dphi_pt5.Draw("same")

    legend = TLegend(.4,.45,.7,.6)
    legend.SetFillColor(kWhite)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.06)
    legend.SetMargin(0.13)
    ##    legend.AddEntry(0,"1.64<|#eta|<2.14:","")
    legend.AddEntry(dphi_pt5,"muon p_{T} = 5 GeV/c","L")
    legend.AddEntry(dphi_pt20,"muon p_{T} = 20 GeV/c","L")
    legend.Draw("same") 

    ## Adding additional information - top right
    """
    tex2 = TLatex(.73,.85,'   L1 Trigger')
    tex2.SetTextSize(0.06)
    tex2.SetNDC()
    tex2.Draw("same")
    """

    tex3 = TLatex(.72,.85,'1.64<|#eta|<2.14')
    tex3.SetTextSize(0.06)
    tex3.SetNDC()
    tex3.Draw("same")

    ## hardcore nitpicking over here!
    if oddEven == "even":
        xpos = 0.2##0.61
    else:
        xpos = 0.25##0.64

    tex = TLatex(xpos,.85,"%s chamber pairs"%(closeFar))
    tex.Draw("same")
    tex.SetTextSize(0.06)
    tex.SetNDC()

    if useReverseOrdering:
        c.SaveAs("%sGEMCSCdPhi_%s_chambers_reverse%s"%(plotDir, oddEven, ext))
    else:
        c.SaveAs("%sGEMCSCdPhi_%s_chambers%s"%(plotDir, oddEven, ext))

def bendingAngleParametrization(filesDir, plotDir, oddEven = "even", ext = ".png"):
    """Bending angle parametrization vs eta"""
    
    pt_values = [5,10,15,20,30,40]
    maxi = [0.03,0.015,0.01,0.007,0.005,0.005]

    for i in range(len(pt_values)):
        t1 = getTree("%sgem_csc_delta_pt%d_pad4.root"%(filesDir,pt_values[i]))

        if oddEven == "even":
            ok_pad_lct = ok_pad2_lct2       
            var = "dphi_pad_even"
        else:
            ok_pad_lct = ok_pad1_lct1
            var = "dphi_pad_odd"
            
        c = TCanvas("c","c",700,450)
        c.Clear()
        ##    c.SetGridx(1)
        ##    c.SetGridy(1)

        gStyle.SetTitleStyle(0)
        gStyle.SetTitleAlign(13) ##// coord in top left
        gStyle.SetTitleX(0.)
        gStyle.SetTitleY(1.)
        gStyle.SetTitleW(1)
        gStyle.SetTitleH(0.058)
        gStyle.SetTitleBorderSize(0)
        
        gStyle.SetPadLeftMargin(0.126)
        gStyle.SetPadRightMargin(0.04)
        gStyle.SetPadTopMargin(0.06)
        gStyle.SetPadBottomMargin(0.13)
        gStyle.SetOptStat(0)
        gStyle.SetMarkerStyle(1)

        dphi_pt = TH2F("dphi_pt",";#eta;#Delta#phi_{(GEM,CSC)} [rad]",6,1.6,2.14,100,0.,0.03)
        t1.Draw("TMath::Abs(%s):TMath::Abs(eta)>>dphi_pt_bin"%(var) , ok_pad_lct)
        dphi_pt_bin = TH1F("dphi_pt",";#Delta#phi_{(GEM,CSC)} [rad]",100,0.,0.03)
        dphi_pt_bin = dphi_pt.ProjectionY("",0,1)
        
        
        #dphi_pt.GetYaxis().SetRangeUser(0,maxi[i])
        #c.SaveAs("%sbendingAnglePar_%d_%d_%s%s"%(plotDir, pt_values[i], j, oddEven, ext))
        c.SaveAs("%sbendingAnglePar_%d_%s%s"%(plotDir, pt_values[i],oddEven, ext))

        """
        ## for each eta bin, construct a 1D histogram and redo analysis
        for j in range(0,7):
        dphi_pt_bin = TH1F("dphi_pt",";#Delta#phi_{(GEM,CSC)} [rad]",100,0.,0.03)
        dphi_pt_bin = dphi_pt.ProjectionY("",j,j+1)
        t1.Draw("TMath::Abs(%s):TMath::Abs(eta)>>dphi_pt_bin"%(var) , ok_pad_lct)
        #dphi_pt.GetYaxis().SetRangeUser(0,maxi[i])
        c.SaveAs("%sbendingAnglePar_%d_%d_%s%s"%(plotDir, pt_values[i], j, oddEven, ext))
        """


if __name__ == "__main__":  
    input_dir = "files/"
    output_dir = "plots_cmssw_601_postls1/bending/"

    plotGEMCSCdPhi(input_dir, output_dir, "even", ".eps", False)
    plotGEMCSCdPhi(input_dir, output_dir, "odd",  ".eps", False)
    plotGEMCSCdPhi(input_dir, output_dir, "even", ".pdf", False)
    plotGEMCSCdPhi(input_dir, output_dir, "odd",  ".pdf", False)
    plotGEMCSCdPhi(input_dir, output_dir, "even", ".png", False)
    plotGEMCSCdPhi(input_dir, output_dir, "odd",  ".png", False)

    plotGEMCSCdPhi(input_dir, output_dir, "even", ".eps", True)
    plotGEMCSCdPhi(input_dir, output_dir, "odd",  ".eps", True)
    plotGEMCSCdPhi(input_dir, output_dir, "even", ".pdf", True)
    plotGEMCSCdPhi(input_dir, output_dir, "odd",  ".pdf", True)
    plotGEMCSCdPhi(input_dir, output_dir, "even", ".png", True)
    plotGEMCSCdPhi(input_dir, output_dir, "odd",  ".png", True)

    bendingAngleParametrization(input_dir, output_dir, "odd",  ".png")
