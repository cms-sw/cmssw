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

def plotGEMCSCdPhi(filesDir, plotDir, oddEven = "even", ext = ".png"):
    """Plot the GEM-CSC bending angles"""
    
    t = getTree("%sgem_csc_delta_pt5_pad4.root"%(filesDir));
    t1 = getTree("%sgem_csc_delta_pt20_pad4.root"%(filesDir));
    
    dphi_pt5 = TH1F("dphi_pt5","",600,0.0,0.03);
    dphi_pt20 = TH1F("dphi_pt20","",600,0.0,0.03);
    
    c = TCanvas("cDphi","cDphi",700,450);
    c.Clear()
    c.SetGridx(1)
    c.SetGridy(1)

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
        closeFar = "Close"
    else:
        ok_pad_lct = ok_pad1_lct1
        var = "dphi_pad_odd"
        closeFar = "Far"
        
    t.Draw("TMath::Abs(%s)>>dphi_pt5"%(var) , ok_pad_lct);
    t1.Draw("TMath::Abs(%s)>>dphi_pt20"%(var) , ok_pad_lct);
    
    dphi_pt5.Scale(1/dphi_pt5.Integral());
    dphi_pt20.Scale(1/dphi_pt20.Integral());
    
    dphi_pt5.SetLineColor(kRed);
    dphi_pt20.SetLineColor(kBlue);
    dphi_pt5.SetLineWidth(2);
    dphi_pt20.SetLineWidth(2);

    dphi_pt20.GetXaxis().SetTitle("#Delta#Phi(GEM,CSC) [rad]");
    dphi_pt20.GetYaxis().SetTitle("Arbitrary units");
    dphi_pt20.SetTitle("CMS Simulation: GEM-CSC bending angle");

    dphi_pt20.Draw();
    dphi_pt5.Draw("same");

    legend = TLegend(.4,.5,.7,.7);
    legend.SetFillColor(kWhite);
    legend.SetFillStyle(0);
    legend.SetBorderSize(0);
    legend.SetTextSize(0.06);
    legend.SetMargin(0.13);
    legend.AddEntry(dphi_pt5,"muon p_{T} = 5 GeV/c","L");
    legend.AddEntry(dphi_pt20,"muon p_{T} = 20 GeV/c","L");
    legend.Draw("same");

    tex = TLatex(.65,.85,"%s chambers"%(closeFar))
    tex.SetTextSize(0.06)
    tex.SetNDC()
    tex.Draw("same")

    tex2 = TLatex(.65,.75,"1.64<|#eta|<2.14")
    tex2.SetTextSize(0.06)
    tex2.SetNDC()
    tex2.Draw("same")

    c.SaveAs("%sGEMCSCdPhi_%s_chambers%s"%(plotDir, oddEven, ext))

if __name__ == "__main__":  
    plotGEMCSCdPhi("files/", "plots/bending/", "even", ".png")
    plotGEMCSCdPhi("files/", "plots/bending/", "odd",  ".png")
    plotGEMCSCdPhi("files/", "plots/bending/", "even", ".pdf")
    plotGEMCSCdPhi("files/", "plots/bending/", "odd",  ".pdf")
    plotGEMCSCdPhi("files/", "plots/bending/", "even", ".eps")
    plotGEMCSCdPhi("files/", "plots/bending/", "odd",  ".eps")


