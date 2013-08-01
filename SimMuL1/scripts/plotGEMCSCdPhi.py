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

def plotGEMCSCdPhi(oddEven = "even", ext = ".pdf"):
    """Plot the GEM-CSC bending angles"""
    
    txtHeader = TLegend(.13,.935,.97,1.);
    txtHeader.SetFillColor(kWhite);
    txtHeader.SetFillStyle(0);
    txtHeader.SetBorderSize(0);
    txtHeader.SetTextFont(42);
    txtHeader.SetTextSize(0.045);
    txtHeader.SetTextAlign(22);
    txtHeader.SetHeader("|#phi(CSC strip)-#phi(GEM Pad4)| in %s numbered chambers"%(oddEven));
    
    legend = TLegend(.4,.60,1.2,.92);
    legend.SetFillColor(kWhite);
    legend.SetFillStyle(0);
    legend.SetBorderSize(0);
    legend.SetTextSize(0.045);
    legend.SetMargin(0.13);
    
    
    t = getTree("files/gem_csc_delta_pt5_pad4.root");
    t1 = getTree("files/gem_csc_delta_pt20_pad4.root");
    
    dphi_pt5 = TH1F("dphi_pt5","",600,0.0,0.03);
    dphi_pt20 = TH1F("dphi_pt20","",600,0.0,0.03);
    
    c = TCanvas("cDphi","cDphi",700,450);
    c.Clear()

    if oddEven == "even":
        ok_pad_lct = ok_pad2_lct2       
        var = "dphi_pad_even"
    else:
        ok_pad_lct = ok_pad1_lct1
        var = "dphi_pad_odd"

    t.Draw("TMath::Abs(%s)>>dphi_pt5"%(var) , ok_pad_lct);
    t1.Draw("TMath::Abs(%s)>>dphi_pt20"%(var) , ok_pad_lct);
    
    dphi_pt5.Scale(1/dphi_pt5.Integral());
    dphi_pt20.Scale(1/dphi_pt20.Integral());
    
    dphi_pt5.SetLineColor(kRed);
    dphi_pt20.SetLineColor(kBlue);
    dphi_pt5.SetLineWidth(2);
    dphi_pt20.SetLineWidth(2);

    dphi_pt20.GetXaxis().SetTitle("|#phi(CSC half-strip) - #phi(GEM pad)| [rad]");
    dphi_pt20.GetYaxis().SetTitle("A.U.");

    legend.AddEntry(dphi_pt5,"Muons with p_{T}=5 GeV","L");
    legend.AddEntry(dphi_pt20,"Muons with p_{T}=20 GeV","L");
    
    dphi_pt20.Draw();
    dphi_pt5.Draw("same");
    txtHeader.Draw("same");
    legend.Draw("same");
    c.SaveAs("plots/GEMCSCdPhi_%s_chambers%s"%(oddEven,ext));

if __name__ == "__main__":  
    plotGEMCSCdPhi("even")
    plotGEMCSCdPhi("odd")
