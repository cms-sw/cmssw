## custom modules
from effFunctions import *
from cuts import *
from tdrStyle import *
from GEMCSCdPhiDict import *

## ROOT modules
from ROOT import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT
ROOT.gROOT.SetBatch(1)

gStyle.SetStatW(0.07)
gStyle.SetStatH(0.06)

gStyle.SetOptStat(0)

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

gStyle.SetMarkerStyle(1)

#_______________________________________________________________________________
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


#_______________________________________________________________________________
def getDphi(eff,pt,evenOdd):
    """Return the delta Phi cut value"""

    return dphi_lct_pad["%s"%(eff)]["%s"%(pt)]["%s"%(evenOdd)]


#_______________________________________________________________________________
def gemTurnOn(filesDir, plotDir, eff, oddEven, ext):
    """Produce GEM turn-on curve"""
    
    pt = ["pt10","pt20","pt30","pt40"]    
    pt_labels = ["10","20","30","40"]
#    pt_labels = ["10 GeV/c","20 GeV/c","30 GeV/c","40 GeV/c"]
    dphis = [0.,0.,0.,0.]

    marker_colors = [kRed, kViolet+1, kAzure+2, kGreen-2]
    marker_styles = [20,21,23,22]

    t = getTree("%sgem_csc_eff_pt2pt50_pad4.root"%(filesDir));

    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","          GEM-CSC bending Angle                       CMS Simulation Preliminary;Generated muon p_{T} [GeV/c];",50,0.,50.)
    superscript = "p_{T}>p_{T}^{min}"
    subscript = "0"
    h.GetYaxis().SetTitle("    |#Delta#phi_{{}^{(GEM,CSC)}}|<|#Delta#phi_{0}^{WP}| Cut Efficiency");
##    h.GetYaxis().SetTitle("    |#Delta#phi_{{}^{(GEM,CSC)}}|<|#Delta#phi_{%s}^{%s}| Cut Efficiency"%(subscript,superscript));
    h.GetYaxis().SetTitleOffset(.9)
    h.SetStats(0)

    ## label sizes more leglible
    h.GetXaxis().SetLabelSize(.05)
    h.GetXaxis().SetLabelOffset(.005)
    h.GetYaxis().SetLabelSize(.05)
    h.GetYaxis().SetLabelOffset(.005)

    h.Draw("")
    histoList = []
    for i in range(len(pt)):
        dphi = getDphi("%s"%(eff),"%s"%(pt[i]),"%s"%(oddEven))
        dphis[i] = dphi
        if oddEven=="even":
            ok_dphi = TCut("TMath::Abs(dphi_pad_even) < %f"%(dphi))
            denom_cut = ok_pad2_lct2_eta
            closeFar = "Close"
        else:
            ok_dphi = TCut("TMath::Abs(dphi_pad_odd) < %f"%(dphi))
            denom_cut = ok_pad1_lct1_eta
            closeFar = "Far"

        h2 = draw_eff(t, "", "h2", "(50,0.,50.)", "pt", 
                         denom_cut, ok_dphi, marker_colors[i], marker_styles[i])
        histoList.append(h2)
        h2.SetMarkerSize(1)
        h2.Draw("same")
    
    ## add legend
    leg = TLegend(0.52,0.17,.93,0.57, "High efficiency patterns:", "brNDC")
    for n in range(len(pt)):
#        superscript = "p_{T}>%s"%(pt_labels[n])
        superscript = "\"%s\""%(pt_labels[n])
        subscript = "0"
        #leg.AddEntry(histoList[n], "#Delta#phi_{%s}^{%s} = %.1f mrad"%(subscript,superscript,dphis[n]*1000), "p")
        leg.AddEntry(histoList[n], "WP = %s"%(pt_labels[n]), "p")


    leg.SetBorderSize(0)
    leg.SetMargin(0.1)
    leg.SetFillStyle(0)
    leg.SetFillStyle(1001)
    leg.SetFillColor(kWhite)
    leg.SetTextSize(0.05)
    leg.Draw("same")

    ## Adding additional information - top right
    """
    tex2 = TLatex(.75,.82,'   L1 Trigger')
    tex2.SetTextSize(0.05)
    tex2.SetNDC()
    tex2.Draw("same")
    """

    tex3 = TLatex(.735,.82,'1.64<|#eta|<2.14')
    tex3.SetTextSize(0.05)
    tex3.SetNDC()
    tex3.Draw("same")

    ## hardcore nitpicking over here!
    if closeFar == "Close":
        xpos = 0.57
    else:
        xpos = 0.611

    tex = TLatex(xpos,.75,'"%s" chamber pairs'%(closeFar))
    tex.Draw("same")
    tex.SetTextSize(0.05)
    tex.SetNDC()

    ## save the file
    c.SaveAs("%sGEM_turnon_%s_%s%s"%(plotDir, eff,oddEven,ext))


#_______________________________________________________________________________
def efficiency_1(f_name, p_name, pt, overlap):
    """efficiency vs half-strip  - separate odd-even""" 
 
    """
    gStyle->SetTitleStyle(0);
    gStyle->SetTitleAlign(13); // coord in top left
    gStyle->SetTitleX(0.);
    gStyle->SetTitleY(1.);
    gStyle->SetTitleW(1);
    gStyle->SetTitleH(0.058);
    gStyle->SetTitleBorderSize(0);
    
    gStyle->SetPadLeftMargin(0.126);
    gStyle->SetPadRightMargin(0.04);
    gStyle->SetPadTopMargin(0.06);
    gStyle->SetPadBottomMargin(0.13);
    gStyle->SetOptStat(0);
    gStyle->SetMarkerStyle(1);
    """

    ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12"
    if (overlap):
        cut1 = ok_pad1_overlap
        cut2 = ok_pad2_overlap
    else:
      cut1 = ok_pad1
      cut2 = ok_pad2

    t = getTree(f_name)
    ho = draw_eff(t, "         GEM reconstruction efficiency               CMS Simulation Preliminary;LCT half-strip number;Efficiency", 
                  "h_odd", "(130,0.5,130.5)", "hs_lct_odd", TCut("%s&&%s"(ok_lct1.GetTitle), ok_eta.GetTitle()), cut1, "", kRed)
    he = draw_eff(t, "         GEM reconstruction efficiency               CMS Simulation Preliminary;LCT half-strip number;Efficiency", 
                  "h_evn", "(130,0.5,130.5)", "hs_lct_even", TCut("%s&&%s"(ok_lct1.GetTitle), ok_eta.GetTitle()), cut2, "same")
    ho.SetMinimum(0.)
    ho.GetXaxis().SetLabelSize(0.05)
    ho.GetYaxis().SetLabelSize(0.05)
    
    leg = TLegend(0.25,0.23,.75,0.5, "", "brNDC");
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.06)
    leg.AddEntry(0,"muon p_{T} = %s GeV/c"%(pt),"")
    leg.AddEntry(he, "\"Close\" chamber pairs","l")
    leg.AddEntry(ho, "\"Far\" chamber pairs","l")
    leg.Draw();
    
    tex2 = TLatex(.67,.8,"   L1 Trigger")
    tex2.SetTextSize(0.05)
    tex2.SetNDC()
    tex2.Draw()
    
    tex = TLatex(.66,.73,"1.64<|#eta|<2.12")
    tex.SetTextSize(0.05)
    tex.SetNDC()
    tex.Draw()
    
## this has to be fixed
    gPad.Print(p_name)


"""
void efficiency_2(TString f_name, TString p_name, TString pt, bool overlap)
{
  // efficiency vs half-strip  - including overlaps in odd&even
  TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12";
  TCut cut1;
  TCut cut2;
  if (overlap)
  {
    cut1 = ok_pad1_overlap;
    cut2 = ok_pad2_overlap;
  }
  else
  {
    cut1 = ok_pad1;
    cut2 = ok_pad2;
  }
  
  TTree *t = getTree(f_name);
  TH1F* ho = draw_eff(t, "         GEM reconstruction efficiency               CMS Simulation Preliminary;local #phi [deg];Efficiency", "h_odd", "(130,-5,5)", "fmod(180*phi/TMath::Pi(),5)", ok_lct1 && ok_eta , cut1, "", kRed);
  TH1F* he = draw_eff(t, "         GEM reconstruction efficiency               CMS Simulation Preliminary;local #phi [deg];Efficiency", "h_evn", "(130,-5,5)", "fmod(180*phi/TMath::Pi(),5)", ok_lct2 && ok_eta , cut2, "same");
  ho->SetMinimum(0.);
  ho->GetXaxis()->SetLabelSize(0.05);
  ho->GetYaxis()->SetLabelSize(0.05);

  TLegend *leg = new TLegend(0.25,0.23,.75,0.5, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.06);
  leg->AddEntry((TObject*)0,"muon p_{T} = " + pt + " GeV/c",""); 
  leg->AddEntry(he, "\"Close\" chamber pairs","l");
  leg->AddEntry(ho, "\"Far\" chamber pairs","l");
  leg->Draw();

  // Print additional information
  TLatex* tex2 = new TLatex(.67,.8,"   L1 Trigger");
  tex2->SetTextSize(0.05);
  tex2->SetNDC();
  tex2->Draw();
    
  TLatex *  tex = new TLatex(.66,.73,"1.64<|#eta|<2.12");
  tex->SetTextSize(0.05);
  tex->SetNDC();
  tex->Draw();

  gPad->Print(p_name);
}
"""











































def efficiency_1(f_name, p_name, overlap):
    """Efficiency to match LCT to GEM pad vs half-strip for even and odd numbered chambers. Option to include chamber overlaps"""
   
    t = getTree(f_name)
    if overlap:
        cut1 = ok_pad1_overlap
        cut2 = ok_pad2_overlap
    else:
        cut1 = ok_pad1
        cut2 = ok_pad2

    ho = draw_eff(t, "Efficiency for track with LCT to have GEM pad in chamber;LCT half-strip;Efficiency",
                  "h_odd", "(130,0.5,130.5)", "hs_lct_odd", TCut("%s && %s" %(ok_lct1.GetTitle(),ok_eta.GetTitle())), cut1, kRed, 5)
    he = draw_eff(t, "Efficiency for track with LCT to have GEM pad in chamber;LCT half-strip;Efficiency",
                  "h_evn", "(130,0.5,130.5)", "hs_lct_even", TCut("%s && %s" %(ok_lct2.GetTitle(),ok_eta.GetTitle())), cut2, kBlue, 5)

    c = TCanvas("c","c",700,500)
    c.Clear()
    c.SetGridx(1)
    c.SetGridy(1)
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber",130,0.5,130.5)
    h.SetTitle("Efficiency for track with LCT to have GEM pad in chamber")
    h.GetXaxis().SetTitle("LCT half-strip")
    h.GetYaxis().SetTitle("Efficiency")
    h.SetStats(0)
    h.Draw()
    ho.Draw("same")
    he.Draw("same")

    pt = f_name[f_name.find('pt'):]
    pt = pt[2:]
    pt = pt[:pt.find('_pad')]
    leg = TLegend(0.40,0.2,.7,0.5,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(.05)
    leg.AddEntry(0, "p_{T} = %s GeV/c"%(pt), "")
    leg.AddEntry(0, "a pad spans 4 strips", "")
    leg.AddEntry(ho,"odd chambers","l")
    leg.AddEntry(he,"even chambers","l")
    leg.Draw("same")
    c.SaveAs(p_name)

def drawplot_eff_eta(f_name, plotDir, ext = ".pdf"):
    """Efficiency to match LCT to GEM pad vs LCT eta for even and odd numbered chambers. Option to include chamber overlaps"""
    
    c = TCanvas("c","c",700,500)
    c.Clear()
    c.SetGridx(1)
    c.SetGridy(1)

    gt = getTree(f_name)
    pt = f_name[f_name.find('pt'):]
    pt = pt[2:]
    pt = pt[:pt.find('_pad')]

    ho = draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",
                  "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_lct_odd)", ok_lct1, ok_pad1, kRed, 5)
    he = draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",
                  "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta_lct_even)", ok_lct2, ok_pad2, kBlue, 5)
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber",140,1.5,2.2)
    h.SetTitle("Efficiency for track with LCT to have GEM pad in chamber")
    h.GetXaxis().SetTitle("LCT |#eta|")
    h.GetYaxis().SetTitle("Efficiency")
    h.SetStats(0)
    h.Draw()
    ho.Draw("same")
    he.Draw("same")
    leg = TLegend(0.4,0.2,.7,0.5,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(.05)
    leg.AddEntry(0, "p_{T} = %s GeV/c"%(pt), "")
    leg.AddEntry(0, "a pad spans 4 strips", "")
    leg.AddEntry(ho,"odd chambers","l")
    leg.AddEntry(he,"even chambers","l")
    leg.Draw()
    c.SaveAs("%sgem_pad_eff_for_LCT_vsLCTEta_pt%s%s"%(plotDir,pt,ext))

    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Efficiency",
             "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1, kRed, 5)
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Efficiency",
             "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2, kBlue, 5)
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber",140,1.5,2.2)
    h.SetTitle("Efficiency for track with LCT to have GEM pad in chamber")
    h.GetXaxis().SetTitle("SimTrack |#eta|")
    h.GetYaxis().SetTitle("Efficiency")
    h.SetStats(0)
    h.Draw()
    ho.Draw("same")
    he.Draw("same")
    leg = TLegend(0.40,0.2,.7,0.5,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(.05)
    leg.AddEntry(0, "p_{T} = %s GeV/c"%(pt), "")
    leg.AddEntry(0, "a pad spans 4 strips", "")
    leg.AddEntry(ho,"odd chambers","l")
    leg.AddEntry(he,"even chambers","l")
    leg.Draw()
    c.SaveAs("%sgem_pad_eff_for_LCT_vsTrkEta_pt%s%s"%(plotDir,pt,ext))

    """ 
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;z SimTrack |#eta|;Efficiency", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_gemsh_odd)", ok_gsh1, ok_gdg1, "P", kRed)
    h1 = draw_eff(gt, "Efficiency for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_pad1 || ok_pad2, "P", kViolet)
    h2 = draw_eff(gt, "Efficiency for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_2pad1 || ok_2pad2, "P same", kViolet-6)
    TLegend *leg = new TLegend(0.42,0.23,.96,0.4, NULL, "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h1, "at least one pad","l")
    leg.AddEntry(he, "two pads in two GEMs","l")
    leg.Draw()
    cEff.Print("gem_pad_eff_for_Trk_vsTrkEta_pt40%s"%(ext))
    
    draw_eff(gt, "Efficiency for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_gsh1 || ok_gsh2, "P", kViolet)
    draw_eff(gt, "Efficiency for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_g2sh1 || ok_g2sh2 , "P", kOrange)
    draw_eff(gt, "Efficiency for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_copad1 || ok_copad2 , "P same", kRed)
    
    
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",
             "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_lct_odd)", ok_lct1, ok_pad1_overlap, kRed, 5)
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",
             "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta_lct_even)", ok_lct2, ok_pad2_overlap, kBlue, 5)
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber",130,0.5,130.5)
    h.SetTitle("Efficiency for track with LCT to have GEM pad in chamber")
    h.GetXaxis().SetTitle("LCT |#eta|")
    h.GetYaxis().SetTitle("Efficiency")
    h.SetStats(0)
    h.Draw()
    ho.Draw("same")
    he.Draw("same")
    leg = TLegend(0.50,0.23,.9,0.4,"","brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(.04)
    leg.AddEntry(0, "p_{T} = %s GeV/c"%(pt), "")
    leg.AddEntry(0, "a pad spans 4 strips", "")
    leg.AddEntry(ho,"odd chambers","l")
    leg.AddEntry(he,"even chambers","l")
    leg.Draw()
    c.SaveAs("gem_pad_eff_for_LCT_vsLCTEta_pt40_overlap%s"%(ext))
    
    
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Efficiency", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1_overlap, "P", kRed)
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Efficiency", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2_overlap, "P same")
    leg.Draw()
    cEff.Print("gem_pad_eff_for_LCT_vsTrkEta_pt40_overlap%s"%(ext))
    
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;z SimTrack |#eta|;Efficiency", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1 && Ep, ok_pad1_overlap, "P", kRed)
    draw_eff(gt, "Efficiency for track with LCT to have GEM pad in chamber;z SimTrack |#eta|;Efficiency", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2 && Ep, ok_pad2_overlap, "P same")
    """

    
def halfStripEfficiencies(filesDir, plotDir, ext):
    """Plot the halfstrip efficiencies"""
    
    eff_halfStrip("%sgem_csc_delta_pt5_pad4.root"%(filesDir), "%sgem_pad_eff_for_LCT_vs_HS_pt05%s"%(plotDir,ext))
    eff_halfStrip("%sgem_csc_delta_pt10_pad4.root"%(filesDir), "%sgem_pad_eff_for_LCT_vs_HS_pt10%s"%(plotDir,ext))
    eff_halfStrip("%sgem_csc_delta_pt15_pad4.root"%(filesDir), "%sgem_pad_eff_for_LCT_vs_HS_pt15%s"%(plotDir,ext))
    eff_halfStrip("%sgem_csc_delta_pt20_pad4.root"%(filesDir), "%sgem_pad_eff_for_LCT_vs_HS_pt20%s"%(plotDir,ext))
    eff_halfStrip("%sgem_csc_delta_pt30_pad4.root"%(filesDir), "%sgem_pad_eff_for_LCT_vs_HS_pt30%s"%(plotDir,ext))
    eff_halfStrip("%sgem_csc_delta_pt40_pad4.root"%(filesDir), "%sgem_pad_eff_for_LCT_vs_HS_pt40%s"%(plotDir,ext))

    eff_halfStrip_overlap("%sgem_csc_delta_pt5_pad4.root"%(filesDir),  "%sgem_pad_eff_for_LCT_vs_HS_pt05_overlap%s"%(plotDir,ext))
    eff_halfStrip_overlap("%sgem_csc_delta_pt10_pad4.root"%(filesDir), "%sgem_pad_eff_for_LCT_vs_HS_pt10_overlap%s"%(plotDir,ext))
    eff_halfStrip_overlap("%sgem_csc_delta_pt15_pad4.root"%(filesDir), "%sgem_pad_eff_for_LCT_vs_HS_pt15_overlap%s"%(plotDir,ext))
    eff_halfStrip_overlap("%sgem_csc_delta_pt20_pad4.root"%(filesDir), "%sgem_pad_eff_for_LCT_vs_HS_pt20_overlap%s"%(plotDir,ext))
    eff_halfStrip_overlap("%sgem_csc_delta_pt30_pad4.root"%(filesDir), "%sgem_pad_eff_for_LCT_vs_HS_pt30_overlap%s"%(plotDir,ext))
    eff_halfStrip_overlap("%sgem_csc_delta_pt40_pad4.root"%(filesDir), "%sgem_pad_eff_for_LCT_vs_HS_pt40_overlap%s"%(plotDir,ext))


def etaMatchingEfficiencies(filesDir, plotDir, ext = ".pdf"):
    """Plot the simtrack to LCT,Pad matching efficiency vs eta"""

    drawplot_eff_eta("%sgem_csc_delta_pt5_pad4.root"%(filesDir),  plotDir, ext)
    drawplot_eff_eta("%sgem_csc_delta_pt10_pad4.root"%(filesDir), plotDir, ext)
    drawplot_eff_eta("%sgem_csc_delta_pt15_pad4.root"%(filesDir), plotDir, ext)
    drawplot_eff_eta("%sgem_csc_delta_pt20_pad4.root"%(filesDir), plotDir, ext)
    drawplot_eff_eta("%sgem_csc_delta_pt30_pad4.root"%(filesDir), plotDir, ext)
    drawplot_eff_eta("%sgem_csc_delta_pt40_pad4.root"%(filesDir), plotDir, ext)





def eff_hs_1(filesDir, plotDir, f_name, ext):
    """Halfstrip matching efficiency dphi"""
    
    t = getTree("%s%s"%(filesDir, f_name));
#    dphi = getDphi("%s"%(eff),"%s"%(pt[i]),"%s"%(oddEven))
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_gsh1_lct1_eta , ok_pad1, kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_gsh2_lct2_eta, ok_pad2)
    
    c.SaveAs("%stest%s"%(plotDir, ext))


def eff_hs_2(filesDir, plotDir, f_name, ext):
    """Comment to be added here"""

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1_eta, ok_gsh1, kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2_eta, ok_gsh2)
    c.SaveAs("%stest%s"%(plotDir, ext))
    
def eff_hs_3(filesDir, plotDir, f_name, ext):
    """Comment to be added here"""

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1_eta_Qn, ok_pad1_dphi1, kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2_eta_Qn, ok_pad2_dphi2)
    c.SaveAs("%stest%s"%(plotDir, ext))

def eff_hs_4(filesDir, plotDir, f_name, ext):
    """Comment to be added here"""

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,-0.2,0.2)", "fmod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta, ok_lct1, kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,-0.2,0.2)", "fmod(phi+TMath::Pi()/36., TMath::Pi()/18.)",ok_eta, ok_lct2)
    c.SaveAs("%stest%s"%(plotDir, ext))


def eff_hs_5(filesDir, plotDir, f_name, ext):
    """Comment to be added here"""

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(384,0.,384.)", "strip_gemsh_odd", ok_gsh1_eta, ok_pad1, kRed);
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(384,0.,384.)", "strip_gemsh_even", ok_gsh2_eta, ok_pad2);
    c.SaveAs("%stest%s"%(plotDir, ext))


def eff_hs_6(filesDir, plotDir, f_name, ext):
    """Comment to be added here"""

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(384,0.,384.)", "strip_gemsh_odd", ok_gsh1_eta, ok_gdg1, kRed);
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(384,0.,384.)", "strip_gemsh_even", ok_gsh2_eta, ok_gdg2);
    c.SaveAs("%stest%s"%(plotDir, ext))

def eff_hs_7(filesDir, plotDir, f_name, ext):
    """Comment to be added here"""

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with GEM digi to have GEM pad in chamber;digi strip;Eff.", "h_odd", "(384,0.5,384.5)", "strip_gemdg_odd", ok_gdg1_eta, ok_pad1,kRed);
    he = draw_eff(t, "Eff. for track with GEM digi to have GEM pad in chamber;digi strip;Eff.", "h_evn", "(384,0.5,384.5)", "strip_gemdg_even", ok_gdg2_eta, ok_pad2);
    c.SaveAs("%stest%s"%(plotDir, ext))


def eff_hs_8(filesDir, plotDir, f_name, ext):
    """Comment to be added here"""

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;|#eta|;Eff.", "hname", "(90,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2, kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;|#eta|;Eff.", "hname", "(90,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1)
    c.SaveAs("%stest%s"%(plotDir, ext))

def eff_hs_9(filesDir, plotDir, f_name, ext):
    """Comment to be added here"""

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;|#phi|;Eff.", "hname", "(128,0,3.2)", "TMath::Abs(phi)", 
                  TCut("%s && %s && %s"%(ok_lct2.GetTitle(), ok_eta.GetTitle(), ok_pt.GetTitle())), ok_pad2,kRed)
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;|#phi|;Eff.", "hname", "(128,0,3.2)", "TMath::Abs(phi)", 
                  TCut("%s && %s && %s"%(ok_lct2.GetTitle(), ok_eta.GetTitle(), ok_pt.GetTitle())), ok_pad1)
    c.SaveAs("%stest%s"%(plotDir, ext))


def eff_hs_10(filesDir, plotDir, f_name, ext):
    """Comment to be added here"""

    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;p_{T} [GeV/c];Eff.", "h_odd", "(50,0.,50.)", "pt", ok_lct1_eta, ok_pad1, kRed)
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;p_{T} [GeV/c];Eff.", "h_evn", "(50,0.,50.)", "pt", ok_lct2_eta, ok_pad2)
    c.SaveAs("%stest%s"%(plotDir, ext))

"""
def eff_hs_11(filesDir, plotDir, f_name, ext):
    Comment to be added here
    
    t = getTree("%s%s"%(filesDir, f_name));
    c = TCanvas("c","c",800,600)
    c.SetGridx(1)
    c.SetGridy(1)
    c.cd()
    h = TH1F("","Efficiency for track with LCT to have GEM pad in chamber;LCT |\#eta|;Efficiency",50,0.,50.)
    h.SetStats(0)
    h.Draw("")
    he.Draw()
    ho.Draw("same")
    ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,-0.2,0.2)", "fmod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta, ok_lct1 || ok_lct2, "", kRed);
    hg = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,-0.2,0.2)", "fmod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta, ok_pad1 || ok_pad2, "same");
    hgp = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,-0.2,0.2)", "fmod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta&&Qpos, ok_pad1 || ok_pad2, "same", kGreen);
    c.SaveAs("%stest%s"%(plotDir, ext))
"""


def eff_hs_all(filesDir, plotDir, f_name, ext):

    eff_hs_1(filesDir, plotDir, f_name, ext)
    eff_hs_2(filesDir, plotDir, f_name, ext)
    eff_hs_3(filesDir, plotDir, f_name, ext)
    eff_hs_4(filesDir, plotDir, f_name, ext)
    eff_hs_5(filesDir, plotDir, f_name, ext)
    eff_hs_6(filesDir, plotDir, f_name, ext)
    eff_hs_7(filesDir, plotDir, f_name, ext)
    eff_hs_8(filesDir, plotDir, f_name, ext)
    eff_hs_9(filesDir, plotDir, f_name, ext)
    """
    eff_hs_10(filesDir, plotDir, f_name, ext)
    eff_hs_11(filesDir, plotDir, f_name, ext)
    """

def eff_hs(filesDir, plotDir, ext):

    eff_hs_all(filesDir, plotDir, "gem_csc_delta_pt5_pad4.root", ext)
    eff_hs_all(filesDir, plotDir, "gem_csc_delta_pt10_pad4.root", ext)
    eff_hs_all(filesDir, plotDir, "gem_csc_delta_pt15_pad4.root", ext)
    eff_hs_all(filesDir, plotDir, "gem_csc_delta_pt20_pad4.root", ext)
    eff_hs_all(filesDir, plotDir, "gem_csc_delta_pt30_pad4.root", ext)
    eff_hs_all(filesDir, plotDir, "gem_csc_delta_pt40_pad4.root", ext)









## gt = getTree("gem_csc_delta_pt40_pad4.root");


## draw_eff(gt, "Eff. ;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_sh1 && ok_eta , ok_digi1)
## draw_eff(gt, "Eff. ;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_digi1 && ok_eta , ok_lct1)
## draw_eff(gt, "Eff. ;p_{T}, GeV/c;Eff.", "hname", "(50,0.,50.)", "pt", ok_sh1 && ok_eta , ok_lct1)


## draw_eff(gt, "Eff. of |CLCT pattern bend| selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname2", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<2 || TMath::Abs(bend_lct_even)<2")
## draw_eff(gt, "Eff. of |CLCT pattern bend| selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname1", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<1 || TMath::Abs(bend_lct_even)<1","same",kBlack)
## draw_eff(gt, "Eff. of |CLCT bend|<3 selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname3", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<3 || TMath::Abs(bend_lct_even)<3","same",kGreen+2)
## draw_eff(gt, "Eff. of |CLCT bend|<4 selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname4", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<4 || TMath::Abs(bend_lct_even)<4","same",kRed)


## // efficiency vs half-strip  - separate odd-even
## TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<1.9"
## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta , ok_pad1, "", kRed)
## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta , ok_pad2, "same")

## // efficiency vs half-strip  - including overlaps in odd&even
## TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<1.9"
## TCut ok_pad1_overlap = ok_pad1 || (ok_lct2 && ok_pad2);
## TCut ok_pad2_overlap = ok_pad2 || (ok_lct1 && ok_pad1);
## TTree *t = getTree("gem_csc_delta_pt20_pad4.root");

## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta , ok_pad1_overlap, "", kRed)
## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip number;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta , ok_pad2_overlap, "same")


## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |\#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_lct_odd)", ok_lct1, ok_pad1, "", kRed)
## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |\#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta_lct_even)", ok_lct2, ok_pad2, "same")

## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;trk |\#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1, "", kRed)
## draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;trk |\#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2, "same")


## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_env", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")

## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd", "(50,0.,50.)", "pt", ok_lct1 && ok_eta && ok_pad1, ok_dphi1)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn", "(50,0.,50.)", "pt", ok_lct2 && ok_eta && ok_pad2, ok_dphi2, "same")





## // 98% pt10
## TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.01076"
## TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.004863"
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_10", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "", kRed)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_10", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")
## // 98% pt30
## TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.00571"
## TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.00306"
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_20", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "same", kRed)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_20", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")
## // 98% pt30
## TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.00426"
## TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.00256"
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_30", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "same", kRed)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_30", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")
## // 98% pt40
## TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.00351"
## TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.00231"
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_40", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "same", kRed)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_40", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")


## |#Delta#phi_{odd}(LCT,Pad)| < 5.5 mrad
## |#Delta#phi_{even}(LCT,Pad)| < 3.1 mrad
## |#Delta\#eta(LCT,Pad)| < 0.08

## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_sh1 && ok_eta , ok_lct1 && ok_pad1 && ok_dphi1)
## draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_sh2 && ok_eta , ok_lct2 && ok_pad2 && ok_dphi2, "same")




## draw_eff(gt, "title;pt;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1)
## draw_eff(gt, "title;pt;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")
## draw_eff(gt, "title;|\#eta|;Eff.", "hname", "(45,1.5,2.2)", "TMath::Abs(eta)", ok_sh1 , ok_lct1 && ok_pad1 )
## draw_eff(gt, "title;|#phi|;Eff.", "hname", "(128,0,3.2)", "TMath::Abs(phi)", ok_sh2 , ok_lct2 && ok_pad2 )




## gt.Draw("TMath::Abs(eta)", ok_sh1, "");
## gt.Draw("TMath::Abs(eta)", ok_sh1 && ok_lct1, "same");
## gt.Draw("TMath::Abs(eta)", ok_sh1 && ok_lct1 && ok_pad1, "same");

## gt.Draw("TMath::Abs(phi)", ok_sh1 && ok_lct1 && ok_pad1, "");
## gt.Draw("TMath::Abs(phi)", ok_sh1, "");
## gt.Draw("TMath::Abs(eta)", ok_sh1 && ok_lct1, "same");

## gt.Draw("TMath::Abs(phi)", ok_sh1, "");
## gt.Draw("TMath::Abs(phi)", ok_sh1 && ok_lct1, "same");
## gt.Draw("TMath::Abs(phi)", ok_sh1 && ok_lct1 && ok_pad1, "same");


## gt.Draw("pt", ok_sh1, "");
## gt.Draw("pt", ok_sh1 && ok_lct1, "same");
## gt.Draw("pt", ok_sh1 && ok_lct1 && ok_pad1, "same");
## gt.Draw("pt", ok_sh1, "");
## gt.Draw("pt", ok_sh1 && ok_lct1 && ok_pad1, "same");

## gt.Draw("pt>>h1", ok_sh1, "");
## dn=(TH1F*)h1.Clone("dn")
## gt.Draw("pt>>h2", ok_sh1 && ok_lct1 && ok_pad1, "same");
## nm=(TH1F*)h2.Clone("nm")

## dn.Draw()
## nm.Draw("same")

## nm.Divide(dn)
## nm.Draw()

## gt.Draw("pt>>h2", ok_sh1 && ok_lct1, "same");
## nmlct=(TH1F*)h2.Clone("nmlct")
## nmlct.Divide(dn)

## nm.Draw()
## nmlct.Draw("same")

if __name__ == "__main__":  
    """
    halfStripEfficiencies("files/", "plots/efficiency/", ".pdf")
    halfStripEfficiencies("files/", "plots/efficiency/", ".eps")
    halfStripEfficiencies("files/", "plots/efficiency/", ".png")

    etaMatchingEfficiencies("files/", "plots/efficiency/", ".pdf")
    etaMatchingEfficiencies("files/", "plots/efficiency/", ".eps")
    etaMatchingEfficiencies("files/", "plots/efficiency/", ".png")
    """
    gemTurnOn("files/", "plots/efficiency/", "98", "even", ".eps")
    gemTurnOn("files/", "plots/efficiency/", "98", "odd", ".eps")

