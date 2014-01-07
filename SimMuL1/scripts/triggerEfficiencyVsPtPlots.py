from ROOT import * 
from triggerPlotHelpers import *
from mkdir import mkdir

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT
ROOT.gROOT.SetBatch(1)


#_______________________________________________________________________________
def setEffHisto(num_name, den_name, dir, nrebin, lcolor, lstyle, lwidth,
                htitle, xtitle, ytitle, x_range, y_range):
    """Set efficiency histogram"""

    hd = getH(dir, den_name)
    hn = getH(dir, num_name)
    
    hd.Sumw2()
    hn.Sumw2()

    myRebin(hd, nrebin)
    myRebin(hn, nrebin)
    heff = hn.Clone(num_name+"_eff")

    hd.Sumw2()
    heff.Sumw2()
    heff.Divide(heff,hd)
    heff.SetLineColor(lcolor)
    heff.SetLineStyle(lstyle)
    heff.SetLineWidth(lwidth)
    heff.SetTitle(htitle)
    heff.GetXaxis().SetTitle(xtitle)
    heff.GetYaxis().SetTitle(ytitle)
    heff.GetXaxis().SetRangeUser(x_range[0],x_range[1])
    heff.GetYaxis().SetRangeUser(y_range[0],y_range[1])
    heff.GetXaxis().SetTitleSize(0.05)
    heff.GetXaxis().SetTitleOffset(1.)
    heff.GetYaxis().SetTitleSize(0.05)
    heff.GetYaxis().SetTitleOffset(1.)
    heff.GetXaxis().SetLabelOffset(0.015)
    heff.GetYaxis().SetLabelOffset(0.015)
    heff.GetXaxis().SetLabelSize(0.05)
    heff.GetYaxis().SetLabelSize(0.05)
    return heff


#_______________________________________________________________________________
def getEffHisto(fname, hdir, num_name, den_name, nrebin, lcolor, lstyle, lwidth, 
                title, x_range, y_range):
    fh = TFile.Open(fname)
    
    hd0 = fh.Get(hdir + "/" + den_name)
    hn0 = fh.Get(hdir + "/" + num_name)
    
    hd = hd0.Clone(den_name+"_cln_"+fname)
    hn = hn0.Clone(num_name+"_cln_"+fname)
    hd.Sumw2()
    hn.Sumw2()
    
    myRebin(hd, nrebin)
    myRebin(hn, nrebin)
    
    heff = hn.Clone(num_name+"_eff_"+fname)
    
    hd.Sumw2()
    heff.Sumw2()
    
    heff.Divide(heff,hd)
    
    heff.SetLineColor(lcolor)
    heff.SetLineStyle(lstyle)
    heff.SetLineWidth(lwidth)
    
    heff.SetTitle(title)
    ##heff.GetXaxis().SetTitle(xtitle)
    ##heff.GetYaxis().SetTitle(ytitle)
    heff.GetXaxis().SetRangeUser(x_range[0],x_range[1])
    heff.GetYaxis().SetRangeUser(y_range[0],y_range[1])
    
    heff.GetXaxis().SetTitleSize(0.07)
    heff.GetXaxis().SetTitleOffset(0.7)
    heff.GetYaxis().SetLabelOffset(0.015)
    
    heff.GetXaxis().SetLabelSize(0.05)
    heff.GetYaxis().SetLabelSize(0.05)
    
    h1 = hn0
    h2 = hd0
    he = heff
    
    ##fh.Close()
    return heff


#_______________________________________________________________________________
def gem_eff_draw(input_dir, file_name, output_dir, ext):
    """Rate the efficiency plots"""
    gStyle.SetOptStat(0)
    gStyle.SetTitleStyle(0)
    
    ptreb=2

    filesDir = "files/"
    plotDir = "plots/trigger_eff_vs_pt/"
    ext = ".pdf"

    
    hdir = "SimMuL1StrictAll"

  ##f_def = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_def_pat2.root"
    f_def =      filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem_dphi0_pat2.root"
    f_g98_pt10 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt10_pat2.root"
    f_g98_pt15 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt15_pat2.root"
    f_g98_pt20 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt20_pat2.root"
    f_g98_pt30 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt30_pat2.root"
    f_g98_pt40 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt40_pat2.root"
    
    f_g95_pt10 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem95_pt10_pat2.root"
    f_g95_pt20 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem95_pt20_pat2.root"
    f_g95_pt30 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem95_pt30_pat2.root"
    f_g95_pt40 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem95_pt40_pat2.root"
    
    rpt = [0.,49.99]
    yrange = [0.,1.04]
    
    htitle = "Efficiency for #mu in 1.6<|#eta|<2.12 to have TF trackp_{T}^{MC}"
    
    hini = "h_pt_initial_1b"
    h2s = "h_pt_after_tfcand_eta1b_2s"
    h3s = "h_pt_after_tfcand_eta1b_3s"
    h2s1b = "h_pt_after_tfcand_eta1b_2s1b"
    h3s1b = "h_pt_after_tfcand_eta1b_3s1b"
        
    h_eff_tf0_2s  = getEffHisto(f_def, hdir, h2s, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange)
    h_eff_tf0_3s  = getEffHisto(f_def, hdir, h3s, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange)
    h_eff_tf0_2s1b  = getEffHisto(f_def, hdir, h2s1b, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange)
    h_eff_tf0_3s1b  = getEffHisto(f_def, hdir, h3s1b, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange)
    
    h_eff_tf10_2s  = getEffHisto(f_def, hdir, h2s + "_pt10", hini, ptreb, kGreen+4, 1, 2, htitle, rpt,yrange)
    h_eff_tf10_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange)
    h_eff_tf10_3s  = getEffHisto(f_def, hdir, h3s + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange)
    h_eff_tf10_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange)

    ##h_eff_tf15_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt15", hini, ptreb, kBlue, 1, 2, htitle, rpt,yrange)
    ##h_eff_tf15_3s  = getEffHisto(f_def, hdir, h3s + "_pt15", hini, ptreb, kBlue, 1, 2, htitle, rpt,yrange)
    ##h_eff_tf15_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt15", hini, ptreb, kBlue, 1, 2, htitle, rpt,yrange)

    h_eff_tf20_2s  = getEffHisto(f_def, hdir, h2s + "_pt20", hini, ptreb, kOrange+4, 1, 2, htitle, rpt,yrange)
    h_eff_tf20_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt20", hini, ptreb, kOrange, 1, 2, htitle, rpt,yrange)
    h_eff_tf20_3s  = getEffHisto(f_def, hdir, h3s + "_pt20", hini, ptreb, kOrange, 1, 2, htitle, rpt,yrange)
    h_eff_tf20_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt20", hini, ptreb, kOrange, 1, 2, htitle, rpt,yrange)

    h_eff_tf30_2s  = getEffHisto(f_def, hdir, h2s + "_pt30", hini, ptreb, kRed+4, 1, 2, htitle, rpt,yrange)
    h_eff_tf30_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt30", hini, ptreb, kRed, 1, 2, htitle, rpt,yrange)
    h_eff_tf30_3s  = getEffHisto(f_def, hdir, h3s + "_pt30", hini, ptreb, kRed, 1, 2, htitle, rpt,yrange)
    h_eff_tf30_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt30", hini, ptreb, kRed, 1, 2, htitle, rpt,yrange)

    h_eff_tf40_2s  = getEffHisto(f_def, hdir, h2s + "_pt40", hini, ptreb, kViolet+4, 1, 2, htitle, rpt,yrange)
    h_eff_tf40_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt40", hini, ptreb, kViolet, 1, 2, htitle, rpt,yrange)
    h_eff_tf40_3s  = getEffHisto(f_def, hdir, h3s + "_pt40", hini, ptreb, kViolet, 1, 2, htitle, rpt,yrange)
    h_eff_tf40_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt40", hini, ptreb, kViolet, 1, 2, htitle, rpt,yrange)

    h_eff_tf10_gpt10_2s1b  = getEffHisto(f_g98_pt10, hdir, h2s1b + "_pt10", hini, ptreb, kGreen+2, 7, 2, htitle, rpt,yrange)
    h_eff_tf10_gpt10_3s1b  = getEffHisto(f_g98_pt10, hdir, h3s1b + "_pt10", hini, ptreb, kGreen+2, 7, 2, htitle, rpt,yrange)

    h_eff_tf15_gpt15_2s1b  = getEffHisto(f_g98_pt15, hdir, h2s1b + "_pt15", hini, ptreb, kBlue, 7, 2, htitle, rpt,yrange)
    h_eff_tf15_gpt15_3s1b  = getEffHisto(f_g98_pt15, hdir, h3s1b + "_pt15", hini, ptreb, kBlue, 7, 2, htitle, rpt,yrange)

    h_eff_tf20_gpt20_2s1b  = getEffHisto(f_g98_pt20, hdir, h2s1b + "_pt20", hini, ptreb, kOrange, 7, 2, htitle, rpt,yrange)
    h_eff_tf20_gpt20_3s1b  = getEffHisto(f_g98_pt20, hdir, h3s1b + "_pt20", hini, ptreb, kOrange, 7, 2, htitle, rpt,yrange)

    h_eff_tf30_gpt30_2s1b  = getEffHisto(f_g98_pt30, hdir, h2s1b + "_pt30", hini, ptreb, kRed, 7, 2, htitle, rpt,yrange)
    h_eff_tf30_gpt30_3s1b  = getEffHisto(f_g98_pt30, hdir, h3s1b + "_pt30", hini, ptreb, kRed, 7, 2, htitle, rpt,yrange)

    h_eff_tf40_gpt40_2s1b  = getEffHisto(f_g98_pt40, hdir, h2s1b + "_pt40", hini, ptreb, kViolet, 7, 2, htitle, rpt,yrange)
    h_eff_tf40_gpt40_3s1b  = getEffHisto(f_g98_pt40, hdir, h3s1b + "_pt40", hini, ptreb, kViolet, 7, 2, htitle, rpt,yrange)


    c2s1b = TCanvas("c2s1b","c2s1b",800,600) 

    """
    h_eff_gmt20_1b  = getEffHisto(f_def, hdir, "h_pt_after_gmt_eta1b_1mu_pt20", hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange)
    h_eff_gmt30_1b  = getEffHisto(f_def, hdir, "h_pt_after_gmt_eta1b_1mu_pt30", hini, ptreb, kBlack-1, 1, 2, htitle, rpt, yrange)
    h_eff_gmt40_1b  = getEffHisto(f_def, hdir, "h_pt_after_gmt_eta1b_1mu_pt40", hini, ptreb, kBlack-2, 1, 2, htitle, rpt, yrange)
    h_eff_gmt20_1b.Draw("hist")
    h_eff_gmt30_1b.Draw("hist same")
    h_eff_gmt40_1b.Draw("hist same")
    return

    h_eff_tf40_3s.Draw("hist")
    h_eff_tf40_3s1b.Draw("hist same")
    h_eff_tf40_gpt40_3s1b.Draw("hist same")
    return
    """


    h_eff_tf10_gpt15_2s1b  = getEffHisto(f_g98_pt15, hdir, h2s1b + "_pt10", hini, ptreb, kGreen+2, 3, 2, htitle, rpt,yrange)
    h_eff_tf10_gpt15_3s1b  = getEffHisto(f_g98_pt15, hdir, h3s1b + "_pt10", hini, ptreb, kGreen+2, 3, 2, htitle, rpt,yrange)

    h_eff_tf15_gpt20_2s1b  = getEffHisto(f_g98_pt20, hdir, h2s1b + "_pt15", hini, ptreb, kViolet, 3, 2, htitle, rpt,yrange)
    h_eff_tf15_gpt20_3s1b  = getEffHisto(f_g98_pt20, hdir, h3s1b + "_pt15", hini, ptreb, kViolet, 3, 2, htitle, rpt,yrange)

    h_eff_tf20_gpt30_2s1b  = getEffHisto(f_g98_pt30, hdir, h2s1b + "_pt20", hini, ptreb, kOrange, 3, 2, htitle, rpt,yrange)
    h_eff_tf20_gpt30_3s1b  = getEffHisto(f_g98_pt30, hdir, h3s1b + "_pt20", hini, ptreb, kOrange, 3, 2, htitle, rpt,yrange)

    h_eff_tf30_gpt40_2s1b  = getEffHisto(f_g98_pt40, hdir, h2s1b + "_pt30", hini, ptreb, kRed, 3, 2, htitle, rpt,yrange)
    h_eff_tf30_gpt40_3s1b  = getEffHisto(f_g98_pt40, hdir, h3s1b + "_pt30", hini, ptreb, kRed, 3, 2, htitle, rpt,yrange)


    h_eff_tf10_gpt20_2s1b  = getEffHisto(f_g98_pt20, hdir, h2s1b + "_pt10", hini, ptreb, kGreen+2, 3, 2, htitle, rpt,yrange)
    h_eff_tf10_gpt20_3s1b  = getEffHisto(f_g98_pt20, hdir, h3s1b + "_pt10", hini, ptreb, kGreen+2, 3, 2, htitle, rpt,yrange)

    h_eff_tf15_gpt30_2s1b  = getEffHisto(f_g98_pt30, hdir, h2s1b + "_pt15", hini, ptreb, kViolet, 3, 2, htitle, rpt,yrange)
    h_eff_tf15_gpt30_3s1b  = getEffHisto(f_g98_pt30, hdir, h3s1b + "_pt15", hini, ptreb, kViolet, 3, 2, htitle, rpt,yrange)

    h_eff_tf20_gpt40_2s1b  = getEffHisto(f_g98_pt40, hdir, h2s1b + "_pt20", hini, ptreb, kOrange, 3, 2, htitle, rpt,yrange)
    h_eff_tf20_gpt40_3s1b  = getEffHisto(f_g98_pt40, hdir, h3s1b + "_pt20", hini, ptreb, kOrange, 3, 2, htitle, rpt,yrange)



    c2s1b = TCanvas("c2s1b","c2s1b",800,600) 

    ##h_eff_tf0_2s1b.Draw("hist")
    h_eff_tf10_2s1b.Draw("hist")
    ##h_eff_tf15_2s1b.Draw("hist same")
    h_eff_tf20_2s1b.Draw("hist same")
    h_eff_tf30_2s1b.Draw("hist same")
    ##h_eff_tf40_2s1b.Draw("hist same")

    h_eff_tf10_gpt10_2s1b.Draw("hist same")
    h_eff_tf20_gpt20_2s1b.Draw("hist same")
    h_eff_tf30_gpt30_2s1b.Draw("hist same")

    leg = TLegend(0.50,0.17,.999,0.57, "", "brNDC")
    leg.SetNColumns(2)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track requires 2+ stubs, one from ME1")
    leg.AddEntry(h_eff_tf10_2s1b, "Trigger p_{T}:", "")
    leg.AddEntry(h_eff_tf10_gpt10_2s1b, "with GEM:", "")
    leg.AddEntry(h_eff_tf10_2s1b, "p_{T}^{TF}>=10", "l")
    leg.AddEntry(h_eff_tf10_gpt10_2s1b, "#Delta#phi for p_{T}=10", "l")
    leg.AddEntry(h_eff_tf20_2s1b, "p_{T}^{TF}>=20", "l")
    leg.AddEntry(h_eff_tf20_gpt20_2s1b, "#Delta#phi for p_{T}=20", "l")
    leg.AddEntry(h_eff_tf30_2s1b, "p_{T}^{TF}>=30", "l")
    leg.AddEntry(h_eff_tf30_gpt30_2s1b, "#Delta#phi for p_{T}=30", "l")
    leg.Draw()

    c2s1b.Print(plotDir + "eff_2s1b" + ext)


    c3s1b = TCanvas("c3s1b","c3s1b",800,600) 

    h_eff_tf10_3s1b.Draw("hist")
    h_eff_tf20_3s1b.Draw("hist same")
    h_eff_tf30_3s1b.Draw("hist same")

    h_eff_tf10_gpt10_3s1b.Draw("hist same")
    h_eff_tf20_gpt20_3s1b.Draw("hist same")
    h_eff_tf30_gpt30_3s1b.Draw("hist same")

    leg = TLegend(0.50,0.17,.999,0.57, "", "brNDC")
    leg.SetNColumns(2)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track requires 3+ stubs, one from ME1")
    leg.AddEntry(h_eff_tf10_3s1b, "Trigger p_{T}:", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "with GEM:", "")
    leg.AddEntry(h_eff_tf10_3s1b, "p_{T}^{TF}>=10", "l")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "#Delta#phi for p_{T}=10", "l")
    leg.AddEntry(h_eff_tf20_3s1b, "p_{T}^{TF}>=20", "l")
    leg.AddEntry(h_eff_tf20_gpt20_3s1b, "#Delta#phi for p_{T}=20", "l")
    leg.AddEntry(h_eff_tf30_3s1b, "p_{T}^{TF}>=30", "l")
    leg.AddEntry(h_eff_tf30_gpt30_3s1b, "#Delta#phi for p_{T}=30", "l")
    leg.Draw()

    c3s1b.Print(plotDir + "eff_3s1b" + ext)



    c3s_2s1b = TCanvas("c3s_2s1b","c3s_2s1b",800,600)

    h_eff_tf10_3s.Draw("hist")
    h_eff_tf20_3s.Draw("hist same")
    h_eff_tf30_3s.Draw("hist same")

    h_eff_tf10_gpt10_2s1b.Draw("hist same")
    h_eff_tf20_gpt20_2s1b.Draw("hist same")
    h_eff_tf30_gpt30_2s1b.Draw("hist same")

    leg = TLegend(0.50,0.17,.999,0.57, "", "brNDC")
    leg.SetNColumns(2)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track requires")
    leg.AddEntry(h_eff_tf10_3s, "3+ stubs", "")
    leg.AddEntry(h_eff_tf10_gpt10_2s1b, "2+ stubs with GEM in ME1", "")
    leg.AddEntry(h_eff_tf10_3s, "p_{T}^{TF}>=10", "l")
    leg.AddEntry(h_eff_tf10_gpt10_2s1b, "#Delta#phi for p_{T}=10", "l")
    leg.AddEntry(h_eff_tf20_3s, "p_{T}^{TF}>=20", "l")
    leg.AddEntry(h_eff_tf20_gpt20_2s1b, "#Delta#phi for p_{T}=20", "l")
    leg.AddEntry(h_eff_tf30_3s, "p_{T}^{TF}>=30", "l")
    leg.AddEntry(h_eff_tf30_gpt30_2s1b, "#Delta#phi for p_{T}=30", "l")
    leg.Draw()

    c3s_2s1b.Print(plotDir + "eff_3s_2s1b" + ext)




    c3s_def = TCanvas("c3s_def","c3s_def",800,600)

    h_eff_tf10_3s.Draw("hist")
    h_eff_tf20_3s.Draw("hist same")
    h_eff_tf30_3s.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track requires 3+ stubs and")
    leg.AddEntry(h_eff_tf10_3s, "p_{T}^{TF}>=10", "l")
    leg.AddEntry(h_eff_tf20_3s, "p_{T}^{TF}>=20", "l")
    leg.AddEntry(h_eff_tf30_3s, "p_{T}^{TF}>=30", "l")
    leg.Draw()

    c3s_def.Print(plotDir + "eff_3s_def" + ext)


    c3s1b_def = TCanvas("c3s1b_def","c3s1b_def",800,600)

    h_eff_tf10_3s1b.Draw("hist")
    h_eff_tf20_3s1b.Draw("hist same")
    h_eff_tf30_3s1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track requires 3+ stubs with ME1 and")
    leg.AddEntry(h_eff_tf10_3s, "p_{T}^{TF}>=10", "l")
    leg.AddEntry(h_eff_tf20_3s, "p_{T}^{TF}>=20", "l")
    leg.AddEntry(h_eff_tf30_3s, "p_{T}^{TF}>=30", "l")
    leg.Draw()

    c3s1b_def.Print(plotDir + "eff_3s1b_def" + ext)



    h_eff_tf10_2s.SetLineColor(kAzure+2)
    h_eff_tf10_2s1b.SetLineColor(kAzure+6)
    h_eff_tf10_3s.SetLineColor(kAzure+3)
    h_eff_tf10_3s1b.SetLineColor(kAzure+7)
    h_eff_tf10_gpt10_2s1b.SetLineColor(kAzure+6)
    h_eff_tf10_gpt10_3s1b.SetLineColor(kAzure+7)

    h_eff_tf20_2s.SetLineColor(kAzure+2)
    h_eff_tf20_2s1b.SetLineColor(kAzure+6)
    h_eff_tf20_3s.SetLineColor(kAzure+3)
    h_eff_tf20_3s1b.SetLineColor(kAzure+7)
    h_eff_tf20_gpt20_2s1b.SetLineColor(kAzure+6)
    h_eff_tf20_gpt20_3s1b.SetLineColor(kAzure+7)

    h_eff_tf30_2s.SetLineColor(kAzure+2)
    h_eff_tf30_2s1b.SetLineColor(kAzure+6)
    h_eff_tf30_3s.SetLineColor(kAzure+3)
    h_eff_tf30_3s1b.SetLineColor(kAzure+7)
    h_eff_tf30_gpt30_2s1b.SetLineColor(kAzure+6)
    h_eff_tf30_gpt30_3s1b.SetLineColor(kAzure+7)

    h_eff_tf40_2s.SetLineColor(kAzure+2)
    h_eff_tf40_2s1b.SetLineColor(kAzure+6)
    h_eff_tf40_3s.SetLineColor(kAzure+3)
    h_eff_tf40_3s1b.SetLineColor(kAzure+7)
    h_eff_tf40_gpt40_2s1b.SetLineColor(kAzure+6)
    h_eff_tf40_gpt40_3s1b.SetLineColor(kAzure+7)


    c2s_pt10_def = TCanvas("c2s_pt10_def","c2s_pt10_def",800,600)

    h_eff_tf10_2s.Draw("hist")
    h_eff_tf10_2s1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track: p_{T}^{TF}>=10 and 2+ stubs")
    leg.AddEntry(h_eff_tf10_2s, "anywhere", "l")
    leg.AddEntry(h_eff_tf10_2s1b, "with ME1", "l")
    leg.Draw()

    c2s_pt10_def.Print(plotDir + "eff_2s_pt10_def" + ext)

    h_eff_tf10_gpt10_2s1b.Draw("hist same")
    leg.AddEntry(h_eff_tf10_gpt10_2s1b, "with (ME1 + GEM)", "l")
    c2s_pt10_def.Print(plotDir + "eff_2s_pt10_gem" + ext)



    c3s_pt10_def = TCanvas("c3s_pt10_def","c3s_pt10_def",800,600)

    h_eff_tf10_3s.Draw("hist")
    h_eff_tf10_3s1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track: p_{T}^{TF}>=10 and 3+ stubs")
    leg.AddEntry(h_eff_tf10_3s, "anywhere", "l")
    leg.AddEntry(h_eff_tf10_3s1b, "with ME1", "l")
    leg.Draw()

    c3s_pt10_def.Print(plotDir + "eff_3s_pt10_def" + ext)

    h_eff_tf10_gpt10_3s1b.Draw("hist same")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "with (ME1 + GEM)", "l")
    c3s_pt10_def.Print(plotDir + "eff_3s_pt10_gem" + ext)




    c2s_pt20_def = TCanvas("c2s_pt20_def","c2s_pt20_def",800,600)

    h_eff_tf20_2s.Draw("hist")
    h_eff_tf20_2s1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track: p_{T}^{TF}>=20 and 2+ stubs")
    leg.AddEntry(h_eff_tf20_2s, "anywhere", "l")
    leg.AddEntry(h_eff_tf20_2s1b, "with ME1", "l")
    leg.Draw()

    c2s_pt20_def.Print(plotDir + "eff_2s_pt20_def" + ext)

    h_eff_tf20_gpt20_2s1b.Draw("hist same")
    leg.AddEntry(h_eff_tf20_gpt20_2s1b, "with (ME1 + GEM)", "l")
    c2s_pt20_def.Print(plotDir + "eff_2s_pt20_gem" + ext)



    c3s_pt20_def = TCanvas("c3s_pt20_def","c3s_pt20_def",800,600)

    h_eff_tf20_3s.Draw("hist")
    h_eff_tf20_3s1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track: p_{T}^{TF}>=20 and 3+ stubs")
    leg.AddEntry(h_eff_tf20_3s, "anywhere", "l")
    leg.AddEntry(h_eff_tf20_3s1b, "with ME1", "l")
    leg.Draw()

    c3s_pt20_def.Print(plotDir + "eff_3s_pt20_def" + ext)

    h_eff_tf20_gpt20_3s1b.Draw("hist same")
    leg.AddEntry(h_eff_tf20_gpt20_3s1b, "with (ME1 + GEM)", "l")
    c3s_pt20_def.Print(plotDir + "eff_3s_pt20_gem" + ext)



    c2s_pt30_def = TCanvas("c2s_pt30_def","c2s_pt30_def",800,600)

    h_eff_tf30_2s.Draw("hist")
    h_eff_tf30_2s1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track: p_{T}^{TF}>=30 and 2+ stubs")
    leg.AddEntry(h_eff_tf30_2s, "anywhere", "l")
    leg.AddEntry(h_eff_tf30_2s1b, "with ME1", "l")
    leg.Draw()

    c2s_pt30_def.Print(plotDir + "eff_2s_pt30_def" + ext)

    h_eff_tf30_gpt30_2s1b.Draw("hist same")
    leg.AddEntry(h_eff_tf30_gpt30_2s1b, "with (ME1 + GEM)", "l")
    c2s_pt30_def.Print(plotDir + "eff_2s_pt30_gem" + ext)



    c3s_pt30_def = TCanvas("c3s_pt30_def","c3s_pt30_def",800,600)

    h_eff_tf30_3s.Draw("hist")
    h_eff_tf30_3s1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track: p_{T}^{TF}>=30 and 3+ stubs")
    leg.AddEntry(h_eff_tf30_3s, "anywhere", "l")
    leg.AddEntry(h_eff_tf30_3s1b, "with ME1", "l")
    leg.Draw()

    c3s_pt30_def.Print(plotDir + "eff_3s_pt30_def" + ext)

    h_eff_tf30_gpt30_3s1b.Draw("hist same")
    leg.AddEntry(h_eff_tf30_gpt30_3s1b, "with (ME1 + GEM)", "l")
    c3s_pt30_def.Print(plotDir + "eff_3s_pt30_gem" + ext)



    c2s_pt40_def = TCanvas("c2s_pt40_def","c2s_pt40_def",800,600)

    h_eff_tf40_2s.Draw("hist")
    h_eff_tf40_2s1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track: p_{T}^{TF}>=40 and 2+ stubs")
    leg.AddEntry(h_eff_tf40_2s, "anywhere", "l")
    leg.AddEntry(h_eff_tf40_2s1b, "with ME1", "l")
    leg.Draw()

    c2s_pt40_def.Print(plotDir + "eff_2s_pt40_def" + ext)

    h_eff_tf40_gpt40_2s1b.Draw("hist same")
    leg.AddEntry(h_eff_tf40_gpt40_2s1b, "with (ME1 + GEM)", "l")
    c2s_pt40_def.Print(plotDir + "eff_2s_pt40_gem" + ext)



    c3s_pt40_def = TCanvas("c3s_pt40_def","c3s_pt40_def",800,600)

    h_eff_tf40_3s.Draw("hist")
    h_eff_tf40_3s1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("TF track: p_{T}^{TF}>=40 and 3+ stubs")
    leg.AddEntry(h_eff_tf40_3s, "anywhere", "l")
    leg.AddEntry(h_eff_tf40_3s1b, "with ME1", "l")
    leg.Draw()

    c3s_pt40_def.Print(plotDir + "eff_3s_pt40_def" + ext)

    h_eff_tf40_gpt40_3s1b.Draw("hist same")
    leg.AddEntry(h_eff_tf40_gpt40_3s1b, "with (ME1 + GEM)", "l")
    c3s_pt40_def.Print(plotDir + "eff_3s_pt40_gem" + ext)



    ##return

    h_eff_tf10_gpt10_3s1b.SetLineColor(kBlue)
    h_eff_tf10_gpt15_3s1b.SetLineColor(kMagenta)
    h_eff_tf20_gpt20_3s1b.SetLineColor(kBlue+2)
    h_eff_tf20_gpt30_3s1b.SetLineColor(kMagenta+2)
    h_eff_tf30_gpt30_3s1b.SetLineColor(kBlue+4)
    h_eff_tf30_gpt40_3s1b.SetLineColor(kMagenta+4)

    c3s_tight = TCanvas("c3s_tight","c3s_tight",800,600)

    h_eff_tf10_gpt10_3s1b.Draw("hist")
    h_eff_tf10_gpt15_3s1b.Draw("hist same")

    ##h_eff_tf15_gpt15_3s1b.Draw("hist same")
    ##h_eff_tf15_gpt20_3s1b.Draw("hist same")

    h_eff_tf20_gpt20_3s1b.Draw("hist same")
    h_eff_tf20_gpt30_3s1b.Draw("hist same")

    h_eff_tf30_gpt30_3s1b.Draw("hist same")
    h_eff_tf30_gpt40_3s1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("TF track: 3+ stubs with ME1")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T}^{TF} cut and", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T}^{TF} cut and", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T} for #Delta#phi(GEM,LCT)", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T} for #Delta#phi(GEM,LCT)", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "#geq10 and 10", "l")
    leg.AddEntry(h_eff_tf10_gpt15_3s1b, "#geq10 and 15", "l")
    leg.AddEntry(h_eff_tf20_gpt20_3s1b, "#geq20 and 20", "l")
    leg.AddEntry(h_eff_tf20_gpt30_3s1b, "#geq20 and 30", "l")
    leg.AddEntry(h_eff_tf30_gpt30_3s1b, "#geq30 and 30", "l")
    leg.AddEntry(h_eff_tf30_gpt40_3s1b, "#geq30 and 40", "l")
    leg.Draw()

    c3s_tight.Print(plotDir + "eff_3s_gemtight" + ext)



    h_eff_tf10_gpt10_3s1b.SetLineColor(kBlue)
    h_eff_tf10_gpt20_3s1b.SetLineColor(kMagenta)
    h_eff_tf15_gpt15_3s1b.SetLineColor(kBlue+2)
    h_eff_tf15_gpt30_3s1b.SetLineColor(kMagenta+2)
    h_eff_tf20_gpt20_3s1b.SetLineColor(kBlue+4)
    h_eff_tf20_gpt40_3s1b.SetLineColor(kMagenta+4)

    c3s_tight = TCanvas("c3s_tight","c3s_tight",800,600)

    h_eff_tf10_gpt10_3s1b.Draw("hist")
    h_eff_tf10_gpt20_3s1b.Draw("hist same")

    h_eff_tf15_gpt15_3s1b.Draw("hist same")
    h_eff_tf15_gpt30_3s1b.Draw("hist same")

    h_eff_tf20_gpt20_3s1b.Draw("hist same")
    h_eff_tf20_gpt40_3s1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("TF track: 3+ stubs with ME1")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T}^{TF} cut and", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T}^{TF} cut and", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T} for #Delta#phi(GEM,LCT)", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T} for #Delta#phi(GEM,LCT)", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "#geq10 and 10", "l")
    leg.AddEntry(h_eff_tf10_gpt20_3s1b, "#geq10 and 20", "l")
    leg.AddEntry(h_eff_tf15_gpt15_3s1b, "#geq15 and 15", "l")
    leg.AddEntry(h_eff_tf15_gpt30_3s1b, "#geq15 and 30", "l")
    leg.AddEntry(h_eff_tf20_gpt20_3s1b, "#geq20 and 20", "l")
    leg.AddEntry(h_eff_tf20_gpt40_3s1b, "#geq20 and 40", "l")
    leg.Draw()

    c3s_tight.Print(plotDir + "eff_3s_gemtightX" + ext)



#_______________________________________________________________________________
def gem_eff_draw_gem1b(input_dir, file_name, output_dir, ext):
    """Draw trigger efficiency plots"""
    gStyle.SetOptStat(0)
    gStyle.SetTitleStyle(0)

    filesDir = "files/"
    plotDir = "plots/trigger_eff_vs_pt/"
    ext = ".pdf"


    ptreb=2

    hdir = "SimMuL1StrictAll"

    ##f_def = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_def_pat2.root"
    f_def = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem_dphi0_pat2.root"

    rpt = [0.,49.99]
    yrange07 = [0.7,1.04]



    hini = "h_pt_initial_gem_1b"
    hini_g = "h_pt_gem_1b"
    hini_gl = "h_pt_lctgem_1b"

    h2g_00 = "h_pt_after_tfcand_gem1b_2s1b"
    h2g_00_123 = "h_pt_after_tfcand_gem1b_2s123"
    h2g_00_13  = "h_pt_after_tfcand_gem1b_2s13"
    h3g_00 = "h_pt_after_tfcand_gem1b_3s1b"
    h2p_00 = "h_pt_after_tfcand_dphigem1b_2s1b"
    h2p_00_123 = "h_pt_after_tfcand_dphigem1b_2s123"
    h2p_00_13 = "h_pt_after_tfcand_dphigem1b_2s13"
    h3p_00 = "h_pt_after_tfcand_dphigem1b_3s1b"

    h2g_15 = "h_pt_after_tfcand_gem1b_2s1b_pt15"
    h2g_15_123 = "h_pt_after_tfcand_gem1b_2s123_pt15"
    h2g_15_13 = "h_pt_after_tfcand_gem1b_2s13_pt15"
    h3g_15 = "h_pt_after_tfcand_gem1b_3s1b_pt15"
    h2p_15 = "h_pt_after_tfcand_dphigem1b_2s1b_pt15"
    h2p_15_123 = "h_pt_after_tfcand_dphigem1b_2s123_pt15"
    h2p_15_13 = "h_pt_after_tfcand_dphigem1b_2s13_pt15"
    h3p_15 = "h_pt_after_tfcand_dphigem1b_3s1b_pt15"

    h2g_20 = "h_pt_after_tfcand_gem1b_2s1b_pt20"
    h2g_20_123 = "h_pt_after_tfcand_gem1b_2s123_pt20"
    h2g_20_13 = "h_pt_after_tfcand_gem1b_2s13_pt20"
    h3g_20 = "h_pt_after_tfcand_gem1b_3s1b_pt20"
    h2p_20 = "h_pt_after_tfcand_dphigem1b_2s1b_pt20"
    h2p_20_123 = "h_pt_after_tfcand_dphigem1b_2s123_pt20"
    h2p_20_13 = "h_pt_after_tfcand_dphigem1b_2s13_pt20"
    h3p_20 = "h_pt_after_tfcand_dphigem1b_3s1b_pt20"

    h2g_30 = "h_pt_after_tfcand_gem1b_2s1b_pt30"
    h2g_30_123 = "h_pt_after_tfcand_gem1b_2s123_pt30"
    h2g_30_13 = "h_pt_after_tfcand_gem1b_2s13_pt30"
    h3g_30 = "h_pt_after_tfcand_gem1b_3s1b_pt30"
    h2p_30 = "h_pt_after_tfcand_dphigem1b_2s1b_pt30"
    h2p_30_123 = "h_pt_after_tfcand_dphigem1b_2s123_pt30"
    h2p_30_13 = "h_pt_after_tfcand_dphigem1b_2s13_pt30"
    h3p_30 = "h_pt_after_tfcand_dphigem1b_3s1b_pt30"


    c2 = TCanvas("c2","c2",800,600) 
    gPad.SetGridx(1)
    gPad.SetGridy(1)


    htitle = "Efficiency for #mu (GEM) in 1.64<|#eta|<2.05 to have TF track with ME1/b stubp_{T}^{MC}"

    hel = getEffHisto(f_def, hdir, hini_gl, hini_g, ptreb, kBlack, 1, 2, htitle, rpt, yrange07)
    hel.Draw("hist")
    het2 = getEffHisto(f_def, hdir, h2g_00, hini_g, ptreb, kGreen+2, 1, 2, htitle, rpt, yrange07)
    het2.Draw("same hist")
    het3 = getEffHisto(f_def, hdir, h3g_00, hini_g, ptreb, kGreen+2, 2, 2, htitle, rpt, yrange07)
    het3.Draw("same hist")
    het2pt20 = getEffHisto(f_def, hdir, h2g_20, hini_g, ptreb, kBlue, 1, 2, htitle, rpt, yrange07)
    het2pt20.Draw("same hist")
    het3pt20 = getEffHisto(f_def, hdir, h3g_20, hini_g, ptreb, kBlue, 2, 2, htitle, rpt, yrange07)
    het3pt20.Draw("same hist")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    ##leg.SetHeader("TF track: 3+ stubs with ME1")
    leg.AddEntry(hel, "ME1/b LCT stub", "l")
    leg.AddEntry(hel, " ", "")
    leg.AddEntry(het2, "any p_{T}^{TF}, 2+ stubs", "l")
    leg.AddEntry(het2pt20, "p_{T}^{TF}#geq20, 2+ stubs", "l")
    leg.AddEntry(het3, "any p_{T}^{TF}, 3+ stubs", "l")
    leg.AddEntry(het3pt20, "p_{T}^{TF}#geq20, 3+ stubs", "l")
    leg.Draw()

    c2.Print(plotDir + "eff_gem1b_basegem" + ext)



    htitle = "Efficiency for #mu (GEM+LCT) in 1.64<|#eta|<2.05 to have TF track with ME1/b stubp_{T}^{MC}"

    helt2pt20 = getEffHisto(f_def, hdir, h2g_20, hini_gl, ptreb, kMagenta-3, 1, 2, htitle, rpt, yrange07)
    helt2pt20.Draw("hist")
    helt3pt20 = getEffHisto(f_def, hdir, h3g_20, hini_gl, ptreb, kMagenta-3, 2, 2, htitle, rpt, yrange07)
    helt3pt20.Draw("same hist")
    het2pt20.Draw("same hist")
    het3pt20.Draw("same hist")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("TF track p_{T}^{TF}#geq20 with ME1")
    leg.AddEntry(het2pt20, "GEM baseline", "")
    leg.AddEntry(helt2pt20, "GEM+LCT baseline", "")
    leg.AddEntry(het2pt20, "2+ stubs", "l")
    leg.AddEntry(helt2pt20, "2+ stubs", "l")
    leg.AddEntry(het3pt20, "3+ stubs", "l")
    leg.AddEntry(helt3pt20, "3+ stubs", "l")
    leg.Draw()
    c2.Print(plotDir + "eff_gem1b_baselctgem" + ext)

    ##return

    htitle = "Efficiency for #mu (GEM) in 1.64<|#eta|<2.05 to have TF track with ME1/b stubp_{T}^{MC}"

    het2pt20.Draw("hist")
    het3pt20.Draw("same hist")
    het2pt20p = getEffHisto(f_def, hdir, h2p_20, hini_g, ptreb, kGray+2, 1, 2, htitle, rpt, yrange07)
    het2pt20p.Draw("same hist")
    het3pt20p = getEffHisto(f_def, hdir, h3p_20, hini_g, ptreb, kGray+2, 2, 2, htitle, rpt, yrange07)
    het3pt20p.Draw("same hist")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("TF track p_{T}^{TF}#geq20 with ME1")
    leg.AddEntry(het2pt20, "no GEM #Delta#phi", "")
    leg.AddEntry(het2pt20p, "with GEM #Delta#phi", "l")
    leg.AddEntry(het2pt20, "2+ stubs", "l")
    leg.AddEntry(het2pt20p, "2+ stubs", "l")
    leg.AddEntry(het3pt20, "3+ stubs", "l")
    leg.AddEntry(het3pt20p, "3+ stubs", "l")
    leg.Draw()
    c2.Print(plotDir + "eff_gem1b_basegem_dphi" + ext)


    htitle = "Efficiency for #mu (GEM+LCT) in 1.64<|#eta|<2.05 to have TF track with ME1/b stubp_{T}^{MC}"

    helt2pt20.Draw("hist")
    helt3pt20.Draw("same hist")
    helt2pt20p = getEffHisto(f_def, hdir, h2p_20, hini_gl, ptreb, kGray+2, 1, 2, htitle, rpt, yrange07)
    helt2pt20p.Draw("same hist")
    helt3pt20p = getEffHisto(f_def, hdir, h3p_20, hini_gl, ptreb, kGray+2, 2, 2, htitle, rpt, yrange07)
    helt3pt20p.Draw("same hist")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("TF track p_{T}^{TF}#geq20 with ME1")
    leg.AddEntry(helt2pt20, "no GEM #Delta#phi", "")
    leg.AddEntry(helt2pt20p, "with GEM #Delta#phi", "l")
    leg.AddEntry(helt2pt20, "2+ stubs", "l")
    leg.AddEntry(helt2pt20p, "2+ stubs", "l")
    leg.AddEntry(helt3pt20, "3+ stubs", "l")
    leg.AddEntry(helt3pt20p, "3+ stubs", "l")
    leg.Draw()
    c2.Print(plotDir + "eff_gem1b_baselpcgem_dphi" + ext)

#_______________________________________________________________________________
def eff_pt_tf(output_dir, ext, dir_name = "GEMCSCTriggerEfficiency"):

    c = TCanvas("c","c",1000,600 ) 
    c.cd()


    helt2pt20.Draw("hist")
    helt3pt20.Draw("same hist")
    helt2pt20_123 = getEffHisto(f_def, hdir, h2g_20_123, hini_gl, ptreb, kMagenta-3, 9, 2, htitle, rpt, yrange07)
    helt2pt20_123.Draw("same hist")
    helt3pt20_13 = getEffHisto(f_def, hdir, h2g_20_13, hini_gl, ptreb, kMagenta-3, 7, 2, htitle, rpt, yrange07)
    helt3pt20_13.Draw("same hist")

    leg = TLegend(0.5,0.17,.999,0.55, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    ##leg.SetNColumns(2)
    leg.SetHeader("TF track p_{T}^{TF}#geq20 with ME1")
    ##leg.AddEntry(helt2pt20, "no GEM #Delta#phi", "")
    ##leg.AddEntry(helt2pt20p, "with GEM #Delta#phi", "")
    leg.AddEntry(helt2pt20, "2+ stubs", "l")
    leg.AddEntry(helt2pt20_123, "2+ stubs (no ME1-4 tracks)", "l")
    leg.AddEntry(helt3pt20_13, "2+ stubs (no ME1-2 and ME1-4)", "l")
    leg.AddEntry(helt3pt20, "3+ stubs", "l")
    leg.Draw()
    c2.Print(plotDir + "eff_gem1b_baselpcgem_123" + ext)


#_______________________________________________________________________________
def eff_pt_tf(output_dir, ext, dir_name = "GEMCSCTriggerEfficiency"):

    c = TCanvas("c","c",1000,600 ) 
    c.cd()

    hegl = getEffHisto(f_def, hdir, hgl, hini, ptreb, kRed, 1, 2, htitle, rpt, yrange)
    hegl.Draw("same hist")
    heg = getEffHisto(f_def, hdir, hg, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange)
    heg.Draw("same hist")

    """
    hini = "h_pt_initial_1b"
    h2s = "h_pt_after_tfcand_eta1b_2s"
    h3s = "h_pt_after_tfcand_eta1b_3s"
    h2s1b = "h_pt_after_tfcand_eta1b_2s1b"
    h3s1b = "h_pt_after_tfcand_eta1b_3s1b"


    h_eff_tf0_2s  = getEffHisto(f_def, hdir, h2s, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange)
    h_eff_tf0_3s  = getEffHisto(f_def, hdir, h3s, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange)
    h_eff_tf0_2s1b  = getEffHisto(f_def, hdir, h2s1b, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange)
    h_eff_tf0_3s1b  = getEffHisto(f_def, hdir, h3s1b, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange)


    h_eff_tf10_2s  = getEffHisto(f_def, hdir, h2s + "_pt10", hini, ptreb, kGreen+4, 1, 2, htitle, rpt,yrange)
    h_eff_tf10_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange)
    h_eff_tf10_3s  = getEffHisto(f_def, hdir, h3s + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange)
    h_eff_tf10_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange)
    """
    c.SaveAs("test.png")


########################################################################################

# Requires only a single input file

########################################################################################

#_______________________________________________________________________________
def eff_pt_tf():

    dir = getRootDirectory(input_dir, file_name, dir_name)

    c = TCanvas("c","c",1000,600 ) 
    c.cd()

    h_eff_pt_after_mpc_ok_plus = setEffHisto("h_pt_after_mpc_ok_plus","h_pt_initial",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (1.2<#eta<2.1)","p_{T}","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus      = setEffHisto("h_pt_after_tfcand_ok_plus","h_pt_initial",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_pt10 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10","h_pt_initial",dir, ptreb, kBlue, 2,2, "","","",xrangept,yrange)
    
    h_eff_pt_after_mpc_ok_plus.Draw("hist")
    h_eff_pt_after_tfcand_ok_plus.Draw("same hist")
    h_eff_pt_after_tfcand_ok_plus_pt10.Draw("same hist")
    
    leg1 = TLegend(0.347,0.19,0.926,0.45,"","brNDC")
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetHeader("Eff. for #mu crossing ME1+one more station in 1.2<#eta<2.1 with")
    leg1.AddEntry(h_eff_pt_after_mpc_ok_plus,"MPC matched in 2stations","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_ok_plus,"TF track with matched stubs in 2st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10,"p_{T}^{TF}>10 TF track with matched stubs in 2st","pl")
    leg1.Draw()
    
    c.SaveAs("%sh_eff_pt_tf%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_pt_tf_eta1b_2s():

    dir = getRootDirectory(input_dir, file_name, dir_name)

    c = TCanvas("c","c",1000,600 ) 
    c.cd()
    
    h_eff_pt_after_tfcand_eta1b_2s  = setEffHisto("h_pt_after_tfcand_eta1b_2s","h_pt_initial_1b",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (1.64<#eta<2.14)","p_{T}^{MC}","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_2s_pt10 = setEffHisto("h_pt_after_tfcand_eta1b_2s_pt10","h_pt_initial_1b",dir, ptreb, kGreen+2, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_2s_pt20 = setEffHisto("h_pt_after_tfcand_eta1b_2s_pt20","h_pt_initial_1b",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_2s_pt25 = setEffHisto("h_pt_after_tfcand_eta1b_2s_pt25","h_pt_initial_1b",dir, ptreb, kOrange, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_2s_pt30 = setEffHisto("h_pt_after_tfcand_eta1b_2s_pt30","h_pt_initial_1b",dir, ptreb, kRed, 1,2, "","","",xrangept,yrange)
    
    h_eff_pt_after_tfcand_eta1b_2s.GetXaxis().SetRangeUser(0.,49.99)
    
    h_eff_pt_after_tfcand_eta1b_2s.Draw("hist")
    h_eff_pt_after_tfcand_eta1b_2s_pt10.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_2s_pt20.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_2s_pt25.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_2s_pt30.Draw("same hist")
    
    leg1 = TLegend(0.5,0.15,0.99,0.5,"","brNDC")
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetHeader("Eff. for #mu in 1.64<#eta<2.14 to have TF track with")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_2s,"stubs in (2+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_2s_pt10,"p_{T}^{TF}>=10, stubs in (2+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_2s_pt20,"p_{T}^{TF}>=20, stubs in (2+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_2s_pt25,"p_{T}^{TF}>=25, stubs in (2+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_2s_pt30,"p_{T}^{TF}>=30, stubs in (2+)st","pl")
    leg1.Draw()
    
    c.SaveAs("%sh_eff_pt_tf_eta1b_2s%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_pt_tf_eta1b_2s1b():

    dir = getRootDirectory(input_dir, file_name, dir_name)

    c = TCanvas("c","c",1000,600 ) 
    c.cd()
    
    h_eff_pt_after_tfcand_eta1b_2s1b  = setEffHisto("h_pt_after_tfcand_eta1b_2s1b","h_pt_initial_1b",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (1.64<#eta<2.14)","p_{T}^{MC}","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_2s1b_pt10 = setEffHisto("h_pt_after_tfcand_eta1b_2s1b_pt10","h_pt_initial_1b",dir, ptreb, kGreen+2, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_2s1b_pt20 = setEffHisto("h_pt_after_tfcand_eta1b_2s1b_pt20","h_pt_initial_1b",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_2s1b_pt25 = setEffHisto("h_pt_after_tfcand_eta1b_2s1b_pt25","h_pt_initial_1b",dir, ptreb, kOrange, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_2s1b_pt30 = setEffHisto("h_pt_after_tfcand_eta1b_2s1b_pt30","h_pt_initial_1b",dir, ptreb, kRed, 1,2, "","","",xrangept,yrange)
    
    h_eff_pt_after_tfcand_eta1b_2s1b.GetXaxis().SetRangeUser(0.,49.99)
    
    h_eff_pt_after_tfcand_eta1b_2s1b.Draw("hist")
    h_eff_pt_after_tfcand_eta1b_2s1b_pt10.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_2s1b_pt20.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_2s1b_pt25.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_2s1b_pt30.Draw("same hist")
    
    leg1 = TLegend(0.5,0.15,0.99,0.5,"","brNDC")
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetHeader("Eff. for #mu in 1.64<#eta<2.14 to have TF track with")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_2s1b,"stubs in ME1+(1+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_2s1b_pt10,"p_{T}^{TF}>=10, stubs in ME1+(1+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_2s1b_pt20,"p_{T}^{TF}>=20, stubs in ME1+(1+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_2s1b_pt25,"p_{T}^{TF}>=25, stubs in ME1+(1+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_2s1b_pt30,"p_{T}^{TF}>=30, stubs in ME1+(1+)st","pl")
    leg1.Draw()
    
    c.SaveAs("%sh_eff_pt_tf_eta1b_2s1b%s"%(output_dir,ext))
    

#_______________________________________________________________________________
def eff_pt_tf_eta1b_3s():
    
    dir = getRootDirectory(input_dir, file_name, dir_name)

    c = TCanvas("c","c",1000,600 ) 
    c.cd()

    h_eff_pt_after_tfcand_eta1b_3s  = setEffHisto("h_pt_after_tfcand_eta1b_3s","h_pt_initial_1b",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (1.64<#eta<2.14)","p_{T}^{MC}","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_3s_pt10 = setEffHisto("h_pt_after_tfcand_eta1b_3s_pt10","h_pt_initial_1b",dir, ptreb, kGreen+2, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_3s_pt20 = setEffHisto("h_pt_after_tfcand_eta1b_3s_pt20","h_pt_initial_1b",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_3s_pt25 = setEffHisto("h_pt_after_tfcand_eta1b_3s_pt25","h_pt_initial_1b",dir, ptreb, kOrange, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_3s_pt30 = setEffHisto("h_pt_after_tfcand_eta1b_3s_pt30","h_pt_initial_1b",dir, ptreb, kRed, 1,2, "","","",xrangept,yrange)
    
    h_eff_pt_after_tfcand_eta1b_3s.GetXaxis().SetRangeUser(0.,49.99)
    
    h_eff_pt_after_tfcand_eta1b_3s.Draw("hist")
    h_eff_pt_after_tfcand_eta1b_3s_pt10.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_3s_pt20.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_3s_pt25.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_3s_pt30.Draw("same hist")
    
    leg1 = TLegend(0.5,0.15,0.99,0.5,"","brNDC")
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetHeader("Eff. for #mu in 1.64<#eta<2.14 to have TF track with")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_3s,"stubs in (3+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_3s_pt10,"p_{T}^{TF}>=10, stubs in (3+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_3s_pt20,"p_{T}^{TF}>=20, stubs in (3+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_3s_pt25,"p_{T}^{TF}>=25, stubs in (3+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_3s_pt30,"p_{T}^{TF}>=30, stubs in (3+)st","pl")
    leg1.Draw()
    
    c.SaveAs("%sh_eff_pt_tf_eta1b_3s%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_pt_tf_eta1b_3s1b():
    
    dir = getRootDirectory(input_dir, file_name, dir_name)

    c = TCanvas("c","c",1000,600 ) 
    c.cd()

    h_eff_pt_after_tfcand_eta1b_3s1b  = setEffHisto("h_pt_after_tfcand_eta1b_3s1b","h_pt_initial_1b",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (1.64<#eta<2.14)","p_{T}^{MC}","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_3s1b_pt10 = setEffHisto("h_pt_after_tfcand_eta1b_3s1b_pt10","h_pt_initial_1b",dir, ptreb, kGreen+2, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_3s1b_pt20 = setEffHisto("h_pt_after_tfcand_eta1b_3s1b_pt20","h_pt_initial_1b",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_3s1b_pt25 = setEffHisto("h_pt_after_tfcand_eta1b_3s1b_pt25","h_pt_initial_1b",dir, ptreb, kOrange, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_3s1b_pt30 = setEffHisto("h_pt_after_tfcand_eta1b_3s1b_pt30","h_pt_initial_1b",dir, ptreb, kRed, 1,2, "","","",xrangept,yrange)
    
    h_eff_pt_after_tfcand_eta1b_3s1b.GetXaxis().SetRangeUser(0.,49.99)
    
    h_eff_pt_after_tfcand_eta1b_3s1b.Draw("hist")
    h_eff_pt_after_tfcand_eta1b_3s1b_pt10.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_3s1b_pt20.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_3s1b_pt25.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_3s1b_pt30.Draw("same hist")
    
    leg1 = TLegend(0.5,0.15,0.99,0.5,"","brNDC")
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetHeader("Eff. for #mu in 1.64<#eta<2.14 to have TF track with")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_3s1b,"stubs in ME1+(2+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_3s1b_pt10,"p_{T}^{TF}>=10, stubs in ME1+(2+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_3s1b_pt20,"p_{T}^{TF}>=20, stubs in ME1+(2+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_3s1b_pt25,"p_{T}^{TF}>=25, stubs in ME1+(2+)st","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_eta1b_3s1b_pt30,"p_{T}^{TF}>=30, stubs in ME1+(2+)st","pl")
    leg1.Draw()
    
    c.SaveAs("%sh_eff_pt_tf_eta1b_3s1b%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_pth_tf():
    
    dir = getRootDirectory(input_dir, file_name, dir_name)

    c = TCanvas("c","c",1000,600 ) 
    c.cd()
        
    h_eff_pth_after_mpc_ok_plus = setEffHisto("h_pth_after_mpc_ok_plus","h_pth_initial",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (2.1<#eta<2.4)","p_{T}","",xrangept,yrange)
    h_eff_pth_after_tfcand_ok_plus = setEffHisto("h_pth_after_tfcand_ok_plus","h_pth_initial",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange)
    h_eff_pth_after_tfcand_ok_plus_pt10 = setEffHisto("h_pth_after_tfcand_ok_plus_pt10","h_pth_initial",dir, ptreb, kBlue, 2,2, "","","",xrangept,yrange)
    
    h_eff_pth_after_mpc_ok_plus.Draw("hist")
    h_eff_pth_after_tfcand_ok_plus.Draw("same hist")
    h_eff_pth_after_tfcand_ok_plus_pt10.Draw("same hist")
    
    leg1 = TLegend(0.347,0.19,0.926,0.45,"","brNDC")
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetHeader("Eff. for #mu crossing ME1+one more station in 2.1<#eta<2.4 with")
    leg1.AddEntry(h_eff_pth_after_mpc_ok_plus,"MPC matched in 2stations","pl")
    leg1.AddEntry(h_eff_pth_after_tfcand_ok_plus,"TF track with matched stubs in 2st","pl")
    leg1.AddEntry(h_eff_pth_after_tfcand_ok_plus_pt10,"p_{T}^{TF}>10 TF track with matched stubs in 2st","pl")
    leg1.Draw()

    c.SaveAs("%sh_eff_pth_tf%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_pth_tf_3st1a():
    
    dir = getRootDirectory(input_dir, file_name, dir_name)
    
    c = TCanvas("c","c",1000,600 ) 
    c.cd()

    h_eff_pth_after_mpc_ok_plus = setEffHisto("h_pth_after_mpc_ok_plus","h_pth_initial",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (2.1<#eta<2.4)","p_{T}","",xrangept,yrange)
    h_eff_pth_after_tfcand_ok_plus_3st1a      = setEffHisto("h_pth_after_tfcand_ok_plus_3st1a","h_pth_initial",dir, ptreb, kBlue, 1,2, "eff(p_{T}^{MC}): TF studies (denom: 2MPCs, 2.1<#eta<2.4)","p_{T}","",xrangept,yrange)
    h_eff_pth_after_tfcand_ok_plus_pt10_3st1a = setEffHisto("h_pth_after_tfcand_ok_plus_pt10_3st1a","h_pth_initial",dir, ptreb, kBlue, 2,2, "","","",xrangept,yrange)
    
    h_eff_pth_after_mpc_ok_plus.Draw("hist")
    h_eff_pth_after_tfcand_ok_plus_3st1a.Draw("same hist")
    h_eff_pth_after_tfcand_ok_plus_pt10_3st1a.Draw("same hist")
    
    leg1 = TLegend(0.347,0.19,0.926,0.45,"","brNDC")
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetHeader("Eff. for #mu crossing ME1+one more station in 2.1<#eta<2.4 with")
    leg1.AddEntry(h_eff_pth_after_mpc_ok_plus,"MPC matched in 2stations","pl")
    leg1.AddEntry(h_eff_pth_after_tfcand_ok_plus_3st1a,"TF track with matched stubs in 3st","pl")
    leg1.AddEntry(h_eff_pth_after_tfcand_ok_plus_pt10_3st1a,"p_{T}^{TF}>10 TF track with matched stubs in 3st","pl")
    leg1.Draw()
    
    c.SaveAs("%sh_eff_pth_tf_3st1a%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_pt_tf_q():

    dir = getRootDirectory(input_dir, file_name, dir_name)

    c = TCanvas("c","c",1000,600) 
    c.cd()
    
    h_eff_pt_after_tfcand_ok_plus_q1   = setEffHisto("h_pt_after_tfcand_ok_plus_q1","h_pt_after_mpc_ok_plus",dir, ptreb, kBlue, 1,1, "eff(p_{T}^{MC}): TF quality studies (denom: 2MPCs, 1.2<#eta<2.1)","p_{T}^{MC}","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_q2   = setEffHisto("h_pt_after_tfcand_ok_plus_q2","h_pt_after_mpc_ok_plus",dir, ptreb, kCyan+2, 1,1, "eff(p_{T}^{MC}): TF quality studies (denom: 2MPCs, 1.2<#eta<2.1)","p_{T}^{MC}","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_q3   = setEffHisto("h_pt_after_tfcand_ok_plus_q3","h_pt_after_mpc_ok_plus",dir, ptreb, kMagenta+1, 1,1, "","","",xrangept,yrange)
    
    ##h_eff_pt_after_tfcand_ok_plus_pt10_q1   = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q1","h_pt_after_mpc_ok_plus",dir, ptreb, kBlue, 2,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_pt10_q2   = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q2","h_pt_after_mpc_ok_plus",dir, ptreb, kCyan+2, 2,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_pt10_q3   = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q3","h_pt_after_mpc_ok_plus",dir, ptreb, kMagenta+1, 2,2, "","","",xrangept,yrange)

    h_eff_pt_after_tfcand_ok_plus_q1.Draw("hist")
    ##h_eff_pt_after_tfcand_ok_plus_pt10_q1.Draw("same hist")
    h_eff_pt_after_tfcand_ok_plus_q2.Draw("same hist")
    ##h_eff_pt_after_tfcand_ok_plus_pt10_q2.Draw("same hist")
    h_eff_pt_after_tfcand_ok_plus_q3.Draw("same hist")
    h_eff_pt_after_tfcand_ok_plus_pt10_q3.Draw("same hist")
    
    leg1 = TLegend(0.347,0.19,0.926,0.45,"","brNDC")
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetNColumns(2)
    leg1.SetHeader("TF track with matched stubs in 2st and ")
    leg1.AddEntry(h_eff_pt_after_tfcand_ok_plus_q1,"Q#geq1","pl")
    ##leg1.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q1,"Q#geq1, p_{T}^{TF}>10","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_ok_plus_q2,"Q#geq2","pl")
    ##leg1.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q2,"Q#geq2, p_{T}^{TF}>10","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_ok_plus_q3,"Q=3","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q3,"Q=3, p_{T}^{TF}>10","pl")
    leg1.Draw()
    
    c.SaveAs("%sh_eff_pt_tf_q%s"%(output_dir,ext))
        

#_______________________________________________________________________________
def do_h_pt_after_tfcand_ok_plus_pt10():

    dir = getRootDirectory(input_dir, file_name, dir_name)
    
    c = TCanvas("c","c",1000,600) 
    c.cd()
    h_eff_pt_after_tfcand_ok_plus_pt10 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10","h_pt_after_tfcand_ok_plus",dir, ptreb, kBlue, 1,2, "p_{T}^{TF}>10 assignment eff(p_{T}^{MC}) studies (denom: any p_{T}^{TF}, 1.2<#eta<2.1)","p_{T}^{MC}","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_pt10_q2 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q2","h_pt_after_tfcand_ok_plus_q2",dir, ptreb, kCyan+2, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_pt10_q3 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q3","h_pt_after_tfcand_ok_plus_q3",dir, ptreb, kMagenta+1, 1,2, "","","",xrangept,yrange)
    
    h_eff_pt_after_tfcand_ok_plus_pt10.Draw("hist")
    h_eff_pt_after_tfcand_ok_plus_pt10_q2.Draw("same hist")
    h_eff_pt_after_tfcand_ok_plus_pt10_q3.Draw("same hist")
    
    leg1 = TLegend(0.347,0.19,0.926,0.45,"","brNDC")
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    leg1.SetHeader("for #mu with p_{T}>20 crossing ME1+one station to pass p_{T}^{TF}>10 with")
    leg1.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10,"any Q","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q2,"Q#geq2","pl")
    leg1.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q3,"Q=3","pl")
    leg1.Draw()
    
    c.SaveAs("%sh_eff_ptres_tf%s"%(output_dir,ext))


#_______________________________________________________________________________
if __name__ == "__main__":

    ## some global style settings
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

    input_dir = "files/"
    output_dir = "plots/"
    ext = ".png"

    ## global variables
    gMinEta = 1.45
    gMaxEta = 2.5
    xrangept = [0.,50.]
    yrange = [0,1]
    ptreb = 2

    ## input files
    f_def =      "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem_dphi0_pat2.root"
    f_g98_pt10 = "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt10_pat2.root"
    f_g98_pt15 = "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt15_pat2.root"
    f_g98_pt20 = "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt20_pat2.root"
    f_g98_pt30 = "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt30_pat2.root"
    f_g98_pt40 = "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt40_pat2.root"

    file_name = "hp_dimu_CMSSW_6_2_0_SLHC1_upgrade2019_pu000_w3_gem98_pt2-50_PU0_pt20_new_eff.root"
    file_name = "hp_dimu_CMSSW_6_2_0_SLHC1_upgrade2019_pu000_w3_gem98_pt2-50_PU0_pt0_new_eff_postBuxFix.root"

    
    reuseOutputDirectory = False
    if not reuseOutputDirectory:
        output_dir = mkdir("myTest")
    
    output_dir = "forPresentationPU000_Pt00_postBugFix_20131230_145747/"
    dir_name = 'GEMCSCTriggerEfficiency'
    """
    do_h_pt_after_tfcand_ok_plus_pt10(input_dir, f_g98_pt10, output_dir, ".png", "SimMuL1StrictAll")
    eff_pt_tf_q(input_dir, f_g98_pt10, output_dir, ".png", "SimMuL1StrictAll")
    eff_pth_tf_3st1a(input_dir, f_g98_pt10, output_dir, ".png", "SimMuL1StrictAll")
    eff_pth_tf(input_dir, f_g98_pt10, output_dir, ".png", "SimMuL1StrictAll")
    eff_pt_tf_eta1b_3s1b(input_dir, f_g98_pt10, output_dir, ".png", "SimMuL1StrictAll")
    eff_pt_tf_eta1b_3s(input_dir, f_g98_pt10, output_dir, ".png", "SimMuL1StrictAll")
    eff_pt_tf_eta1b_2s1b(input_dir, f_g98_pt10, output_dir, ".png", "SimMuL1StrictAll")
    eff_pt_tf_eta1b_2s(input_dir, f_g98_pt10, output_dir, ".png", "SimMuL1StrictAll")
    eff_pt_tf(input_dir, f_g98_pt10, output_dir, ".png", "SimMuL1StrictAll")
    """

    do_h_pt_after_tfcand_ok_plus_pt10()
    eff_pt_tf_q()
    eff_pth_tf_3st1a()
    eff_pth_tf()
    eff_pt_tf_eta1b_3s1b()
    eff_pt_tf_eta1b_3s()
    eff_pt_tf_eta1b_2s1b()
    eff_pt_tf_eta1b_2s()
    eff_pt_tf()

    

