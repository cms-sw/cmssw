from helpers import *
from ROOT import *

## Attempt to pyrootize the drawplot_gmtrt.C script

pdir = ""
dir = "SimMuL1StrictAll"

gem_dir = "gem/"
gem_label = "gem98"
pdir = "plots/"

gNPU=100
gNEvt=238000

gdy = [0.1, 2500]

ptscale = [-1.,  0., 1.5,  2., 2.5,  3., 3.5,  4., 4.5,  5.,  6.,  7.,  8.,  10.,  12.,  14.,
            16., 18., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 120., 140.]

ptscaleb = [1.5,  2., 2.5,  3., 3.5,  4., 4.5,  5.,  6.,  7.,  8.,  10.,  12.,  14.,
            16., 18., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 120., 140., 150.]

ptscaleb_ = [1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75,  5.5, 6.5, 7.5,  9., 11.,  13.,  15.,
             17.,  19., 22.5, 27.5, 32.5, 37.5, 42.5, 47.5,  55.,  65., 75., 85., 95., 110., 130., 150.]

def drawLumiBXPULabel():
    tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}")
    tex.SetNDC()
    tex.Draw()


def setHistoPt(f_name, name, cname, title, lcolor, lstyle, lwidth):
    print "Opening ", f_name
    f = TFile.Open(f_name)
    h0 = getH(f, dir, name)
    nb = h0.GetXaxis().GetNbins()
    h = TH1D(name+cname,title,30,ptscaleb)
    for b in range(1,nb+1):
        bc = h0.GetBinContent(b)
        if (bc==0):
            continue
        bin = h.GetXaxis().FindFixBin(h0.GetBinCenter(b))
        ##cout<<b<<" "<<bc<<" "<<bin<<endl
        h.SetBinContent(bin, bc)
    ##h.Sumw2()
    scale(h)

    return h


def setHistoPtRaw(f_name, name, cname, title, lcolor, lstyle, lwidth):
    print "Opening ", f_name
    f = TFile.Open(f_name)
    h0 = getH(f, dir, name)
    nb = h0.GetXaxis().GetNbins()
    h = h0.Clone(name+cname)
    ##h.Sumw2()
    scale(h)

    return h


def setHistoEta(f_name, name, cname, title, lcolor, lstyle, lwidth):
    print "Opening", f_name
    f = TFile.Open(f_name)

    h0 = getH(f, dir, name)
    nb = h0.GetXaxis().GetNbins()

    h = h0.Clone(name+cname)
    h.SetTitle(title)

    h.Sumw2()
    scale(h)
    
    h.SetLineColor(lcolor)
    ##h.SetFillColor(lcolor)
    h.SetLineStyle(lstyle)
    h.SetLineWidth(lwidth)

    h.SetTitle(title)

    ##h.GetXaxis().SetRangeUser(1.2, 2.4)
    h.GetYaxis().SetRangeUser(gdy[0],gdy[1])
    
    h.GetXaxis().SetTitleSize(0.055)
    h.GetXaxis().SetTitleOffset(1.05)
    h.GetXaxis().SetLabelSize(0.045)
    h.GetXaxis().SetLabelOffset(0.003)
    h.GetXaxis().SetTitleFont(62)
    h.GetXaxis().SetLabelFont(62)
    h.GetXaxis().SetMoreLogLabels(1)
    
    h.GetYaxis().SetTitleSize(0.055)
    h.GetYaxis().SetTitleOffset(0.9)
    h.GetYaxis().SetLabelSize(0.045)
    h.GetYaxis().SetTitleFont(62)
    h.GetYaxis().SetLabelFont(62)
    
    ##h.GetYaxis().SetLabelOffset(0.015)

    return h


def getPTHisto(f_name, dir_name, h_name, clone_suffix = "_cln"):
    f = TFile.Open(f_name)
    return f.Get(dir_name + "/" + h_name).Clone(h_name + clone_suffix)

def setPTHisto(h0, title, lcolor, lstyle, lwidth):
    nb = h0.GetXaxis().GetNbins()

    h = TH1D("%s_varpt"%(h0.GetName(), title, 30, ptscaleb_))
    for b in range(1,nb+1):
        bc = h0.GetBinContent(b)
        if (bc==0):
            continue
        bin = h.GetXaxis().FindFixBin(h0.GetBinCenter(b))
        h.SetBinContent(bin, bc)

    ## integrate the bins to get the rate vs pt cut!!
    for b in range(1,31): ## fixme this number is hard-coded to be 30
        ## should be independent of the number of bins!!
        h.SetBinContent(b, h.Integral(b,31))

    h.Sumw2()
    scale(h)
    
    h.SetLineColor(lcolor)
    h.SetFillColor(lcolor)
    h.SetLineStyle(lstyle)
    h.SetLineWidth(lwidth)
    
    h.SetTitle(title)
    
    h.GetXaxis().SetRangeUser(2, 129.)
    h.GetYaxis().SetRangeUser(gdy[0],gdy[1])
    
    h.GetXaxis().SetTitleSize(0.055)
    h.GetXaxis().SetTitleOffset(1.05)
    h.GetXaxis().SetLabelSize(0.045)
    h.GetXaxis().SetLabelOffset(0.003)
    h.GetXaxis().SetTitleFont(62)
    h.GetXaxis().SetLabelFont(62)
    h.GetXaxis().SetMoreLogLabels(1)
    
    h.GetYaxis().SetTitleSize(0.055)
    h.GetYaxis().SetTitleOffset(0.9)
    h.GetYaxis().SetLabelSize(0.045)
    h.GetYaxis().SetTitleFont(62)
    h.GetYaxis().SetLabelFont(62)
    
    return h

def setPTHisto(f_name, dir_name, h_name, clone_suffix, title, lcolor, lstyle, lwidth):
    h0 = getPTHisto(f_name, dir_name, h_name, clone_suffix)
    return setPTHisto(h0, title, lcolor, lstyle, lwidth)

def setHisto(f_name, name, cname, title, lcolor, lstyle, lwidth):
    print "Opening ", f_name
    f = TFile.Open(f_name)
    h0 = getH(f, dir, name)
    nb = h0.GetXaxis().GetNbins()
    ## FIXME -  the number of bins is hard-coded to be 30!!!
    h = TH1D(s_name + cname, title, 30, ptscaleb_)
    
    for b in range(1,nb+1):
        bc = h0.GetBinContent(b)
        if (bc==0): 
            continue
    bin = h.GetXaxis().FindFixBin(h0.GetBinCenter(b))
    h.SetBinContent(bin, bc)
    for b in range(1,31):
        h.SetBinContent(b, h.Integral(b,31))

    h.Sumw2()
    scale(h)
    
    h.SetLineColor(lcolor)
    h.SetFillColor(lcolor)
    h.SetLineStyle(lstyle)
    h.SetLineWidth(lwidth)
    
    h.SetTitle(title)
    
    h.GetXaxis().SetRangeUser(2, 129.)
    h.GetYaxis().SetRangeUser(gdy[0],gdy[1])
    
    h.GetXaxis().SetTitleSize(0.055)
    h.GetXaxis().SetTitleOffset(1.05)
    h.GetXaxis().SetLabelSize(0.045)
    h.GetXaxis().SetLabelOffset(0.003)
    h.GetXaxis().SetTitleFont(62)
    h.GetXaxis().SetLabelFont(62)
    h.GetXaxis().SetMoreLogLabels(1)
    
    h.GetYaxis().SetTitleSize(0.055)
    h.GetYaxis().SetTitleOffset(0.9)
    h.GetYaxis().SetLabelSize(0.045)
    h.GetYaxis().SetTitleFont(62)
    h.GetYaxis().SetLabelFont(62)
    
    ##h.GetYaxis().SetLabelOffset(0.015)
    
    return h

def setHistoRatio(num, denom, title = "", ymin=0.4, ymax=1.6, color = kRed+3):
    ratio = num.Clone("%s--%s_ratio"%(num.GetName(),denom.GetName()))
    ratio.Divide(num, denom, 1., 1.)
    ratio.SetTitle(title)
    
    ratio.GetYaxis().SetRangeUser(ymin, ymax)
    ratio.GetYaxis().SetTitle("ratio: (with GEM)/default")
    ratio.GetYaxis().SetTitleSize(.14)
    ratio.GetYaxis().SetTitleOffset(0.4)
    ratio.GetYaxis().SetLabelSize(.11)
    
    ##ratio.GetXaxis().SetMoreLogLabels(1)
    ratio.GetXaxis().SetTitle("p_{T}^{cut} [GeV/c]")
    ratio.GetXaxis().SetLabelSize(.11)
    ratio.GetXaxis().SetTitleSize(.14)
    ratio.GetXaxis().SetTitleOffset(1.3) 
    
    ratio.SetLineWidth(2)
    ratio.SetFillColor(color)
    ratio.SetLineColor(color)
    ratio.SetMarkerColor(color)
    ratio.SetMarkerStyle(20)
    ##ratio.Draw("e3")
    
    return ratio


def setHistoRatio2(num, denom, title = "", ymin=0.4, ymax=1.6):
    ratio = num.Clone("%s--%s_ratio"%(num.GetName(),denom.GetName()))
    ratio.Divide(num, denom, 1., 1.)
    ratio.SetTitle(title)
    ratio.GetYaxis().SetRangeUser(ymin, ymax)
    ratio.GetYaxis().SetTitle("ratio: with ME1a stub/without")
    ratio.GetYaxis().SetLabelSize(0.07)
    ratio.GetYaxis().SetTitleSize(0.07)
    ratio.GetYaxis().SetTitleOffset(0.6)
    ratio.GetXaxis().SetMoreLogLabels(1)

    ratio.SetLineWidth(2)
    ratio.SetFillColor(kRed+3)
    ratio.SetLineColor(kRed+3)
    ratio.SetMarkerColor(kRed+3)
    ratio.SetMarkerStyle(20)
    ##ratio.Draw("e3")
    
    return ratio


def drawPULabel(x=0.17, y=0.15, font_size=0.):                          
    tex = TLatex(x, y,"L=4*10^{34} (25ns PU100)")
    if (font_size > 0.): 
        tex.SetFontSize(font_size)
    tex.SetNDC()
    tex.Draw()
    return tex


def gem_rate_draw():
    gStyle.SetOptStat(0)
    gStyle.SetTitleStyle(0)
    ##gStyle.SetPadTopMargin(0.08)
    gStyle.SetTitleH(0.06)
    
    ptreb = 2
    hdir = "SimMuL1StrictAll"

    f_def = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat2.root"
    f_g98_pt10 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt10_pat2.root"
    f_g98_pt15 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt15_pat2.root"
    f_g98_pt20 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt20_pat2.root"
    f_g98_pt30 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt30_pat2.root"
    f_g98_pt40 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt40_pat2.root"
    
    sprintf(pdir,"%s", gem_dir.Data())
    
    rpt = [0.,49.99]
    
    htitle = "Efficiency for #mu in 1.6<|#eta|<2.12 to have TF track;p_{T}^{MC}"

    hini = "h_pt_initial_1b"
    h2s = "h_pt_after_tfcand_eta1b_2s"
    h3s = "h_pt_after_tfcand_eta1b_3s"
    h2s1b = "h_pt_after_tfcand_eta1b_2s1b"
    h3s1b = "h_pt_after_tfcand_eta1b_3s1b"
    

    ##gdy[0]=0 gdy[1]=7.
    ##if (vs_eta_minpt=="20") gdy[1]=10.
    miny = 0.01, maxy

    ### Trigger rate plots with PT >= 20
    
    vs_eta_minpt = "20"
    ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";
    
  
    h_rt_tf20_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, kAzure+2, 1, 2);
    h_rt_tf20_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kAzure+5, 1, 2);
    h_rt_tf20_gpt20_2s1b   = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kGreen+1, 7, 2);
    
    h_rt_tf20_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, kAzure+3, 1, 2);
    h_rt_tf20_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kAzure+6, 1, 2);
    h_rt_tf20_gpt20_3s1b   = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kGreen+3, 7, 2);
    
    h_rt_tf20_3s1ab   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kAzure+6, 1, 2);
    h_rt_tf20_gpt20_3s1ab   = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kGreen+3, 7, 2);
    

    ## cAll100 = TCanvas("cAll100","cAll100",800,600) ;
    ## cAll100.SetLogy(1);
    ## maxy = 300;## 45;
    ## h_rt_tf20_2s.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf20_2s1b.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf20_gpt20_2s1b.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf20_2s.Draw("hist e1");
    ## h_rt_tf20_gpt20_2s1b.Draw("hist e1 same");
    ## h_rt_tf20_2s.Draw("hist e1 same");
    ## h_rt_tf20_2s1b.Draw("hist e1 same");
    ## leg = TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
    ## leg.SetBorderSize(0);
    ## leg.SetFillStyle(0);
    ## leg.AddEntry(h_rt_tf20_2s,"Tracks: p_{T}>=20, 2+ stubs","");
    ## leg.AddEntry(h_rt_tf20_2s,"anywhere","l");
    ## leg.AddEntry(h_rt_tf20_2s1b,"with ME1 in 1.6<|#eta|<2.14","l");
    ## leg.AddEntry(h_rt_tf20_gpt20_2s1b,"with (ME1+GEM) in 1.6<|#eta|<2.14","l");
    ## leg.Draw();
    ## tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
    ## tex.SetNDC();
    ## tex.Draw();
    ## Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-2s-2s1b__gem-2s-2s1b.png").Data());

    ## cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
    ## gPad.SetGridx(1);
    ## gPad.SetGridy(1);
    ## gem_ratio = setHistoRatio(h_rt_tf20_gpt20_2s1b, h_rt_tf20_2s1b, "", 0.,1.8);
    ## gem_ratio.Draw("e1");
    ## Print(cAll100r, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-2s-2s1b__gem-2s-2s1b__ratio.png").Data());


    ##========================  3+ Stubs ================================##

    ## ((TCanvas*)gROOT.FindObject("cAll100")).cd();
    ## maxy = 30.;##10;
    ## h_rt_tf20_3s.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf20_3s1b.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf20_gpt20_3s1b.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf20_3s.Draw("hist e1");
    ## h_rt_tf20_gpt20_3s1b.Draw("hist e1 same");
    ## h_rt_tf20_3s.Draw("hist e1 same");
    ## h_rt_tf20_3s1b.Draw("hist e1 same");
    ## leg = TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
    ## leg.SetBorderSize(0);
    ## leg.SetFillStyle(0);
    ## leg.AddEntry(h_rt_tf20_3s,"Tracks: p_{T}>=20, 3+ stubs","");
    ## leg.AddEntry(h_rt_tf20_3s,"anywhere","l");
    ## leg.AddEntry(h_rt_tf20_3s1b,"with ME1 in 1.6<|#eta|<2.14","l");
    ## leg.AddEntry(h_rt_tf20_gpt20_3s1b,"with (ME1+GEM) in 1.6<|#eta|<2.14","l");
    ## leg.Draw();
    ## tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
    ## tex.SetNDC();
    ## tex.Draw();
    ## Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1b__gem-3s-3s1b.png").Data());

    ## cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
    ## gPad.SetGridx(1);gPad.SetGridy(1);
    ## gem_ratio = setHistoRatio(h_rt_tf20_gpt20_3s1b, h_rt_tf20_3s1b, "", 0.,1.8);
    ## gem_ratio.Draw("e1");
    ## Print(cAll100r, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1b__gem-3s-3s1b__ratio.png").Data());


    ##========================  3+ Stubs Including the region ME1/1a ================================##

    ##  (gROOT.FindObject("cAll100")).cd();
    ##  h_rt_tf20_3s.Draw("hist e1");
    ##  h_rt_tf20_gpt20_3s1ab.Draw("hist e1 same");
    ##  h_rt_tf20_3s.Draw("hist e1 same");
    ##  h_rt_tf20_3s1ab.Draw("hist e1 same");
    ##  maxy = 30.;
    ## 10;
    ##  h_rt_tf20_3s1ab.GetYaxis().SetRangeUser(miny,maxy);
    ##  h_rt_tf20_gpt20_3s1ab.GetYaxis().SetRangeUser(miny,maxy);
    ##  leg = TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
    ##  leg.SetBorderSize(0);
    ##  leg.SetFillStyle(0);
    ##  leg.AddEntry(h_rt_tf20_3s,"Tracks: p_{T}>=20, 3+ stubs","");
    ##  leg.AddEntry(h_rt_tf20_3s,"anywhere","l");
    ##  leg.AddEntry(h_rt_tf20_3s1ab,"with ME1 in 1.6<|#eta|","l");
    ##  leg.AddEntry(h_rt_tf20_gpt20_3s1ab,"with (ME1+GEM) in 1.6<|#eta|","l");
    ##  leg.Draw();
    ##   tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
    ##  tex.SetNDC();
    ##  tex.Draw();
    ##  Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab.png").Data());

    ## cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
    ## gPad.SetGridx(1);gPad.SetGridy(1);
    ## gem_ratio = setHistoRatio(h_rt_tf20_gpt20_3s1ab, h_rt_tf20_3s1ab, "", 0.,1.8);
    ## gem_ratio.Draw("e1");
    ##  Print(cAll100r, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab__ratio.png").Data());

    ### Trigger rate plots with PT >= 20

    vs_eta_minpt = "30";
    ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";
    
    ## h_rt_tf30_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, kAzure+2, 1, 2);
    ## h_rt_tf30_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kAzure+5, 1, 2);
    ## h_rt_tf30_gpt30_2s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kGreen+1, 7, 2);
    h_rt_tf30_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, kAzure+3, 1, 2);
    h_rt_tf30_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kAzure+6, 1, 2);
    h_rt_tf30_3s1ab   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kAzure+2, 1, 2);
    h_rt_tf30_gpt30_3s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kGreen+3, 7, 2);
    h_rt_tf30_gpt30_3s1ab   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kGreen, 7, 2);
    
    ## cAll100 = TCanvas("cAll100","cAll100",800,600) ;
    ## cAll100.SetLogy(1);
    ## h_rt_tf30_2s.Draw("hist e1");
    ## h_rt_tf30_gpt30_2s1b.Draw("hist e1 same");
    ## h_rt_tf30_2s.Draw("hist e1 same");
    ## h_rt_tf30_2s1b.Draw("hist e1 same");
    ## maxy = 120.;##35.;
    ## h_rt_tf30_2s.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf30_2s1b.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf30_gpt30_2s1b.GetYaxis().SetRangeUser(miny,maxy);
    ## leg = TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
    ## leg.SetBorderSize(0);
    ## leg.SetFillStyle(0);
    ## leg.AddEntry(h_rt_tf30_2s,"Tracks: p_{T}>=30, 2+ stubs","");
    ## leg.AddEntry(h_rt_tf30_2s,"anywhere","l");
    ## leg.AddEntry(h_rt_tf30_2s1b,"with ME1 in 1.6<|#eta|<2.14","l");
    ## leg.AddEntry(h_rt_tf30_gpt30_2s1b,"with (ME1+GEM) in 1.6<|#eta|<2.14","l");
    ## leg.Draw();
    ##  tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
    ## tex.SetNDC();
    ## tex.Draw();
    ## Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-2s-2s1b__gem-2s-2s1b.png").Data());

    ## cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
    ## gPad.SetGridx(1);gPad.SetGridy(1);
    ## gem_ratio = setHistoRatio(h_rt_tf30_gpt30_2s1b, h_rt_tf30_2s1b, "", 0.,1.8);
    ## gem_ratio.Draw("e1");
    ## Print(cAll100r, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-2s-2s1b__gem-2s-2s1b__ratio.png").Data());


    ## (gROOT.FindObject("cAll100")).cd();
    ## h_rt_tf30_3s.Draw("hist e1");
    ## h_rt_tf30_gpt30_3s1b.Draw("hist e1 same");
    ## h_rt_tf30_3s.Draw("hist e1 same");
    ## h_rt_tf30_3s1b.Draw("hist e1 same");
    ## maxy = 30.;##7.;
    ## h_rt_tf30_3s.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf30_3s1b.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf30_gpt30_3s1b.GetYaxis().SetRangeUser(miny,maxy);
    ## leg = TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
    ## leg.SetBorderSize(0);
    ## leg.SetFillStyle(0);
    ## leg.AddEntry(h_rt_tf30_3s,"Tracks: p_{T}>=30, 3+ stubs","");
    ## leg.AddEntry(h_rt_tf30_3s,"anywhere","l");
    ## leg.AddEntry(h_rt_tf30_3s1b,"with ME1 in 1.6<|#eta|<2.14","l");
    ## leg.AddEntry(h_rt_tf30_gpt30_3s1b,"with (ME1+GEM) in 1.6<|#eta|<2.14","l");
    ## leg.Draw();
    ## tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
    ## tex.SetNDC();
    ## tex.Draw();
    ## Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1b__gem-3s-3s1b.png").Data());

    ## cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
    ## gPad.SetGridx(1);gPad.SetGridy(1);
    ## gem_ratio = setHistoRatio(h_rt_tf30_gpt30_3s1b, h_rt_tf30_3s1b, "", 0.,1.8);
    ## gem_ratio.Draw("e1");
    ## Print(cAll100r, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1b__gem-3s-3s1b__ratio.png").Data());


    ##==========================  Including the   ME1a  ==========================##

    ## (gROOT.FindObject("cAll100")).cd();
    ## h_rt_tf30_3s.Draw("hist e1");
    ## h_rt_tf30_gpt30_3s1ab.Draw("hist e1 same");
    ## h_rt_tf30_3s.Draw("hist e1 same");
    ## h_rt_tf30_3s1ab.Draw("hist e1 same");
    ## h_rt_tf30_3s.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf30_3s1ab.GetYaxis().SetRangeUser(miny,maxy);
    ## h_rt_tf30_gpt30_3s1ab.GetYaxis().SetRangeUser(miny,maxy);
    ## leg = TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
    ## leg.SetBorderSize(0);
    ## leg.SetFillStyle(0);
    ## leg.AddEntry(h_rt_tf30_3s,"Tracks: p_{T}>=30, 3+ stubs","");
    ## leg.AddEntry(h_rt_tf30_3s,"anywhere","l");
    ## leg.AddEntry(h_rt_tf30_3s1ab,"with ME1 in 1.6<|#eta|","l");
    ## leg.AddEntry(h_rt_tf30_gpt30_3s1ab,"with (ME1+GEM) in 1.6<|#eta|<2.14","l");
    ## leg.Draw();
    ## tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
    ## tex.SetNDC();
    ## tex.Draw();
    ## Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab.png").Data());

    ## cAll100r2 = TCanvas("cAll100r2","cAll100r2",800,300) ;
    ## gPad.SetGridx(1);gPad.SetGridy(1);
    ## gem_ratio = setHistoRatio(h_rt_tf30_gpt30_3s1ab, h_rt_tf30_3s1ab, "", 0.,1.8);
    ## gem_ratio.Draw("e1");
    ## Print(cAll100r2, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab__ratio.png").Data());

    ##==========================  Comparison with/withous Stub in ME1a ==========================##

    ##  (gROOT.FindObject("cAll100")).cd();
    ##  h_rt_tf30_3s1b.Draw("hist e1");
    ##  h_rt_tf30_3s1ab.Draw("hist e1 same");
    ##  leg = TLegend(0.2,0.65,.80,0.90,NULL,"brNDC");
    ##  leg.SetBorderSize(0);
    ##  leg.SetFillStyle(0);
    ##  leg.AddEntry(h_rt_tf30_3s1b,"Tracks: p_{T}>=30, 3+ stubs","");
    ##  leg.AddEntry(h_rt_tf30_3s1b,"with ME1 in 1.6<|#eta|<2.14","l");
    ##  leg.AddEntry(h_rt_tf30_3s1ab,"with ME1 in 1.6<|#eta|<2.4","l");
    ##  leg.Draw();
    ##  Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab_compstubME1a.png").Data());
  
    ##  cAll100r2 = TCanvas("cAll100r2","cAll100r2",800,300) ;
    ##  gPad.SetGridx(1);gPad.SetGridy(1);
    ##  gem_ratio = setHistoRatio2(h_rt_tf30_3s1ab, h_rt_tf30_3s1b, "", 0.,1.8);
    ##  gem_ratio.Draw("e1");
    ##  Print(cAll100r2, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab__ratio_compstubME1a.png").Data());

    ##==========================  Comparison with/withous Stub in ME1a + GEMS ==========================##
  
    """
    (gROOT.FindObject("cAll100")).cd();
    h_rt_tf30_gpt30_3s1b.Draw("hist e1");
    h_rt_tf30_gpt30_3s1ab.Draw("hist e1 same");
    leg = TLegend(0.2,0.65,.80,0.90,NULL,"brNDC");
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.AddEntry(h_rt_tf30_3s1b,"Tracks: p_{T}>=30, 3+ stubs","");
    leg.AddEntry(h_rt_tf30_3s1b,"with ME1 in 1.6<|#eta|<2.14","l");
    leg.AddEntry(h_rt_tf30_3s1ab,"with ME1 in 1.6<|#eta|<2.4","l");
    leg.Draw();
    Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab_compstubME1a.png").Data());
    
    cAll100r2 = TCanvas("cAll100r2","cAll100r2",800,300) ;
    gPad.SetGridx(1);gPad.SetGridy(1);
    gem_ratio = setHistoRatio2(h_rt_tf30_3s1ab, h_rt_tf30_3s1b, "", 0.,1.8);
    gem_ratio.Draw("e1");
    Print(cAll100r2, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab__ratio_compstubME1a.png").Data());
    """

def drawplot_gmtrt(dname = "", vs_eta_minpt = ""):
        
    ##gStyle.SetStatW(0.13);
    ##gStyle.SetStatH(0.08);
    gStyle.SetStatW(0.07);
    gStyle.SetStatH(0.06);
    gStyle.SetOptStat(0);
    gStyle.SetTitleStyle(0);
    gStyle.SetTitleAlign(13);## coord in top left
    gStyle.SetTitleX(0.);
    gStyle.SetTitleY(1.);
    gStyle.SetTitleW(1);
    gStyle.SetTitleH(0.058);
    gStyle.SetTitleBorderSize(0);
    gStyle.SetPadLeftMargin(0.126);
    gStyle.SetPadRightMargin(0.04);
    gStyle.SetPadTopMargin(0.06);
    gStyle.SetPadBottomMargin(0.13);
    gStyle.SetMarkerStyle(1);

    """
    d1=""
    d2=""
    if (dname != ""):
    sprintf(d1,"_%s", dname.Data());
    ##if (strcmp(pu,"")>0)    sprintf(d2,"_%s",pu);
    ##sprintf(pdir,"pic%s%s",d1,d2);
    
    if (interactive && dname != "") {
    sprintf(pdir,"pic%s%s",d1,d2);
    if( gSystem.AccessPathName(pdir)==0 ) {
    ##cout<<"directory "<<pdir<<" exists, removing it!"<<endl;
    cmd[111];
    ##sprintf(cmd,"rm -r %s",pdir);
    ##if (gSystem.Exec(cmd) != 0) {cout<<"can't remode directory! exiting..."<<endl; return;};
    }
    else {
    cout<<" creating directory "<<pdir<<endl;
    gSystem.MakeDirectory(pdir);
    }  
    }
    ##cout<<"opening "<<fname<<endl;
    ##f = TFile::Open(fname);
    """

    ## directory inside of root file:
    dir = "SimMuL1StrictAll"
    
    f_pu100_pat8 = gem_dir;
    f_pu100_pat8_gem = gem_dir;

    ## f_pu100_pat2 +=     "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_pat2.root";
    ## f_pu100_pat8 +=     "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_pat8.root";
    ## f_pu100_pat2_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem_pat2.root";
    ## f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem_pat8.root";

    if (dname.Contains("_pat8")):      f_pu100_pat8 +=     "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat8.root";
    if (dname == "minbias_pt05_pat8"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt05_pat8.root";
    if (dname == "minbias_pt06_pat8"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt06_pat8.root";
    if (dname == "minbias_pt10_pat8"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt10_pat8.root";
    if (dname == "minbias_pt15_pat8"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt15_pat8.root";
    if (dname == "minbias_pt20_pat8"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt20_pat8.root";
    if (dname == "minbias_pt30_pat8"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt30_pat8.root";
    if (dname == "minbias_pt40_pat8"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt40_pat8.root";
    
    if (dname.Contains("_pat2")):      f_pu100_pat8 +=     "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat2.root";
    if (dname == "minbias_pt05_pat2"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt05_pat2.root";
    if (dname == "minbias_pt06_pat2"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt06_pat2.root";
    if (dname == "minbias_pt10_pat2"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt10_pat2.root";
    if (dname == "minbias_pt15_pat2"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt15_pat2.root";
    if (dname == "minbias_pt20_pat2"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt20_pat2.root";
    if (dname == "minbias_pt30_pat2"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt30_pat2.root";
    if (dname == "minbias_pt40_pat2"): f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt40_pat2.root";
    
    ###################################### PU100 ####################################
    
    ## full eta 1. - 2.4    Default: 3station, 2s & 1b    GEM: 3station, 2s & 1b
    gdy[0]=0 
    gdy[1]=7.
    if (vs_eta_minpt=="20") gdy[1]=10.;
    
    ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";
    hAll100    = setHistoEta(f_pu100_pat8,     "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_2s1b", "_hAll100", ttl, kAzure+9, 1, 2);
    hAll100gem = setHistoEta(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_2s1b", "_hAll100gem", ttl, kGreen+3, 1, 2);
    
     ##gStyle.SetPadTopMargin(0.08);
    gStyle.SetTitleH(0.06);
    cAll100 = TCanvas("cAll100","cAll100",800,600) ;
    hAll100.Draw("hist e1");
    hAll100gem.Draw("hist e1 same");
    leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
    const TObject obj;
    leg_cc100.SetBorderSize(0);
    ##leg_cc100.SetTextSize(0.0368);
    leg_cc100.SetFillStyle(0);
    leg_cc100.AddEntry(hAll100,"default emulator","l");
    leg_cc100.AddEntry(hAll100gem,"with GEM match","l");
    leg_cc100.AddEntry(hAll100,"Tracks req. for both:","");
    leg_cc100.AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
    leg_cc100.AddEntry(hAll100,"except in ME1/b region require","");
    leg_cc100.AddEntry(hAll100,">=2 stubs and one of them from ME1/b","");
    leg_cc100.Draw();
    
    tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
    tex.SetNDC();
    tex.Draw();
    
    Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-2s1b__gem-3s-2s1b.png").Data());
    
    if (do_return) return;


    ## full eta 1. - 2.4    Default: 3station, 3s & 1b    GEM: 3station, 3s & 1b
      gdy[0]=0; gdy[1]=7.;
      if (vs_eta_minpt=="20") gdy[1]=10.;

      ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";
      hAll100    = setHistoEta(f_pu100_pat8,     "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kAzure+9, 1, 2);
      hAll100gem = setHistoEta(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100gem", ttl, kGreen+3, 1, 2);

        ##gStyle.SetPadTopMargin(0.08);
      gStyle.SetTitleH(0.06);
      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      hAll100.Draw("hist e1");
      hAll100gem.Draw("hist e1 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","l");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","l");
      leg_cc100.AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100.AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100.AddEntry(hAll100,"in ME1/b region also require","");
      leg_cc100.AddEntry(hAll100,"one stub to be from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1b__gem-3s-3s1b.png").Data());

    ## full eta 1. - 2.4    Default: 3station    GEM: 3station, 2s & 1b
      gdy[0]=0; gdy[1]=7.;
      if (vs_eta_minpt=="20") gdy[1]=10.;

      ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";
      hAll100    = setHistoEta(f_pu100_pat8,     "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, kAzure+9, 1, 2);
      hAll100gem = setHistoEta(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_2s1b", "_hAll100gem", ttl, kGreen+3, 1, 2);

        ##gStyle.SetPadTopMargin(0.08);
      gStyle.SetTitleH(0.06);
      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      hAll100.Draw("hist e1");
      hAll100gem.Draw("hist e1 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","l");
      leg_cc100.AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","l");
      leg_cc100.AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100.AddEntry(hAll100,"except in ME1/b region require","");
      leg_cc100.AddEntry(hAll100,">=2 stubs and one of them from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s__gem-3s-2s1b.png").Data());

    ## full eta 1. - 2.4    Default: 3station    GEM: 3station, 3s & 1b
      gdy[0]=0; gdy[1]=7.;
      if (vs_eta_minpt=="20") gdy[1]=10.;

      ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";
      hAll100    = setHistoEta(f_pu100_pat8,     "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, kAzure+9, 1, 2);
      hAll100gem = setHistoEta(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100gem", ttl, kGreen+3, 1, 2);

        ##gStyle.SetPadTopMargin(0.08);
      gStyle.SetTitleH(0.06);
      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      hAll100.Draw("hist e1");
      hAll100gem.Draw("hist e1 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","l");
      leg_cc100.AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","l");
      leg_cc100.AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100.AddEntry(hAll100,"in ME1/b region also require","");
      leg_cc100.AddEntry(hAll100,"one stub to be from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s__gem-3s-3s1b.png").Data());

      if (do_return) return;


    ## full eta 1. - 2.4    Default: 3station, 2s & 1b    GEM: 3station, 2s & 1b
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_2s1b", "_hAll100", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_2s1b", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      gPad.SetLogx(1);gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);
      hAll100.Draw("e3");
      hAll100gem.Draw("e3 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","f");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100.AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100.AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100.AddEntry(hAll100,"except in ME1/b region require","");
      leg_cc100.AddEntry(hAll100,">=2 stubs and one of them from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, "rates__1-2.4__PU100__def-3s-2s1b__gem-3s-2s1b.png");


      cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
      gPad.SetLogx(1);##gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.4);
      hAll100gem_ratio.Draw("e1");

      Print(cAll100r, "rates__1-2.4__PU100__def-3s-2s1b__gem-3s-2s1b__ratio.png");

    ## full eta 1. - 2.4    Default: 3station   GEM: 3station, 2s & 1b
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s", "_hAll100s3", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_2s1b", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      gPad.SetLogx(1);gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);
      hAll100.Draw("e3");
      hAll100gem.Draw("e3 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","f");
      leg_cc100.AddEntry(hAll100,"Tracks: with >=3 stubs in 1<|#eta|<2.4","");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100.AddEntry(hAll100,"Tracks: same, except in ME1/b region req.","");
      leg_cc100.AddEntry(hAll100,">=2 stubs and one of them from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, "rates__1-2.4__PU100__def-3s__gem-3s-2s1b.png");


      cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
      gPad.SetLogx(1);##gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.2);
      hAll100gem_ratio.Draw("e1");

      Print(cAll100r, "rates__1-2.4__PU100__def-3s__gem-3s-2s1b__ratio.png");


    ## no ME1/a eta 1. - 2.1    Default: 3station   GEM: 3station, 3s & 1b
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_no1a", "_hAll100s3", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      gPad.SetLogx(1);gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);
      hAll100.Draw("e3");
      hAll100gem.Draw("e3 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","f");
      leg_cc100.AddEntry(hAll100,"Tracks: with >=3 stubs in 1<|#eta|<2.1","");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100.AddEntry(hAll100,"Tracks: same, except in ME1/b region req.","");
      leg_cc100.AddEntry(hAll100,">=3 stubs and one of them from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, "rates__1-2.1__PU100__def-3s__gem-3s-3s1b.png");


      cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
      gPad.SetLogx(1);
      gPad.SetGridx(1);gPad.SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.2);
      hAll100gem_ratio.Draw("e1");

      Print(cAll100r, "rates__1-2.1__PU100__def-3s__gem-3s-3s1b__ratio.png");


      result_gem_eta_no1a = hAll100gem;
      result_def_eta_no1a = hAll100;

    ## no ME1/a eta 1. - 2.1    Default: 3station, 3s & 1b   GEM: 3station, 3s & 1b
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_hAll100s3", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      gPad.SetLogx(1);gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);
      hAll100.Draw("e3");
      hAll100gem.Draw("e3 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","f");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100.AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100.AddEntry(hAll100,"with >=3 stubs in 1<|#eta|<2.1","");
      leg_cc100.AddEntry(hAll100,"for ME1/b etas require one stub from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, "rates__1-2.1__PU100__def-3s-3s1b__gem-3s-3s1b.png");


      cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
      gPad.SetLogx(1);
      gPad.SetGridx(1);gPad.SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.2);
      hAll100gem_ratio.Draw("e1");

      Print(cAll100r, "rates__1-2.1__PU100__def-3s-3s1b__gem-3s-3s1b__ratio.png");

        ##result_gem_eta_no1a = hAll100gem;
      result_def_eta_no1a_3s1b = hAll100;


    ## Full eta 1. - 2.4    Default: 3station   GEM: 3station, 3s & 1b
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s", "_hAll100s3", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      gPad.SetLogx(1);gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);
      hAll100.Draw("e3");
      hAll100gem.Draw("e3 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","f");
      leg_cc100.AddEntry(hAll100,"Tracks: with >=3 stubs in 1<|#eta|<2.4","");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100.AddEntry(hAll100,"Tracks: same, except","");
      leg_cc100.AddEntry(hAll100,"for ME1/b etas require one stub from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, "rates__1-2.4__PU100__def-3s__gem-3s-3s1b.png");


      cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
      gPad.SetLogx(1);
      gPad.SetGridx(1);gPad.SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.2);
      hAll100gem_ratio.Draw("e1");

      Print(cAll100r, "rates__1-2.4__PU100__def-3s__gem-3s-3s1b__ratio.png");

      result_gem_eta_all = hAll100gem;
      result_def_eta_all = hAll100;


    ## Full eta 1. - 2.4    Default: 3station, 3s & 1b   GEM: 3station, 3s & 1b
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_3s1b", "_hAll100s3", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      gPad.SetLogx(1);gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);
      hAll100.Draw("e3");
      hAll100gem.Draw("e3 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","f");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100.AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100.AddEntry(hAll100,"with >=3 stubs in 1<|#eta|<2.4","");
      leg_cc100.AddEntry(hAll100,"for ME1/b etas require one stub from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, "rates__1-2.4__PU100__def-3s-3s1b__gem-3s-3s1b.png");


      cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
      gPad.SetLogx(1);
      gPad.SetGridx(1);gPad.SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.6);
      hAll100gem_ratio.Draw();

      Print(cAll100r, "rates__1-2.4__PU100__def-3s-3s1b__gem-3s-3s1b__ratio.png");

        ##result_gem_eta_all = hAll100gem;
      result_def_eta_all_3s1b = hAll100;



    ## ME1b eta 1.64 - 2.14    Default: 3station, 3s & 1b   GEM: 3station, 3s & 1b
      gdy[0]=0.02; gdy[1]=1000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_hAll100s3", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_hAll100gem", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      gPad.SetLogx(1);gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);
      hAll100.Draw("e3");
      hAll100gem.Draw("e3 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","f");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100.AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100.AddEntry(hAll100,"with >=3 stubs in 1.64<|#eta|<2.14","");
      leg_cc100.AddEntry(hAll100,"and require one stub to be from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b.png");


      cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
      gPad.SetLogx(1);
      gPad.SetGridx(1);gPad.SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.,2.1);
      hAll100gem_ratio.Draw("e1");

      Print(cAll100r, "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__ratio.png");

      result_gem = hAll100gem;
      result_def_3s1b = hAll100;


    ## ME1b eta 1.64 - 2.14    Default: 3station, 3s   GEM: 3station, 3s & 1b
      gdy[0]=0.02; gdy[1]=1000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_1b", "_hAll100s3", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_hAll100gem", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      gPad.SetLogx(1);gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);
      hAll100.Draw("e3");
      hAll100gem.Draw("e3 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","f");
      leg_cc100.AddEntry(hAll100,"Tracks: with >=3 stubs in 1.64<|#eta|<2.14","");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100.AddEntry(hAll100,"Tracks: same, plus req. one stub from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b.png");


      cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
      gPad.SetLogx(1);
      gPad.SetGridx(1);gPad.SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.,1.1);
      hAll100gem_ratio.Draw("e1");

      Print(cAll100r, "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__ratio.png");

        ##result_gem = hAll100gem;
      result_def = hAll100;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_2s_1b", "_hAll100s2", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      result_def_2s = hAll100;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_2s_1b", "_hAll100s2", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      result_def_2s = hAll100;


    ## ME1b eta 1.64 - 2.14    Default: 3station, 3s   GMT single trigg
      gdy[0]=0.02; gdy[1]=1000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_1b", "_hAll100s3", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_ptmax_sing_1b", "_hAll100gem", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+1, 1, 1);

      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      gPad.SetLogx(1);gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);
      hAll100.Draw("e3");
      hAll100gem.Draw("e3 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"Tracks: with >=3 stubs in 1.64<|#eta|<2.14","");
      leg_cc100.AddEntry(hAll100gem,"GMT selection for Single Trigger","f");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, "rates__1.6-2.1_PU100__def-3s__gmtsing.png");

      result_def_gmtsing = hAll100gem;

    ## ME1b eta 1.64 - 2.14    Default: 3station, 2s & 1b   GEM: 3station, 2s & 1b
      gdy[0]=0.02; gdy[1]=1000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_hAll100s3", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_hAll100gem", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      cAll100 = TCanvas("cAll100","cAll100",800,600) ;
      gPad.SetLogx(1);gPad.SetLogy(1);
      gPad.SetGridx(1);gPad.SetGridy(1);
      hAll100.Draw("e3");
      hAll100gem.Draw("e3 same");
      leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100.SetBorderSize(0);
        ##leg_cc100.SetTextSize(0.0368);
      leg_cc100.SetFillStyle(0);
      leg_cc100.AddEntry(hAll100,"default emulator","f");
      leg_cc100.AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100.AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100.AddEntry(hAll100,"with >=2 stubs in 1.64<|#eta|<2.14","");
      leg_cc100.AddEntry(hAll100,"and require one stub to be from ME1/b","");
      leg_cc100.Draw();

       tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex.SetNDC();
      tex.Draw();

      Print(cAll100, "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b.png");


      cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
      gPad.SetLogx(1);
      gPad.SetGridx(1);gPad.SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.,2.1);
      hAll100gem_ratio.Draw("e1");

      Print(cAll100r, "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b__ratio.png");

      result_gem_2s1b = hAll100gem;
      result_def_2s1b = hAll100;

 
  """

    ## ME1b eta 1.64 - 2.14    Default: 3station, 3s   GEM: 3station, 2s & 1b
  if (1)
  {
  gdy[0]=0.02; gdy[1]=1000.;

  hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_1b", "_hAll100s3", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
  hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_hAll100gem", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

  cAll100 = TCanvas("cAll100","cAll100",800,600) ;
  gPad.SetLogx(1);gPad.SetLogy(1);
  gPad.SetGridx(1);gPad.SetGridy(1);
  hAll100.Draw("e3");
  hAll100gem.Draw("e3 same");
  leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
  const TObject obj;
  leg_cc100.SetBorderSize(0);
    ##leg_cc100.SetTextSize(0.0368);
  leg_cc100.SetFillStyle(0);
  leg_cc100.AddEntry(hAll100,"default emulator","f");
  leg_cc100.AddEntry(hAll100,"Tracks: with >=3 stubs in 1.64<|#eta|<2.14","");
  leg_cc100.AddEntry(hAll100gem,"with GEM match","f");
  leg_cc100.AddEntry(hAll100gem,"Tracks: with >=2 stubs in 1.64<|#eta|<2.14","");
  leg_cc100.AddEntry(hAll100gem,"plus req. one stub from ME1/b","");
  leg_cc100.Draw();

   tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
  tex.SetNDC();
  tex.Draw();

  Print(cAll100, "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b.png");


  cAll100r = TCanvas("cAll100r","cAll100r",800,300) ;
  gPad.SetLogx(1);
  gPad.SetGridx(1);gPad.SetGridy(1);

  hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.,1.1);
  hAll100gem_ratio.Draw("e1");

  Print(cAll100r, "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b__ratio.png");

  result_gem_2s1b = hAll100gem;
    ##result_def = hAll100;

  if (do_return) return;
  }

  """



"""

  .L drawplot_gmtrt.C
  drawplot_gmtrt("minbias_pt10_pat2")
  hh = (TH1D*)result_gem.Clone("gem_new")
  hh.SetFillColor(kGreen+4)
  for (b = hh.FindBin(15); b <= hh.GetNbinsX(); ++b) hh.SetBinContent(b, 0);
  drawplot_gmtrt("minbias_pt15_pat2")
  h15 = (TH1D*)result_gem.Clone("gem15")
  for (b = h15.FindBin(15); b < h15.FindBin(20); ++b) hh.SetBinContent(b, h15.GetBinContent(b));
  drawplot_gmtrt("minbias_pt20_pat2")
  h20 = (TH1D*)result_gem.Clone("gem20")
  for (b = h20.FindBin(20); b < h20.FindBin(30); ++b) hh.SetBinContent(b, h20.GetBinContent(b));
  drawplot_gmtrt("minbias_pt30_pat2")
  h30 = (TH1D*)result_gem.Clone("gem30")
  for (b = h30.FindBin(30); b <= h30.GetNbinsX(); ++b) hh.SetBinContent(b, h30.GetBinContent(b));
  for (b = 1; b <= hh.GetNbinsX(); ++b) if (hh.GetBinContent(b)==0) hh.SetBinError(b, 0.);

  (gROOT.FindObject("cAll100")).cd();
  result_def.Draw("e3");
  hh.Draw("same e3");

  leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
  const TObject obj;
  leg_cc100.SetBorderSize(0);
  leg_cc100.SetFillStyle(0);
  leg_cc100.AddEntry(result_def,"default emulator","f");
  leg_cc100.AddEntry(result_def,"Tracks: with >=3 stubs in 1.14<|#eta|<2.14","");
  leg_cc100.AddEntry(hh,"with GEM match","f");
  leg_cc100.AddEntry(result_def,"Tracks: same, plus req. one stub from ME1/b","");
  leg_cc100.Draw();

   tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
  tex.SetNDC();
  tex.Draw();

  gPad.Print("gem/rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2.png")


  (gROOT.FindObject("cAll100r")).cd();
  hh_ratio = setHistoRatio(hh, result_def, "", 0.,1.1);
  hh_ratio.Draw("e1");
  gPad.Print("gem/rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio.png")


  (gROOT.FindObject("cAll100")).cd();
  result_def_3s1b.Draw("e3")
  hh.Draw("same e3")

  leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
  const TObject obj;
  leg_cc100.SetBorderSize(0);
  leg_cc100.SetFillStyle(0);
  leg_cc100.AddEntry(result_def_3s1b,"default emulator","f");
  leg_cc100.AddEntry(hh,"with GEM match","f");
  leg_cc100.AddEntry(result_def_3s1b,"Tracks req. for both:","");
  leg_cc100.AddEntry(result_def_3s1b,"with >=3 stubs in 1.64<|#eta|<2.14","");
  leg_cc100.AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
  leg_cc100.Draw();

   tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
  tex.SetNDC();
  tex.Draw();

  gPad.Print("gem/rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2.png")


  (gROOT.FindObject("cAll100r")).cd();
  hh_ratio = setHistoRatio(hh, result_def_3s1b, "", 0.,1.1);
  hh_ratio.Draw("e1");
  gPad.Print("gem/rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio.png")




  .L drawplot_gmtrt.C
  drawplot_gmtrt("minbias_pt10_pat8")
  hh = (TH1D*)result_gem.Clone("gem_new")
  hh.SetFillColor(kGreen+4)
  for (b = hh.FindBin(15); b <= hh.GetNbinsX(); ++b) hh.SetBinContent(b, 0);
  drawplot_gmtrt("minbias_pt15_pat8")
  h15 = (TH1D*)result_gem.Clone("gem15")
  for (b = h15.FindBin(15); b < h15.FindBin(20); ++b) hh.SetBinContent(b, h15.GetBinContent(b));
  drawplot_gmtrt("minbias_pt20_pat8")
  h20 = (TH1D*)result_gem.Clone("gem20")
  for (b = h20.FindBin(20); b < h20.FindBin(30); ++b) hh.SetBinContent(b, h20.GetBinContent(b));
  drawplot_gmtrt("minbias_pt30_pat8")
  h30 = (TH1D*)result_gem.Clone("gem30")
  for (b = h30.FindBin(30); b <= h30.GetNbinsX(); ++b) hh.SetBinContent(b, h30.GetBinContent(b));
  for (b = 1; b <= hh.GetNbinsX(); ++b) if (hh.GetBinContent(b)==0) hh.SetBinError(b, 0.);


  (gROOT.FindObject("cAll100")).cd();
  result_def.Draw("e3")
  hh.Draw("same e3")

  leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
  const TObject obj;
  leg_cc100.SetBorderSize(0);
  leg_cc100.SetFillStyle(0);
  leg_cc100.AddEntry(result_def,"default emulator","f");
  leg_cc100.AddEntry(result_def,"Tracks: with >=3 stubs in 1.14<|#eta|<2.14","");
  leg_cc100.AddEntry(hh,"with GEM match","f");
  leg_cc100.AddEntry(result_def,"Tracks: same, plus req. one stub from ME1/b","");
  leg_cc100.Draw();

   tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
  tex.SetNDC();
  tex.Draw();

  gPad.Print("gem/rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8.png")


  (gROOT.FindObject("cAll100r")).cd();
  hh_ratio = setHistoRatio(hh, result_def, "", 0.,1.1);
  hh_ratio.Draw("e1");
  gPad.Print("gem/rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio.png")



  (gROOT.FindObject("cAll100")).cd();
  result_def_3s1b.Draw("e3")
  hh.Draw("same e3")

  leg_cc100 = TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
  const TObject obj;
  leg_cc100.SetBorderSize(0);
  leg_cc100.SetFillStyle(0);
  leg_cc100.AddEntry(result_def_3s1b,"default emulator","f");
  leg_cc100.AddEntry(hh,"with GEM match","f");
  leg_cc100.AddEntry(result_def_3s1b,"Tracks req. for both:","");
  leg_cc100.AddEntry(result_def_3s1b,"with >=3 stubs in 1.64<|#eta|<2.14","");
  leg_cc100.AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
  leg_cc100.Draw();

   tex = TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
  tex.SetNDC();
  tex.Draw();

  gPad.Print("gem/rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8.png")


  (gROOT.FindObject("cAll100r")).cd();
  hh_ratio = setHistoRatio(hh, result_def_3s1b, "", 0.,1.1);
  hh_ratio.Draw("e1");
  gPad.Print("gem/rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio.png")


"""


if __name__ == "__main__":
    print "It works!"
