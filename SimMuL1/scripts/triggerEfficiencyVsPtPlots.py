from ROOT import * 
from triggerPlotHelpers import *
from mkdir import mkdir

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT
ROOT.gROOT.SetBatch(1)


#_______________________________________________________________________________
def drawEtaLabel(minEta, maxEta, x=0.75, y=0.17, font_size=0.):
  """Label for pile-up"""
  tex = TLatex(x, y,"%.2f < |#eta| < %.2f"%(minEta, maxEta))
  if (font_size > 0.):
      tex.SetFontSize(font_size)
  tex.SetTextSize(0.05);
  tex.SetNDC()
  tex.Draw("same")
  return tex


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


################################################################################
################ Plots that require only a single input file ###################
################################################################################

#_______________________________________________________________________________
def eff_Ns1b_def(N, do1b, dir_name = "GEMCSCTriggerEfficiency"):

    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600)
    c.cd()
    
    if do1b:
      arg1b = "1b"
    else:
      arg1b = ""
      
    h_eff_tf10_Ns = setEffHisto("h_pt_after_tfcand_eta1b_%ds%s_pt10"%(N,arg1b), "h_pt_initial_1b", dir, ptreb, kGreen+2, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf20_Ns = setEffHisto("h_pt_after_tfcand_eta1b_%ds%s_pt20"%(N,arg1b), "h_pt_initial_1b", dir, ptreb, kOrange, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf30_Ns = setEffHisto("h_pt_after_tfcand_eta1b_%ds%s_pt30"%(N,arg1b), "h_pt_initial_1b", dir, ptreb, kRed, 1,2, title, xTitle, yTitle, xrangept,yrange)

    h_eff_tf10_Ns.Draw("hist")
    h_eff_tf20_Ns.Draw("hist same")
    h_eff_tf30_Ns.Draw("hist same")

    if do1b:
      leg1b = ", 1 in ME1/b"
    else:
      leg1b = ""

    leg = TLegend(0.55,0.2,.999,0.6, "", "brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have a TF track with"%(minSimPt,maxSimPt))
    leg.AddEntry(h_eff_tf10_Ns, "p_{T}^{TF}#geq10, #geq %d stubs%s"%(N,leg1b), "l")
    leg.AddEntry(h_eff_tf20_Ns, "p_{T}^{TF}#geq20, #geq %d stubs%s"%(N,leg1b), "l")
    leg.AddEntry(h_eff_tf30_Ns, "p_{T}^{TF}#geq30, #geq %d stubs%s"%(N,leg1b), "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.12)    

    c.SaveAs("%seff_%ds%s_def%s"%(output_dir,N,arg1b,ext))

#_______________________________________________________________________________
def eff_Ns_ptX_def(N, X, dir_name = "GEMCSCTriggerEfficiency"):
  
    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600)
    c.cd()

    h_eff_tfX_Ns =  setEffHisto("h_pt_after_tfcand_eta1b_%ds_pt%d"%(N,X), "h_pt_initial_1b", dir, ptreb, kAzure+3, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tfX_Ns1b =  setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt%d"%(N,X), "h_pt_initial_1b", dir, ptreb, kAzure+7, 1,2, title, xTitle, yTitle, xrangept,yrange)

    h_eff_tfX_Ns.Draw("hist")
    h_eff_tfX_Ns1b.Draw("hist same")
    
    leg = TLegend(0.55,0.2,.999,0.6, "", "brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have a TF track with p_{T}^{TF}#geq %d"%(minSimPt,maxSimPt,X))
    leg.AddEntry(h_eff_tfX_Ns, "#geq %d stubs"%(N), "l")
    leg.AddEntry(h_eff_tfX_Ns1b, "#geq %d stubs, 1 in ME1/b"%(N), "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.12)    

    c.SaveAs("%seff_%ds_pt%d_def%s"%(output_dir,N,X,ext))

#_______________________________________________________________________________
def eff_gem1b_basegem(dir_name = "GEMCSCTriggerEfficiency"):

    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600) 
    c.cd()

    gPad.SetGridx(1)
    gPad.SetGridy(1)

    hel =  setEffHisto("h_pt_lctgem_1b", "h_pt_gem_1b", dir, ptreb, kBlack, 1,2, title, xTitle, yTitle, xrangept,yrange)
    het2 =  setEffHisto("h_pt_after_tfcand_gem1b_2s1b", "h_pt_gem_1b", dir, ptreb, kBlue+1, 1,2, title, xTitle, yTitle, xrangept,yrange)
    het3 =  setEffHisto("h_pt_after_tfcand_gem1b_3s1b", "h_pt_gem_1b", dir, ptreb, kBlue+1, 2,2, title, xTitle, yTitle, xrangept,yrange)
    het2pt20 =  setEffHisto("h_pt_after_tfcand_gem1b_2s1b_pt20", "h_pt_gem_1b", dir, ptreb, kGreen+1, 1,2, title, xTitle, yTitle, xrangept,yrange)
    het3pt20 =  setEffHisto("h_pt_after_tfcand_gem1b_3s1b_pt20", "h_pt_gem_1b", dir, ptreb, kGreen+1, 2,2, title, xTitle, yTitle, xrangept,yrange)

    hel.Draw("hist")
    het2.Draw("same hist")
    het3.Draw("same hist")
    het2pt20.Draw("same hist")
    het3pt20.Draw("same hist")

    leg = TLegend(0.45,0.2,.95,0.6, "", "brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have"%(minSimPt, maxSimPt)) 
    leg.AddEntry(hel, "a stub in ME1/b", "l")
    leg.AddEntry(0, "", "")    
    leg.AddEntry(hel, "a TF track with a stub in ME1/b", "")
    leg.AddEntry(0, "", "")        
    leg.AddEntry(het2,     "p_{T}^{TF}#geq0, #geq 2 stubs", "l")
    leg.AddEntry(het2pt20, "p_{T}^{TF}#geq20, #geq 2 stubs", "l")
    leg.AddEntry(het3,     "p_{T}^{TF}#geq0, #geq 3 stubs", "l")
    leg.AddEntry(het3pt20, "p_{T}^{TF}#geq20, #geq 3 stubs", "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.05)    

    c.SaveAs("%seff_gem1b_basegem%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_gem1b_baselctgem(dir_name = "GEMCSCTriggerEfficiency"):

    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"

    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600 ) 
    c.cd()

    het2pt20 =  setEffHisto("h_pt_after_tfcand_gem1b_2s1b_pt20", "h_pt_gem_1b", dir, ptreb, kBlue+1, 1,2, title, xTitle, yTitle, xrangept,yrange)
    het3pt20 =  setEffHisto("h_pt_after_tfcand_gem1b_3s1b_pt20", "h_pt_gem_1b", dir, ptreb, kMagenta+1, 1,2, title, xTitle, yTitle, xrangept,yrange)
    helt2pt20 =  setEffHisto("h_pt_after_tfcand_gem1b_2s1b_pt20", "h_pt_lctgem_1b", dir, ptreb, kBlue+1, 2,2, title, xTitle, yTitle, xrangept,yrange)
    helt3pt20 =  setEffHisto("h_pt_after_tfcand_gem1b_3s1b_pt20", "h_pt_lctgem_1b", dir, ptreb, kMagenta+1, 2,2, title, xTitle, yTitle, xrangept,yrange)

    helt2pt20.Draw("hist")
    het2pt20.Draw("same hist")
    het3pt20.Draw("same hist")
    helt3pt20.Draw("same hist")

    leg = TLegend(0.45,0.2,.95,0.6, "", "brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Efficiency for muon with %d < p_{T} < %d to have TF track p_{T}^{TF}#geq20"%(minSimPt, maxSimPt))
    leg.AddEntry(het2pt20, "GEM baseline", "")
    leg.AddEntry(helt2pt20, "GEM+CSC baseline", "")
    leg.AddEntry(het2pt20, "#geq 2 stubs, 1 in ME1/b", "l")
    leg.AddEntry(helt2pt20, "#geq 2 stubs, 1 in ME1/b", "l")
    leg.AddEntry(het3pt20, "#geq 3 stubs, 1 in ME1/b", "l")
    leg.AddEntry(helt3pt20, "#geq 3 stubs, 1 in ME1/b", "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.05)    

    c.SaveAs("%seff_gem1b_baselctgem%s"%(output_dir,ext))

#_______________________________________________________________________________
def eff_gem1b_basegem_dphi(dir_name = "GEMCSCTriggerEfficiency"):

    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"

    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600 ) 
    c.cd()

    helt2pt20 =  setEffHisto("h_pt_after_tfcand_gem1b_2s1b_pt20", "h_pt_gem_1b", dir, ptreb, kBlue+1, 1,2, title, xTitle, yTitle, xrangept,yrange)
    helt3pt20 =  setEffHisto("h_pt_after_tfcand_gem1b_3s1b_pt20", "h_pt_gem_1b", dir, ptreb, kMagenta+1, 1,2, title, xTitle, yTitle, xrangept,yrange)
    helt2pt20p = setEffHisto("h_pt_after_tfcand_dphigem1b_2s1b_pt20", "h_pt_gem_1b", dir, ptreb, kBlue, 2,2, title, xTitle, yTitle, xrangept,yrange)
    helt3pt20p = setEffHisto("h_pt_after_tfcand_dphigem1b_3s1b_pt20", "h_pt_gem_1b", dir, ptreb, kMagenta, 2,2, title, xTitle, yTitle, xrangept,yrange)

    helt2pt20.Draw("hist")
    helt3pt20.Draw("same hist")
    helt2pt20p.Draw("same hist")
    helt3pt20p.Draw("same hist")

    leg = TLegend(0.45,0.2,.95,0.6, "", "brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have a TF track p_{T}^{TF}#geq20"%(minSimPt, maxSimPt))
    leg.AddEntry(helt2pt20, "without GEM #Delta#phi cut", "")
    leg.AddEntry(helt2pt20p, "with GEM #Delta#phi cut", "")
    leg.AddEntry(helt2pt20, "#geq 2 stubs, 1 in ME1/b", "l")
    leg.AddEntry(helt2pt20p, "#geq 2 stubs, 1 in ME1/b", "l")
    leg.AddEntry(helt3pt20, "#geq 3 stubs, 1 in ME1/b", "l")
    leg.AddEntry(helt3pt20p, "#geq 3 stubs, 1 in ME1/b", "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.05)    

    c.SaveAs("%seff_gem1b_basegem_dphi%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_gem1b_baselpcgem_dphi(dir_name = "GEMCSCTriggerEfficiency"):

    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"

    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600 ) 
    c.cd()

    ## the denominator also requires a GEM stub to be present!!!
    helt2pt20 =  setEffHisto("h_pt_after_tfcand_gem1b_2s1b_pt20", "h_pt_lctgem_1b", dir, ptreb, kBlue+1, 2,2, title, xTitle, yTitle, xrangept,yrange)
    helt3pt20 =  setEffHisto("h_pt_after_tfcand_gem1b_3s1b_pt20", "h_pt_lctgem_1b", dir, ptreb, kMagenta+1, 1,2, title, xTitle, yTitle, xrangept,yrange)
    helt2pt20p = setEffHisto("h_pt_after_tfcand_dphigem1b_2s1b_pt20", "h_pt_lctgem_1b", dir, ptreb, kBlue, 2,2, title, xTitle, yTitle, xrangept,yrange)
    helt3pt20p = setEffHisto("h_pt_after_tfcand_dphigem1b_3s1b_pt20", "h_pt_lctgem_1b", dir, ptreb, kMagenta, 1,2, title, xTitle, yTitle, xrangept,yrange)

    helt2pt20.Draw("hist")
    helt3pt20.Draw("same hist")
    helt2pt20p.Draw("same hist")
    helt3pt20p.Draw("same hist")

    leg = TLegend(0.45,0.2,.95,0.6, "", "brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d (ME1/b baseline) to have a TF track p_{T}^{TF}#geq20"%(minSimPt, maxSimPt))
    leg.AddEntry(helt2pt20, "without GEM #Delta#phi cut", "")
    leg.AddEntry(helt2pt20p, "with GEM #Delta#phi cut", "")
    leg.AddEntry(helt2pt20, "#geq 2 stubs, 1 in ME1/b", "l")
    leg.AddEntry(helt2pt20p, "#geq 2 stubs, 1 in ME1/b", "l")
    leg.AddEntry(helt3pt20, "#geq 3 stubs, 1 in ME1/b", "l")
    leg.AddEntry(helt3pt20p, "#geq 3 stubs, 1 in ME1/b", "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.05)    

    c.SaveAs("%seff_gem1b_baselpcgem_dphi%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_gem1b_baselpcgem_123(dir_name = "GEMCSCTriggerEfficiency"):

    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    htitle = "Efficiency for #mu (GEM+LCT) in 1.64<|#eta|<2.05 to have TF track with ME1/b stubp_{T}^{MC}"

    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600 ) 
    c.cd()

    helt2pt20 = setEffHisto("h_pt_after_tfcand_gem1b_2s1b_pt20", "h_pt_lctgem_1b", dir, ptreb, kBlue+1, 1,2, title, xTitle, yTitle, xrangept,yrange)
    helt2pt20_123 = setEffHisto("h_pt_after_tfcand_dphigem1b_2s123_pt20", "h_pt_lctgem_1b", dir, ptreb, kBlue, 2,2, title, xTitle, yTitle, xrangept,yrange)
    helt3pt20_13 = setEffHisto("h_pt_after_tfcand_dphigem1b_2s13_pt20", "h_pt_lctgem_1b", dir, ptreb, kMagenta+1, 1,2, title, xTitle, yTitle, xrangept,yrange)
    helt3pt20 = setEffHisto("h_pt_after_tfcand_gem1b_3s1b_pt20", "h_pt_lctgem_1b", dir, ptreb, kMagenta, 2,2, title, xTitle, yTitle, xrangept,yrange)

    helt2pt20.Draw("hist")
    helt3pt20.Draw("same hist")
    helt2pt20_123.Draw("same hist")
    helt3pt20_13.Draw("same hist")

    leg = TLegend(0.5,0.2,.999,0.6, "", "brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have a TF track with p_{T}^{TF}#geq20"%(minSimPt, maxSimPt))
    leg.AddEntry(helt2pt20, "#geq 2 stubs, 1 stub in ME1/b", "l")
    leg.AddEntry(helt2pt20_123, "#geq 2 stubs (no ME1-4 tracks)", "l")
    leg.AddEntry(helt3pt20_13, "#geq 2 stubs (no ME1-2 and ME1-4)", "l")
    leg.AddEntry(helt3pt20, "#geq 3 stubs", "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.12)    

    c.SaveAs("%seff_gem1b_baselpcgem_123%s"%(output_dir,ext))

#_______________________________________________________________________________
def eff_pt_tf():

    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"
    c = TCanvas("c","c",1000,600 ) 
    c.cd()

    h_eff_pt_after_mpc_ok_plus = setEffHisto("h_pt_after_mpc_ok_plus","h_pt_initial",dir, ptreb, kBlack, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus = setEffHisto("h_pt_after_tfcand_ok_plus","h_pt_initial",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_pt10 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10","h_pt_initial",dir, ptreb, kBlue, 2,2, "","","",xrangept,yrange)
    
    h_eff_pt_after_mpc_ok_plus.Draw("hist")
    h_eff_pt_after_tfcand_ok_plus.Draw("same hist")
    h_eff_pt_after_tfcand_ok_plus_pt10.Draw("same hist")
    
    leg = TLegend(0.3,0.19,0.926,0.45,"","brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have"%(minSimPt, maxSimPt))
    leg.AddEntry(h_eff_pt_after_mpc_ok_plus,"#geq 2 stubs","pl")
    leg.AddEntry(h_eff_pt_after_tfcand_ok_plus,"a TF track with #geq 2 stubs","pl")
    leg.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10,"a TF track with p_{T}^{TF}>10 and #geq 2 stubs","pl")
    leg.Draw()
    etalabel = drawEtaLabel(minEta, maxEta)    
    
    c.SaveAs("%seff_pt_tf%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_pt_tf_eta1b_Ns(N, do1b):

    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    if do1b:
      arg1b = "1b"
    else:
      arg1b = ""
      
    if do1b:
      leg1b = ", 1 in ME1/b"
    else:
      leg1b = ""

    c = TCanvas("c","c",1000,600 ) 
    c.cd()
    
    h_eff_pt_after_tfcand_eta1b_Ns      = setEffHisto("h_pt_after_tfcand_eta1b_%ds%s"%(N,arg1b),"h_pt_initial_1b",dir, ptreb, kBlack, 1,2, title, xTitle, yTitle ,xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_Ns_pt10 = setEffHisto("h_pt_after_tfcand_eta1b_%ds%s_pt10"%(N,arg1b),"h_pt_initial_1b",dir, ptreb, kGreen+2, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_Ns_pt20 = setEffHisto("h_pt_after_tfcand_eta1b_%ds%s_pt20"%(N,arg1b),"h_pt_initial_1b",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_Ns_pt25 = setEffHisto("h_pt_after_tfcand_eta1b_%ds%s_pt25"%(N,arg1b),"h_pt_initial_1b",dir, ptreb, kOrange, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_eta1b_Ns_pt30 = setEffHisto("h_pt_after_tfcand_eta1b_%ds%s_pt30"%(N,arg1b),"h_pt_initial_1b",dir, ptreb, kRed, 1,2, "","","",xrangept,yrange)
    
    h_eff_pt_after_tfcand_eta1b_Ns.Draw("hist")
    h_eff_pt_after_tfcand_eta1b_Ns_pt10.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_Ns_pt20.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_Ns_pt25.Draw("same hist")
    h_eff_pt_after_tfcand_eta1b_Ns_pt30.Draw("same hist")
    
    leg = TLegend(0.5,0.15,0.99,0.5,"","brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have a TF track with "%(minSimPt, maxSimPt))
    leg.AddEntry(h_eff_pt_after_tfcand_eta1b_Ns,"#geq %d stubs"%(N),"pl")
    leg.AddEntry(h_eff_pt_after_tfcand_eta1b_Ns_pt10,"p_{T}^{TF} #geq 10, #geq %d stubs%s"%(N,leg1b),"pl")
    leg.AddEntry(h_eff_pt_after_tfcand_eta1b_Ns_pt20,"p_{T}^{TF} #geq 20, #geq %d stubs%s"%(N,leg1b),"pl")
    leg.AddEntry(h_eff_pt_after_tfcand_eta1b_Ns_pt25,"p_{T}^{TF} #geq 25, #geq %d stubs%s"%(N,leg1b),"pl")
    leg.AddEntry(h_eff_pt_after_tfcand_eta1b_Ns_pt30,"p_{T}^{TF} #geq 30, #geq %d stubs%s"%(N,leg1b),"pl")
    leg.Draw()
    
    etalabel = drawEtaLabel(1.6, 2.12, 0.75, 0.55)    
    c.SaveAs("%sh_eff_pt_tf_eta1b_%ds%s%s"%(output_dir,N,arg1b,ext))

#_______________________________________________________________________________
def eff_pth_tf():
    
    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600 ) 
    c.cd()
        
    h_eff_pth_after_mpc_ok_plus = setEffHisto("h_pth_after_mpc_ok_plus","h_pth_initial",dir, ptreb, kBlack, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_pth_after_tfcand_ok_plus = setEffHisto("h_pth_after_tfcand_ok_plus","h_pth_initial",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange)
    h_eff_pth_after_tfcand_ok_plus_pt10 = setEffHisto("h_pth_after_tfcand_ok_plus_pt10","h_pth_initial",dir, ptreb, kBlue, 2,2, "","","",xrangept,yrange)
    
    h_eff_pth_after_mpc_ok_plus.Draw("hist")
    h_eff_pth_after_tfcand_ok_plus.Draw("same hist")
    h_eff_pth_after_tfcand_ok_plus_pt10.Draw("same hist")
    
    leg = TLegend(0.3,0.19,0.9,0.45,"","brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have"%(minSimPt, maxSimPt))
    leg.AddEntry(h_eff_pth_after_mpc_ok_plus,"#geq 2 stubs","pl")
    leg.AddEntry(h_eff_pth_after_tfcand_ok_plus,"a TF track with #geq 2 stubs","pl")
    leg.AddEntry(h_eff_pth_after_tfcand_ok_plus_pt10,"a TF track with p_{T}^{TF}>10 and #geq 2 stubs","pl")
    leg.Draw()

    etalabel = drawEtaLabel(2.1, 2.45)    
    c.SaveAs("%sh_eff_pth_tf_2st%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_pth_tf_3st1a():
    
    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"
    
    c = TCanvas("c","c",1000,600 ) 
    c.cd()

    h_eff_pth_after_mpc_ok_plus = setEffHisto("h_pth_after_mpc_ok_plus","h_pth_initial",dir, ptreb, kBlack, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_pth_after_tfcand_ok_plus_3st1a = setEffHisto("h_pth_after_tfcand_ok_plus_3st1a","h_pth_initial",dir, ptreb, kBlue, 1,2, "","","", xrangept,yrange)
    h_eff_pth_after_tfcand_ok_plus_pt10_3st1a = setEffHisto("h_pth_after_tfcand_ok_plus_pt10_3st1a","h_pth_initial",dir, ptreb, kBlue, 2,2, "","","",xrangept,yrange)
    
    h_eff_pth_after_mpc_ok_plus.Draw("hist")
    h_eff_pth_after_tfcand_ok_plus_3st1a.Draw("same hist")
    h_eff_pth_after_tfcand_ok_plus_pt10_3st1a.Draw("same hist")
    
    leg = TLegend(0.3,0.19,0.926,0.45,"","brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have"%(minSimPt, maxSimPt))
    leg.AddEntry(h_eff_pth_after_mpc_ok_plus,"#geq 2 stubs","pl")
    leg.AddEntry(h_eff_pth_after_tfcand_ok_plus_3st1a,"a TF track with #geq 3 stubs, 1 in ME1/a","pl")
    leg.AddEntry(h_eff_pth_after_tfcand_ok_plus_pt10_3st1a,"a TF track with p_{T}^{TF}>10 and #geq 3 stubs, 1 in ME1/a","pl")
    leg.Draw()
    etalabel = drawEtaLabel(2.1, 2.45)    
    
    c.SaveAs("%sh_eff_pth_tf_3st1a%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_pt_tf_q():

    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600) 
    c.cd()
    
    h_eff_pt_after_tfcand_ok_plus_q1 = setEffHisto("h_pt_after_tfcand_ok_plus_q1","h_pt_after_mpc_ok_plus",dir, ptreb, kBlue, 1,1, title, xTitle, yTitle, xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_q2 = setEffHisto("h_pt_after_tfcand_ok_plus_q2","h_pt_after_mpc_ok_plus",dir, ptreb, kCyan+2, 1,1, title, xTitle, yTitle, xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_q3 = setEffHisto("h_pt_after_tfcand_ok_plus_q3","h_pt_after_mpc_ok_plus",dir, ptreb, kMagenta+1, 1,1, title, xTitle, yTitle, xrangept,yrange)    
    h_eff_pt_after_tfcand_ok_plus_pt10_q1 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q1","h_pt_after_mpc_ok_plus",dir, ptreb, kBlue, 2,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_pt10_q2 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q2","h_pt_after_mpc_ok_plus",dir, ptreb, kCyan+2, 2,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_pt10_q3 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q3","h_pt_after_mpc_ok_plus",dir, ptreb, kMagenta+1, 2,2, "","","",xrangept,yrange)

    h_eff_pt_after_tfcand_ok_plus_q1.Draw("hist")
    h_eff_pt_after_tfcand_ok_plus_q2.Draw("same hist")
    h_eff_pt_after_tfcand_ok_plus_q3.Draw("same hist")
    h_eff_pt_after_tfcand_ok_plus_pt10_q1.Draw("same hist")
    h_eff_pt_after_tfcand_ok_plus_pt10_q2.Draw("same hist")
    h_eff_pt_after_tfcand_ok_plus_pt10_q3.Draw("same hist")
    
    leg = TLegend(0.3,0.19,0.99,0.5,"","brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have a TFTrack with #geq 2 stubs"%(minSimPt, maxSimPt))
    leg.AddEntry(h_eff_pt_after_tfcand_ok_plus_q1,"Q#geq1","pl")
    leg.AddEntry(h_eff_pt_after_tfcand_ok_plus_q2,"Q#geq2","pl")
    leg.AddEntry(h_eff_pt_after_tfcand_ok_plus_q3,"Q=3","pl")
    leg.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q1,"Q#geq1, p_{T}^{TF}>10","pl")
    leg.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q2,"Q#geq2, p_{T}^{TF}>10","pl")
    leg.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q3,"Q=3, p_{T}^{TF}>10","pl")
    leg.Draw()
    etalabel = drawEtaLabel(minEta, maxEta)    
    
    c.SaveAs("%sh_eff_pt_tf_q%s"%(output_dir,ext))
        

#_______________________________________________________________________________
def do_h_pt_after_tfcand_ok_plus_pt10():

    dir = getRootDirectory(input_dir, file_name, dir_name)
    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    #"p_{T}^{TF}>10 assignment eff(p_{T}^{MC}) studies (denom: any p_{T}^{TF}, 1.2<#eta<2.1)","p_{T}^{MC}",""
    #for #mu with p_{T}>20 crossing ME1+one station to pass p_{T}^{TF}>10 with")
    
    c = TCanvas("c","c",1000,600) 
    c.cd()
    h_eff_pt_after_tfcand_ok_plus_pt10 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10","h_pt_after_tfcand_ok_plus",dir, ptreb, kBlue, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_pt10_q2 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q2","h_pt_after_tfcand_ok_plus_q2",dir, ptreb, kCyan+2, 1,2, "","","",xrangept,yrange)
    h_eff_pt_after_tfcand_ok_plus_pt10_q3 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q3","h_pt_after_tfcand_ok_plus_q3",dir, ptreb, kMagenta+1, 1,2, "","","",xrangept,yrange)
    
    h_eff_pt_after_tfcand_ok_plus_pt10.Draw("hist")
    h_eff_pt_after_tfcand_ok_plus_pt10_q2.Draw("same hist")
    h_eff_pt_after_tfcand_ok_plus_pt10_q3.Draw("same hist")
    
    leg = TLegend(0.3,0.19,0.99,0.45,"","brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Efficiency for a muon  with %d < p_{T} < %d and a TFTrack with 2 stubs to pass p_{T}^{TF}>10"%(minSimPt, maxSimPt))
    leg.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10,"any Q","pl")
    leg.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q2,"Q#geq2","pl")
    leg.AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q3,"Q=3","pl")
    leg.Draw("same")
    etalabel = drawEtaLabel(minEta, maxEta)    

    c.SaveAs("%sh_eff_ptres_tf%s"%(output_dir,ext))



################################################################################
################## Plots that require multiple input files #####################
## GEM-CSC bending angle cut applied at L1CSC and used by the CSC TrackFinder ##
################################################################################

#_______________________________________________________________________________
def eff_Ns1b(N, file_gem10, file_gem20, file_gem30, dir_name = "GEMCSCTriggerEfficiency"):
  
    dir = getRootDirectory(input_dir, file_name, dir_name)
    dir_gem10 = getRootDirectory(input_dir, file_gem10, dir_name)
    dir_gem20 = getRootDirectory(input_dir, file_gem20, dir_name)
    dir_gem30 = getRootDirectory(input_dir, file_gem30, dir_name)

    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600)
    c.cd()

#    h_eff_tf10_Ns =   setEffHisto("h_pt_after_tfcand_eta1b_%ds_pt%d"%(N), "h_pt_initial_1b", dir, ptreb, kAzure+3, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf10_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt10"%(N), "h_pt_initial_1b", dir, ptreb, kAzure+7, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf20_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt20"%(N), "h_pt_initial_1b", dir, ptreb, kAzure+7, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf30_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt30"%(N), "h_pt_initial_1b", dir, ptreb, kAzure+7, 1,2, title, xTitle, yTitle, xrangept,yrange)  
    h_eff_tf10_gpt10_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt10"%(N), "h_pt_initial_1b", dir_gem10, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf20_gpt20_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt20"%(N), "h_pt_initial_1b", dir_gem20, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf30_gpt30_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt30"%(N), "h_pt_initial_1b", dir_gem30, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)

    h_eff_tf10_Ns1b.Draw("hist")
    h_eff_tf20_Ns1b.Draw("hist same")
    h_eff_tf30_Ns1b.Draw("hist same")
    h_eff_tf10_gpt10_Ns1b.Draw("hist same")
    h_eff_tf20_gpt20_Ns1b.Draw("hist same")
    h_eff_tf30_gpt30_Ns1b.Draw("hist same")

    leg = TLegend(0.50,0.17,.999,0.57, "", "brNDC")
    leg.SetMargin(0.15)
    leg.SetNColumns(2)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have a TF track with #geq %d stubs, 1 in ME1/b"%(minSimPt, maxSimPt,N))
    leg.AddEntry(h_eff_tf10_Ns1b, "Trigger p_{T}:", "")
    leg.AddEntry(h_eff_tf10_gpt10_Ns1b, "with GEM:", "")
    leg.AddEntry(h_eff_tf10_Ns1b, "p_{T}^{TF}#geq10", "l")
    leg.AddEntry(h_eff_tf10_gpt10_Ns1b, "#Delta#phi for p_{T}=10", "l")
    leg.AddEntry(h_eff_tf20_Ns1b, "p_{T}^{TF}#geq20", "l")
    leg.AddEntry(h_eff_tf20_gpt20_Ns1b, "#Delta#phi for p_{T}=20", "l")
    leg.AddEntry(h_eff_tf30_Ns1b, "p_{T}^{TF}#geq30", "l")
    leg.AddEntry(h_eff_tf30_gpt30_Ns1b, "#Delta#phi for p_{T}=30", "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.12)    

    c.SaveAs("%seff_%ds1b%s"%(output_dir,N,ext))


#_______________________________________________________________________________
def eff_3s_2s1b(file_gem10, file_gem20, file_gem30, dir_name = "GEMCSCTriggerEfficiency"):
  
    dir = getRootDirectory(input_dir, file_name, dir_name)
    dir_gem10 = getRootDirectory(input_dir, file_gem10, dir_name)
    dir_gem20 = getRootDirectory(input_dir, file_gem20, dir_name)
    dir_gem30 = getRootDirectory(input_dir, file_gem30, dir_name)

    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600)
    c.cd()

    h_eff_tf10_3s = setEffHisto("h_pt_after_tfcand_eta1b_3s1b_pt10", "h_pt_initial_1b", dir, ptreb, kAzure+7, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf20_3s = setEffHisto("h_pt_after_tfcand_eta1b_3s1b_pt20", "h_pt_initial_1b", dir, ptreb, kAzure+7, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf30_3s = setEffHisto("h_pt_after_tfcand_eta1b_3s1b_pt30", "h_pt_initial_1b", dir, ptreb, kAzure+7, 1,2, title, xTitle, yTitle, xrangept,yrange)  
    h_eff_tf10_gpt10_2s1b = setEffHisto("h_pt_after_tfcand_eta1b_2s1b_pt10", "h_pt_initial_1b", dir_gem10, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf20_gpt20_2s1b = setEffHisto("h_pt_after_tfcand_eta1b_2s1b_pt20", "h_pt_initial_1b", dir_gem20, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf30_gpt30_2s1b = setEffHisto("h_pt_after_tfcand_eta1b_2s1b_pt30", "h_pt_initial_1b", dir_gem30, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)

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
    leg.SetHeader("Efficiency for a muon with %d < p_{T} < %d to have a TF track with"%(minSimPt, maxSimPt,N))
    leg.AddEntry(h_eff_tf10_3s, "#geq 3 stubs", "")
    leg.AddEntry(h_eff_tf10_gpt10_2s1b, "#geq 3 stubs, 1 in ME1/b, and with GEM", "")
    leg.AddEntry(h_eff_tf10_3s, "p_{T}^{TF}#geq10", "l")
    leg.AddEntry(h_eff_tf10_gpt10_2s1b, "#Delta#phi for p_{T}=10", "l")
    leg.AddEntry(h_eff_tf20_3s, "p_{T}^{TF}#geq20", "l")
    leg.AddEntry(h_eff_tf20_gpt20_2s1b, "#Delta#phi for p_{T}=20", "l")
    leg.AddEntry(h_eff_tf30_3s, "p_{T}^{TF}#geq30", "l")
    leg.AddEntry(h_eff_tf30_gpt30_2s1b, "#Delta#phi for p_{T}=30", "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.12)    

    c.SaveAs("%seff_3s_2s1b%s"%(output_dir,ext))


#_______________________________________________________________________________
def eff_Ns_gemtight(N, file_gem10, file_gem15, file_gem20, file_gem30, file_gem40, dir_name = "GEMCSCTriggerEfficiency"):

    dir = getRootDirectory(input_dir, file_name, dir_name)
    dir_gem10 = getRootDirectory(input_dir, file_gem10, dir_name)
    dir_gem15 = getRootDirectory(input_dir, file_gem15, dir_name)
    dir_gem20 = getRootDirectory(input_dir, file_gem20, dir_name)
    dir_gem30 = getRootDirectory(input_dir, file_gem30, dir_name)
    dir_gem40 = getRootDirectory(input_dir, file_gem40, dir_name)

    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600)
    c.cd()

    h_eff_tf10_gpt10_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt10"%(N), "h_pt_initial_1b", dir_gem10, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf10_gpt15_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt10"%(N), "h_pt_initial_1b", dir_gem15, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf15_gpt15_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt15"%(N), "h_pt_initial_1b", dir_gem15, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf15_gpt20_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt15"%(N), "h_pt_initial_1b", dir_gem20, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf20_gpt20_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt20"%(N), "h_pt_initial_1b", dir_gem20, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf20_gpt30_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt20"%(N), "h_pt_initial_1b", dir_gem30, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf30_gpt30_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt30"%(N), "h_pt_initial_1b", dir_gem30, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf30_gpt40_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt30"%(N), "h_pt_initial_1b", dir_gem40, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)

    h_eff_tf10_gpt10_Ns1b.Draw("hist")
    h_eff_tf10_gpt15_Ns1b.Draw("hist same")
    h_eff_tf15_gpt15_Ns1b.Draw("hist same")
    h_eff_tf15_gpt20_Ns1b.Draw("hist same")
    h_eff_tf20_gpt20_Ns1b.Draw("hist same")
    h_eff_tf20_gpt30_Ns1b.Draw("hist same")
    h_eff_tf30_gpt30_Ns1b.Draw("hist same")
    h_eff_tf30_gpt40_Ns1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Eff. for a muon with %d < p_{T} < %d to have a TF track with #geq %d stubs, 1 in ME1/b"%(minSimPt, maxSimPt,N))
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T}^{TF} cut and", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T}^{TF} cut and", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T} for #Delta#phi(GEM,CSC)", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T} for #Delta#phi(GEM,CSC)", "")
    leg.AddEntry(h_eff_tf10_gpt10_3s1b, "#geq10 and 10", "l")
    leg.AddEntry(h_eff_tf10_gpt15_3s1b, "#geq10 and 15", "l")
    leg.AddEntry(h_eff_tf20_gpt20_3s1b, "#geq20 and 20", "l")
    leg.AddEntry(h_eff_tf20_gpt30_3s1b, "#geq20 and 30", "l")
    leg.AddEntry(h_eff_tf30_gpt30_3s1b, "#geq30 and 30", "l")
    leg.AddEntry(h_eff_tf30_gpt40_3s1b, "#geq30 and 40", "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.12)    

    c.SaveAs("%seff_%ds_gemtight%s"%(output_dir,N,ext))


#_______________________________________________________________________________
def eff_Ns_gemtightX(N, file_gem10, file_gem15, file_gem20, file_gem30, file_gem40, dir_name = "GEMCSCTriggerEfficiency"):

    dir = getRootDirectory(input_dir, file_name, dir_name)
    dir_gem10 = getRootDirectory(input_dir, file_gem10, dir_name)
    dir_gem15 = getRootDirectory(input_dir, file_gem15, dir_name)
    dir_gem20 = getRootDirectory(input_dir, file_gem20, dir_name)
    dir_gem30 = getRootDirectory(input_dir, file_gem30, dir_name)
    dir_gem40 = getRootDirectory(input_dir, file_gem40, dir_name)

    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600)
    c.cd()

    h_eff_tf10_gpt10_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt10"%(N), "h_pt_initial_1b", dir_gem10, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf10_gpt20_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt10"%(N), "h_pt_initial_1b", dir_gem20, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf15_gpt15_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt15"%(N), "h_pt_initial_1b", dir_gem15, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf15_gpt30_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt15"%(N), "h_pt_initial_1b", dir_gem30, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf20_gpt20_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt20"%(N), "h_pt_initial_1b", dir_gem20, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tf20_gpt40_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt20"%(N), "h_pt_initial_1b", dir_gem40, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)

    h_eff_tf10_gpt10_Ns1b.Draw("hist")
    h_eff_tf10_gpt20_Ns1b.Draw("hist same")
    h_eff_tf15_gpt15_Ns1b.Draw("hist same")
    h_eff_tf15_gpt30_Ns1b.Draw("hist same")
    h_eff_tf20_gpt20_Ns1b.Draw("hist same")
    h_eff_tf20_gpt40_Ns1b.Draw("hist same")

    leg = TLegend(0.55,0.17,.999,0.57, "", "brNDC")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetHeader("Eff. for a muon with %d < p_{T} < %d to have a TF track with #geq %d stubs, 1 in ME1/b"%(minSimPt, maxSimPt,N))
    leg.AddEntry(h_eff_tf10_gpt10_Ns1b, "p_{T}^{TF} cut and", "")
    leg.AddEntry(h_eff_tf10_gpt10_Ns1b, "p_{T}^{TF} cut and", "")
    leg.AddEntry(h_eff_tf10_gpt10_Ns1b, "p_{T} for #Delta#phi(GEM,CSC)", "")
    leg.AddEntry(h_eff_tf10_gpt10_Ns1b, "p_{T} for #Delta#phi(GEM,CSC)", "")
    leg.AddEntry(h_eff_tf10_gpt10_Ns1b, "#geq10 and 10", "l")
    leg.AddEntry(h_eff_tf10_gpt20_Ns1b, "#geq10 and 20", "l")
    leg.AddEntry(h_eff_tf15_gpt15_Ns1b, "#geq15 and 15", "l")
    leg.AddEntry(h_eff_tf15_gpt30_Ns1b, "#geq15 and 30", "l")
    leg.AddEntry(h_eff_tf20_gpt20_Ns1b, "#geq20 and 20", "l")
    leg.AddEntry(h_eff_tf20_gpt40_Ns1b, "#geq20 and 40", "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.12)    

    c.SaveAs("%seff_%ds_gemtightX%s"%(output_dir,N,ext))


#_______________________________________________________________________________
def eff_Ns_ptX_gem_ptY(N, X, Y, gem_file, dir_name = "GEMCSCTriggerEfficiency"):

    dir = getRootDirectory(input_dir, file_name, dir_name)
    dir_gem = getRootDirectory(input_dir, gem_name, dir_name)

    title = " " * 11 + "CSCTF Track efficiency" + " " * 35 + "CMS Simulation Preliminary"
    xTitle = "Simtrack p_{T}"
    yTitle = "Efficiency"

    c = TCanvas("c","c",1000,600)
    c.cd()

    h_eff_tfX_Ns =  setEffHisto("h_pt_after_tfcand_eta1b_%ds_pt%d"%(N,X), "h_pt_initial_1b", dir, ptreb, kAzure+3, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tfX_Ns1b =  setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt%d"%(N,X), "h_pt_initial_1b", dir, ptreb, kAzure+7, 1,2, title, xTitle, yTitle, xrangept,yrange)
    h_eff_tfX_gptY_Ns1b = setEffHisto("h_pt_after_tfcand_eta1b_%ds1b_pt%d"%(N,X), "h_pt_initial_1b", dir_gem, ptreb, kAzure+7, 2,2, title, xTitle, yTitle, xrangept,yrange)

    h_eff_tfX_Ns.Draw("hist")
    h_eff_tfX_Ns1b.Draw("hist same")
    h_eff_tfX_gptY_Ns1b.Draw("hist same")
    
    leg = TLegend(0.55,0.2,.999,0.6, "", "brNDC")
    leg.SetMargin(0.15)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    leg.SetHeader("Efficiency for muon with %d < p_{T} < %d to have a TF track with p_{T}^{TF}#geq %d"%(minSimPt, maxSimPt, X))
    leg.AddEntry(h_eff_tfX_Ns, "#geq %d stubs"%(N), "l")
    leg.AddEntry(h_eff_tfX_Ns1b, "#geq %d stubs, 1 in ME1/b"%(N), "l")
    leg.AddEntry(h_eff_tfX_gptY_Ns1b, "#geq %d stubs, 1 in ME1/b, #Delta#phi(GEM,CSC) for p_{T}=%d"%(N,Y), "l")
    leg.Draw()
    etalabel = drawEtaLabel(1.64, 2.12)    

    c.SaveAs("%seff_%ds_pt%d_gem_pt%d%s"%(output_dir,N,X,Y,ext))


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
    minEta = 1.45
    maxEta = 2.5
    minSimPt = 2
    maxSimPt = 50

    ## input file without GEM,CSC bending angle cut
    file_name = "hp_dimu_CMSSW_6_2_0_SLHC1_upgrade2019_pu000_w3_gem98_pt2-50_PU0_pt20_new_eff.root"
    file_name = "hp_dimu_CMSSW_6_2_0_SLHC1_upgrade2019_pu000_w3_gem98_pt2-50_PU0_pt0_new_eff_postBuxFix.root"
    ## input files with various GEM,CSC bending angle cuts
    file_gem10 = ""
    file_gem15 = ""
    file_gem20 = ""
    file_gem30 = ""
    file_gem40 = ""
    
    ## output directory definition
    reuseOutputDirectory = True
    if not reuseOutputDirectory:
        output_dir = mkdir("myTest")
    else:
        output_dir = "myTest_20140107_044508/"
        
    ## default name of the directory 
    dir_name = 'GEMCSCTriggerEfficiency'

    ## plots that require a single input file
    eff_pth_tf_3st1a()
    eff_pth_tf()
    eff_pt_tf()
    do_h_pt_after_tfcand_ok_plus_pt10()
    eff_pt_tf_q()
    
    ## effect of GEM
    eff_gem1b_baselpcgem_123()
    eff_gem1b_baselpcgem_dphi()
    eff_gem1b_basegem_dphi()
    eff_gem1b_baselctgem()
    eff_gem1b_basegem()

    ## TFTrack efficiencies
    ## 1st argument is min number of stubs
    ## 2nd argument is min pt TFTrack 
    eff_Ns_ptX_def(2,10)
    eff_Ns_ptX_def(2,15)
    eff_Ns_ptX_def(2,20)
    eff_Ns_ptX_def(2,30)
    eff_Ns_ptX_def(2,40)

    eff_Ns_ptX_def(3,10)
    eff_Ns_ptX_def(3,15)
    eff_Ns_ptX_def(3,20)
    eff_Ns_ptX_def(3,30)
    eff_Ns_ptX_def(3,40)
  
    ## TFTrack efficiencies
    ## 1st argument is min number of stubs
    ## 2nd argument is check stubs in ME1/b
    eff_Ns1b_def(2,False)
    eff_Ns1b_def(2,True)
    eff_Ns1b_def(3,False)
    eff_Ns1b_def(3,True)

    eff_pt_tf_eta1b_Ns(2, False)
    eff_pt_tf_eta1b_Ns(2, True)
    eff_pt_tf_eta1b_Ns(3, False)
    eff_pt_tf_eta1b_Ns(3, True)
    
    ## These functions require also the gem bending angle cut files
    ## Needs testing!!!

    withMultipleFiles = False
    if withMultipleFiles:      
      eff_Ns1b(2, file_gem10, file_gem20, file_gem30)
      eff_Ns1b(3, file_gem10, file_gem20, file_gem30)
      eff_3s_2s1b(file_gem10, file_gem20, file_gem30)
      eff_Ns_gemtight(2, file_gem10, file_gem15, file_gem20, file_gem30, file_gem40)
      eff_Ns_gemtight(3, file_gem10, file_gem15, file_gem20, file_gem30, file_gem40)
      eff_Ns_gemtightX(2, file_gem10, file_gem15, file_gem20, file_gem30, file_gem40)
      eff_Ns_gemtightX(3, file_gem10, file_gem15, file_gem20, file_gem30, file_gem40)

      eff_Ns_ptX_gem_ptY(2, 10, 10, file_gem10)
      eff_Ns_ptX_gem_ptY(2, 15, 15, file_gem15)
      eff_Ns_ptX_gem_ptY(2, 20, 20, file_gem20)
      eff_Ns_ptX_gem_ptY(2, 30, 30, file_gem30)
      eff_Ns_ptX_gem_ptY(2, 40, 40, file_gem40)

      eff_Ns_ptX_gem_ptY(3, 10, 10, file_gem10)
      eff_Ns_ptX_gem_ptY(3, 15, 15, file_gem15)
      eff_Ns_ptX_gem_ptY(3, 20, 20, file_gem20)
      eff_Ns_ptX_gem_ptY(3, 30, 30, file_gem30)
      eff_Ns_ptX_gem_ptY(3, 40, 40, file_gem40)
