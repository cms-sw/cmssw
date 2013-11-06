from ROOT import *

ptscale = [-1.,  0., 1.5,  2., 2.5,  3., 3.5,  4., 4.5,  5.,  6.,  7.,  8.,  10.,  12.,  14.,
            16., 18., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 120., 140.]

ptscaleb = [1.5,  2., 2.5,  3., 3.5,  4., 4.5,  5.,  6.,  7.,  8.,  10.,  12.,  14.,
            16., 18., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 120., 140., 150.]

ptscaleb_ = [1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75,  5.5, 6.5, 7.5,  9., 11.,  13.,  15.,
             17.,  19., 22.5, 27.5, 32.5, 37.5, 42.5, 47.5,  55.,  65., 75., 85., 95., 110., 130., 150.]

def GetStat(h):
  """Get the statistics options"""
  return h.FindObject("stats")

def SetOptStat(h, op):
  """Set the statistics options"""
  stat = GetStat(h)
  stat.SetOptStat(op)
  return stat
    
def GetH(f, dir, name):
  """Get the histogram"""
  dir = "SimMuL1StrictAll"
  return f.Get("%s/%s;1"%(dir,name))

def Print(c, name):
  """Print the histogram"""
  c.Print("%s/%s"%(pdir,name))

def myRebin(h, n):
  """Custom rebin function"""
  nb = h.GetNbinsX()
  entr = h.GetEntries()
  bin0 = h.GetBinContent(0)
  binN1 = h.GetBinContent(nb+1)
  if (nb % n):
    binN1 += h.Integral(nb - nb%n + 1, nb)
  h.Rebin(n)
  nb = h.GetNbinsX()
  h.SetBinContent(0, bin0)
  h.SetBinContent(nb+1, binN1)
  h.SetEntries(entr)

def scale(h):
  """Calculate the trigger rate"""
  rate = 40000.
  nevents = 238000
  bx_window = 3
  bx_filling = 0.795
  h.Scale(rate*bx_filling/(bx_window*nevents))

def drawLumiBXPULabel():
  """Draw the luminosity + BX label -- not for TDR"""
  tex = TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}")
  tex.SetNDC()
  tex.Draw()

def drawPULabel(x=0.17, y=0.15, font_size=0.):                          
    tex = TLatex(x, y,"L=4*10^{34} (25ns PU100)")
    if (font_size > 0.): 
        tex.SetFontSize(font_size)
    tex.SetNDC()
    tex.Draw()
    return tex

def setHistoPt(f_name, name, cname, title, lcolor, lstyle, lwidth):
  """Set rate vs pt histogram""" 
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
    h.SetBinContent(bin, bc)
  ##h.Sumw2()
  scale(h)
  return h

def setHistoPtRaw(f_name, name, cname, title, lcolor, lstyle, lwidth):
  """Set rate vs pt histogram""" 
  print "Opening ", f_name
  f = TFile.Open(f_name)
  h0 = getH(f, dir, name)
  nb = h0.GetXaxis().GetNbins()
  h = h0.Clone(name+cname)
  ##h.Sumw2()
  scale(h)
  return h

def setHistoEta(f_name, name, cname, title, lcolor, lstyle, lwidth):
  """Set rate vs eta histogram"""
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
  """Get rate vs pt histogram"""
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
    h = TH1D(s_name + cname, title, len(ptscaleb_), ptscaleb_)
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


if __name__ == "__main__":
    print "It's Working!"
