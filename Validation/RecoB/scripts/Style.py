#____________________________________________________________
#
#  cuy
#
# A very simple way to make plots with ROOT via an XML file
#
# Francisco Yumiceva
# yumiceva@fnal.gov
#
# Fermilab, 2008
#
#____________________________________________________________

import ROOT

class Style:

    def SetStyle(self):

	ROOT.gStyle.SetFrameBorderMode(0)
	ROOT.gStyle.SetCanvasBorderMode(0)
	ROOT.gStyle.SetPadBorderMode(0)

	ROOT.gStyle.SetFrameFillColor(0)
	ROOT.gStyle.SetPadColor(0)
	ROOT.gStyle.SetCanvasColor(0)
	ROOT.gStyle.SetTitleColor(1)
	ROOT.gStyle.SetStatColor(0)

	# set the paper & margin sizes
	ROOT.gStyle.SetPaperSize(20,26)
	ROOT.gStyle.SetPadTopMargin(0.06)
	ROOT.gStyle.SetPadRightMargin(0.04)
	ROOT.gStyle.SetPadBottomMargin(0.14)
	ROOT.gStyle.SetPadLeftMargin(0.16)
	ROOT.gStyle.SetPadTickX(1)
	ROOT.gStyle.SetPadTickY(1)
    
	ROOT.gStyle.SetTextFont(42) #132
	ROOT.gStyle.SetTextSize(0.09)
	ROOT.gStyle.SetLabelFont(42,"xyz")
	ROOT.gStyle.SetTitleFont(42,"xyz")
	ROOT.gStyle.SetLabelSize(0.045,"xyz") #0.035
	ROOT.gStyle.SetTitleSize(0.045,"xyz")
	ROOT.gStyle.SetTitleOffset(1.5,"y")
    
	# use bold lines and markers
	ROOT.gStyle.SetMarkerStyle(8)
	ROOT.gStyle.SetHistLineWidth(2)
	ROOT.gStyle.SetLineWidth(1)
    #ROOT.gStyle.SetLineStyleString(2,"[12 12]") // postscript dashes

	# do not display any of the standard histogram decorations
	ROOT.gStyle.SetOptTitle(1)
	ROOT.gStyle.SetOptStat(0) #("m")
	ROOT.gStyle.SetOptFit(0)
    
    #ROOT.gStyle.SetPalette(1,0)
	ROOT.gStyle.cd()
	ROOT.gROOT.ForceStyle()
