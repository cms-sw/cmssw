#chad harrington 2019
#execute as python3 test/offsetStack.py -f HS_old:tmp/HS_old/DQMTotal.root HS:tmp/HS/DQMTotal.root -o .
import argparse
import ROOT

from Validation.RecoParticleFlow.defaults_cfi import candidateType,muHighOffset,npvHighOffset

ROOT.gROOT.SetBatch(True)

def arg2List( args ) :
  list = []
  for pair in args :
    split = pair.split(":")
    l = len(split)
    if l < 2 : raise Exception("Must provide a label and file, as in label:file")

    label, file = split[0], ":".join(split[1:l]) #join in the case file contains ':'
    list += [(label, file)]
  return list

def main() :
  parser = argparse.ArgumentParser()
  parser.add_argument( "-f", "--files", nargs='+', required=True, help="must provide at least one file. Eg, label1:file1 label2:file2" )
  parser.add_argument( "-v", "--var",    type=str,   action='store', default="npv", help="variable to bin offset eT" )
  parser.add_argument( "-r", "--deltaR", type=float, action='store', default=0.4,   help="deltaR value" )
  parser.add_argument( "-o", "--outDir", type=str,   action='store', required=True, help="output directory" )
  args = parser.parse_args()

  offsetStack( arg2List(args.files), args.var, args.deltaR, args.outDir )

def offsetStack( files, var, r, outdir ) :

  file1 = ROOT.TFile.Open( files[0][1] )
  if file1 == None : raise Exception( "{} does not exist!".format(files[0][1]) )
  label1 = files[0][0]
  label = label1

  h1 = file1.FindObjectAny( var )
  if h1 == None : raise Exception( "Could not find {} histogram in {}!".format(var, files[0][1]) )
  avg1 = int( h1.GetMean()+0.5 )

  d_hist1 = getHists( file1, var, avg1, r )

  setStyle()
  c = ROOT.TCanvas("c", "c", 600, 600)

  stack1 = ROOT.THStack( "stack1", ";#eta;<Offset Energy_{{T}}(#pi{:.1f}^{{2}})>/<{}> [GeV]".format(r, "N_{PV}" if var=="npv" else "#mu") )
  setStack( stack1, d_hist1 )
  stack1.Draw("hist")
  stack1.SetMaximum( stack1.GetMaximum()*1.4 )
  stack1.Draw("hist")

  legPF = "F"
  leg = ROOT.TLegend(.4,.67,.65,.92)

  #file2 included
  if len(files) > 1 :
    file2 = ROOT.TFile.Open( files[1][1] )
    if file2 == None : raise Exception( "{} does not exist!".format(files[1][1]) )
    label2 = files[1][0]
    label = label1 + "_vs_" + label2

    h2 = file2.FindObjectAny( var )
    if h2 == None : raise Exception( "Could not find {} histogram in {}!".format(var, files[1][1]) )
    avg2 = int( h2.GetMean()+0.5 )

    d_hist2 = getHists( file2, var, avg2, r )

    stack2 = ROOT.THStack( "stack2", "stack2" )
    setStack( stack2, d_hist2 )
    stack2.Draw("samepe");

    legPF = "PF"
    leg.SetHeader( "#bf{{Markers: {}, Histograms: {}}}".format(label2, label1) )

    #Draw Markers for EM Deposits and Hadronic Deposits in two separate regions
    hfe_clone = d_hist2["hfe"].Clone("hfe_clone")
    hfh_clone = d_hist2["hfh"].Clone("hfh_clone")

    cloneStack = ROOT.THStack( "cloneStack", "cloneStack" )
    cloneStack.Add(d_hist2["ne"])
    cloneStack.Add(hfe_clone)
    cloneStack.Add(d_hist2["nh"])
    cloneStack.Add(hfh_clone)
    cloneStack.Draw("samepe")

    #
    # Disable plotting restrictions for fixing Phase 2 offset plots.
    # Better to have busy plots than missing info...
    # Some clever trick for skipping zero fractions could be implemented
    # for the second PR
    # -Juska 24 June 2019
    #
    
    #d_hist2["ne"] .SetAxisRange(-2.9,2.9)
    #d_hist2["hfe"].SetAxisRange(-5,-2.6)
    #d_hist2["nh"] .SetAxisRange(-2.9,2.9)
    #d_hist2["hfh"].SetAxisRange(-5,-2.6)
    #d_hist2["chu"].SetAxisRange(-2.9,2.9)
    #d_hist2["chm"].SetAxisRange(-2.9,2.9)

    #hfe_clone     .SetAxisRange(2.6,5)
    #hfh_clone     .SetAxisRange(2.6,5)

  leg.SetBorderSize(0)
  leg.SetFillColor(0)
  leg.SetFillStyle(0)
  leg.SetTextSize(0.04)
  leg.SetTextFont(42)

  leg.AddEntry( d_hist1["ne"],  "Photons",                  legPF )
  leg.AddEntry( d_hist1["hfe"], "EM Deposits",              legPF )
  leg.AddEntry( d_hist1["nh"],  "Neutral Hadrons",          legPF )
  leg.AddEntry( d_hist1["hfh"], "Hadronic Deposits",        legPF )
  leg.AddEntry( d_hist1["chu"], "Unassoc. Charged Hadrons", legPF )
  leg.AddEntry( d_hist1["chm"], "Assoc. Charged Hadrons",   legPF )

  leg.Draw()

  text = ROOT.TLatex()
  text.SetNDC()

  text.SetTextSize(0.065)
  text.SetTextFont(61)
  text.DrawLatex(0.2, 0.87, "CMS")

  text.SetTextSize(0.045)
  text.SetTextFont(42)
  text.DrawLatex(1-len(label)/41., 0.96, label)

  outName = outdir + "/stack_" + label + ".pdf"
  c.Print( outName )
  return outName

def getHists( file, var, var_val, r ) :
  dict = {}

  var_val_range=var_val;
  if var=="mu" and var_val>=muHighOffset: var_val_range=muHighOffset-1
  if var=="npv" and var_val>=npvHighOffset: var_val_range=npvHighOffset-1
  
  for pf in candidateType :

    name = "p_offset_eta_{}{}_{}".format( var, var_val_range, pf )
    p = file.FindObjectAny(name)
    if p == None : raise Exception( "Could not find {} profile in {}!".format(name, file.GetName()) )
    dict[pf] = p.ProjectionX( pf )
    dict[pf].Scale( r*r / 2 / var_val )
    
    xbins = p.GetXaxis().GetXbins().GetArray()
    for i in range(1, p.GetNbinsX()+1) :
      dict[pf].SetBinContent( i, dict[pf].GetBinContent(i) / (xbins[i]-xbins[i-1]) )
      dict[pf].SetBinError( i, dict[pf].GetBinError(i) / (xbins[i]-xbins[i-1]) )

  return dict

def setStack( stack, hists ) :

  stack.Add( hists["ne"] )
  stack.Add( hists["hfe"] )
  stack.Add( hists["nh"] )
  stack.Add( hists["hfh"] )
  stack.Add( hists["chu"] )
  stack.Add( hists["chm"] )

  hists["ne"] .SetMarkerStyle(ROOT.kMultiply)
  hists["hfe"].SetMarkerStyle(ROOT.kOpenStar)
  hists["nh"] .SetMarkerStyle(ROOT.kOpenDiamond)
  hists["hfh"].SetMarkerStyle(ROOT.kOpenTriangleUp)
  hists["chu"].SetMarkerStyle(ROOT.kOpenCircle)
  hists["chm"].SetMarkerStyle(ROOT.kOpenCircle)

  hists["ne"] .SetFillColor(ROOT.kBlue)
  hists["hfe"].SetFillColor(ROOT.kViolet+2)
  hists["nh"] .SetFillColor(ROOT.kGreen)
  hists["hfh"].SetFillColor(ROOT.kPink+6)
  hists["chu"].SetFillColor(ROOT.kRed-9)
  hists["chm"].SetFillColor(ROOT.kRed)

  hists["ne"] .SetLineColor(ROOT.kBlack)
  hists["hfe"].SetLineColor(ROOT.kBlack)
  hists["nh"] .SetLineColor(ROOT.kBlack)
  hists["hfh"].SetLineColor(ROOT.kBlack)
  hists["chu"].SetLineColor(ROOT.kBlack)
  hists["chm"].SetLineColor(ROOT.kBlack)

def setStyle() :

  ROOT.gStyle.SetPadTopMargin(0.05)
  ROOT.gStyle.SetPadBottomMargin(0.1)
  ROOT.gStyle.SetPadLeftMargin(0.16)
  ROOT.gStyle.SetPadRightMargin(0.02)

  ROOT.gStyle.SetOptStat(0)
  ROOT.gStyle.SetOptTitle(0)

  ROOT.gStyle.SetTitleFont(42, "XYZ")
  ROOT.gStyle.SetTitleSize(0.05, "XYZ")
  ROOT.gStyle.SetTitleXOffset(0.9)
  ROOT.gStyle.SetTitleYOffset(1.4)

  ROOT.gStyle.SetLabelFont(42, "XYZ")
  ROOT.gStyle.SetLabelOffset(0.007, "XYZ")
  ROOT.gStyle.SetLabelSize(0.04, "XYZ")

  ROOT.gStyle.SetPadTickX(1)
  ROOT.gStyle.SetPadTickY(1)

if __name__ == "__main__":
    main()
