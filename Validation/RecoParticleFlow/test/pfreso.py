
import ROOT as r
import sys
import array as ar


shf=-7; shf2=1
colors = [r.kRed-4,r.kRed+shf,r.kBlue+shf,r.kGreen+shf+1,r.kCyan+shf+1,
            r.kMagenta+shf,r.kGray+1,r.kOrange,r.kCyan]
darkcolors = [r.kRed-2,r.kRed+shf2,r.kBlue+shf2,r.kGreen+shf2+1,r.kCyan+shf2+1,
            r.kMagenta+shf2,r.kGray+2,r.kOrange+1,r.kCyan+1]

fullmarkers = [r.kFullCircle,r.kFullSquare,r.kFullDiamond,r.kFullTriangleUp,
                     r.kFullTriangleDown,r.kFullStar,r.kFullCross,31]
openmarkers = [r.kOpenCircle,r.kOpenSquare,r.kOpenDiamond,
               r.kOpenTriangleUp,r.kOpenTriangleDown,r.kOpenStar,r.kOpenCross]
               
def pf_resolution(forig, fdev, path, apdx):
   
   # Eta bin from apdx
   if (apdx == "eta05"):
      info = "|#eta| < 0.5"
   if (apdx == "eta13"):
      info = "|#eta| < 1.3"
   if (apdx == "eta21"):
      info = "1.3 < |#eta| < 2.1"
   if (apdx == "eta25"):
      info = "2.1 < |#eta| < 2.5"
   if (apdx == "eta30"):
      info = "2.5 < |#eta| < 3.0"
   
   creso = r.TCanvas("resolution","Jet pT resolution", 600, 600)
   
   # Loop responses i.e. dist widths to an array. Or TProfile.
   ptbins = ar.array('d',[10,24,32,43,56,74,97,133,174,245,300,362,430,
                  507,592,686,846,1032,1248,1588,2000,2500,3000,4000,6000])
   nptbins = len(ptbins)-1
   
   #etabins = [[0,0.5],[0,1.3],[1.3,2.1],[2.1,2.5],[2.5,3.0]]
   
   preso = r.TProfile("preso","Jet pT resolution",nptbins,ptbins)
   presodev = r.TProfile("presodev","Jet pT resolution",nptbins,ptbins)
   
   leg = r.TLegend(0.5,0.75,0.9,0.85)
   leg.SetFillStyle(0);
   leg.SetBorderSize(0);
   leg.SetEntrySeparation(0);
   leg.AddEntry(preso,"Original PF","pl")
   leg.AddEntry(presodev,"Modified PF","pl")
   
   
   
   for idx in range(nptbins):
      elow = ptbins[idx]
      ehigh = ptbins[idx+1]
      h = forig.Get("%sreso_dist_%i_%i_%s" % (path,elow,ehigh,apdx))
      hdev = fdev.Get("%sreso_dist_%i_%i_%s" % (path,elow,ehigh,apdx))
      std = h.GetStdDev()
      err = h.GetStdDevError()
      stddev = hdev.GetStdDev()
      errdev = hdev.GetStdDevError()
      preso.Fill(elow,std)
      presodev.Fill(elow,stddev)
      # Set errors
      preso.SetBinError(idx,err)
      presodev.SetBinError(idx,errdev)
      
   preso.SetXTitle("Jet pT [GeV]")
   preso.SetYTitle("#sigma(p_{T}^{reco}/p_{T}^{gen})")
   preso.GetYaxis().SetTitleOffset(1.3)
   
   preso.SetMarkerStyle(fullmarkers[0])
   preso.SetMarkerColor(colors[1])
   preso.SetLineColor(colors[1])
   
   presodev.SetMarkerStyle(openmarkers[1])
   presodev.SetMarkerColor(colors[2])
   presodev.SetLineColor(colors[2])
   
   
   r.gStyle.SetOptStat(0)
   preso.SetAxisRange(10,3000,"X")
   preso.SetAxisRange(0,0.5,"Y")
   preso.Draw("p")
   presodev.Draw("psame")
   leg.Draw()
   creso.SetLogx()
   preso.GetXaxis().SetMoreLogLabels()
   preso.GetXaxis().SetNoExponent()
   
   t = r.TLatex()
   t.SetTextColor(1)
   t.SetTextSize(0.05)
   t.SetNDC()
   
   t.DrawLatex(0.15,0.15,info)
   
   creso.SaveAs("res/pfreso_%s.pdf" % apdx)
      
   
   
def main():

   forig = r.TFile("/home/juska/Dropbox/postdoc/pfdev/files/"
      "DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root","read")

   path = "DQMData/Run 1/Physics/Run summary/JetResponse/"

   #dresoorig = f.Get(path)
   #print dresoorig
   
   fdev = r.TFile("/home/juska/Dropbox/postdoc/pfdev/files/"
      "DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root","read")
   
   #print fdev
   
   dresodev = fdev.Get(path)
   
   
   #print dresodev
#   h0 = fdev.Get("DQMData/Run 1/Physics/Run summary/JetResponse/reso_dist_1032_1248_eta05")
   
   
   
   
   pf_resolution(forig, fdev, path, "eta05")
   pf_resolution(forig, fdev, path, "eta13")
   pf_resolution(forig, fdev, path, "eta21")
   pf_resolution(forig, fdev, path, "eta25")
   pf_resolution(forig, fdev, path, "eta30")
   
   #input("Press enter to quit")
main()
