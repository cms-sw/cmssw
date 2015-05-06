from ROOT import TFile, TCanvas, TPad, TH1F, TProfile, TProfile2D, TLine, gStyle, kOrange, kOpenCircle, kFullCircle
from math import sqrt

def isTrackerMuon(chain, index):
  if chain.muon_type[index] & 4 == 4:
    return True

def inBarrel(chain, index):
  """
  Establish if the outer hit of a muon is in the barrel region.
  """
  if abs(chain.muon_outerPositionz[index]) < 108:
    return True

def inForward(chain, index):
  """
  Establish if the outer hit of a muon is in the last layer of TEC in
  the forward region.
  """
  if abs(chain.muon_outerPositionz[index]) > 260:
    return True

def isWithinZ(chain, index, value):
  """
  Establish if a muon is close to 0 in Z by value @value.
  """
  if abs(chain.muon_dz[index]) <= value:
    return True

def overlayAndRatio(canvas, min_ratio, max_ratio, h1, h2):
  canvas.ResetAttPad()
  canvas.Clear()
  pad = TPad("main","main", 0, 0.3, 1, 1)
  pad.SetBottomMargin(0.05);
  pad.Draw()
  pad.cd()
  h1.Draw()
  h2.SetLineColor(kOrange+10)
  h2.Draw('SAME')
  canvas.cd()
  ratio = TPad("ratio","ratio", 0, 0.05, 1, 0.3);
  ratio.SetTopMargin(0.05);
  ratio.Draw()
  ratio.cd()
  if isinstance(h1, TProfile) and isinstance(h2, TProfile):
    h1_p = h1.ProjectionX()
    h2_p = h2.ProjectionX()
    h1_p.Divide(h2_p)
    h1_p.SetMaximum(max_ratio)
    h1_p.SetMinimum(min_ratio)
    h1_p.SetMarkerStyle(kFullCircle)
    h1_p.SetMarkerSize(0.6)
    h1_p.SetTitle('')
    h1_p.GetXaxis().SetLabelFont(42)
    h1_p.GetXaxis().SetLabelSize(0.1)
    h1_p.GetYaxis().SetLabelFont(42)
    h1_p.GetYaxis().SetLabelSize(0.1)
    h1_p.GetYaxis().SetNdivisions(505)
    equality = TLine()
    h1_p.Draw("SAME HIST P")
    equality.SetLineColor(kOrange+10)
    equality.DrawLine(h1_p.GetXaxis().GetXmin(), 1, h1_p.GetXaxis().GetXmax(), 1)
  


gStyle.SetOptStat(0)

files = ['trackTupla.root', 'trackTuplaNewMaterial.root']
destinations = ['/afs/cern.ch/work/r/rovere/public/temporary/materialEffects/CurrentGeometry',
                '/afs/cern.ch/work/r/rovere/public/temporary/materialEffects/NewMaterialGeometry']

delta_p_xy = []
delta_p_xy.append(TProfile2D('Delta_p_Old', 'Delta_p', 240, -120, 120, 240, -120, 120))
delta_p_xy.append(TProfile2D('Delta_p_New', 'Delta_p', 240, -120, 120, 240, -120, 120))
outermost_z = []
outermost_z.append(TH1F("outermost_z_Old", "outermost_z", 560, -280, 280))
outermost_z.append(TH1F("outermost_z_New", "outermost_z", 560, -280, 280))
delta_p_outermost_z = []
delta_p_outermost_z.append(TProfile("Delta_p_outermost_z_Old", "Delta_p_outermost_z", 560, -280, 280))
delta_p_outermost_z.append(TProfile("Delta_p_outermost_z_New", "Delta_p_outermost_z", 560, -280, 280))
delta_p_rz = []
delta_p_rz.append(TProfile2D("Delta_p_rz_Old", "Delta_p_rz", 600, -300, 300, 120, 0, 120))
delta_p_rz.append(TProfile2D("Delta_p_rz_New", "Delta_p_rz", 600, -300, 300, 120, 0, 120))
delta_p_eta = []
delta_p_eta.append(TProfile("Delta_p_eta_Old", "Delta_p_eta", 100, -2.5, 2.5))
delta_p_eta.append(TProfile("Delta_p_eta_New", "Delta_p_eta", 100, -2.5, 2.5))
delta_pt_eta = []
delta_pt_eta.append(TProfile("Delta_pt_eta_Old", "Delta_pt_eta", 100, -2.5, 2.5))
delta_pt_eta.append(TProfile("Delta_pt_eta_New", "Delta_pt_eta", 100, -2.5, 2.5))
delta_pl_eta = []
delta_pl_eta.append(TProfile("Delta_pl_eta_Old", "Delta_pl_eta", 100, -2.5, 2.5))
delta_pl_eta.append(TProfile("Delta_pl_eta_New", "Delta_pl_eta", 100, -2.5, 2.5))

c = TCanvas("c", "c", 1024, 1024)
counter = 0
for f in files:
  fh = TFile.Open(f)
  chain = fh.Get('Tracks')
  entries = chain.GetEntriesFast()

  for e in xrange(entries):
    nb = chain.GetEntry(e)
    num_muons = chain.nmuon
    for m in xrange(num_muons):
      delta_p_value = abs(chain.muon_innerMom[m] - chain.muon_outerMom[m])
      delta_pt_value = sqrt((chain.muon_innerMomx[m] - chain.muon_outerMomx[m])**2 +
                            (chain.muon_innerMomy[m] - chain.muon_outerMomy[m])**2)
      delta_pl_value = abs(chain.muon_innerMomz[m] - chain.muon_outerMomz[m])
      if isTrackerMuon(chain, m):
        outermost_z[counter].Fill(chain.muon_outerPositionz[m])
        delta_p_outermost_z[counter].Fill(chain.muon_outerPositionz[m], delta_p_value)
        delta_p_xy[counter].Fill(chain.muon_outerPositionx[m],
                                 chain.muon_outerPositiony[m],
                                 delta_p_value)
        delta_p_rz[counter].Fill(chain.muon_outerPositionz[m],
                                 sqrt(chain.muon_outerPositionx[m]**2+chain.muon_outerPositiony[m]**2),
                                 delta_p_value)
        delta_p_eta[counter].Fill(chain.muon_eta[m], delta_p_value)
        delta_pt_eta[counter].Fill(chain.muon_eta[m], delta_pt_value)
        delta_pl_eta[counter].Fill(chain.muon_eta[m], delta_pl_value)

  outermost_z[counter].Draw()
  c.SaveAs("%s/OuterMostMuonHitInTracker.png" % destinations[counter])

  delta_p_outermost_z[counter].SetMaximum(0.2)
  delta_p_outermost_z[counter].Draw()
  c.SaveAs("%s/DeltaP_OuterMostMuonHitInTracker.png" % destinations[counter])

  delta_p_rz[counter].SetMaximum(4)
  delta_p_rz[counter].Draw('COLZ')
  c.SaveAs("%s/DeltaP_RZ.png" % destinations[counter])
  
  delta_p_eta[counter].SetMaximum(0.2)
  delta_p_eta[counter].Draw('COLZ')
  c.SaveAs("%s/DeltaP_ETA.png" % destinations[counter])

  for val in [1, 0.5, 0.1]:
    delta_p_xy[counter].SetMaximum(val)
    delta_p_xy[counter].Draw("COLZ")
    c.SaveAs("%s/DeltaP_XY_Max%f.png" % (destinations[counter],val))
  counter +=1

overlayAndRatio(c, 0.85, 1.15, *delta_p_outermost_z)
c.SaveAs("%s/DeltaP_OuterMostMuonHitInTracker_Comparison.png" % destinations[0])
c.SaveAs("%s/DeltaP_OuterMostMuonHitInTracker_Comparison.png" % destinations[1])

overlayAndRatio(c, 0.85, 1.15, *delta_p_eta)
c.SaveAs("%s/DeltaP_Eta_Comparison.png" % destinations[0])
c.SaveAs("%s/DeltaP_Eta_Comparison.png" % destinations[1])

overlayAndRatio(c, 0.999, 1.001, *delta_pt_eta)
c.SaveAs("%s/DeltaPt_Eta_Comparison.png" % destinations[0])
c.SaveAs("%s/DeltaPt_Eta_Comparison.png" % destinations[1])

delta_pl_eta[0].SetMaximum(0.2)
delta_pl_eta[1].SetMaximum(0.2)
overlayAndRatio(c, 0.85, 1.15, *delta_pl_eta)
c.SaveAs("%s/DeltaPl_Eta_Comparison.png" % destinations[0])
c.SaveAs("%s/DeltaPl_Eta_Comparison.png" % destinations[1])


