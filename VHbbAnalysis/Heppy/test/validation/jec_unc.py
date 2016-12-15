import ROOT, sys, os
from PhysicsTools.Heppy.physicsutils.JetReCalibrator import JetReCalibrator

files = os.environ["FILE_NAMES"].split()
files = ["root://storage01.lcg.cscs.ch//pnfs/lcg.cscs.ch/cms/trivcat/" + fi for fi in files]

of = ROOT.TFile("out.root", "RECREATE")
tt = ROOT.TChain("vhbb/tree")
for fi in files:
    fi = fi.strip()
    print fi
    tt.AddFile(fi)

of.cd()
tt.Draw("Jet_pt[0] >>+ nominal(200,20,520)")
tt.Draw("Jet_pt[0]*Jet_corr[0] >>+ corr(200,20,520)")
for jet_corr in JetReCalibrator.factorizedJetCorrections + ["JEC"]:
    s1 = "(Jet_pt[0] * Jet_corr_{0}Up[0]) >>+ h__{0}__Up(200,20,520)".format(jet_corr)
    s2 = "(Jet_pt[0] * Jet_corr_{0}Down[0]) >>+ h__{0}__Down(200,20,520)".format(jet_corr)
    print s1
    print s2
    tt.Draw(s1)
    tt.Draw(s2)
of.Write()
of.Close()
