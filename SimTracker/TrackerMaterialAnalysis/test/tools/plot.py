#! /usr/bin/env python

import sys
import material
import ROOT

def usage():
  print "Usage..."

def plot():
  if (len(sys.argv) < 2) or (sys.argv[1] not in ("r", "z", "eta")):
    usage()
    sys.exit(1)
  dir_name  = sys.argv[1]
  direction = material.Element.directions[dir_name]
  dir_label = material.Element.dir_labels[dir_name]
  layers = sys.argv[2:]

  elements = []
  for layer in layers:
    elements += material.parse(file("layers/" + layer))
  if len(elements) == 0:
    sys.exit(1)

  positions = set()
  for element in elements:
    positions.add(element.position[direction])
  positions = sorted(positions)

  data   = ROOT.TFile("TrackerRecMaterial.root")
  canvas = ROOT.TCanvas("material", "", 800, 600)
  if dir_name == "r":
    modules = ROOT.TH1F("modules", "material along R",   1200,    0, 120)
  elif dir_name == "z":
    modules = ROOT.TH1F("modules", "material along Z",   6000, -300, 300)
  else:
    modules = ROOT.TH1F("modules", "material along Eta",   60,   -3,   3)
  for position in positions:
    modules.SetBinContent(modules.GetXaxis().FindBin(position), 1)
  modules.SetLineColor(ROOT.kBlack)
  modules.SetFillStyle(1001)
  modules.SetFillColor(ROOT.kBlack)
  modules.SetMaximum(1)
  modules.Draw()
  dedx   = dict( (layer, data.Get("%s_dedx_vs_%s"   % (layer, dir_name))) for layer in layers )
  radlen = dict( (layer, data.Get("%s_radlen_vs_%s" % (layer, dir_name))) for layer in layers )
  for layer in layers:
    dedx[layer].SetLineColor(ROOT.kRed)
    dedx[layer].Draw("same")
    radlen[layer].SetLineColor(ROOT.kGreen)
    radlen[layer].Draw("same")

  canvas.Update()

  while(canvas):
    pass


if __name__ == "__main__":
  plot()
