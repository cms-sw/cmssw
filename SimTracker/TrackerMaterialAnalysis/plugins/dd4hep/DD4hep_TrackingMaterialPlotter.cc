#include <iostream>

#include <TROOT.h>
#include <TObjArray.h>
#include <TColor.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TFrame.h>
#include "DD4hep_XHistogram.h"
#include "DD4hep_TrackingMaterialPlotter.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"

void DD4hep_TrackingMaterialPlotter::fill_color(void) {
  m_color.push_back(kBlack);        // unassigned
  m_color.push_back(kAzure);        // PixelBarrel
  m_color.push_back(kAzure + 1);    //
  m_color.push_back(kAzure + 1);    //
  m_color.push_back(kAzure + 3);    //
  m_color.push_back(kAzure + 3);    //
  m_color.push_back(kGreen);        // TIB
  m_color.push_back(kGreen);        //
  m_color.push_back(kGreen + 2);    //
  m_color.push_back(kGreen + 2);    //
  m_color.push_back(kGreen - 3);    //
  m_color.push_back(kGreen - 3);    //
  m_color.push_back(kGreen - 1);    //
  m_color.push_back(kGreen - 1);    //
  m_color.push_back(kRed);          // TOB
  m_color.push_back(kRed);          //
  m_color.push_back(kRed);          //
  m_color.push_back(kRed + 3);      //
  m_color.push_back(kRed + 3);      //
  m_color.push_back(kRed + 3);      //
  m_color.push_back(kRed - 3);      //
  m_color.push_back(kRed - 3);      //
  m_color.push_back(kRed - 3);      //
  m_color.push_back(kOrange + 9);   //
  m_color.push_back(kOrange + 9);   //
  m_color.push_back(kOrange + 9);   //
  m_color.push_back(kOrange + 7);   //
  m_color.push_back(kOrange + 7);   //
  m_color.push_back(kOrange + 7);   //
  m_color.push_back(kOrange + 5);   //
  m_color.push_back(kOrange + 5);   //
  m_color.push_back(kOrange + 5);   //
  m_color.push_back(kOrange + 8);   // PixelEndcap Z-
  m_color.push_back(kOrange + 10);  //
  m_color.push_back(kOrange - 3);   //
  m_color.push_back(kOrange - 1);   // PixelEndcap Z+
  m_color.push_back(kOrange - 8);   //
  m_color.push_back(kYellow);       // TID Z-
  m_color.push_back(kYellow);       //
  m_color.push_back(kYellow + 2);   //
  m_color.push_back(kYellow + 2);   //
  m_color.push_back(kYellow + 2);   //
  m_color.push_back(kYellow + 3);   //
  m_color.push_back(kMagenta);      //
  m_color.push_back(kMagenta);      //
  m_color.push_back(kMagenta);      //
  m_color.push_back(kMagenta);      //
  m_color.push_back(kMagenta);      //
  m_color.push_back(kMagenta + 1);  //
  m_color.push_back(kMagenta + 2);  //
  m_color.push_back(kMagenta + 3);  //
  m_color.push_back(kMagenta + 4);  //
  m_color.push_back(kMagenta + 5);  //
  m_color.push_back(kMagenta + 6);  //
  m_color.push_back(kMagenta + 7);  //
  m_color.push_back(kMagenta + 8);  //
}

unsigned int DD4hep_TrackingMaterialPlotter::fill_gradient(const TColor& first,
                                                           const TColor& last,
                                                           unsigned int steps /*= 100*/,
                                                           unsigned int index /* = 0*/) {
  if (index == 0) {
    // if no index was given, find the highest used one and start from that plus one
    index = ((TObjArray*)gROOT->GetListOfColors())->GetLast() + 1;
  }

  float r1, g1, b1, r2, g2, b2;
  first.GetRGB(r1, g1, b1);
  last.GetRGB(r2, g2, b2);
  float delta_r = (r2 - r1) / (steps - 1);
  float delta_g = (g2 - g1) / (steps - 1);
  float delta_b = (b2 - b1) / (steps - 1);

  m_gradient.resize(steps);
  for (unsigned int i = 0; i < steps; ++i) {
    new TColor(static_cast<Int_t>(index + i), r1 + delta_r * i, g1 + delta_g * i, b1 + delta_b * i);
    m_gradient[i] = index + i;
  }

  return index;
}

unsigned int DD4hep_TrackingMaterialPlotter::fill_gradient(unsigned int first,
                                                           unsigned int last,
                                                           unsigned int steps /*= 100*/,
                                                           unsigned int index /* = 0*/) {
  return fill_gradient(
      *(TColor*)gROOT->GetListOfColors()->At(first), *(TColor*)gROOT->GetListOfColors()->At(last), steps, index);
}

DD4hep_TrackingMaterialPlotter::DD4hep_TrackingMaterialPlotter(float maxZ, float maxR, float resolution) {
  const float rzMinZ = -maxZ;
  const float rzMaxZ = maxZ;
  const float rzMinR = 0.;
  const float rzMaxR = maxR;
  const int rzBinsZ = (int)(2. * maxZ * resolution);
  const int rzBinsR = (int)(maxR * resolution);

  std::vector<double> max;
  max.push_back(0.08);
  max.push_back(0.00016);
  m_tracker = DD4hep_XHistogram(
      2, rzBinsZ, rzBinsR, std::make_pair(rzMinZ, rzMaxZ), std::make_pair(rzMinR, rzMaxR), m_color.size(), max);

  TColor::InitializeColors();
  fill_color();
  fill_gradient(kWhite, kBlack, 100);  // 100-steps gradient from white to black
}

void DD4hep_TrackingMaterialPlotter::plotSegmentUnassigned(const MaterialAccountingStep& step) {
  std::vector<double> w(2);
  w[0] = step.radiationLengths();
  w[1] = step.energyLoss();
  m_tracker.fill(std::make_pair(step.in().z(), step.out().z()),
                 std::make_pair(step.in().perp(), step.out().perp()),
                 w,
                 step.length(),
                 1);  // 0 is empty, 1 is unassigned
}

void DD4hep_TrackingMaterialPlotter::plotSegmentInLayer(const MaterialAccountingStep& step, int layer) {
  std::vector<double> w(2);
  w[0] = step.radiationLengths();
  w[1] = step.energyLoss();
  m_tracker.fill(std::make_pair(step.in().z(), step.out().z()),
                 std::make_pair(step.in().perp(), step.out().perp()),
                 w,
                 step.length(),
                 layer + 1);  // layer is 1-based, but plot uses: 0 is empty, 1 is unassigned
}

void DD4hep_TrackingMaterialPlotter::draw(void) {
  const double scale = 10.;
  TCanvas* canvas;

  DD4hep_XHistogram::Histogram* radlen = m_tracker.get(0);
  canvas = new TCanvas("radlen_rz", "RadiationLengths - RZ view", (int)(600 * scale * 1.25), (int)(120 * scale * 1.50));
  gStyle->SetOptStat(0);
  gStyle->SetPalette(m_gradient.size(), &m_gradient.front());
  gStyle->SetNumberContours(m_gradient.size());
  canvas->GetFrame()->SetFillColor(kWhite);
  radlen->Draw("colz");
  radlen->Draw("same axis y+");
  radlen->SaveAs("radlen.root");
  canvas->SaveAs("radlen.png");
  // Replicate RainBow palette, with White in the first white_slots
  // positions
  int white_slots = 1;
  int MyPalette[100];
  double stops[9] = {0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750, 1.0000};
  double red[9] = {
      0. / 255., 5. / 255., 15. / 255., 35. / 255., 102. / 255., 196. / 255., 208. / 255., 199. / 255., 110. / 255.};
  double green[9] = {
      0. / 255., 48. / 255., 124. / 255., 192. / 255., 206. / 255., 226. / 255., 97. / 255., 16. / 255., 0. / 255.};
  double blue[9] = {
      99. / 255., 142. / 255., 198. / 255., 201. / 255., 90. / 255., 22. / 255., 13. / 255., 8. / 255., 2. / 255.};
  int palette_index = TColor::CreateGradientColorTable(9, stops, red, green, blue, 100 - white_slots);
  for (int i = 0; i < white_slots; i++)
    MyPalette[i] = kWhite;
  for (int i = 0; i < 100 - white_slots; i++)
    MyPalette[i + white_slots] = palette_index + i;
  canvas->Clear();
  gStyle->SetNumberContours(100);
  gStyle->SetPalette(100, MyPalette);  // ROOT Rainbow color palette
  radlen->Draw("colz");
  radlen->Draw("same axis y+");
  canvas->SaveAs("radlenColor.png");

  delete canvas;

  DD4hep_XHistogram::Histogram* dedx = m_tracker.get(1);
  canvas = new TCanvas("dedx_rz", "-dE/dx term - RZ view", (int)(600 * scale * 1.25), (int)(120 * scale * 1.50));
  canvas->GetFrame()->SetFillColor(kWhite);
  gStyle->SetOptStat(0);
  gStyle->SetPalette(m_gradient.size(), &m_gradient.front());
  gStyle->SetNumberContours(m_gradient.size());
  dedx->Draw("colz");
  dedx->Draw("same axis y+");
  dedx->SaveAs("dedx.root");
  canvas->SaveAs("dedx.png");
  canvas->Clear();
  gStyle->SetNumberContours(100);
  gStyle->SetPalette(100, MyPalette);  // ROOT Rainbow color palette
  dedx->Draw("colz");
  dedx->Draw("same axis y+");
  canvas->SaveAs("dedxColor.png");
  delete canvas;

  DD4hep_XHistogram::ColorMap* colormap = m_tracker.colormap();
  canvas = new TCanvas("layer_rz", "Layers - RZ view", (int)(600 * scale * 1.25), (int)(120 * scale * 1.50));
  canvas->GetFrame()->SetFillColor(kWhite);
  gStyle->SetOptStat(0);
  gStyle->SetPalette(m_color.size(), &m_color.front());
  gStyle->SetNumberContours(m_color.size());
  colormap->SetMinimum(1);
  colormap->SetMaximum(m_color.size());
  colormap->Draw("col");
  colormap->Draw("same axis y+");
  colormap->SaveAs("layers.root");
  canvas->SaveAs("layers.png");
  delete canvas;
}
