#include <iostream>

#include <TColor.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TFrame.h>
#include "XHistogram.h"
#include "TrackingMaterialPlotter.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"

void TrackingMaterialPlotter::fill_color(void)
{
  m_color.push_back(kBlack);          // unassigned
  m_color.push_back(kBlue);           // PixelBarrel
  m_color.push_back(kBlue  + 100);    //
  m_color.push_back(kBlue);           //
  m_color.push_back(kGreen);          // TIB
  m_color.push_back(kGreen + 100);    //
  m_color.push_back(kGreen);          //
  m_color.push_back(kGreen + 100);    //
  m_color.push_back(kRed);            // TOB
  m_color.push_back(kRed   + 100);    //
  m_color.push_back(kRed);            //
  m_color.push_back(kRed   + 100);    //
  m_color.push_back(kRed);            //
  m_color.push_back(kRed   + 100);    //
  m_color.push_back(kBlue);           // PixelEndcap Z-
  m_color.push_back(kBlue  + 100);    //
  m_color.push_back(kGreen);          // TID Z-
  m_color.push_back(kGreen + 100);    //
  m_color.push_back(kGreen);          //
  m_color.push_back(kRed);            // TEC Z-
  m_color.push_back(kRed   + 100);    //
  m_color.push_back(kRed);            //
  m_color.push_back(kRed   + 100);    //
  m_color.push_back(kRed);            //
  m_color.push_back(kRed   + 100);    //
  m_color.push_back(kRed);            //
  m_color.push_back(kRed   + 100);    //
  m_color.push_back(kRed);            //
  m_color.push_back(kBlue);           // PixelEndcap Z+
  m_color.push_back(kBlue  + 100);    //
  m_color.push_back(kGreen);          // TID Z+
  m_color.push_back(kGreen + 100);    //
  m_color.push_back(kGreen);          //
  m_color.push_back(kRed);            // TEC Z+
  m_color.push_back(kRed   + 100);    //
  m_color.push_back(kRed);            //
  m_color.push_back(kRed   + 100);    //
  m_color.push_back(kRed);            //
  m_color.push_back(kRed   + 100);    //
  m_color.push_back(kRed);            //
  m_color.push_back(kRed   + 100);    //
  m_color.push_back(kRed);            //
}


void TrackingMaterialPlotter::fill_gradient(void)
{
  const unsigned int index = 1000;
  const unsigned int steps =  100;
  m_gradient.resize(steps);
  for (unsigned int i = 0; i < steps; ++i)
    m_gradient[i] = index + i;
  
  float r1 = 1.0, g1 = 1.0, b1 = 1.0;   // white
  float r2 = 0.0, g2 = 0.0, b2 = 0.0;   // black
  float delta_r = (r2 - r1) / (steps - 1);
  float delta_g = (g2 - g1) / (steps - 1);
  float delta_b = (b2 - b1) / (steps - 1);
  
  for (unsigned int i = 0; i < steps; ++i)
    new TColor(m_gradient[i], r1 + delta_r * i, g1 + delta_g * i, b1 + delta_b * i);
}


TrackingMaterialPlotter::TrackingMaterialPlotter( float maxZ, float maxR, float resolution )
{ 
  const float rzMinZ  = -maxZ;
  const float rzMaxZ  =  maxZ;
  const float rzMinR  =    0.;
  const float rzMaxR  =  maxR;
  const int   rzBinsZ = (int) (2. * maxZ * resolution);
  const int   rzBinsR = (int) (     maxR * resolution);

  std::vector<double> max;
  max.push_back( 0.02 ); 
  max.push_back( 0.04 ); 
  m_tracker = XHistogram( 2, rzBinsZ, rzBinsR, std::make_pair(rzMinZ, rzMaxZ), std::make_pair(rzMinR, rzMaxR), m_color.size(), max);

  fill_color();
  fill_gradient();
}
 
void TrackingMaterialPlotter::plotSegmentUnassigned( const MaterialAccountingStep & step )
{ 
  std::vector<double> w(2);
  w[0] = step.radiationLengths();
  w[1] = step.energyLoss();
  m_tracker.fill( std::make_pair(step.in().z(), step.out().z()),
                  std::make_pair(step.in().perp(), step.out().perp()),
                  w, step.length(), 1 );             // 0 is empty, 1 is unassigned
}

void TrackingMaterialPlotter::plotSegmentInLayer( const MaterialAccountingStep & step, int layer )
{ 
  std::vector<double> w(2);
  w[0] = step.radiationLengths();
  w[1] = step.energyLoss();
  m_tracker.fill( std::make_pair(step.in().z(), step.out().z()),
                  std::make_pair(step.in().perp(), step.out().perp()),
                  w, step.length(), layer+1 );       // layer is 1-based, but plot uses: 0 is empty, 1 is unassigned
}


void TrackingMaterialPlotter::draw( void ) 
{
  const double scale = 10.;
  TCanvas* canvas;
 
  XHistogram::Histogram* radlen = m_tracker.get(0);
  canvas = new TCanvas("radlen_rz", "RadiationLengths - RZ view", (int) (600 * scale * 1.25),  (int) (120 * scale * 1.50));
  gStyle->SetOptStat(0);
  gStyle->SetPalette( m_gradient.size(), & m_gradient.front() );
  gStyle->SetNumberContours( m_gradient.size() );
  canvas->GetFrame()->SetFillColor(kWhite);
  radlen->Draw("colz");
  radlen->Draw("same axis y+");
  canvas->SaveAs("radlen.png");

  XHistogram::Histogram* dedx = m_tracker.get(1);
  canvas = new TCanvas("dedx_rz", "-dE/dx term - RZ view", (int) (600 * scale * 1.25),  (int) (120 * scale * 1.50));
  canvas->GetFrame()->SetFillColor(kWhite);
  gStyle->SetOptStat(0);
  gStyle->SetPalette( m_gradient.size(), & m_gradient.front() );
  gStyle->SetNumberContours( m_gradient.size() );
  dedx->Draw("colz");
  dedx->Draw("same axis y+");
  canvas->SaveAs("dedx.png");

  XHistogram::ColorMap* colormap = m_tracker.colormap();
  canvas = new TCanvas("layer_rz", "Layers - RZ view", (int) (600 * scale * 1.25),  (int) (120 * scale * 1.50));
  canvas->GetFrame()->SetFillColor(kWhite);
  gStyle->SetOptStat(0);
  gStyle->SetPalette( m_color.size(), & m_color.front() );
  gStyle->SetNumberContours( m_color.size() );
  colormap->SetMinimum( 1 );
  colormap->SetMaximum( m_color.size() );
  colormap->Draw("col");
  colormap->Draw("same axis y+");
  canvas->SaveAs("layers.png");
}

