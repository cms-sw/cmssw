#include <sstream>
#include <string>

#include <TH1F.h>
#include <TCanvas.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingDetector.h"
#include "MaterialAccountingLayer.h"

MaterialAccountingLayer::MaterialAccountingLayer( const DetLayer & layer ) :
  m_layers( 1, & layer ),
  m_accounting(),
  m_errors(),
  m_tracks( 0 ),
  m_counted( false )
{
  m_dedx_spectrum   = new TH1F(0, "Energy loss spectrum", 1000,  0, 1);
  m_dedx_vs_eta     = new TH1F(0, "Energy loss vs. eta", 1000, -5, 5);
  m_dedx_vs_z       = new TH1F(0, "Energy loss vs. Z", 6000, -300, 300);
  m_dedx_vs_r       = new TH1F(0, "Energy loss vs. R", 1200, 0, 120);
  m_radlen_spectrum = new TH1F(0, "Radiation lengths spectrum", 1000,  0, 1);
  m_radlen_vs_eta   = new TH1F(0, "Radiation lengths vs. eta", 1000, -5, 5);
  m_radlen_vs_z     = new TH1F(0, "Radiation lengths vs. Z", 6000, -300, 300);
  m_radlen_vs_r     = new TH1F(0, "Radiation lengths vs. R", 1200, 0, 120);
}

MaterialAccountingLayer::MaterialAccountingLayer( std::vector<const DetLayer *> layers ) :
  m_layers( layers ),
  m_accounting(),
  m_errors(),
  m_tracks( 0 ),
  m_counted( false )
{
  m_dedx_spectrum   = new TH1F(0, "Energy loss spectrum", 1000,  0, 1);
  m_dedx_vs_eta     = new TH1F(0, "Energy loss vs. eta", 1000, -5, 5);
  m_dedx_vs_z       = new TH1F(0, "Energy loss vs. Z", 6000, -300, 300);
  m_dedx_vs_r       = new TH1F(0, "Energy loss vs. R", 1200, 0, 120);
  m_radlen_spectrum = new TH1F(0, "Radiation lengths spectrum", 1000,  0, 1);
  m_radlen_vs_eta   = new TH1F(0, "Radiation lengths vs. eta", 1000, -5, 5);
  m_radlen_vs_z     = new TH1F(0, "Radiation lengths vs. Z", 6000, -300, 300);
  m_radlen_vs_r     = new TH1F(0, "Radiation lengths vs. R", 1200, 0, 120);
}



MaterialAccountingLayer::~MaterialAccountingLayer(void)
{
  delete m_dedx_spectrum;
  delete m_dedx_vs_eta;
  delete m_dedx_vs_z;
  delete m_dedx_vs_r;
  delete m_radlen_spectrum;
  delete m_radlen_vs_eta;
  delete m_radlen_vs_z;
  delete m_radlen_vs_r;
}

bool MaterialAccountingLayer::addDetector( const MaterialAccountingDetector& detector ) {
  if (not inside(detector))
    return false;

  // multiple hits in the same layer (overlaps, etc.) from a single track still count as one for averaging,
  // since the energy deposits from the track have been already split between the different detectors
  m_buffer += detector.material();
  m_counted = true;

  return true;
}

void MaterialAccountingLayer::endOfTrack(void) {
  if (m_counted) {
    m_accounting += m_buffer;
    m_errors     += m_buffer * m_buffer;
    ++m_tracks;

    GlobalPoint average( (m_buffer.in().x() + m_buffer.out().x()) / 2.,
                         (m_buffer.in().y() + m_buffer.out().y()) / 2., 
                         (m_buffer.in().z() + m_buffer.out().z()) / 2. );
    m_dedx_spectrum->Fill(   m_buffer.energyLoss() );
    m_dedx_vs_eta->Fill(     average.eta(),  m_buffer.energyLoss() );
    m_dedx_vs_z->Fill(       average.z(),    m_buffer.energyLoss() );
    m_dedx_vs_r->Fill(       average.perp(), m_buffer.energyLoss() );
    m_radlen_spectrum->Fill( m_buffer.radiationLengths() );
    m_radlen_vs_eta->Fill(   average.eta(),  m_buffer.radiationLengths() );
    m_radlen_vs_z->Fill(     average.z(),    m_buffer.radiationLengths() );
    m_radlen_vs_r->Fill(     average.perp(), m_buffer.radiationLengths() );
  }
  m_counted = false;
  m_buffer  = MaterialAccountingStep();
}

std::string MaterialAccountingLayer::getName(void) const {
  // extract the subdetector name using the overloaded operator<<
  // use the first layer only (for simplicity and backward compatibility)
  std::stringstream name;
  name << m_layers[0]->subDetector();
  return name.str();
}


void MaterialAccountingLayer::savePlots(const std::string & name) const
{
  TCanvas * canvas;

  canvas = new TCanvas ("cavas", "Energy loss", 800, 600);
  m_dedx_spectrum->Draw("");
  canvas->SaveAs((name + "_dedx_spectrum.png").c_str(),  "");
  delete canvas;
  
  canvas = new TCanvas ("canvas", "Energy loss vs. eta", 800, 600);
  //if (double integral = m_dedx_vs_eta->Integral())
  //  m_dedx_vs_eta->Scale( 1. / integral );
  m_dedx_vs_eta->Draw("");
  canvas->SaveAs((name + "_dedx_vs_eta.png").c_str(),  "");
  delete canvas;
  
  canvas = new TCanvas ("canvas", "Energy loss vs. Z", 800, 600);
  //if (double integral = m_dedx_vs_z->Integral())
  //  m_dedx_vs_z->Scale( 1. / integral );
  m_dedx_vs_z->Draw("");
  canvas->SaveAs((name + "_dedx_vs_z.png").c_str(),  "");
  delete canvas;
  
  canvas = new TCanvas ("canvas", "Energy loss vs. R", 800, 600);
  //if (double integral = m_dedx_vs_r->Integral())
  //  m_dedx_vs_r->Scale( 1. / integral );
  m_dedx_vs_r->Draw("");
  canvas->SaveAs((name + "_dedx_vs_r.png").c_str(),  "");
  delete canvas;
  
  canvas = new TCanvas ("canvas", "Radiation lenghts", 800, 600);
  m_radlen_spectrum->Draw("");
  canvas->SaveAs((name + "_radlen_spectrum.png").c_str(),  "");
  delete canvas;
  
  canvas = new TCanvas ("canvas", "Radiation lenghts vs. eta", 800, 600);
  //if (double integral = m_radlen_vs_eta->Integral())
  //  m_radlen_vs_eta->Scale( 1. / integral );
  m_radlen_vs_eta->Draw("");
  canvas->SaveAs((name + "_radlen_vs_eta.png").c_str(),  "");
  delete canvas;
  
  canvas = new TCanvas ("canvas", "Radiation lenghts vs. Z", 800, 600);
  //if (double integral = m_radlen_vs_z->Integral())
  //  m_radlen_vs_z->Scale( 1. / integral );
  m_radlen_vs_z->Draw("");
  canvas->SaveAs((name + "_radlen_vs_z.png").c_str(),  "");
  delete canvas;
  
  canvas = new TCanvas ("canvas", "Radiation lenghts vs. R", 800, 600);
  //if (double integral = m_radlen_vs_r->Integral())
  //  m_radlen_vs_r->Scale( 1. / integral );
  m_radlen_vs_r->Draw("");
  canvas->SaveAs((name + "_radlen_vs_r.png").c_str(),  "");
  delete canvas;
}

