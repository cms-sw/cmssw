#include <sstream>
#include <iomanip>
#include <string>
#include <stdexcept>

#include <TFile.h>
#include <TH1F.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <TFrame.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingDetector.h"
#include "MaterialAccountingGroup.h"

double const MaterialAccountingGroup::s_tolerance = 0.01; // 100um should be small enough that no elements from different layers/groups are so close

MaterialAccountingGroup::MaterialAccountingGroup( const std::string & name, const DDCompactView & geometry ) :
  m_name( name ),
  m_elements(),
  m_boundingbox(),
  m_accounting(),
  m_errors(),
  m_tracks( 0 ),
  m_counted( false ),
  m_file( 0 )
{
  // retrieve the elements from DDD
  DDFilteredView fv( geometry );
  DDSpecificsFilter filter;
  filter.setCriteria(DDValue("TrackingMaterialGroup", name), DDCompOp::equals);
  fv.addFilter(filter);
  while (fv.next()) {
    // DD3Vector and DDTranslation are the same type as math::XYZVector
    math::XYZVector position = fv.translation() / 10.;  // mm -> cm
    m_elements.push_back( GlobalPoint(position.x(), position.y(), position.z()) );
  }

  // grow the bounding box
  for (unsigned int i = 0; i < m_elements.size(); ++i) {
    m_boundingbox.grow(m_elements[i].perp(), m_elements[i].z());
  }
  m_boundingbox.grow(s_tolerance);

  // initialize the histograms 
  m_dedx_spectrum   = new TH1F((m_name + "_dedx_spectrum").c_str(),     "Energy loss spectrum",       1000,    0,   1);
  m_radlen_spectrum = new TH1F((m_name + "_radlen_spectrum").c_str(),   "Radiation lengths spectrum", 1000,    0,   1);
  m_dedx_vs_eta     = new TProfile((m_name + "_dedx_vs_eta").c_str(),   "Energy loss vs. eta",         600,   -3,   3);
  m_dedx_vs_z       = new TProfile((m_name + "_dedx_vs_z").c_str(),     "Energy loss vs. Z",          6000, -300, 300);
  m_dedx_vs_r       = new TProfile((m_name + "_dedx_vs_r").c_str(),     "Energy loss vs. R",          1200,    0, 120);
  m_radlen_vs_eta   = new TProfile((m_name + "_radlen_vs_eta").c_str(), "Radiation lengths vs. eta",   600,   -3,   3);
  m_radlen_vs_z     = new TProfile((m_name + "_radlen_vs_z").c_str(),   "Radiation lengths vs. Z",    6000, -300, 300);
  m_radlen_vs_r     = new TProfile((m_name + "_radlen_vs_r").c_str(),   "Radiation lengths vs. R",    1200,    0, 120);
  m_dedx_spectrum->SetDirectory( 0 );
  m_radlen_spectrum->SetDirectory( 0 );
  m_dedx_vs_eta->SetDirectory( 0 );
  m_dedx_vs_z->SetDirectory( 0 );
  m_dedx_vs_r->SetDirectory( 0 );
  m_radlen_vs_eta->SetDirectory( 0 );
  m_radlen_vs_z->SetDirectory( 0 );
  m_radlen_vs_r->SetDirectory( 0 );
}

MaterialAccountingGroup::~MaterialAccountingGroup(void)
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

// TODO the inner check could be sped up in many ways
// (sorting the m_elements, partitioning the bounding box, ...)
// but is it worth? 
// especially with the segmentation of the layers ?
bool MaterialAccountingGroup::inside( const MaterialAccountingDetector& detector ) const
{
  const GlobalPoint & position = detector.position();
  // first check to see if the point is inside the bounding box
  if (not m_boundingbox.inside(position.perp(), position.z())) {
    return false;
  } else {
    // now check if the point is actually close enough to any element
    for (unsigned int i = 0; i < m_elements.size(); ++i)
      if ((position - m_elements[i]).mag2() < (s_tolerance * s_tolerance))
        return true;
    return false;
  }
}

bool MaterialAccountingGroup::addDetector( const MaterialAccountingDetector& detector ) 
{
  if (not inside(detector))
    return false;

  // multiple hits in the same layer (overlaps, etc.) from a single track still count as one for averaging,
  // since the energy deposits from the track have been already split between the different detectors
  m_buffer += detector.material();
  m_counted = true;

  return true;
}

void MaterialAccountingGroup::endOfTrack(void) {
  // add a detector
  if (m_counted) {
    m_accounting += m_buffer;
    m_errors     += m_buffer * m_buffer;
    ++m_tracks;

    GlobalPoint average( (m_buffer.in().x() + m_buffer.out().x()) / 2.,
                         (m_buffer.in().y() + m_buffer.out().y()) / 2., 
                         (m_buffer.in().z() + m_buffer.out().z()) / 2. );
    m_dedx_spectrum->Fill(   m_buffer.energyLoss() );
    m_radlen_spectrum->Fill( m_buffer.radiationLengths() );
    m_dedx_vs_eta->Fill(   average.eta(),  m_buffer.energyLoss(),       1. );
    m_dedx_vs_z->Fill(     average.z(),    m_buffer.energyLoss(),       1. );
    m_dedx_vs_r->Fill(     average.perp(), m_buffer.energyLoss(),       1. );
    m_radlen_vs_eta->Fill( average.eta(),  m_buffer.radiationLengths(), 1. );
    m_radlen_vs_z->Fill(   average.z(),    m_buffer.radiationLengths(), 1. );
    m_radlen_vs_r->Fill(   average.perp(), m_buffer.radiationLengths(), 1. );
  }
  m_counted = false;
  m_buffer  = MaterialAccountingStep();
}

void MaterialAccountingGroup::savePlot(TH1F * plot, const std::string & name)
{
  TCanvas canvas(name.c_str(), plot->GetTitle(), 1280, 1024);
  plot->SetFillColor(15);       // grey
  plot->SetLineColor(1);        // black
  plot->Draw("c e");
  canvas.GetFrame()->SetFillColor(kWhite);
  canvas.Draw();
  canvas.SaveAs((name + ".png").c_str(), "");

  // store te plot into m_file
  plot->SetDirectory(m_file);
}

void MaterialAccountingGroup::savePlot(TProfile * plot, float average, const std::string & name)
{
  // Nota Bene:
  // these "line" plots are not deleted explicitly since
  //   - deleting them before saving them to a TFile will not save them
  //   - deleting them after the TFile they're stored into results in a SEGV
  // ROOT is probably "taking care" (read: messing things up) somehow...
  TH1F * line = new TH1F((name + "_par").c_str(), "Parametrization", 1, plot->GetXaxis()->GetXmin(), plot->GetXaxis()->GetXmax());
  line->SetBinContent(1, average);

  TCanvas canvas(name.c_str(), plot->GetTitle(), 1280, 1024);
  plot->SetFillColor(15);       // grey
  plot->SetLineColor(1);        // black
  plot->SetLineWidth(2);
  plot->Draw("c e6");
  line->SetLineColor(2);        // red
  line->SetLineWidth(2);
  line->Draw("same");
  canvas.GetFrame()->SetFillColor(kWhite);
  canvas.Draw();
  canvas.SaveAs((name + ".png").c_str(), "");

  // store te plots into m_file
  plot->SetDirectory(m_file);
  line->SetDirectory(m_file);
}

/// get some infos
std::string MaterialAccountingGroup::info(void) const
{
  std::stringstream out;
  out << std::setw(48) << std::left << m_name << std::right << std::fixed;;
  out << "BBox: " << std::setprecision(1) << std::setw(6) << m_boundingbox.range_z().first << " < Z < " << std::setprecision(1) << std::setw(6) << m_boundingbox.range_z().second;
  out << ", "     << std::setprecision(1) << std::setw(5) << m_boundingbox.range_r().first << " < R < " << std::setprecision(1) << std::setw(5) << m_boundingbox.range_r().second;
  out << "   Elements: " << std::setw(6) << m_elements.size();
  return out.str();
}


void MaterialAccountingGroup::savePlots(void)
{
  m_file = new TFile((m_name + ".root").c_str(), "RECREATE");
  savePlot(m_dedx_spectrum,   m_name + "_dedx_spectrum");
  savePlot(m_radlen_spectrum, m_name + "_radlen_spectrum");
  savePlot(m_dedx_vs_eta,     averageEnergyLoss(),       m_name + "_dedx_vs_eta");
  savePlot(m_dedx_vs_z,       averageEnergyLoss(),       m_name + "_dedx_vs_z");
  savePlot(m_dedx_vs_r,       averageEnergyLoss(),       m_name + "_dedx_vs_r");
  savePlot(m_radlen_vs_eta,   averageRadiationLengths(), m_name + "_radlen_vs_eta");
  savePlot(m_radlen_vs_z,     averageRadiationLengths(), m_name + "_radlen_vs_z");
  savePlot(m_radlen_vs_r,     averageRadiationLengths(), m_name + "_radlen_vs_r");
  m_file->Write();
  m_file->Close();

  delete m_file;
}
