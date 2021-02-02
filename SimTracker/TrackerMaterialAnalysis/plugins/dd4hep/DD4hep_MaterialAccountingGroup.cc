#include <sstream>
#include <iomanip>
#include <string>
#include <stdexcept>

#include <TFile.h>
#include <TH1F.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <TFrame.h>

#include <DD4hep/DD4hepUnits.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingDetector.h"
#include "DD4hep_MaterialAccountingGroup.h"

DD4hep_MaterialAccountingGroup::DD4hep_MaterialAccountingGroup(const std::string& name,
                                                               const cms::DDCompactView& geometry)
    : m_name(name),
      m_elements(),
      m_boundingbox(),
      m_accounting(),
      m_errors(),
      m_tracks(0),
      m_counted(false),
      m_file(nullptr) {
  cms::DDFilter filter("TrackingMaterialGroup", {name.data(), name.size()});
  cms::DDFilteredView fv(geometry, filter);
  bool firstChild = fv.firstChild();

  edm::LogVerbatim("TrackingMaterialAnalysis") << "Elements within: " << name;

  for (const auto& j : fv.specpars()) {
    for (const auto& k : j.second->paths) {
      if (firstChild) {
        std::vector<std::vector<cms::Node*>> children = fv.children(k);
        for (auto const& path : children) {
          cms::Translation trans = fv.translation(path) / dd4hep::cm;
          GlobalPoint gp = GlobalPoint(trans.x(), trans.y(), trans.z());
          m_elements.emplace_back(gp);
          edm::LogVerbatim("TrackerMaterialAnalysis")
              << "MaterialAccountingGroup:\t"
              << "Adding element at (r,z) " << gp.perp() << "," << gp.z() << std::endl;
        }
      }
    }
  }

  for (unsigned int i = 0; i < m_elements.size(); ++i) {
    m_boundingbox.grow(m_elements[i].perp(), m_elements[i].z());
  }

  m_boundingbox.grow(s_tolerance);

  edm::LogVerbatim("TrackerMaterialAnalysis")
      << "MaterialAccountingGroup:\t"
      << "Final BBox r_range: " << m_boundingbox.range_r().first << ", " << m_boundingbox.range_r().second << std::endl
      << "Final BBox z_range: " << m_boundingbox.range_z().first << ", " << m_boundingbox.range_z().second << std::endl;

  m_dedx_spectrum = std::make_shared<TH1F>((m_name + "_dedx_spectrum").c_str(), "Energy loss spectrum", 1000, 0., 1.);
  m_radlen_spectrum =
      std::make_shared<TH1F>((m_name + "_radlen_spectrum").c_str(), "Radiation lengths spectrum", 1000, 0., 1.);
  m_dedx_vs_eta = std::make_shared<TProfile>((m_name + "_dedx_vs_eta").c_str(), "Energy loss vs. eta", 600, -3., 3.);
  m_dedx_vs_z = std::make_shared<TProfile>((m_name + "_dedx_vs_z").c_str(), "Energy loss vs. Z", 6000, -300., 300.);
  m_dedx_vs_r = std::make_shared<TProfile>((m_name + "_dedx_vs_r").c_str(), "Energy loss vs. R", 1200, 0., 120.);
  m_radlen_vs_eta =
      std::make_shared<TProfile>((m_name + "_radlen_vs_eta").c_str(), "Radiation lengths vs. eta", 600, -3., 3.);
  m_radlen_vs_z =
      std::make_shared<TProfile>((m_name + "_radlen_vs_z").c_str(), "Radiation lengths vs. Z", 6000, -300., 300.);
  m_radlen_vs_r =
      std::make_shared<TProfile>((m_name + "_radlen_vs_r").c_str(), "Radiation lengths vs. R", 1200, 0., 120.);

  m_dedx_spectrum->SetDirectory(nullptr);
  m_radlen_spectrum->SetDirectory(nullptr);
  m_dedx_vs_eta->SetDirectory(nullptr);
  m_dedx_vs_z->SetDirectory(nullptr);
  m_dedx_vs_r->SetDirectory(nullptr);
  m_radlen_vs_eta->SetDirectory(nullptr);
  m_radlen_vs_z->SetDirectory(nullptr);
  m_radlen_vs_r->SetDirectory(nullptr);
}

bool DD4hep_MaterialAccountingGroup::isInside(const MaterialAccountingDetector& detector) const {
  const GlobalPoint& position = detector.position();

  edm::LogVerbatim("MaterialAccountingGroup")
      << "Testing position: (x, y, z, r) = " << position.x() << ", " << position.y() << ", " << position.z() << ", "
      << position.perp() << std::endl;

  if (not m_boundingbox.inside(position.perp(), position.z())) {
    edm::LogVerbatim("MaterialAccountingGroup")
        << "r outside of: (" << m_boundingbox.range_r().first << ", " << m_boundingbox.range_r().second
        << "), Z ouside of: (" << m_boundingbox.range_z().first << ", " << m_boundingbox.range_z().second << ")"
        << std::endl;
    return false;
  } else {
    edm::LogVerbatim("MaterialAccountingGroup")
        << "r within: (" << m_boundingbox.range_r().first << ", " << m_boundingbox.range_r().second << "), Z within: ("
        << m_boundingbox.range_z().first << ", " << m_boundingbox.range_z().second << ")" << std::endl;

    for (unsigned int i = 0; i < m_elements.size(); ++i) {
      edm::LogVerbatim("MaterialAccountingGroup")
          << "Closest testing agains(x, y, z, r): (" << m_elements[i].x() << ", " << m_elements[i].y() << ", "
          << m_elements[i].z() << ", " << m_elements[i].perp() << ") --> " << (position - m_elements[i]).mag()
          << " vs tolerance: " << s_tolerance << std::endl;
      if ((position - m_elements[i]).mag2() < (s_tolerance * s_tolerance))
        return true;
    }
    return false;
  }
}

bool DD4hep_MaterialAccountingGroup::addDetector(const MaterialAccountingDetector& detector) {
  if (!isInside(detector))
    return false;

  m_buffer += detector.material();
  m_counted = true;

  return true;
}

void DD4hep_MaterialAccountingGroup::endOfTrack(void) {
  if (m_counted) {
    m_accounting += m_buffer;
    m_errors += m_buffer * m_buffer;
    ++m_tracks;

    GlobalPoint average((m_buffer.in().x() + m_buffer.out().x()) / 2.,
                        (m_buffer.in().y() + m_buffer.out().y()) / 2.,
                        (m_buffer.in().z() + m_buffer.out().z()) / 2.);
    m_dedx_spectrum->Fill(m_buffer.energyLoss());
    m_radlen_spectrum->Fill(m_buffer.radiationLengths());
    m_dedx_vs_eta->Fill(average.eta(), m_buffer.energyLoss(), 1.);
    m_dedx_vs_z->Fill(average.z(), m_buffer.energyLoss(), 1.);
    m_dedx_vs_r->Fill(average.perp(), m_buffer.energyLoss(), 1.);
    m_radlen_vs_eta->Fill(average.eta(), m_buffer.radiationLengths(), 1.);
    m_radlen_vs_z->Fill(average.z(), m_buffer.radiationLengths(), 1.);
    m_radlen_vs_r->Fill(average.perp(), m_buffer.radiationLengths(), 1.);
  }
  m_counted = false;
  m_buffer = MaterialAccountingStep();
}

void DD4hep_MaterialAccountingGroup::savePlots(void) {
  m_file = std::make_unique<TFile>((m_name + ".root").c_str(), "RECREATE");
  savePlot(m_dedx_spectrum, m_name + "_dedx_spectrum");
  savePlot(m_radlen_spectrum, m_name + "_radlen_spectrum");
  savePlot(m_dedx_vs_eta, averageEnergyLoss(), m_name + "_dedx_vs_eta");
  savePlot(m_dedx_vs_z, averageEnergyLoss(), m_name + "_dedx_vs_z");
  savePlot(m_dedx_vs_r, averageEnergyLoss(), m_name + "_dedx_vs_r");
  savePlot(m_radlen_vs_eta, averageRadiationLengths(), m_name + "_radlen_vs_eta");
  savePlot(m_radlen_vs_z, averageRadiationLengths(), m_name + "_radlen_vs_z");
  savePlot(m_radlen_vs_r, averageRadiationLengths(), m_name + "_radlen_vs_r");
  m_file->Write();
  m_file->Close();
}

void DD4hep_MaterialAccountingGroup::savePlot(std::shared_ptr<TH1F>& plot, const std::string& name) {
  TCanvas canvas(name.c_str(), plot->GetTitle(), 1280, 1024);
  plot->SetFillColor(15);  // grey
  plot->SetLineColor(1);   // black
  plot->Draw("c e");
  canvas.GetFrame()->SetFillColor(kWhite);
  canvas.Draw();
  canvas.SaveAs((name + ".png").c_str(), "");
  plot->SetDirectory(m_file.get());
}

void DD4hep_MaterialAccountingGroup::savePlot(std::shared_ptr<TProfile>& plot, float average, const std::string& name) {
  std::unique_ptr<TH1F> line = std::make_unique<TH1F>(
      (name + "_par").c_str(), "Parametrization", 1, plot->GetXaxis()->GetXmin(), plot->GetXaxis()->GetXmax());

  line->SetBinContent(1, average);

  TCanvas canvas(name.c_str(), plot->GetTitle(), 1280, 1024);
  plot->SetFillColor(15);  // grey
  plot->SetLineColor(1);   // black
  plot->SetLineWidth(2);
  plot->Draw("c e6");
  line->SetLineColor(2);  // red
  line->SetLineWidth(2);
  line->Draw("same");
  canvas.GetFrame()->SetFillColor(kWhite);
  canvas.Draw();
  canvas.SaveAs((name + ".png").c_str(), "");
  plot->SetDirectory(m_file.get());
  line->SetDirectory(m_file.get());
  line->Write();
}

std::string DD4hep_MaterialAccountingGroup::info(void) const {
  std::stringstream out;
  out << std::setw(48) << std::left << m_name << std::right << std::fixed;

  out << "BBox: " << std::setprecision(1) << std::setw(6) << m_boundingbox.range_z().first << " < Z < "
      << std::setprecision(1) << std::setw(6) << m_boundingbox.range_z().second;
  out << ", " << std::setprecision(1) << std::setw(5) << m_boundingbox.range_r().first << " < R < "
      << std::setprecision(1) << std::setw(5) << m_boundingbox.range_r().second;
  out << "   Elements: " << std::setw(6) << m_elements.size();
  return out.str();
}

MaterialAccountingStep DD4hep_MaterialAccountingGroup::average(void) const {
  return m_tracks ? m_accounting / m_tracks : MaterialAccountingStep();
}

double DD4hep_MaterialAccountingGroup::averageLength(void) const {
  return m_tracks ? m_accounting.length() / m_tracks : 0.;
}

double DD4hep_MaterialAccountingGroup::averageEnergyLoss(void) const {
  return m_tracks ? m_accounting.energyLoss() / m_tracks : 0.;
}

double DD4hep_MaterialAccountingGroup::sigmaLength(void) const {
  return m_tracks ? std::sqrt(m_errors.length() / m_tracks - averageLength() * averageLength()) : 0.;
}

double DD4hep_MaterialAccountingGroup::sigmaRadiationLengths(void) const {
  return m_tracks
             ? std::sqrt(m_errors.radiationLengths() / m_tracks - averageRadiationLengths() * averageRadiationLengths())
             : 0.;
}

double DD4hep_MaterialAccountingGroup::sigmaEnergyLoss(void) const {
  return m_tracks ? std::sqrt(m_errors.energyLoss() / m_tracks - averageEnergyLoss() * averageEnergyLoss()) : 0.;
}

double DD4hep_MaterialAccountingGroup::averageRadiationLengths(void) const {
  return m_tracks ? m_accounting.radiationLengths() / m_tracks : 0.;
}
