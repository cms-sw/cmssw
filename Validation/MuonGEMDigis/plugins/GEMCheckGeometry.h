#ifndef Validation_MuonGEMDigis_GEMCheckGeometry_H
#define Validation_MuonGEMDigis_GEMCheckGeometry_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

class GEMCheckGeometry : public DQMEDAnalyzer {
public:
  explicit GEMCheckGeometry(const edm::ParameterSet& gc);
  ~GEMCheckGeometry() override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;

private:
  std::map<UInt_t, MonitorElement*> theStdPlots;
  std::map<UInt_t, MonitorElement*> the_st_dphi;
  double GE11PhiBegin_;
  double GE11PhiStep_;
  double minPhi_;
  double maxPhi_;
  bool detailPlot_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;
};

#endif
