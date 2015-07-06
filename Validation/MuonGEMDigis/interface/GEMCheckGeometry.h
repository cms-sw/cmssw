#ifndef GEMCheckGeometry_H
#define GEMCheckGeometry_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include <TMath.h>

class GEMCheckGeometry : public DQMEDAnalyzer
{
public:
  explicit GEMCheckGeometry(const edm::ParameterSet& gc);
  ~GEMCheckGeometry();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event& e, const edm::EventSetup&) override;

 private:

  std::map< UInt_t , MonitorElement* > theStdPlots;
  std::map< UInt_t , MonitorElement* > the_st_dphi;
  double GE11PhiBegin_; 
  double GE11PhiStep_;
  double minPhi_;
  double maxPhi_;
  bool detailPlot_; 
};

#endif
