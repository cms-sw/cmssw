#include "Validation/RecoTau/plugins/DQMSimpleFileSaver.h"

// framework & common header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>

TauDQMSimpleFileSaver::TauDQMSimpleFileSaver(const edm::ParameterSet& cfg) {
  //std::cout << "<TauDQMSimpleFileSaver::TauDQMSimpleFileSaver>:" << std::endl;

  usesResource("DQMStore");

  cfgError_ = 0;

  outputFileName_ = cfg.getParameter<std::string>("outputFileName");
  if (outputFileName_.empty()) {
    edm::LogError("TauDQMSimpleFileSaver") << " No outputFileName specified --> histograms will NOT be saved !!";
    cfgError_ = 1;
  } else if (outputFileName_.find(".root") == std::string::npos) {
    edm::LogError("TauDQMSimpleFileSaver")
        << " Invalid outputFileName = " << outputFileName_ << " --> histograms will NOT be saved !!";
    cfgError_ = 1;
  }
}

TauDQMSimpleFileSaver::~TauDQMSimpleFileSaver() {
  // nothing to be done yet...
}

void TauDQMSimpleFileSaver::analyze(const edm::Event&, const edm::EventSetup&) {
  // nothing to be done yet...
}

void TauDQMSimpleFileSaver::endJob() {
  std::cout << "<TauDQMSimpleFileSaver::endJob>:" << std::endl;

  //--- check that configuration parameters contain no errors
  if (cfgError_) {
    edm::LogError("endJob") << " Error in Configuration ParameterSet --> histograms will NOT be saved !!";
    return;
  }

  //--- check that DQMStore service is available
  if (!edm::Service<DQMStore>().isAvailable()) {
    edm::LogError("endJob") << " Failed to access dqmStore --> histograms will NOT be saved !!";
    return;
  }

  DQMStore& dqmStore = (*edm::Service<DQMStore>());
  dqmStore.save(outputFileName_);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TauDQMSimpleFileSaver);
