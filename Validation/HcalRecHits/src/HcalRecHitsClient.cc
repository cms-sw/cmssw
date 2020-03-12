#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/HcalRecHits/interface/HcalRecHitsClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

HcalRecHitsClient::HcalRecHitsClient(const edm::ParameterSet &iConfig) : conf_(iConfig) {
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");

  debug_ = false;
  verbose_ = false;

  dirName_ = iConfig.getParameter<std::string>("DQMDirName");
}

HcalRecHitsClient::~HcalRecHitsClient() {}

void HcalRecHitsClient::dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) { ig.setCurrentFolder(dirName_); }

// called after entering the HcalRecHitsV/HcalRecHitTask directory
// hcalMEs are within that directory
int HcalRecHitsClient::HcalRecHitsEndjob(const std::vector<MonitorElement *> &hcalMEs) {
  return 1;  // Removed all actions
}

DEFINE_FWK_MODULE(HcalRecHitsClient);
