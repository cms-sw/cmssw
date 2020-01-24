#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonGEMHits/plugins/MuonGEMHitsHarvestor.h"

using namespace std;
MuonGEMHitsHarvestor::MuonGEMHitsHarvestor(const edm::ParameterSet& ps) : MuonGEMBaseHarvestor(ps) {
  folder_ = ps.getParameter<std::string>("folder");
}

MuonGEMHitsHarvestor::~MuonGEMHitsHarvestor() {}

void MuonGEMHitsHarvestor::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  igetter.setCurrentFolder(folder_);
}
