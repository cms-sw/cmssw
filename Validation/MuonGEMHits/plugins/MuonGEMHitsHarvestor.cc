#include "Validation/MuonGEMHits/plugins/MuonGEMHitsHarvestor.h"


MuonGEMHitsHarvestor::MuonGEMHitsHarvestor(const edm::ParameterSet& pset)
    : MuonGEMBaseHarvestor(pset, "MuonGEMHitsHarvestor") {}


MuonGEMHitsHarvestor::~MuonGEMHitsHarvestor() {}


void MuonGEMHitsHarvestor::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {}
