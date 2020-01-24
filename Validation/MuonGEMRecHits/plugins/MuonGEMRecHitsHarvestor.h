#ifndef Validation_MuonGEMRecHits_MuonGEMDigisHarvestor_H
#define Validation_MuonGEMRecHits_MuonGEMDigisHarvestor_H

#include "Validation/MuonGEMHits/interface/MuonGEMBaseHarvestor.h"

class MuonGEMRecHitsHarvestor : public MuonGEMBaseHarvestor {
public:
  /// constructor
  explicit MuonGEMRecHitsHarvestor(const edm::ParameterSet&);
  /// destructor
  ~MuonGEMRecHitsHarvestor() override;

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

  std::vector<Int_t> region_ids_, station_ids_, layer_ids_;
  std::string folder_;
};
#endif
