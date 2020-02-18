#ifndef Validation_MuonGEMHits_MuonGEMHitsHarvestor_h
#define Validation_MuonGEMHits_MuonGEMHitsHarvestor_h

#include "Validation/MuonGEMHits/interface/MuonGEMBaseHarvestor.h"


class MuonGEMHitsHarvestor : public MuonGEMBaseHarvestor {
 public:
  explicit MuonGEMHitsHarvestor(const edm::ParameterSet&);
  ~MuonGEMHitsHarvestor() override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
};

#endif // Validation_MuonGEMHits_MuonGEMHitsHarvestor_h
