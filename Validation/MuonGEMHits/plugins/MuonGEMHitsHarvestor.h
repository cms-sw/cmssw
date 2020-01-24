#ifndef Validation_MuonGEMHits_MuonGEMHitsHarvestor_H
#define Validation_MuonGEMHits_MuonGEMHitsHarvestor_H

#include "Validation/MuonGEMHits/interface/MuonGEMBaseHarvestor.h"

class MuonGEMHitsHarvestor : public MuonGEMBaseHarvestor {
public:
  /// constructor
  explicit MuonGEMHitsHarvestor(const edm::ParameterSet&);
  /// destructor
  ~MuonGEMHitsHarvestor() override;

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  std::string folder_;
};

#endif
