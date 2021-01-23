#ifndef Validation_MuonGEMDigis_MuonGEMDigisHarvestor_h
#define Validation_MuonGEMDigis_MuonGEMDigisHarvestor_h

#include "Validation/MuonGEMHits/interface/MuonGEMBaseHarvestor.h"

class MuonGEMDigisHarvestor : public MuonGEMBaseHarvestor {
public:
  /// constructor
  explicit MuonGEMDigisHarvestor(const edm::ParameterSet&);
  /// destructor
  ~MuonGEMDigisHarvestor() override;

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  std::vector<Int_t> region_ids_, station_ids_, layer_ids_;
  Bool_t detail_plot_;
};

#endif  // Validation_MuonGEMDigis_MuonGEMDigisHarvestor_h
