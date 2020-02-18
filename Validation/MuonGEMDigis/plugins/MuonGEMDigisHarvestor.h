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

  // TODO
  // void harvestStripDigi();
  // void harvestPadDigi();
  // void harvestCoPadDigi();

 private:
  std::string strip_folder_, pad_folder_, copad_folder_, cluster_folder;
  std::vector<Int_t> region_ids_, station_ids_, layer_ids_;
};

#endif // Validation_MuonGEMDigis_MuonGEMDigisHarvestor_h
