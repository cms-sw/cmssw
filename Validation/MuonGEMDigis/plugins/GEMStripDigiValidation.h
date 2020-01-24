#ifndef Validation_MuonGEMDigis_GEMStripDigiValidation_H
#define Validation_MuonGEMDigis_GEMStripDigiValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

class GEMStripDigiValidation : public GEMBaseValidation {
public:
  explicit GEMStripDigiValidation(const edm::ParameterSet&);
  ~GEMStripDigiValidation() override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  // ParameterSet
  edm::EDGetTokenT<GEMDigiCollection> inputToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> inputTokenSH_;

  // Monitor elements
  // NOTE muon simhit - strip digi matching
  // occupancy plots for efficiency
  MEMap1Ids me_simhit_occ_eta_;
  MEMap1Ids me_strip_occ_eta_;
  MEMap2Ids me_simhit_occ_phi_;
  MEMap2Ids me_strip_occ_phi_;
  MEMap2Ids me_simhit_occ_det_;
  MEMap2Ids me_strip_occ_det_;

  // bunch crossing
  MEMap3Ids me_detail_bx_;
};

#endif
