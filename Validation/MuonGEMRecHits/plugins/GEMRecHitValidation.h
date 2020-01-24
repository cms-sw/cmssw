#ifndef Validation_MuonGEMRecHits_GEMRecHitValidation_H
#define Validation_MuonGEMRecHits_GEMRecHitValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

class GEMRecHitValidation : public GEMBaseValidation {
public:
  explicit GEMRecHitValidation(const edm::ParameterSet&);
  ~GEMRecHitValidation() override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;

private:
  Bool_t matchRecHitAgainstSimHit(GEMRecHitCollection::const_iterator, Int_t);

  // MonitorElement

  // cluster size of rechit
  MonitorElement* me_cls_;
  MEMap3Ids me_detail_cls_;

  MEMap1Ids me_pull_x_;
  MEMap1Ids me_pull_y_;
  MEMap3Ids me_detail_pull_x_;
  MEMap3Ids me_detail_pull_y_;

  MEMap1Ids me_residual_x_;
  MEMap1Ids me_residual_y_;
  MEMap3Ids me_detail_residual_x_;
  MEMap3Ids me_detail_residual_y_;

  // occupancy of PSimHit and GEMRecHIts for efficiency
  MEMap1Ids me_simhit_occ_eta_;
  MEMap1Ids me_rechit_occ_eta_;
  MEMap2Ids me_simhit_occ_phi_;
  MEMap2Ids me_rechit_occ_phi_;
  MEMap2Ids me_simhit_occ_det_;
  MEMap2Ids me_rechit_occ_det_;

  edm::EDGetTokenT<GEMRecHitCollection> inputToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> inputTokenSH_;
};

#endif
