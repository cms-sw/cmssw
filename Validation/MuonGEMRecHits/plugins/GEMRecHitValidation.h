#ifndef Validation_MuonGEMRecHits_GEMRecHitValidation_h
#define Validation_MuonGEMRecHits_GEMRecHitValidation_h

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

class GEMRecHitValidation : public GEMBaseValidation {
public:
  explicit GEMRecHitValidation(const edm::ParameterSet&);
  ~GEMRecHitValidation() override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  Bool_t matchRecHitAgainstSimHit(GEMRecHitCollection::const_iterator, Int_t);

  // Parameter
  edm::EDGetTokenT<GEMRecHitCollection> rechit_token_;
  edm::EDGetTokenT<edm::PSimHitContainer> simhit_token_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomTokenBeginRun_;

  // MonitorElement
  MonitorElement* me_cls_;
  MEMap3Ids me_detail_cls_;

  MEMap1Ids me_residual_x_;
  MEMap1Ids me_residual_y_;
  MEMap3Ids me_detail_residual_x_;
  MEMap3Ids me_detail_residual_y_;

  MEMap1Ids me_pull_x_;
  MEMap1Ids me_pull_y_;
  MEMap3Ids me_detail_pull_x_;
  MEMap3Ids me_detail_pull_y_;

  // Occupancy
  MEMap1Ids me_occ_zr_;
  MEMap3Ids me_detail_occ_xy_;
  MEMap3Ids me_detail_occ_polar_;
  // occupancy of PSimHit and GEMRecHIts for efficiency
  MEMap1Ids me_simhit_occ_eta_;
  MEMap2Ids me_simhit_occ_phi_;
  MEMap2Ids me_simhit_occ_det_;
  // GEMRecHit that matches PSimHit
  MEMap1Ids me_rechit_occ_eta_;
  MEMap2Ids me_rechit_occ_phi_;
  MEMap2Ids me_rechit_occ_det_;
};

#endif  // Validation_MuonGEMRecHits_GEMRecHitValidation_h
