#ifndef Validation_MuonGEMDigis_GEMStripDigiValidation_h
#define Validation_MuonGEMDigis_GEMStripDigiValidation_h

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

class GEMStripDigiValidation : public GEMBaseValidation {
public:
  explicit GEMStripDigiValidation(const edm::ParameterSet&);
  ~GEMStripDigiValidation() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  // ParameterSet
  edm::EDGetTokenT<GEMDigiCollection> strip_token_;
  edm::EDGetTokenT<edm::PSimHitContainer> simhit_token_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomTokenBeginRun_;

  // NOTE Monitor elements

  // Occupaancy
  MEMap1Ids me_occ_zr_;
  MEMap2Ids me_occ_det_;
  MEMap3Ids me_detail_occ_xy_;
  MEMap3Ids me_detail_occ_strip_;
  MEMap3Ids me_detail_occ_phi_strip_;

  // Bunch Crossing
  MonitorElement* me_bx_;
  MEMap3Ids me_detail_bx_;

  // occupancy plots for efficiency (muon simhit - strip digi matching)
  MEMap1Ids me_simhit_occ_eta_;
  MEMap2Ids me_simhit_occ_phi_;
  MEMap2Ids me_simhit_occ_det_;
  // Strip that matches the SimHit
  MEMap1Ids me_strip_occ_eta_;
  MEMap2Ids me_strip_occ_phi_;
  MEMap2Ids me_strip_occ_det_;
};

#endif  // Validation_MuonGEMDigis_GEMStripDigiValidation_h
