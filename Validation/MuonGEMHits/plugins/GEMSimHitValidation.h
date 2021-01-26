#ifndef Validation_MuonGEMHits_GEMSimHitValidation_h
#define Validation_MuonGEMHits_GEMSimHitValidation_h

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"

#include <tuple>
#include <map>
#include <vector>
#include <string>

class GEMSimHitValidation : public GEMBaseValidation {
public:
  explicit GEMSimHitValidation(const edm::ParameterSet&);
  ~GEMSimHitValidation() override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  std::tuple<Double_t, Double_t> getTOFRange(Int_t station_id);

  // Parameters
  edm::EDGetTokenT<edm::PSimHitContainer> simhit_token_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomTokenBeginRun_;
  std::vector<Double_t> tof_range_;

  // Monitor elemnts
  MEMap2Ids me_tof_mu_;  // time of flight
  MEMap2Ids me_tof_others_;
  MEMap3Ids me_detail_tof_;
  MEMap3Ids me_detail_tof_mu_;

  MEMap1Ids me_eloss_mu_;  // energy loss
  MEMap1Ids me_eloss_others_;
  MEMap3Ids me_detail_eloss_;
  MEMap3Ids me_detail_eloss_mu_;

  MEMap3Ids me_occ_eta_mu_;  // occupancy
  MEMap3Ids me_occ_phi_mu_;
  MEMap3Ids me_occ_pid_;
  MEMap1Ids me_detail_occ_zr_;
  MEMap2Ids me_detail_occ_det_;
  MEMap2Ids me_detail_occ_det_mu_;
  MEMap3Ids me_detail_occ_xy_;

  // Constants
  const Float_t kEnergyCF_ = 1e6f;  // energy loss conversion factor:
};

#endif  // Validation_MuonGEMHits_GEMSimHitValidation_h
