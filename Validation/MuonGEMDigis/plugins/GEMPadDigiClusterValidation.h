#ifndef Validation_MuonGEMDigis_GEMPadDigiClusterValidation_h
#define Validation_MuonGEMDigis_GEMPadDigiClusterValidation_h

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"


class GEMPadDigiClusterValidation : public GEMBaseValidation {
public:
  explicit GEMPadDigiClusterValidation(const edm::ParameterSet&);
  ~GEMPadDigiClusterValidation() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  MEMap2Ids me_occ_det_;
  MEMap1Ids me_occ_zr_;
  MEMap3Ids me_detail_occ_xy_;
  MEMap3Ids me_detail_occ_phi_pad_;
  MEMap3Ids me_detail_occ_pad_;

  MEMap3Ids me_detail_bx_;

  edm::EDGetTokenT<GEMPadDigiClusterCollection> pad_cluster_token_;
};

#endif // Validation_MuonGEMDigis_GEMPadDigiClusterValidation_h
