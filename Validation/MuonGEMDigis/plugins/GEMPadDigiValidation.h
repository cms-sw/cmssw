#ifndef Validation_MuonGEMDigis_GEMPadDigiValidation_h
#define Validation_MuonGEMDigis_GEMPadDigiValidation_h

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"

class GEMPadDigiValidation : public GEMBaseValidation {
public:
  explicit GEMPadDigiValidation(const edm::ParameterSet&);
  ~GEMPadDigiValidation() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  // NOTE Parameters
  edm::EDGetTokenT<GEMPadDigiCollection> pad_token_;

  // NOTE MonitorElemnts
  MEMap2Ids me_occ_det_;
  MEMap1Ids me_occ_zr_;
  MEMap3Ids me_detail_occ_xy_;
  MEMap3Ids me_detail_occ_phi_pad_;
  MEMap3Ids me_detail_occ_pad_;
  MEMap3Ids me_detail_bx_;
};

#endif  // Validation_MuonGEMDigis_GEMPadDigiValidation_h
