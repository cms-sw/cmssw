#ifndef Validation_MuonGEMDigis_GEMCoPadDigiValidation_h
#define Validation_MuonGEMDigis_GEMCoPadDigiValidation_h

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"

class GEMCoPadDigiValidation : public GEMBaseValidation {
public:
  explicit GEMCoPadDigiValidation(const edm::ParameterSet&);
  ~GEMCoPadDigiValidation() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  //
  MEMap1Ids me_detail_occ_zr_;
  MEMap2Ids me_detail_occ_det_;
  MEMap2Ids me_detail_occ_xy_;
  MEMap2Ids me_detail_occ_phi_pad_;
  MEMap2Ids me_detail_occ_pad_;

  MEMap2Ids me_detail_bx_;

  // Parameters
  edm::EDGetTokenT<GEMCoPadDigiCollection> copad_token_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomTokenBeginRun_;

  //
  int gem_bx_min_, gem_bx_max_;
};

#endif  // Validation_MuonGEMDigis_GEMCoPadDigiValidation_h
