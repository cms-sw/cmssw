#ifndef Validation_MuonGEMDigis_GEMPadDigiValidation_h
#define Validation_MuonGEMDigis_GEMPadDigiValidation_h

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/GEMDigiSimLink/interface/GEMDigiSimLink.h"

class GEMPadDigiValidation : public GEMBaseValidation {
public:
  explicit GEMPadDigiValidation(const edm::ParameterSet&);
  ~GEMPadDigiValidation() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  // NOTE Parameters
  edm::EDGetTokenT<GEMPadDigiCollection> pad_token_;
  edm::EDGetTokenT<edm::PSimHitContainer> simhit_token_;
  edm::EDGetTokenT<edm::DetSetVector<GEMDigiSimLink>> digisimlink_token_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomTokenBeginRun_;

  // NOTE MonitorElemnts
  MEMap3Ids me_occ_total_pad_;
  MEMap3Ids me_pad_occ_eta_;
  MEMap3Ids me_pad_occ_phi_;
  MEMap2Ids me_detail_occ_det_;
  MEMap2Ids me_detail_pad_occ_det_;
  MEMap3Ids me_detail_occ_xy_;
  MEMap1Ids me_detail_occ_zr_;
  MEMap3Ids me_detail_occ_phi_pad_;
  MEMap3Ids me_detail_occ_pad_;
  MEMap3Ids me_detail_bx_;
};

#endif  // Validation_MuonGEMDigis_GEMPadDigiValidation_h
