#ifndef Validation_MuonGEMDigis_GEMPadDigiClusterValidation_h
#define Validation_MuonGEMDigis_GEMPadDigiClusterValidation_h

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/GEMDigiSimLink/interface/GEMDigiSimLink.h"

class GEMPadDigiClusterValidation : public GEMBaseValidation {
public:
  explicit GEMPadDigiClusterValidation(const edm::ParameterSet&);
  ~GEMPadDigiClusterValidation() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  Bool_t matchClusterAgainstSimHit(GEMPadDigiClusterCollection::const_iterator, Int_t);

  edm::EDGetTokenT<GEMPadDigiClusterCollection> pad_cluster_token_;
  edm::EDGetTokenT<edm::PSimHitContainer> simhit_token_;
  edm::EDGetTokenT<edm::DetSetVector<GEMDigiSimLink>> digisimlink_token_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomTokenBeginRun_;

  MonitorElement* me_cls_;
  MEMap3Ids me_total_cluster_;
  MEMap3Ids me_pad_cluster_occ_eta_;
  MEMap3Ids me_pad_cluster_occ_phi_;
  MEMap2Ids me_detail_occ_det_;
  MEMap2Ids me_detail_pad_cluster_occ_det_;
  MEMap1Ids me_detail_occ_zr_;
  MEMap3Ids me_detail_occ_xy_;
  MEMap3Ids me_detail_occ_phi_pad_;
  MEMap3Ids me_detail_occ_pad_;

  MEMap3Ids me_detail_bx_;
};

#endif  // Validation_MuonGEMDigis_GEMPadDigiClusterValidation_h
