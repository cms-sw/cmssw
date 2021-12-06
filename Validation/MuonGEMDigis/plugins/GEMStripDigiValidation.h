#ifndef Validation_MuonGEMDigis_GEMStripDigiValidation_h
#define Validation_MuonGEMDigis_GEMStripDigiValidation_h

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/GEMDigiSimLink/interface/GEMDigiSimLink.h"

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
  edm::EDGetTokenT<edm::DetSetVector<GEMDigiSimLink>> digisimlink_token_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomTokenBeginRun_;

  // NOTE Monitor elements

  // Occupaancy
  MonitorElement* me_detail_total_strip_all_;
  MEMap2Ids me_detail_total_strip_;
  MEMap1Ids me_detail_occ_zr_;
  MEMap2Ids me_detail_occ_det_;
  MEMap3Ids me_detail_occ_xy_;
  MEMap3Ids me_detail_occ_strip_;

  // Bunch Crossing
  MonitorElement* me_detail_bx_;
  MEMap3Ids me_detail_bx_layer_;

  // Strip that matches the SimHit
  MEMap2Ids me_occ_pid_;
  MEMap3Ids me_occ_pid_eta_;
  MEMap3Ids me_detail_strip_occ_eta_;
  MEMap3Ids me_detail_strip_occ_phi_;
  MEMap2Ids me_detail_strip_occ_det_;
};

#endif  // Validation_MuonGEMDigis_GEMStripDigiValidation_h
