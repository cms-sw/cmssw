#ifndef GEMCosmicMuonStandSim_H
#define GEMCosmicMuonStandSim_H

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "TFile.h"
#include "TTree.h"

#include <vector>

class GEMCosmicMuonStandSim : public DQMEDAnalyzer {
 public:
  explicit GEMCosmicMuonStandSim( const edm::ParameterSet& );
  ~GEMCosmicMuonStandSim();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  void analyze(const edm::Event& e, const edm::EventSetup&) override;

  TString ConvertTitleToName(TString const& title);

  MonitorElement* BookHist1D(DQMStore::IBooker &ibooker,
                             TString title,
                             Int_t nchX, Double_t lowX, Double_t highX);

  MonitorElement* BookHist2D(DQMStore::IBooker &ibooker,
                             TString title,
                             Int_t nchX, Double_t lowX, Double_t highX,
                             Int_t nchY, Double_t lowY, Double_t highY);


  Int_t GetVFATId(Float_t x, const GEMEtaPartition* roll);

  // Conver Id to Index
  Int_t GetChamberIndex(Int_t chamber_id) {return (chamber_id - 1) / 2; }

  // conversion_factor = 10 deg * ( TMath::Pi() / 180 deg ) / 384 = 0.00045451283
  Float_t GetLocalPhi(Float_t strip) {return 0.00045451283 * (strip - 192);}

 private:
  edm::EDGetTokenT<edm::PSimHitContainer> sim_hit_token_;
  edm::EDGetTokenT<GEMRecHitCollection> rec_hit_token_;

  const Int_t kMinCLS_ = 1, kMaxCLS_ = 10;
  const Int_t kNumEtaSegments_ = 8, kNumPhiSegments_ = 3;
  const Int_t kNumChambers_ = 15, kMinChamberId_ = 1, kMaxChamberId_ = 29;
  const Int_t kMinRollId_ = 1, kMaxRollId_ = 8;
  const Int_t kMinVFATId_ = 1, kMaxVFATId_ = 3;
  const Int_t kNumStrips_ = 384, kMinStripId_ = 0, kMaxStripId_ = 384;

  MonitorElement *me_vfat_passed_, *me_vfat_total_;
  MonitorElement *me_vfat_passed_per_chamber_[15];
  MonitorElement *me_vfat_total_per_chamber_[15];
  MonitorElement *me_vfat_occupancy_per_chamber_[15];

  MonitorElement *me_residual_local_x_, *me_residual_local_y_, *me_residual_local_phi_;
  MonitorElement *me_error_x_, *me_error_y_;
  MonitorElement *me_pull_local_x_, *me_pull_local_y_;

  MonitorElement *me_cls_, *me_cls_vs_chamber_, *me_num_clusters_vs_chamber_, *meNumClusters_, *me_num_sim_hits_, *me_num_rec_hits_;
  MonitorElement *me_sim_hit_bare_local_phi_, *me_sim_hit_local_phi_, *me_rec_hit_local_phi_;

  MonitorElement *me_mat_chamber_, *me_mis_chamber_;
};

DEFINE_FWK_MODULE (GEMCosmicMuonStandSim) ;
#endif
