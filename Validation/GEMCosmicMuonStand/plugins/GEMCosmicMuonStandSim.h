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

  // conversion_factor = 10 deg * ( TMath::Pi() / 180 deg ) / 384
  Float_t GetLocalPhi(Float_t strip) {return 0.00045451283 * (strip - 192);}

 private:
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<GEMRecHitCollection> recHitToken_;

  const Int_t kMinCLS_ = 1, kMaxCLS_ = 10;
  const Int_t kNumEtaSegments_ = 8, kNumPhiSegments_ = 3;
  const Int_t kNumChambers_ = 15, kMinChamberId_ = 1, kMaxChamberId_ = 29;
  const Int_t kMinRollId_ = 1, kMaxRollId_ = 8;
  const Int_t kMinVFATId_ = 1, kMaxVFATId_ = 3;
  const Int_t kNumStrips_ = 384, kMinStripId_ = 0, kMaxStripId_ = 384;

  MonitorElement *meVFATPassed_, *meVFATTotal_;
  MonitorElement *meVFATPassedPerChamber_[15];
  MonitorElement *meVFATTotalPerChamber_[15];
  MonitorElement *meVFATOccupancyPerChamber_[15];

  MonitorElement *meResidualLocalX_, *meResidualLocalY_, *meResidualLocalPhi_;
  MonitorElement *meLocalPosErrorX_, *meLocalPosErrorY_;
  MonitorElement *mePullLocalX_, *mePullLocalY_;

  MonitorElement *meCLS_, *meCLSvsChamber_, *meNumClustersvsChamber_, *meNumClusters_, *meNumSimHits_, *meNumRecHits_;
  MonitorElement *meSimHitBareLocalPhi_, *meSimHitLocalPhi_, *meRecHitLocalPhi_;

  MonitorElement *meMatChamber_, *meMisChamber_;
};

DEFINE_FWK_MODULE (GEMCosmicMuonStandSim) ;
#endif
