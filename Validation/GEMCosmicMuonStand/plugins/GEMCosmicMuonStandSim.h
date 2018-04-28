#ifndef GEMCosmicMuonStandSim_H
#define GEMCosmicMuonStandSim_H

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

class GEMCosmicMuonStandSim : public DQMEDAnalyzer {
 public:
  explicit GEMCosmicMuonStandSim( const edm::ParameterSet& );
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  void analyze(const edm::Event& e, const edm::EventSetup&) override;

  MonitorElement* BookHist1D(DQMStore::IBooker &,
                             const char* name, const char* label,
                             unsigned int row, unsigned int coll,
                             unsigned int layer_num, unsigned int vfat_num,
                             const unsigned int Nbin, const Float_t xMin, const Float_t xMax);

  MonitorElement* BookHist1D(DQMStore::IBooker &,
                             const char* name, const char* label,
                             const unsigned int Nbin, const Float_t xMin, const Float_t xMax);

  Int_t GetVFATId(Float_t x, const GEMEtaPartition* roll);

  // Conver Id to Index
  Int_t GetChamberIndex(Int_t chamber_id) {return (chamber_id - 1) / 2; }


 private:
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<GEMRecHitCollection> recHitToken_;

  const Int_t kNumEtaSegments = 3;
  const Int_t kNumPhiSegments = 8;

  MonitorElement* gem_vfat_passed_[15]; // 2D (roll_id, vfat_id)
  MonitorElement* gem_vfat_total_[15];

};

DEFINE_FWK_MODULE (GEMCosmicMuonStandSim) ;
#endif
