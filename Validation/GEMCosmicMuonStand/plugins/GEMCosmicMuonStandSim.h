#ifndef GEMCosmicMuonStandSim_H
#define GEMCosmicMuonStandSim_H

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

class GEMCosmicMuonStandSim : public DQMEDAnalyzer
{
 public:
  explicit GEMCosmicMuonStandSim( const edm::ParameterSet& );
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
  MonitorElement* BookHist1D( DQMStore::IBooker &, const char* name, const char* label, unsigned int row, unsigned int coll, unsigned int layer_num, unsigned int vfat_num, const unsigned int Nbin, const Float_t xMin, const Float_t xMax);
  MonitorElement* BookHist1D( DQMStore::IBooker &, const char* name, const char* label, const unsigned int Nbin, const Float_t xMin, const Float_t xMax);

 private:

  MonitorElement* gem_vfat_eff[3][5];
  MonitorElement* gem_vfat_total_eff;

  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<GEMRecHitCollection> recHitToken_;

};

DEFINE_FWK_MODULE (GEMCosmicMuonStandSim) ;
#endif
