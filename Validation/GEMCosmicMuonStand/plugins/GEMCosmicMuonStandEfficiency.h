#ifndef GEMCosmicMuonStandEfficiency_H
#define GEMCosmicMuonStandEfficiency_H

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

class GEMCosmicMuonStandEfficiency : public DQMEDAnalyzer
{
public:
explicit GEMCosmicMuonStandEfficiency( const edm::ParameterSet& );
void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
void analyze(const edm::Event& e, const edm::EventSetup&) override;
MonitorElement* BookHist1D( DQMStore::IBooker &, const char* name, const char* label, unsigned int row, unsigned int coll, unsigned int layer_num, unsigned int vfat_num, const unsigned int Nbin, const Float_t xMin, const Float_t xMax);
MonitorElement* BookHist1D( DQMStore::IBooker &, const char* name, const char* label, const unsigned int Nbin, const Float_t xMin, const Float_t xMax);

private:

MonitorElement* gem_vfat_eff[30];
MonitorElement* gem_vfat_tot[30];
MonitorElement* gem_vfat_total_eff;
MonitorElement* isuperChamber;
MonitorElement* ilayers;
MonitorElement* ichamber;
MonitorElement* iroll;
MonitorElement* ipartition;
MonitorElement* ichi2;

edm::EDGetTokenT<reco::TrackCollection> insideOutTracks_, outsideInTracks_;

};

DEFINE_FWK_MODULE (GEMCosmicMuonStandEfficiency) ;
#endif

/* Track -> seedRef() :::: RefToBase<TrajectorySeed> :::: range
 * recHits
 */
