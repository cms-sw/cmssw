#ifndef TrackerHitAssociator_h
#define TrackerHitAssociator_h

/* \class TrackerHitAssociator
 *
 ** Associates SimHits and RecHits based on information produced during
 *  digitisation (StripDigiSimLinks).
 *  The association works in both ways: from a SimHit to RecHits and
 *  from a RecHit to SimHits.
 *
 * \author Patrizia Azzi (INFN PD), Vincenzo Chiochia (Uni Zuerich)
 *
 * \version   1st version: April 2006. Add configurable switch: August 2006   
 *
 *
 ************************************************************/

//#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

//--- for SimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

//--- for RecHit
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include <string>
#include <vector>

typedef std::pair<uint32_t, EncodedEventId> SimHitIdpr;
class TrackerHitAssociator {
  
 public:
  struct Config {
    Config(const edm::ParameterSet& conf, edm::ConsumesCollector && iC);
    Config(edm::ConsumesCollector && iC);
    bool doPixel_, doStrip_, doTrackAssoc_, assocHitbySimTrack_;
    edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > stripToken_;
    edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > pixelToken_;
    std::vector<edm::EDGetTokenT<CrossingFrame<PSimHit> > > cfTokens_;
    std::vector<edm::EDGetTokenT<std::vector<PSimHit> > > simHitTokens_;
  };

  // The constructor supporting the consumes interface and tokens
  TrackerHitAssociator(const edm::Event& e, const Config& config);

  // Destructor
  virtual ~TrackerHitAssociator(){}
  
  typedef std::pair<unsigned int, unsigned int> simHitCollectionID;
  typedef std::pair<simHitCollectionID, unsigned int> simhitAddr;

  std::vector<PSimHit> associateHit(const TrackingRecHit & thit) const;
  //for PU events
  std::vector<SimHitIdpr> associateHitId(const TrackingRecHit & thit) const;
  void associateHitId(const TrackingRecHit & thit,std::vector<SimHitIdpr> &simhitid, std::vector<simhitAddr>* simhitCFPos=0) const;
  template<typename T>
    void associateSiStripRecHit(const T *simplerechit, std::vector<SimHitIdpr>& simtrackid, std::vector<simhitAddr>* simhitCFPos=0) const;

  // Method for obtaining simTracks and simHits from a cluster
  void associateCluster(const SiStripCluster* clust,
			const DetId& detid,
			std::vector<SimHitIdpr>& simtrackid, std::vector<PSimHit>& simhit) const;

  // Obtain simTracks, and optionally simHit addresses, from a cluster 
  void associateSimpleRecHitCluster(const SiStripCluster* clust,
				    const DetId& detid,
				    std::vector<SimHitIdpr>& simtrackid, std::vector<simhitAddr>* simhitCFPos=0) const;

  std::vector<SimHitIdpr> associateMatchedRecHit(const SiStripMatchedRecHit2D * matchedrechit, std::vector<simhitAddr>* simhitCFPos=0) const;
  std::vector<SimHitIdpr> associateProjectedRecHit(const ProjectedSiStripRecHit2D * projectedrechit, std::vector<simhitAddr>* simhitCFPos=0) const;
  void associatePixelRecHit(const SiPixelRecHit * pixelrechit, std::vector<SimHitIdpr> & simhitid, std::vector<simhitAddr>* simhitCFPos=0) const;
  std::vector<SimHitIdpr> associateFastRecHit(const FastTrackerRecHit * rechit) const;
  std::vector<SimHitIdpr> associateMultiRecHitId(const SiTrackerMultiRecHit * multirechit, std::vector<simhitAddr>* simhitCFPos=0) const;
  std::vector<PSimHit>    associateMultiRecHit(const SiTrackerMultiRecHit * multirechit) const;
  
  
  typedef std::map<unsigned int, std::vector<PSimHit> > simhit_map;
  simhit_map SimHitMap;
  typedef std::map<simHitCollectionID, std::vector<PSimHit> > simhit_collectionMap;
  simhit_collectionMap SimHitCollMap;
 
 private:
  typedef std::vector<std::string> vstring;

  void makeMaps(const edm::Event& theEvent, const Config& config);
  edm::Handle< edm::DetSetVector<StripDigiSimLink> >  stripdigisimlink;
  edm::Handle< edm::DetSetVector<PixelDigiSimLink> >  pixeldigisimlink;
  bool doPixel_, doStrip_, doTrackAssoc_, assocHitbySimTrack_;
};

#endif
