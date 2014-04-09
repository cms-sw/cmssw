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
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include <string>
#include <vector>

typedef std::pair<uint32_t, EncodedEventId> SimHitIdpr;

class TrackerHitAssociator {
  
 public:
  
  // Simple constructor
  TrackerHitAssociator(const edm::Event& e);
  // Constructor with configurables
  TrackerHitAssociator(const edm::Event& e, const edm::ParameterSet& conf);
  // Destructor
  virtual ~TrackerHitAssociator(){}
  
  std::vector<PSimHit> associateHit(const TrackingRecHit & thit);
  //for PU events
  std::vector<SimHitIdpr> associateHitId(const TrackingRecHit & thit);
  void associateHitId(const TrackingRecHit & thit,std::vector<SimHitIdpr> &simhitid);
  template<typename T>
    void associateSiStripRecHit(const T *simplerechit, std::vector<SimHitIdpr>& simtrackid);
  void associateSimpleRecHitCluster(const SiStripCluster* clust,
				    const uint32_t& detID,
				    std::vector<SimHitIdpr>& simtrackid);

  std::vector<SimHitIdpr> associateMatchedRecHit(const SiStripMatchedRecHit2D * matchedrechit);
  std::vector<SimHitIdpr> associateProjectedRecHit(const ProjectedSiStripRecHit2D * projectedrechit);
  void associatePixelRecHit(const SiPixelRecHit * pixelrechit, std::vector<SimHitIdpr> & simhitid);
  std::vector<SimHitIdpr> associateGSRecHit(const SiTrackerGSRecHit2D * gsrechit);
  std::vector<SimHitIdpr> associateMultiRecHitId(const SiTrackerMultiRecHit * multirechit);
  std::vector<PSimHit>    associateMultiRecHit(const SiTrackerMultiRecHit * multirechit);
  std::vector<SimHitIdpr> associateGSMatchedRecHit(const SiTrackerGSMatchedRecHit2D * gsmrechit);
  
  typedef std::map<unsigned int, std::vector<PSimHit> > simhit_map;
  simhit_map SimHitMap;
 
 private:
  const edm::Event& myEvent_;
  typedef std::vector<std::string> vstring;
  vstring trackerContainers;

  edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
  std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;

  edm::Handle< edm::DetSetVector<StripDigiSimLink> >  stripdigisimlink;
  edm::Handle< edm::DetSetVector<PixelDigiSimLink> >  pixeldigisimlink;
  //vector with the trackIds
  std::vector<SimHitIdpr> simtrackid; 
  
  bool doPixel_, doStrip_, doTrackAssoc_;
  
};  


#endif

