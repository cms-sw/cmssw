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
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/EDProduct.h"

//--- for SimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

//--- for RecHit
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

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
  /*  std::vector<unsigned int> associateHitId(const TrackingRecHit & thit);
      std::vector<unsigned int> associateSimpleRecHit(const SiStripRecHit2D * simplerechit);
      std::vector<unsigned int> associateMatchedRecHit(const SiStripMatchedRecHit2D * matchedrechit);
      std::vector<unsigned int> associatePixelRecHit(const SiPixelRecHit * pixelrechit);
  */
  //for PU events
  std::vector<SimHitIdpr> associateHitId(const TrackingRecHit & thit);
  std::vector<SimHitIdpr> associateSimpleRecHit(const SiStripRecHit2D * simplerechit);
  std::vector<SimHitIdpr> associateMatchedRecHit(const SiStripMatchedRecHit2D * matchedrechit);
  std::vector<SimHitIdpr> associateProjectedRecHit(const ProjectedSiStripRecHit2D * projectedrechit);
  std::vector<SimHitIdpr> associatePixelRecHit(const SiPixelRecHit * pixelrechit);
  
  std::vector<PSimHit> theStripHits;
  typedef std::map<unsigned int, std::vector<PSimHit> > simhit_map;
  typedef simhit_map::iterator simhit_map_iterator;
  simhit_map SimHitMap;
  std::vector<PSimHit> thePixelHits;
 
 private:
  const edm::Event& myEvent_;
  typedef std::vector<std::string> vstring;
  vstring trackerContainers;

  edm::Handle< edm::DetSetVector<StripDigiSimLink> >  stripdigisimlink;
  edm::Handle< edm::DetSetVector<PixelDigiSimLink> >  pixeldigisimlink;
  //vector with the trackIds
  //  std::vector<unsigned int> simtrackid; 
  std::vector<SimHitIdpr> simtrackid; 
  
  bool doPixel_, doStrip_, doTrackAssoc_;
  
};  


#endif

