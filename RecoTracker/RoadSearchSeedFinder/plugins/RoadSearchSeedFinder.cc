//
// Package:         RecoTracker/RoadSearchSeedFinder
// Class:           RoadSearchSeedFinder
// 
// Description:     Calls RoadSeachSeedFinderAlgorithm
//                  to find RoadSearchSeeds.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: wmtan $
// $Date: 2010/02/11 00:14:44 $
// $Revision: 1.18 $
//

#include <iostream>
#include <memory>
#include <string>

#include "RoadSearchSeedFinder.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "DataFormats/RoadSearchSeed/interface/RoadSearchSeedCollection.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoTracker/SpecialSeedGenerators/interface/ClusterChecker.h"

RoadSearchSeedFinder::RoadSearchSeedFinder(edm::ParameterSet const& conf) : 
  roadSearchSeedFinderAlgorithm_(conf) ,
  conf_(conf)
{
  produces<RoadSearchSeedCollection>();

}


// Virtual destructor needed.
RoadSearchSeedFinder::~RoadSearchSeedFinder() { }  

// Functions that gets called by framework every event
void RoadSearchSeedFinder::produce(edm::Event& e, const edm::EventSetup& es)
{

  // retrieve InputTags for strip rechits
  edm::InputTag matchedStripRecHitsInputTag = conf_.getParameter<edm::InputTag>("matchedStripRecHits");
  edm::InputTag rphiStripRecHitsInputTag    = conf_.getParameter<edm::InputTag>("rphiStripRecHits");
  edm::InputTag stereoStripRecHitsInputTag  = conf_.getParameter<edm::InputTag>("stereoStripRecHits");
  edm::InputTag clusterCollectionInputTag   = conf_.getParameter<edm::InputTag>("ClusterCollectionLabel");
  
  // get Inputs
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedRecHits;
  e.getByLabel(matchedStripRecHitsInputTag ,matchedRecHits);
  edm::Handle<SiStripRecHit2DCollection> rphiRecHits;
  e.getByLabel(rphiStripRecHitsInputTag ,rphiRecHits);
  edm::Handle<SiStripRecHit2DCollection> stereoRecHits;
  e.getByLabel(stereoStripRecHitsInputTag ,stereoRecHits);
 
  // retrieve InputTag for pixel rechits
  edm::InputTag pixelRecHitsInputTag  = conf_.getParameter<edm::InputTag>("pixelRecHits");

  // special treatment for getting pixel collection
  // if collection exists in file, use collection from file
  // if collection does not exist in file, create empty collection
  static const SiPixelRecHitCollection s_empty;
  const SiPixelRecHitCollection *pixelRecHitCollection = &s_empty;
  edm::Handle<SiPixelRecHitCollection> pixelRecHits;
  if( e.getByLabel(pixelRecHitsInputTag, pixelRecHits)) {
    pixelRecHitCollection = pixelRecHits.product();
  } else {
    LogDebug("RoadSearch") << "Collection SiPixelRecHitCollection with InputTag " << pixelRecHitsInputTag << " cannot be found, using empty collection of same type. The RoadSearch algorithm is also fully functional without Pixel RecHits.";
  }

  // create empty output collection
  std::auto_ptr<RoadSearchSeedCollection> output(new RoadSearchSeedCollection);

  ClusterChecker check(conf_);
    
  // invoke the seed finding algorithm: check number of clusters per event *only* in cosmic tracking mode
  size_t clustsOrZero = check.tooManyClusters(e);
  if (!clustsOrZero) {

    roadSearchSeedFinderAlgorithm_.run(rphiRecHits.product(),  
				       stereoRecHits.product(),
				       matchedRecHits.product(),
				       pixelRecHitCollection,
				       es,
				       *output);
  } else {
    edm::LogError("TooManyClusters") << "Found too many clusters (" << clustsOrZero << "), bailing out.\n";
  }

  // write output to file
  e.put(output);

}
