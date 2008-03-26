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
// $Author: noeding $
// $Date: 2007/11/07 23:42:18 $
// $Revision: 1.11 $
//

#include <iostream>
#include <memory>
#include <string>

#include "RecoTracker/RoadSearchSeedFinder/interface/RoadSearchSeedFinder.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "DataFormats/RoadSearchSeed/interface/RoadSearchSeedCollection.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

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
    edm::LogWarning("RoadSearch") << "Collection SiPixelRecHitCollection with InputTag " << pixelRecHitsInputTag << " cannot be found, using empty collection of same type. The RoadSearch algorithm is also fully functional without Pixel RecHits.";
  }
  
  //get special input for cluster multiplicity filter
  edm::Handle<edm::DetSetVector<SiStripCluster> > clusterDSV;
  e.getByLabel(clusterCollectionInputTag,clusterDSV);
  const edm::DetSetVector<SiStripCluster> *clusters = clusterDSV.product();

  // create empty output collection
  std::auto_ptr<RoadSearchSeedCollection> output(new RoadSearchSeedCollection);
 

   //special parameters for cosmic track reconstruction
  bool cosmicTracking              = conf_.getParameter<bool>("CosmicTracking");
  unsigned int maxNumberOfClusters = conf_.getParameter<unsigned int>("MaxNumberOfClusters");

  // invoke the seed finding algorithm: check number of clusters per event *only* in cosmic tracking mode
  if(!cosmicTracking 
     || (cosmicTracking && roadSearchSeedFinderAlgorithm_.ClusterCounter(clusters)<maxNumberOfClusters)) {

    roadSearchSeedFinderAlgorithm_.run(rphiRecHits.product(),  
				       stereoRecHits.product(),
				       matchedRecHits.product(),
				       pixelRecHitCollection,
				       es,
				       *output);
  }

  // write output to file
  e.put(output);

}
