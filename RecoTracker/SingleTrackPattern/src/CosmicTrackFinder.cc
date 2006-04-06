// Package:    RecoTracker/SingleTrackPattern
// Class:      CosmicTrackFinder
// Original Author:  Michele Pioppi-INFN perugia
#include <memory>
#include <string>

#include "RecoTracker/SingleTrackPattern/interface/CosmicTrackFinder.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace cms
{

  CosmicTrackFinder::CosmicTrackFinder(edm::ParameterSet const& conf) : 
    cosmicTrajectoryBuilder_(conf) ,
    conf_(conf)
  {
    produces<TrackCandidateCollection>();
  }


  // Virtual destructor needed.
  CosmicTrackFinder::~CosmicTrackFinder() { }  

  // Functions that gets called by framework every event
  void CosmicTrackFinder::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // Step A: Get Inputs 
    
    // retrieve seeds
    edm::Handle<TrajectorySeedCollection> seed;
    e.getByType(seed);
    //retrieve PixelRecHits
    edm::Handle<SiPixelRecHitCollection> pixelHits;
    e.getByType(pixelHits);
    //retrieve StripRecHits
    edm::Handle<SiStripRecHit2DMatchedLocalPosCollection> matchedrecHits;
    e.getByLabel("SiStripRecHits","matchedRecHit" ,matchedrecHits);
    edm::Handle<SiStripRecHit2DLocalPosCollection> rphirecHits;
    e.getByLabel("SiStripRecHits","rphiRecHit" ,rphirecHits);
    edm::Handle<SiStripRecHit2DLocalPosCollection> stereorecHits;
    e.getByLabel("SiStripRecHits","stereoRecHit" ,stereorecHits);
    



    // Step B: create empty output collection
    std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);
    
    cosmicTrajectoryBuilder_.init(es);

    // Step C: Invoke the cloud cleaning algorithm

    if ((*seed).size()>0){
      cosmicTrajectoryBuilder_.run(*seed,
				   *stereorecHits,
				   *rphirecHits,
				   *matchedrecHits,
				   *pixelHits,
				   es,
				   *output);
      
      
      // Step D: write output to file
      if ((*output).size()>0) e.put(output);
    }
  }
  
}
