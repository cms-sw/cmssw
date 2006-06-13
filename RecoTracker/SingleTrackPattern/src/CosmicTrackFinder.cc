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
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
namespace cms
{

  CosmicTrackFinder::CosmicTrackFinder(edm::ParameterSet const& conf) : 
    cosmicTrajectoryBuilder_(conf) ,
    conf_(conf)
  {
    geometry=conf_.getUntrackedParameter<std::string>("GeometricStructure","STANDARD");
    produces<reco::TrackCollection>();
    produces<TrackingRecHitCollection>();
    produces<reco::TrackExtraCollection>();
  }


  // Virtual destructor needed.
  CosmicTrackFinder::~CosmicTrackFinder() { }  

  // Functions that gets called by framework every event
  void CosmicTrackFinder::produce(edm::Event& e, const edm::EventSetup& es)
  {

    std::string hitProducer = conf_.getParameter<std::string>("HitProducer");

  
    // retrieve seeds
    edm::Handle<TrajectorySeedCollection> seed;
    e.getByType(seed);
    //retrieve PixelRecHits
    edm::Handle<SiPixelRecHitCollection> pixelHits;
    if (geometry!="MTCC")  e.getByType(pixelHits);
    //retrieve StripRecHits
    edm::Handle<SiStripRecHit2DMatchedLocalPosCollection> matchedrecHits;
    e.getByLabel(hitProducer,"matchedRecHit" ,matchedrecHits);
    edm::Handle<SiStripRecHit2DLocalPosCollection> rphirecHits;
    e.getByLabel(hitProducer,"rphiRecHit" ,rphirecHits);
    edm::Handle<SiStripRecHit2DLocalPosCollection> stereorecHits;
    e.getByLabel(hitProducer,"stereoRecHit" ,stereorecHits);

    // Step B: create empty output collection
    std::auto_ptr<reco::TrackCollection> output(new reco::TrackCollection);
    std::auto_ptr<TrackingRecHitCollection> outputRHColl (new TrackingRecHitCollection);
    std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);


    edm::ESHandle<TrackerGeometry> tracker;
    es.get<TrackerDigiGeometryRecord>().get(tracker);


  
    if((*seed).size()>0){
      TrajectorySeed::range hRange= (*(*seed).begin()).recHits();
      TrajectorySeed::const_iterator ihit;
      float yy1=0;
      float yy2=0;
      uint iny=1;
      for (ihit = hRange.first; 
	   ihit != hRange.second; ihit++) {

	unsigned int iraw=(*ihit).geographicalId().rawId();
	LocalPoint lp=(*ihit).localPosition();
	if (iny==1)yy1=(tracker->idToDet(DetId(iraw))->surface().toGlobal(lp).y());
	else yy2=(tracker->idToDet(DetId(iraw))->surface().toGlobal(lp).y());
	iny++;
      }


      bool seedplus=((yy1-yy2)>0);

      if (seedplus)
	LogDebug("CosmicTrackFinder")<<"Reconstruction in top-down direction";
      else
	LogDebug("CosmicTrackFinder")<<"Reconstruction in bottom-up direction";
      cosmicTrajectoryBuilder_.init(es,seedplus);
      
      vector<AlgoProduct> algooutput;
      edm::OrphanHandle<reco::TrackExtraCollection> ohTE;
      
    
      cosmicTrajectoryBuilder_.run(*seed,
				   *stereorecHits,
				   *rphirecHits,
				   *matchedrecHits,
				   *pixelHits,
				   es,
				   e,
				   algooutput);
   
     
      if(algooutput.size()>0){
	int cc = 0;	
	vector<AlgoProduct>::iterator ialgo;
	for(ialgo=algooutput.begin();ialgo!=algooutput.end();ialgo++){

	
	  Trajectory  theTraj = (*ialgo).first;
	  //RecHitCollection	
	  const edm::OwnVector< const TransientTrackingRecHit>& transHits = theTraj.recHits();
	  for(edm::OwnVector<const TransientTrackingRecHit>::const_iterator j=transHits.begin();
	      j!=transHits.end(); j++){
	    outputRHColl->push_back( ( (j->hit() )->clone()) );
	  }

	  edm::OrphanHandle <TrackingRecHitCollection> ohRH  = e.put( outputRHColl );

	  //TrackExtra????

	  reco::TrackExtra * theTrackExtra;
	  TSOS outertsos = theTraj.lastMeasurement().updatedState();
	  TSOS Fitsos = theTraj.firstMeasurement().updatedState();

	  GlobalPoint v;
	  GlobalVector p;
	  if (seedplus){
	    p=outertsos.globalMomentum();
	    v=outertsos.globalPosition();
	  }else{
	    p=Fitsos.globalMomentum();
	    v=Fitsos.globalPosition();
	  }


	  math::XYZVector outmom( p.x(), p.y(), p.z() );
	  math::XYZPoint  outpos( v.x(), v.y(), v.z() );   
	  theTrackExtra = new reco::TrackExtra(outpos, outmom, true);
	  for(edm::OwnVector<const TransientTrackingRecHit>::const_iterator j=transHits.begin();
	      j!=transHits.end(); j++){
	    theTrackExtra->add(TrackingRecHitRef(ohRH,cc));
	    cc++;
	  }
	  outputTEColl->push_back(*theTrackExtra);
	  ohTE = e.put(outputTEColl);
	}
	cc = 0;
	reco::Track  theTrack = (*ialgo).second;
	reco::TrackExtraRef  theTrackExtraRef(ohTE,cc);
	theTrack.setExtra(theTrackExtraRef);
	output->push_back(theTrack);
    
	cc++;
	e.put(output);
      }

    }
  }
  
}
