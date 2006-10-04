// Package:    RecoTracker/SingleTrackPattern
// Class:      CosmicTrackFinder
// Original Author:  Michele Pioppi-INFN perugia
#include <memory>
#include <string>

#include "RecoTracker/SingleTrackPattern/interface/CosmicTrackFinder.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

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
    edm::InputTag matchedrecHitsTag = conf_.getParameter<edm::InputTag>("matchedRecHits");
    edm::InputTag rphirecHitsTag = conf_.getParameter<edm::InputTag>("rphirecHits");
    edm::InputTag stereorecHitsTag = conf_.getParameter<edm::InputTag>("stereorecHits");
  
    edm::InputTag seedTag = conf_.getParameter<edm::InputTag>("cosmicSeeds");
    // retrieve seeds
    edm::Handle<TrajectorySeedCollection> seed;
    //   e.getByType(seed);
    e.getByLabel(seedTag,seed);  
  //retrieve PixelRecHits
    edm::Handle<SiPixelRecHitCollection> pixelHits;
    if (geometry!="MTCC")  e.getByType(pixelHits);
    //retrieve StripRecHits
    edm::Handle<SiStripMatchedRecHit2DCollection> matchedrecHits;
    e.getByLabel( matchedrecHitsTag ,matchedrecHits);
    edm::Handle<SiStripRecHit2DCollection> rphirecHits;
    e.getByLabel( rphirecHitsTag ,rphirecHits);
    edm::Handle<SiStripRecHit2DCollection> stereorecHits;
    e.getByLabel( stereorecHitsTag, stereorecHits);

    // Step B: create empty output collection
    std::auto_ptr<reco::TrackCollection> output(new reco::TrackCollection);
    std::auto_ptr<TrackingRecHitCollection> outputRHColl (new TrackingRecHitCollection);
    std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);


    edm::ESHandle<TrackerGeometry> tracker;
    es.get<TrackerDigiGeometryRecord>().get(tracker);

  
    if((*seed).size()>0){

      bool seedplus=((*(*seed).begin()).direction()==alongMomentum);
  
      if (seedplus)
	LogDebug("CosmicTrackFinder")<<"Reconstruction along momentum ";
      else
	LogDebug("CosmicTrackFinder")<<"Reconstruction opposite to momentum";
      cosmicTrajectoryBuilder_.init(es,seedplus);
      
      
      
      std::vector<Trajectory> trajoutput;
      
      cosmicTrajectoryBuilder_.run(*seed,
				   *stereorecHits,
				   *rphirecHits,
				   *matchedrecHits,
				   *pixelHits,
				   es,
				   e,
				   trajoutput);
   
      
      if(trajoutput.size()>0){
	//Trajectory from the algorithm
	const Trajectory  theTraj =(*trajoutput.begin());
	
	//RecHitCollection	
	//RC const edm::OwnVector< const TransientTrackingRecHit>& transHits = theTraj.recHits();
	//RC for(edm::OwnVector<const TransientTrackingRecHit>::const_iterator j=transHits.begin();
	Trajectory::RecHitContainer transHits = theTraj.recHits();
	for(Trajectory::RecHitContainer::const_iterator j=transHits.begin();
	    j!=transHits.end(); j++){
	  //RC outputRHColl->push_back( ( (j->hit() )->clone()) );
	  outputRHColl->push_back( ( ((**j).hit() )->clone()) );
	}

	edm::OrphanHandle <TrackingRecHitCollection> ohRH  = e.put( outputRHColl );

	int cc = 0;	
	TSOS UpState;
	TSOS LowState;
	unsigned int outerId, innerId;
	if (seedplus){
          UpState=theTraj.lastMeasurement().updatedState();
          LowState=theTraj.firstMeasurement().updatedState();
	  outerId = theTraj.lastMeasurement().recHit()->geographicalId().rawId();
	  innerId = theTraj.firstMeasurement().recHit()->geographicalId().rawId();
        }else{
          UpState=theTraj.firstMeasurement().updatedState();
          LowState=theTraj.lastMeasurement().updatedState();
	  outerId = theTraj.firstMeasurement().recHit()->geographicalId().rawId();
	  innerId = theTraj.lastMeasurement().recHit()->geographicalId().rawId();
        }

	//Track construction
	int ndof =theTraj.foundHits()-5;
	if (ndof<0) ndof=0;

	TSCPBuilderNoMaterial tscpBuilder;
	TrajectoryStateClosestToPoint tscp=tscpBuilder(*(UpState.freeState()),
						       UpState.globalPosition());
	PerigeeTrajectoryParameters::ParameterVector param = tscp.perigeeParameters();
	
	PerigeeTrajectoryError::CovarianceMatrix covar = tscp.perigeeError();


	
	reco::Track theTrack(theTraj.chiSquared(),
			     int(ndof),
			     param,tscp.pt(),
			     covar);


	//Track Extra
	GlobalPoint v=UpState.globalPosition();
	GlobalVector p=UpState.globalMomentum();
	math::XYZVector outmom( p.x(), p.y(), p.z() );
	math::XYZPoint  outpos( v.x(), v.y(), v.z() );   
	v=LowState.globalPosition();
	p=LowState.globalMomentum();
	math::XYZVector inmom( p.x(), p.y(), p.z() );
	math::XYZPoint  inpos( v.x(), v.y(), v.z() );   
//	reco::TrackExtra *theTrackExtra = new reco::TrackExtra(outpos, outmom, true);
	reco::TrackExtra *theTrackExtra = new reco::TrackExtra(outpos, outmom, true, inpos, inmom, true,
							       UpState.curvilinearError(), outerId,
							       LowState.curvilinearError(), innerId);
	//RC for(edm::OwnVector<const TransientTrackingRecHit>::const_iterator j=transHits.begin();
	for(Trajectory::RecHitContainer::const_iterator j=transHits.begin();
	    j!=transHits.end(); j++){
	  theTrackExtra->add(TrackingRecHitRef(ohRH,cc));
	  cc++;
	}

	outputTEColl->push_back(*theTrackExtra);
	edm::OrphanHandle<reco::TrackExtraCollection> ohTE = e.put(outputTEColl);

	reco::TrackExtraRef  theTrackExtraRef(ohTE,0);
	theTrack.setExtra(theTrackExtraRef);
	theTrack.setHitPattern((*theTrackExtraRef).recHits());

	output->push_back(theTrack);

      }

    }
	e.put(output);
  
  }
}
