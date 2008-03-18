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
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"


namespace cms
{

  CosmicTrackFinder::CosmicTrackFinder(edm::ParameterSet const& conf) : 
    cosmicTrajectoryBuilder_(conf) ,
    crackTrajectoryBuilder_(conf) ,
    conf_(conf)
  {
    geometry=conf_.getUntrackedParameter<std::string>("GeometricStructure","STANDARD");
    trinevents=conf_.getParameter<bool>("TrajInEvents");
    produces<reco::TrackCollection>();
    produces<TrackingRecHitCollection>();
    produces<reco::TrackExtraCollection>();
    if (trinevents) {
      produces<std::vector<Trajectory> >();
      produces<TrajTrackAssociationCollection>();
    }
  }


  // Virtual destructor needed.
  CosmicTrackFinder::~CosmicTrackFinder() { }  

  // Functions that gets called by framework every event
  void CosmicTrackFinder::produce(edm::Event& e, const edm::EventSetup& es)
  {
    edm::InputTag matchedrecHitsTag = conf_.getParameter<edm::InputTag>("matchedRecHits");
    edm::InputTag rphirecHitsTag = conf_.getParameter<edm::InputTag>("rphirecHits");
    edm::InputTag stereorecHitsTag = conf_.getParameter<edm::InputTag>("stereorecHits");
    edm::InputTag pixelRecHitsTag = conf_.getParameter<edm::InputTag>("pixelRecHits");  


    edm::InputTag seedTag = conf_.getParameter<edm::InputTag>("cosmicSeeds");
    // retrieve seeds
    edm::Handle<TrajectorySeedCollection> seed;
    e.getByLabel(seedTag,seed);  

  //retrieve PixelRecHits
    static const SiPixelRecHitCollection s_empty;
    const SiPixelRecHitCollection *pixelHitCollection = &s_empty;
    edm::Handle<SiPixelRecHitCollection> pixelHits;
    if (geometry!="MTCC" && (geometry!="CRACK" )) {
      if( e.getByLabel(pixelRecHitsTag, pixelHits)) {
	pixelHitCollection = pixelHits.product();
      } else {
	edm::LogWarning("CosmicTrackFinder") << "Collection SiPixelRecHitCollection with InputTag " << pixelRecHitsTag << " cannot be found, using empty collection of same type.";
      }
    }
    
   


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
    std::auto_ptr<std::vector<Trajectory> > outputTJ(new std::vector<Trajectory> );

    edm::ESHandle<TrackerGeometry> tracker;
    es.get<TrackerDigiGeometryRecord>().get(tracker);
    edm::LogVerbatim("CosmicTrackFinder") << "========== Cosmic Track Finder Info ==========";
    edm::LogVerbatim("CosmicTrackFinder") << " Numbers of Seeds " << (*seed).size();
    if((*seed).size()>0){

      std::vector<Trajectory> trajoutput;
      
      if(geometry!="CRACK" ) {
        cosmicTrajectoryBuilder_.run(*seed,
                                   *stereorecHits,
                                   *rphirecHits,
                                   *matchedrecHits,
				   *pixelHitCollection,
                                   es,
                                   e,
                                   trajoutput);
      } else {
        crackTrajectoryBuilder_.run(*seed,
                                   *stereorecHits,
                                   *rphirecHits,
                                   *matchedrecHits,
				   *pixelHitCollection,
                                   es,
                                   e,
                                   trajoutput);
      }

      edm::LogVerbatim("CosmicTrackFinder") << " Numbers of Temp Trajectories " << trajoutput.size();
      edm::LogVerbatim("CosmicTrackFinder") << "========== END Info ==========";
      if(trajoutput.size()>0){
	std::vector<Trajectory*> tmpTraj;
	std::vector<Trajectory>::iterator itr;
	for (itr=trajoutput.begin();itr!=trajoutput.end();itr++)tmpTraj.push_back(&(*itr));

	//The best track is selected
	//FOR MTCC the criteria are:
	//1)# of layers,2) # of Hits,3)Chi2
	if (geometry=="MTCC")  stable_sort(tmpTraj.begin(),tmpTraj.end(),CompareTrajLay());
	else  stable_sort(tmpTraj.begin(),tmpTraj.end(),CompareTrajChi());



	const Trajectory  theTraj = *(*tmpTraj.begin());
	if(trinevents) outputTJ->push_back(theTraj);
	bool seedplus=(theTraj.seed().direction()==alongMomentum);
	PropagationDirection seedDir =theTraj.seed().direction();
	if (seedplus)
	  LogDebug("CosmicTrackFinder")<<"Reconstruction along momentum ";
	else
	  LogDebug("CosmicTrackFinder")<<"Reconstruction opposite to momentum";

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
        if(geometry=="CRACK") ndof++;
	if (ndof<0) ndof=0;

	TSCPBuilderNoMaterial tscpBuilder;
	TrajectoryStateClosestToPoint tscp=tscpBuilder(*(UpState.freeState()),
						       UpState.globalPosition());

// 	PerigeeTrajectoryParameters::ParameterVector param = tscp.perigeeParameters();
	
// 	PerigeeTrajectoryError::CovarianceMatrix covar = tscp.perigeeError();
	GlobalPoint vv = tscp.theState().position();
	math::XYZPoint  pos( vv.x(), vv.y(), vv.z() );
	GlobalVector pp = tscp.theState().momentum();
	math::XYZVector mom( pp.x(), pp.y(), pp.z() );


	reco::Track theTrack(theTraj.chiSquared(),
			     int(ndof),
			     pos, mom, tscp.charge(), tscp.theState().curvilinearError());


	
// 	reco::Track theTrack(theTraj.chiSquared(),
// 			     int(ndof),
// 			     param,tscp.pt(),
// 			     covar);


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
							       LowState.curvilinearError(), innerId,seedDir);
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
	delete theTrackExtra;
      }
      else {
	e.put( outputRHColl );
	e.put(outputTEColl);    
      }
    }
    else {
      e.put( outputRHColl );
      e.put(outputTEColl);    
    }
    edm::OrphanHandle<reco::TrackCollection> rTracks = e.put(output);  
    if (trinevents) {
      edm::OrphanHandle<std::vector<Trajectory> > rTrajs = e.put(outputTJ);
      // Now Create traj<->tracks association map, we have only one track at 0:
      std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection() );
      if (rTracks->size() == 1) {
 	trajTrackMap->insert(edm::Ref<std::vector<Trajectory> > (rTrajs, 0),
			     edm::Ref<reco::TrackCollection>    (rTracks, 0));
      } else if (rTracks->size() != 0) {
	edm::LogError("WrongSize") <<"@SUB=produce" << "Expected <= 1 track, not " 
				   << rTracks->size();
      }
      e.put( trajTrackMap );
    }
  }
}
