#include "RecoTracker/TrackProducer/plugins/GsfTrackRefitter.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/GsfTracking/interface/TrajGsfTrackAssociation.h"
#include "TrackingTools/GsfTracking/interface/GsfTrackConstraintAssociation.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

GsfTrackRefitter::GsfTrackRefitter(const edm::ParameterSet& iConfig):
  GsfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),
		       iConfig.getParameter<bool>("useHitsSplitting")),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( iConfig.getParameter<edm::InputTag>( "src" ), iConfig.getParameter<edm::InputTag>( "beamSpot" ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );
  std::string  constraint_str = iConfig.getParameter<std::string>( "constraint" );

  if (constraint_str == "") constraint_ = none;
//   else if (constraint_str == "momentum") constraint_ = momentum;
  else if (constraint_str == "vertex") {
    constraint_ = vertex;
    gsfTrackVtxConstraintTag_ = iConfig.getParameter<edm::InputTag>("gsfTrackVtxConstraintTag");
  } else {
    edm::LogError("GsfTrackRefitter")<<"constraint: "<<constraint_str<<" not understood. Set it to 'momentum', 'vertex' or leave it empty";    
    throw cms::Exception("GsfTrackRefitter") << "unknown type of contraint! Set it to 'momentum', 'vertex' or leave it empty";    
  }

  //register your products
  produces<reco::GsfTrackCollection>().setBranchAlias( alias_ + "GsfTracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
  produces<reco::GsfTrackExtraCollection>().setBranchAlias( alias_ + "GsfTrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
  produces<std::vector<Trajectory> >() ;
  produces<TrajGsfTrackAssociationCollection>();

}

void GsfTrackRefitter::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  edm::LogInfo("GsfTrackRefitter") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::auto_ptr<TrackingRecHitCollection>   outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::GsfTrackCollection>      outputTColl(new reco::GsfTrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);
  std::auto_ptr<reco::GsfTrackExtraCollection> outputGsfTEColl(new reco::GsfTrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> >   outputTrajectoryColl(new std::vector<Trajectory>);

  //
  //declare and get stuff to be retrieved from ES
  //
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<MeasurementTracker>  theMeasTk;
  //  getFromES(setup,theG,theMF,theFitter,thePropagator);
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  getFromES(setup,theG,theMF,theFitter,thePropagator,theMeasTk,theBuilder);

  //
  //declare and get TrackCollection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;
  switch(constraint_){
  case none :
    {
      edm::Handle<reco::GsfTrackCollection> theTCollection;
      getFromEvt(theEvent,theTCollection,bs);
      if (theTCollection.failedToGet()){
	edm::LogError("GsfTrackRefitter")<<"could not get the reco::GsfTrackCollection."; return;}
      LogDebug("GsfTrackRefitter") << "run the algorithm" << "\n";
      try {
	theAlgo.runWithTrack(theG.product(), theMF.product(), *theTCollection, 
			     theFitter.product(), thePropagator.product(),  
			     theBuilder.product(), bs, algoResults);
      }catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack." << "\n" << e << "\n"; throw; }
      break;
    }
  case vertex :
    {
      edm::Handle<GsfTrackVtxConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByLabel(gsfTrackVtxConstraintTag_, theTCollectionWithConstraint);
      edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
      theEvent.getByLabel(bsSrc_,recoBeamSpotHandle);
      bs = *recoBeamSpotHandle;      
      if (theTCollectionWithConstraint.failedToGet()){
	edm::LogError("TrackRefitter")<<"could not get TrackVtxConstraintAssociationCollection product."; break;}
      LogDebug("TrackRefitter") << "run the algorithm" << "\n";
      try {
      theAlgo.runWithVertex(theG.product(), theMF.product(), *theTCollectionWithConstraint, 
			    theFitter.product(), thePropagator.product(), theBuilder.product(), bs, algoResults);      
      }catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack." << "\n" << e << "\n"; throw; }
    }
    //default... there cannot be any other possibility due to the check in the ctor
  }
  
  //put everything in th event
  putInEvt(theEvent, thePropagator.product(), theMeasTk.product(),
	   outputRHColl, outputTColl, outputTEColl, outputGsfTEColl, outputTrajectoryColl, algoResults, bs);
  LogDebug("GsfTrackRefitter") << "end" << "\n";
}

