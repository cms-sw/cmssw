#include "RecoTracker/TrackProducer/plugins/TrackRefitter.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

TrackRefitter::TrackRefitter(const edm::ParameterSet& iConfig):
  KfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent")),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( iConfig.getParameter<std::string>( "src" ));
  setProducer( iConfig.getParameter<std::string>( "producer" ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );
  std::string  constraint_str = iConfig.getParameter<std::string>( "constraint" );

  if (constraint_str == "") constraint_ = none;
  else if (constraint_str == "momentum") constraint_ = momentum;
  else if (constraint_str == "vertex") constraint_ = vertex;
  else {
    edm::LogError("TrackRefitter")<<"constraint: "<<constraint_str<<" not understood. Set it to 'momentum', 'vertex' or leave it empty";    
    throw cms::Exception("TrackRefitter") << "unknown type of contraint! Set it to 'momentum', 'vertex' or leave it empty";    
  }

  //register your products
  produces<reco::TrackCollection>().setBranchAlias( alias_ + "Tracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
  produces<std::vector<Trajectory> >() ;
  produces<TrajTrackAssociationCollection>();

}

void TrackRefitter::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  edm::LogInfo("TrackRefitter") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::auto_ptr<TrackingRecHitCollection>   outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection>      outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> >   outputTrajectoryColl(new std::vector<Trajectory>);

  //
  //declare and get stuff to be retrieved from ES
  //
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  //  getFromES(setup,theG,theMF,theFitter,thePropagator);
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  getFromES(setup,theG,theMF,theFitter,thePropagator,theBuilder);

  //
  //declare and get TrackCollection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  switch(constraint_){
  case none :
    {
      edm::Handle<reco::TrackCollection> theTCollection;
      getFromEvt(theEvent,theTCollection);
      if (theTCollection.failedToGet()){
	edm::LogError("TrackRefitter")<<"could not get the reco::TrackCollection."; break;}
      LogDebug("TrackRefitter") << "run the algorithm" << "\n";
      try {
	theAlgo.runWithTrack(theG.product(), theMF.product(), *theTCollection, 
			     theFitter.product(), thePropagator.product(), theBuilder.product(), algoResults);
      }catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack." << "\n" << e << "\n";}
      break;
    }
  case momentum :
    {
      edm::Handle<TrackMomConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByType(theTCollectionWithConstraint);
      if (theTCollectionWithConstraint.failedToGet()){
	edm::LogError("TrackRefitter")<<"could not get TrackMomConstraintAssociationCollection product."; break;}
      LogDebug("TrackRefitter") << "run the algorithm" << "\n";
      try {
	theAlgo.runWithMomentum(theG.product(), theMF.product(), *theTCollectionWithConstraint, 
				theFitter.product(), thePropagator.product(), theBuilder.product(), algoResults);
      }catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack." << "\n" << e << "\n";}
      break;}
  case  vertex :
    {
      edm::Handle<TrackVtxConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByType(theTCollectionWithConstraint);
      if (theTCollectionWithConstraint.failedToGet()){
	edm::LogError("TrackRefitter")<<"could not get TrackVtxConstraintAssociationCollection product."; break;}
      LogDebug("TrackRefitter") << "run the algorithm" << "\n";
      try {
      theAlgo.runWithVertex(theG.product(), theMF.product(), *theTCollectionWithConstraint, 
			    theFitter.product(), thePropagator.product(), theBuilder.product(), algoResults);      
      }catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack." << "\n" << e << "\n";}
    }
    //default... there cannot be any other possibility due to the check in the ctor
  }

  
  //put everything in th event
  putInEvt(theEvent, outputRHColl, outputTColl, outputTEColl, outputTrajectoryColl, algoResults);
  LogDebug("TrackRefitter") << "end" << "\n";
}

