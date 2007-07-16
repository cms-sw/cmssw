#include "RecoTracker/TrackProducer/interface/TrackRefitter.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

TrackRefitter::TrackRefitter(const edm::ParameterSet& iConfig):
  TrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent")),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( iConfig.getParameter<std::string>( "src" ));
  setProducer( iConfig.getParameter<std::string>( "producer" ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );
  constraint_ = iConfig.getParameter<std::string>( "constraint" );
  //register your products
  produces<reco::TrackCollection>().setBranchAlias( alias_ + "Tracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
  produces<std::vector<Trajectory> >() ;
  produces<TrajTrackAssociationCollection>();
}

void TrackRefitter::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  edm::LogInfo("TrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";
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
  try {
    if (constraint_==""){
      edm::Handle<reco::TrackCollection> theTCollection;
      getFromEvt(theEvent,theTCollection);
      
      //
      //run the algorithm  
      //
      LogDebug("TrackProducer") << "run the algorithm" << "\n";
      theAlgo.runWithTrack(theG.product(), theMF.product(), *theTCollection, 
			   theFitter.product(), thePropagator.product(), theBuilder.product(), algoResults);
    } else if (constraint_=="momentum"){
      edm::Handle<TrackMomConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByType(theTCollectionWithConstraint);
      LogDebug("TrackProducer") << "run the algorithm" << "\n";
      theAlgo.runWithMomentum(theG.product(), theMF.product(), *theTCollectionWithConstraint, 
			      theFitter.product(), thePropagator.product(), theBuilder.product(), algoResults);
    } else if (constraint_=="vertex"){
      edm::Handle<TrackVtxConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByType(theTCollectionWithConstraint);
      LogDebug("TrackProducer") << "run the algorithm" << "\n";
      theAlgo.runWithVertex(theG.product(), theMF.product(), *theTCollectionWithConstraint, 
			      theFitter.product(), thePropagator.product(), theBuilder.product(), algoResults);      
    } else throw cms::Exception("TrackRefitter") << "unknown type of contraint! Set it to 'momentum', 'vertex' or leave it empty";

  } catch (cms::Exception &e){}
  //
  //put everything in th event
  putInEvt(theEvent, outputRHColl, outputTColl, outputTEColl, outputTrajectoryColl, algoResults);
  LogDebug("TrackProducer") << "end" << "\n";
}

