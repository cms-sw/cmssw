#include "RecoTracker/TrackProducer/plugins/TrackRefitter.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

TrackRefitter::TrackRefitter(const edm::ParameterSet& iConfig):
  KfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),
		      iConfig.getParameter<bool>("useHitsSplitting")),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>( "src" )), 
          consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>( "beamSpot" )),
          consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>( "MeasurementTrackerEvent") ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );
  std::string  constraint_str = iConfig.getParameter<std::string>( "constraint" );
  edm::InputTag trkconstrcoll = iConfig.getParameter<edm::InputTag>( "srcConstr" );
  

  if (constraint_str == "") constraint_ = none;
  else if (constraint_str == "momentum") { constraint_ = momentum; trkconstrcoll_ = consumes<TrackMomConstraintAssociationCollection>(trkconstrcoll); }
  else if (constraint_str == "vertex")   { constraint_ = vertex;   trkconstrcoll_ = consumes<TrackVtxConstraintAssociationCollection>(trkconstrcoll); }
  else if (constraint_str == "trackParameters") { constraint_ = trackParameters;  trkconstrcoll_ = consumes<TrackParamConstraintAssociationCollection>(trkconstrcoll); }
  else {
    edm::LogError("TrackRefitter")<<"constraint: "<<constraint_str<<" not understood. Set it to 'momentum', 'vertex', 'trackParameters' or leave it empty";
    throw cms::Exception("TrackRefitter") << "unknown type of contraint! Set it to 'momentum', 'vertex', 'trackParameters' or leave it empty";    
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
  LogDebug("TrackRefitter") << "Analyzing event number: " << theEvent.id() << "\n";
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
  edm::ESHandle<MeasurementTracker>  theMeasTk;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  getFromES(setup,theG,theMF,theFitter,thePropagator,theMeasTk,theBuilder);

  edm::ESHandle<TrackerTopology> httopo;
  setup.get<TrackerTopologyRcd>().get(httopo);

  //
  //declare and get TrackCollection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;
  switch(constraint_){
  case none :
    {
      edm::Handle<reco::TrackCollection> theTCollection;
      getFromEvt(theEvent,theTCollection,bs);

      LogDebug("TrackRefitter") << "TrackRefitter::produce(none):Number of Trajectories:" << (*theTCollection).size();

      if (bs.position()==math::XYZPoint(0.,0.,0.) && bs.type() == reco::BeamSpot::Unknown) {
	edm::LogError("TrackRefitter") << " BeamSpot is (0,0,0), it is probably because is not valid in the event"; break; }

      if (theTCollection.failedToGet()){
	edm::LogError("TrackRefitter")<<"could not get the reco::TrackCollection."; break;}
      LogDebug("TrackRefitter") << "run the algorithm" << "\n";

      try {
	theAlgo.runWithTrack(theG.product(), theMF.product(), *theTCollection, 
			     theFitter.product(), thePropagator.product(), 
			     theBuilder.product(), bs, algoResults);
      }catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack." << "\n" << e << "\n"; throw; }
      break;
    }
  case momentum :
    {
      edm::Handle<TrackMomConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByToken(trkconstrcoll_,theTCollectionWithConstraint);


      edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
      theEvent.getByToken(bsSrc_,recoBeamSpotHandle);
      if (!recoBeamSpotHandle.isValid()) break;
      bs = *recoBeamSpotHandle;      
      if (theTCollectionWithConstraint.failedToGet()){
	//edm::LogError("TrackRefitter")<<"could not get TrackMomConstraintAssociationCollection product.";
	break;}
      LogDebug("TrackRefitter") << "run the algorithm" << "\n";
      try {
	theAlgo.runWithMomentum(theG.product(), theMF.product(), *theTCollectionWithConstraint, 
				theFitter.product(), thePropagator.product(), theBuilder.product(), bs, algoResults);
      }catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack." << "\n" << e << "\n"; throw; }
      break;
    }
  case  vertex :
    {
      edm::Handle<TrackVtxConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByToken(trkconstrcoll_,theTCollectionWithConstraint);
      edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
      theEvent.getByToken(bsSrc_,recoBeamSpotHandle);
      if (!recoBeamSpotHandle.isValid()) break;
      bs = *recoBeamSpotHandle;      
      if (theTCollectionWithConstraint.failedToGet()){
	edm::LogError("TrackRefitter")<<"could not get TrackVtxConstraintAssociationCollection product."; break;}
      LogDebug("TrackRefitter") << "run the algorithm" << "\n";
      try {
      theAlgo.runWithVertex(theG.product(), theMF.product(), *theTCollectionWithConstraint, 
			    theFitter.product(), thePropagator.product(), theBuilder.product(), bs, algoResults);      
      }catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack." << "\n" << e << "\n"; throw; }
    }
  case trackParameters :
    {
      edm::Handle<TrackParamConstraintAssociationCollection> theTCollectionWithConstraint;
      theEvent.getByToken(trkconstrcoll_,theTCollectionWithConstraint);
      edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
      theEvent.getByToken(bsSrc_,recoBeamSpotHandle);
      if (!recoBeamSpotHandle.isValid()) break;
      bs = *recoBeamSpotHandle;      
      if (theTCollectionWithConstraint.failedToGet()){
	//edm::LogError("TrackRefitter")<<"could not get TrackParamConstraintAssociationCollection product.";
	break;}
      LogDebug("TrackRefitter") << "run the algorithm" << "\n";
      try {
      theAlgo.runWithTrackParameters(theG.product(), theMF.product(), *theTCollectionWithConstraint, 
				     theFitter.product(), thePropagator.product(), theBuilder.product(), bs, algoResults);      
      }catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack." << "\n" << e << "\n"; throw; }
    }
    //default... there cannot be any other possibility due to the check in the ctor
  }

  
  //put everything in th event
  putInEvt(theEvent, thePropagator.product(), theMeasTk.product(), outputRHColl, outputTColl, outputTEColl, outputTrajectoryColl, algoResults,theBuilder.product(), httopo.product());
  LogDebug("TrackRefitter") << "end" << "\n";
}

