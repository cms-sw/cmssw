#include "RecoTracker/TrackProducer/plugins/MTFTrackProducer.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiTrackFilterHitCollector.h"
#include "RecoTracker/Record/interface/MultiRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"


MTFTrackProducer::MTFTrackProducer(const edm::ParameterSet& iConfig):
  KfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),false),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( iConfig.getParameter<edm::InputTag>( "src" ), iConfig.getParameter<edm::InputTag>( "beamSpot" ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );
  //register your products
  produces<reco::TrackCollection>().setBranchAlias( alias_ + "Tracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
  produces<std::vector<Trajectory> >() ;
  produces<TrajTrackAssociationCollection>();
}


void MTFTrackProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  edm::LogInfo("MTFTrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::auto_ptr<TrackingRecHitCollection>    outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection>       outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection>  outputTEColl(new reco::TrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> >    outputTrajectoryColl(new std::vector<Trajectory>);

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
  //get additional es_modules needed by the MTF	
  edm::ESHandle<MultiTrackFilterHitCollector> measurementCollectorHandle;
  edm::ESHandle<SiTrackerMultiRecHitUpdatorMTF> updatorHandle;	
  std::string measurementCollectorName = getConf().getParameter<std::string>("MeasurementCollector");
  setup.get<MultiRecHitRecord>().get(measurementCollectorName, measurementCollectorHandle);
  std::string  updatorName = getConf().getParameter<std::string>("UpdatorName");	
  setup.get<MultiRecHitRecord>().get(updatorName, updatorHandle);	 

  AlgoProductCollection algoResults;

  edm::Handle<std::vector<Trajectory> > theTrajectoryCollection;

  reco::BeamSpot bs;


  getFromEvt(theEvent,theTrajectoryCollection,bs);

  measurementCollectorHandle->updateEvent(theEvent); 	
  //
  //run the algorithm  
    //

  theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTrajectoryCollection,
			   theFitter.product(), theBuilder.product(), measurementCollectorHandle.product(), updatorHandle.product(),bs,algoResults);
  //
  //put everything in the event
  putInEvt(theEvent, thePropagator.product(), theMeasTk.product(), 
	   outputRHColl, outputTColl, outputTEColl, outputTrajectoryColl, algoResults);
  LogDebug("MTFTrackProducer") << "end" << "\n";
}


void MTFTrackProducer::getFromEvt(edm::Event& theEvent,edm::Handle<TrajectoryCollection>& theTrajectoryCollection, reco::BeamSpot& bs)
{

  edm::InputTag src_=getConf().getParameter<edm::InputTag>( "src" );
  theEvent.getByLabel(src_,theTrajectoryCollection );  
  
  //get the BeamSpot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  theEvent.getByLabel(bsSrc_,recoBeamSpotHandle);
  bs = *recoBeamSpotHandle;
}
