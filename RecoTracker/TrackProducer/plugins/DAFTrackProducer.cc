#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/MultiRecHitRecord.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "RecoTracker/TrackProducer/plugins/DAFTrackProducer.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajAnnealing.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"



DAFTrackProducer::DAFTrackProducer(const edm::ParameterSet& iConfig):
  KfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),false),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( consumes<TrackCandidateCollection>(iConfig.getParameter<edm::InputTag>( "src" )),
          consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>( "beamSpot" )),
          consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>( "MeasurementTrackerEvent") ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );

  //register your products
  produces<reco::TrackCollection>().setBranchAlias( alias_ + "Tracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();
  produces<TrajAnnealingCollection>().setBranchAlias( alias_ + "TrajectoryAnnealing" );
}


void DAFTrackProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
//  std::cout << "Analyzing event number: " << theEvent.id() << "\n";
  edm::LogInfo("DAFTrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";
  
  //empty output collections
  std::auto_ptr<TrackingRecHitCollection>    outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection>       outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection>  outputTEColl(new reco::TrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> >    outputTrajectoryColl(new std::vector<Trajectory>);
  std::auto_ptr<TrajAnnealingCollection>     outputTrajAnnColl(new TrajAnnealingCollection);

  //declare and get stuff to be retrieved from ES
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<MeasurementTracker>  theMeasTk;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  getFromES(setup,theG,theMF,theFitter,thePropagator,theMeasTk,theBuilder);

  //get additional es_modules needed by the DAF	
  edm::ESHandle<MultiRecHitCollector> measurementCollectorHandle;
  std::string measurementCollectorName = getConf().getParameter<std::string>("MeasurementCollector");
  setup.get<MultiRecHitRecord>().get(measurementCollectorName, measurementCollectorHandle);
  edm::ESHandle<SiTrackerMultiRecHitUpdator> updatorHandle;
  std::string updatorName = getConf().getParameter<std::string>("UpdatorName");	
  setup.get<MultiRecHitRecord>().get(updatorName, updatorHandle);	 

  //get MeasurementTrackerEvent
  edm::Handle<MeasurementTrackerEvent> mte;
  theEvent.getByToken(mteSrc_, mte);


  //declare and get TrackColection 
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;
  TrajAnnealingCollection trajannResults;
  try{

    edm::Handle<std::vector<Trajectory> > theTrajectoryCollection;
    getFromEvt(theEvent,theTrajectoryCollection,bs);

    //obsolete?
    //measurementCollectorHandle->updateEvent(theEvent);

    //run the algorithm  
    LogDebug("DAFTrackProducer") << "run the DAF algorithm" << "\n";
    theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTrajectoryCollection, &*mte,
                             theFitter.product(), theBuilder.product(), 
			     measurementCollectorHandle.product(), updatorHandle.product(), bs, 
			     algoResults, trajannResults);
    
  } catch (cms::Exception &e){ 
    edm::LogInfo("DAFTrackProducer") << "cms::Exception caught!!!" << "\n" << e << "\n"; 
    throw; 
  }

  //put everything in the event
  putInEvt(theEvent, thePropagator.product(), theMeasTk.product(), outputRHColl, 
	   outputTColl, outputTEColl, 
           outputTrajectoryColl, algoResults);
  putInEvtTrajAnn(theEvent, trajannResults, outputTrajAnnColl);

//  std::cout << "DAFTrackProducer: end the DAF algorithm." << "\n";
  LogDebug("DAFTrackProducer") << "end the DAF algorithm." << "\n";
}

void DAFTrackProducer::getFromEvt(edm::Event& theEvent,edm::Handle<TrajectoryCollection>& theTrajectoryCollection, reco::BeamSpot& bs)
{

  //get the TrajectoryCollection from the event
  //WARNING: src has always to be redefined in cfg file
  edm::InputTag src_=getConf().getParameter<edm::InputTag>( "src" );
  theEvent.getByLabel(src_,theTrajectoryCollection );

  //get the BeamSpot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  theEvent.getByToken(bsSrc_,recoBeamSpotHandle);
  bs = *recoBeamSpotHandle;

}

void DAFTrackProducer::putInEvtTrajAnn(edm::Event& theEvent, TrajAnnealingCollection & trajannResults,
				std::auto_ptr<TrajAnnealingCollection>& outputTrajAnnColl){
  const int size = trajannResults.size();
  outputTrajAnnColl->reserve(size);

  for(unsigned int i = 0; i < trajannResults.size() ; i++){
//    trajannResults.at(i).Debug();
    outputTrajAnnColl->push_back(trajannResults.at(i));
  }

  theEvent.put( outputTrajAnnColl );
}

