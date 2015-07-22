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

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"


DAFTrackProducer::DAFTrackProducer(const edm::ParameterSet& iConfig):
  KfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),false),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( consumes<TrackCandidateCollection>(iConfig.getParameter<edm::InputTag>( "src" )),
          consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>( "beamSpot" )),
          consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>( "MeasurementTrackerEvent") ));
  srcTT_ = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>( "src" ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );

  //register your products
  produces<reco::TrackCollection>().setBranchAlias( alias_ + "Tracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();
  produces<TrajAnnealingCollection>().setBranchAlias( alias_ + "TrajectoryAnnealing" );
  produces<reco::TrackCollection>("beforeDAF").setBranchAlias( alias_ + "TracksBeforeDAF" );
  produces<reco::TrackExtraCollection>("beforeDAF").setBranchAlias( alias_ + "TrackExtrasBeforeDAF" );
  produces<reco::TrackCollection>("afterDAF").setBranchAlias( alias_ + "TracksAfterDAF" );
  produces<reco::TrackExtraCollection>("afterDAF").setBranchAlias( alias_ + "TrackExtrasAfterDAF" );

  TrajAnnSaving_ = iConfig.getParameter<bool>("TrajAnnealingSaving");
}


void DAFTrackProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  edm::LogInfo("DAFTrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";
  
  //empty output collections
  std::auto_ptr<TrackingRecHitCollection>    outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection>       outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection>  outputTEColl(new reco::TrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> >    outputTrajectoryColl(new std::vector<Trajectory>);
  std::auto_ptr<TrajAnnealingCollection>     outputTrajAnnColl(new TrajAnnealingCollection);
 
  //new tracks collections (changes before and after DAF)
  std::auto_ptr<TrackingRecHitCollection>    outputRHCollBeforeDAF (new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection>       outputTCollBeforeDAF(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection>  outputTECollBeforeDAF(new reco::TrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> >    outputTrajectoryCollBeforeDAF(new std::vector<Trajectory>);
  //----
  std::auto_ptr<TrackingRecHitCollection>    outputRHCollAfterDAF (new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection>       outputTCollAfterDAF(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection>  outputTECollAfterDAF(new reco::TrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> >    outputTrajectoryCollAfterDAF(new std::vector<Trajectory>);

  //declare and get stuff to be retrieved from ES
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<MeasurementTracker>  theMeasTk;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  getFromES(setup,theG,theMF,theFitter,thePropagator,theMeasTk,theBuilder);

  edm::ESHandle<TrackerTopology> httopo;
  setup.get<TrackerTopologyRcd>().get(httopo);

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


  //declare and get TrackCollection 
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;
  TrajAnnealingCollection trajannResults;

  //declare and get  new tracks collections
  AlgoProductCollection algoResultsBeforeDAF;
  AlgoProductCollection algoResultsAfterDAF;
  try{

    edm::Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
    getFromEvt(theEvent,trajTrackAssociationHandle,bs);


    //run the algorithm  
    LogDebug("DAFTrackProducer") << "run the DAF algorithm" << "\n";
    theAlgo.runWithCandidate(theG.product(), theMF.product(),  
			     *trajTrackAssociationHandle, 
			     &*mte,
                             theFitter.product(), theBuilder.product(), 
			     measurementCollectorHandle.product(), updatorHandle.product(), bs, 
			     algoResults, trajannResults, TrajAnnSaving_,
			     algoResultsBeforeDAF, algoResultsAfterDAF);
    
  } catch (cms::Exception &e){ 
    edm::LogInfo("DAFTrackProducer") << "cms::Exception caught!!!" << "\n" << e << "\n"; 
    throw; 
  }

  //put everything in the event
  putInEvt(theEvent, thePropagator.product(),theMeasTk.product(), 
           outputRHColl, outputTColl, outputTEColl, 
           outputTrajectoryColl, algoResults, theBuilder.product(), httopo.product());
  putInEvtTrajAnn(theEvent, trajannResults, outputTrajAnnColl);

  //put in theEvent before and after DAF tracks collections
  putInEvt(theEvent, thePropagator.product(),theMeasTk.product(), 
           outputRHCollBeforeDAF, outputTCollBeforeDAF, outputTECollBeforeDAF, 
           outputTrajectoryCollBeforeDAF, algoResultsBeforeDAF, theBuilder.product(), httopo.product(), 1);
  putInEvt(theEvent, thePropagator.product(),theMeasTk.product(), 
           outputRHCollAfterDAF, outputTCollAfterDAF, outputTECollAfterDAF, 
           outputTrajectoryCollAfterDAF, algoResultsAfterDAF, theBuilder.product(), httopo.product(), 2);

  LogDebug("DAFTrackProducer") << "end the DAF algorithm." << "\n";
}
//----------------------------------------------------------------------------------------------------------//
void DAFTrackProducer::getFromEvt(edm::Event& theEvent,edm::Handle<TrajTrackAssociationCollection>& trajTrackAssociationHandle, reco::BeamSpot& bs)
{

  //get the TrajTrackMap from the event
  //WARNING: src has always to be redefined in cfg file
  theEvent.getByToken(srcTT_,trajTrackAssociationHandle);

  //get the BeamSpot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  theEvent.getByToken(bsSrc_,recoBeamSpotHandle);
  bs = *recoBeamSpotHandle;

}
//----------------------------------------------------------------------------------------------------------//
void DAFTrackProducer::putInEvtTrajAnn(edm::Event& theEvent, TrajAnnealingCollection & trajannResults,
				std::auto_ptr<TrajAnnealingCollection>& outputTrajAnnColl){
  const int size = trajannResults.size();
  outputTrajAnnColl->reserve(size);

  for(unsigned int i = 0; i < trajannResults.size() ; i++){
    outputTrajAnnColl->push_back(trajannResults[i]);
  }

  theEvent.put( outputTrajAnnColl );
}
