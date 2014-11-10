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
  src_ = consumes<TrajectoryCollection>(iConfig.getParameter<edm::InputTag>( "src" ));
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
  std::auto_ptr<reco::TrackCollection>       outputTCollbeforeDAF(new reco::TrackCollection);
  std::auto_ptr<reco::TrackCollection>       outputTCollafterDAF(new reco::TrackCollection);
//  std::auto_ptr<reco::TrackExtraCollection>  outputTECollbeforeDAF(new reco::TrackExtraCollection);

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

  //declare and get  new tracks collections
  AlgoProductCollection algoResultsBeforeDAF;
  AlgoProductCollection algoResultsAfterDAF;
  try{

    edm::Handle<std::vector<Trajectory> > theTrajectoryCollection;
    getFromEvt(theEvent,theTrajectoryCollection,bs);

    //run the algorithm  
    LogDebug("DAFTrackProducer") << "run the DAF algorithm" << "\n";
    theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTrajectoryCollection, &*mte,
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
           outputTrajectoryColl, algoResults, theBuilder.product());
  putInEvtTrajAnn(theEvent, trajannResults, outputTrajAnnColl);
  putInEvtTrackDAF(theEvent, outputTCollbeforeDAF, algoResultsBeforeDAF, true);
  putInEvtTrackDAF(theEvent, outputTCollafterDAF, algoResultsAfterDAF, false);


  LogDebug("DAFTrackProducer") << "end the DAF algorithm." << "\n";
}
//----------------------------------------------------------------------------------------------------------//
void DAFTrackProducer::getFromEvt(edm::Event& theEvent,edm::Handle<TrajectoryCollection>& theTrajectoryCollection, reco::BeamSpot& bs)
{

  //get the TrajectoryCollection from the event
  //WARNING: src has always to be redefined in cfg file
  theEvent.getByToken(src_,theTrajectoryCollection );

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
//----------------------------------------------------------------------------------------------------------//
void DAFTrackProducer::putInEvtTrackDAF(edm::Event& theEvent,
					std::auto_ptr<reco::TrackCollection>& selTracks,
//                                   	std::auto_ptr<reco::TrackExtraCollection>& selTrackExtras,
					AlgoProductCollection& algoResults,
					bool before	){
  selTracks->reserve(algoResults.size());
//  selTrackExtras->reserve(algoResults.size());
  for(AlgoProductCollection::iterator i=algoResults.begin(); i!=algoResults.end();i++){

    //put the Track
    reco::Track * theTrack = (*i).second.first;
    selTracks->push_back(std::move(*theTrack));
    delete theTrack;

/*    //build the TrackExtra
    GlobalPoint v = outertsos.globalParameters().position();
    GlobalVector p = outertsos.globalParameters().momentum();
    math::XYZVector outmom( p.x(), p.y(), p.z() );
    math::XYZPoint  outpos( v.x(), v.y(), v.z() );
    v = innertsos.globalParameters().position();
    p = innertsos.globalParameters().momentum();
    math::XYZVector inmom( p.x(), p.y(), p.z() );
    math::XYZPoint  inpos( v.x(), v.y(), v.z() );
    
    reco::TrackExtraRef teref= reco::TrackExtraRef ( rTrackExtras, idx ++ );
    reco::Track & track = selTracks->back();
    track.setExtra( teref );

    selTrackExtras->push_back( reco::TrackExtra (outpos, outmom, true, inpos, inmom, true,
                                                 outertsos.curvilinearError(), outerId,
                                                 innertsos.curvilinearError(), innerId,
                                                 seedDir, theTraj->seedRef()));
*/

  }

  //ERICA :: why?
  selTracks->shrink_to_fit();
  //selTrackExtras->shrink_to_fit();

  if( before == true ) { theEvent.put( selTracks , "beforeDAF"); }
  else { theEvent.put( selTracks , "afterDAF"); }
 // theEvent.put( selTrackExtras );

}
