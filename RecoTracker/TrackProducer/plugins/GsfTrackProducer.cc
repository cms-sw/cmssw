#include "RecoTracker/TrackProducer/plugins/GsfTrackProducer.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TrackingTools/GsfTracking/interface/TrajGsfTrackAssociation.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfComponent5D.h"

GsfTrackProducer::GsfTrackProducer(const edm::ParameterSet& iConfig):
  GsfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),
		       iConfig.getParameter<bool>("useHitsSplitting")),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( consumes<TrackCandidateCollection>(iConfig.getParameter<edm::InputTag>( "src" )), 
          consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>( "beamSpot" )),
          consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>( "MeasurementTrackerEvent") ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );
//   string a = alias_;
//   a.erase(a.size()-6,a.size());
  //register your products
  produces<reco::GsfTrackCollection>().setBranchAlias( alias_ + "GsfTracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
  produces<reco::GsfTrackExtraCollection>().setBranchAlias( alias_ + "GsfTrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
  produces<std::vector<Trajectory> >() ;
  produces<TrajGsfTrackAssociationCollection>();

}


void GsfTrackProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  edm::LogInfo("GsfTrackProducer") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::auto_ptr<TrackingRecHitCollection> outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::GsfTrackCollection> outputTColl(new reco::GsfTrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);
  std::auto_ptr<reco::GsfTrackExtraCollection> outputGsfTEColl(new reco::GsfTrackExtraCollection);
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

  //
  //declare and get TrackColection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;
  try{  
    edm::Handle<TrackCandidateCollection> theTCCollection;
    getFromEvt(theEvent,theTCCollection,bs);
    
    //
    //run the algorithm  
    //
    LogDebug("GsfTrackProducer") << "run the algorithm" << "\n";
    theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTCCollection, 
			     theFitter.product(), thePropagator.product(), theBuilder.product(), bs, algoResults);
  } catch (cms::Exception &e){ edm::LogInfo("GsfTrackProducer") << "cms::Exception caught!!!" << "\n" << e << "\n"; throw; }
  //
  //put everything in the event
  putInEvt(theEvent, thePropagator.product(), theMeasTk.product(), outputRHColl, outputTColl, outputTEColl, outputGsfTEColl,
	   outputTrajectoryColl, algoResults, theBuilder.product(), bs);
  LogDebug("GsfTrackProducer") << "end" << "\n";
}

void GsfTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("CkfElectronCandidates"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("producer", std::string(""));
  desc.add<std::string>("Fitter", std::string("GsfElectronFittingSmoother"));
  desc.add<bool>("useHitsSplitting", false);
  desc.add<bool>("TrajectoryInEvent", false);
  desc.add<std::string>("TTRHBuilder", std::string("WithTrackAngle"));
  desc.add<std::string>("Propagator", std::string("fwdElectronPropagator"));
  desc.add<std::string>("NavigationSchool", std::string("SimpleNavigationSchool"));
  desc.add<std::string>("MeasurementTracker", std::string(""));                   
  desc.add<edm::InputTag>("MeasurementTrackerEvent", edm::InputTag("MeasurementTrackerEvent")); 
  desc.add<bool>("GeometricInnerState", false);
  desc.add<std::string>("AlgorithmName", std::string("gsf"));

  descriptions.add("gsfTrackProducer", desc);
}
