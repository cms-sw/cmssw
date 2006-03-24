#include "RecoTracker/TrackProducer/interface/TrackProducer.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

// constructors and destructor
TrackProducer::TrackProducer(const edm::ParameterSet& iConfig):
  theAlgo(iConfig), conf_(iConfig), src_( iConfig.getParameter<std::string>( "src" ))
{
  //register your products
  produces<TrackingRecHitCollection>();
  produces<reco::TrackCollection>();
  produces<reco::TrackExtraCollection>();
}

TrackProducer::~TrackProducer(){ }

// member functions
// ------------ method called to produce the data  ------------
void TrackProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  // create empty output collections
  std::auto_ptr<TrackingRecHitCollection> outputRHColl;
  std::auto_ptr<reco::TrackCollection> outputTColl;
  std::auto_ptr<reco::TrackExtraCollection> outputTEColl;
  //get geometry
  edm::ESHandle<TrackerGeometry> theG;
  setup.get<TrackerDigiGeometryRecord>().get(theG);
  //get magnetic field
  edm::ESHandle<MagneticField> theMF;
  setup.get<IdealMagneticFieldRecord>().get(theMF);  
  //
  // get the fitter from the ES
  //
  std::string fitterName = conf_.getParameter<std::string>("Fitter");   
  edm::ESHandle<TrajectoryFitter> theFitter;
  setup.get<TrackingComponentsRecord>().get(fitterName,theFitter);
  //
  // get also the propagator
  //
  std::string propagatorName = conf_.getParameter<std::string>("Propagator");   
  edm::ESHandle<Propagator> thePropagator;
  setup.get<TrackingComponentsRecord>().get(propagatorName,thePropagator);

  edm::Handle<TrackCandidateCollection> theTCCollection;
  //   theEvent.getByLabel( "minchipink", theTCCollection );
  theEvent.getByLabel( src_, theTCCollection );

  //run the algorithm  
  AlgoProductCollection algoResults;
  theAlgo.run(theG.product(), theMF.product(), *theTCCollection, theFitter.product(), 
	      thePropagator.product(), algoResults);

  

  //
  //first loop: create the full collection of TrackingRecHit
  //
  for(AlgoProductCollection::iterator i=algoResults.begin();
      i!=algoResults.end();i++){
  
    Trajectory * theTraj = (*i).first;
    
    const edm::OwnVector<TransientTrackingRecHit>& transHits = theTraj->recHits();
    for(edm::OwnVector<TransientTrackingRecHit>::const_iterator j=transHits.begin();
	j!=transHits.end(); j++){
      outputRHColl->push_back( ( j->hit() ) );
    }
    
  }
  //put the collection of TrackingRecHit in the event
  edm::OrphanHandle <TrackingRecHitCollection> ohRH = theEvent.put( outputRHColl );


  //
  //second loop: create the collection of TrackExtra
  //
  int cc = 0;	
  for(AlgoProductCollection::iterator i=algoResults.begin();
      i!=algoResults.end();i++){

    Trajectory * theTraj = (*i).first;
   
    reco::TrackExtra * theTrackExtra;
    //sets the outermost and innermost TSOSs
    TrajectoryStateOnSurface outertsos;
    TrajectoryStateOnSurface innertsos;
    if (theTraj->direction() == alongMomentum) {
      outertsos = theTraj->lastMeasurement().updatedState();
      innertsos = theTraj->firstMeasurement().updatedState();
    } else { 
      outertsos = theTraj->firstMeasurement().updatedState();
      innertsos = theTraj->lastMeasurement().updatedState();
    }
    //build the TrackExtra
    GlobalPoint v = outertsos.globalParameters().position();
    GlobalVector p = outertsos.globalParameters().momentum();
    math::XYZVector outmom( p.x(), p.y(), p.z() );
    math::XYZPoint  outpos( v.x(), v.y(), v.z() );   
    theTrackExtra = new reco::TrackExtra(outpos, outmom, true);
    
    
    //fill the TrackExtra with TrackingRecHitRef	
    const edm::OwnVector<TransientTrackingRecHit>& transHits = theTraj->recHits();
    for(edm::OwnVector<TransientTrackingRecHit>::const_iterator j=transHits.begin();
	j!=transHits.end(); j++){
      theTrackExtra->add(TrackingRecHitRef(ohRH,cc));
      cc++;
    }
    
    //fill the TrackExtraCollection
    outputTEColl->push_back(*theTrackExtra);
  }
  //put the collection of TrackExtra in the event
  edm::OrphanHandle<reco::TrackExtraCollection> ohTE = theEvent.put(outputTEColl);


  //
  //third loop: create the collection of Tracks
  //
  cc = 0;
  for(AlgoProductCollection::iterator i=algoResults.begin();
      i!=algoResults.end();i++){

    reco::Track * theTrack = (*i).second;
    
    //create a TrackExtraRef
    reco::TrackExtraRef  theTrackExtraRef(ohTE,cc);
    
    //use the TrackExtraRef to assign the TrackExtra to the Track
    theTrack->setExtra(theTrackExtraRef);
    
    //fill the TrackCollection
    outputTColl->push_back(*theTrack);
    
    cc++;
  }
  //put the TrackCollection in the event
  theEvent.put(outputTColl);

}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackProducer)
