#include "RecoTracker/TrackProducer/interface/GsfTrackProducerBase.h"

/// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"


//destructor
GsfTrackProducerBase::~GsfTrackProducerBase(){ }

// member functions
// ------------ method called to produce the data  ------------

void GsfTrackProducerBase::getFromES(const edm::EventSetup& setup,
				  edm::ESHandle<TrackerGeometry>& theG,
				  edm::ESHandle<MagneticField>& theMF,
				  edm::ESHandle<TrajectoryFitter>& theFitter,
				  edm::ESHandle<Propagator>& thePropagator,
				  edm::ESHandle<TransientTrackingRecHitBuilder>& theBuilder)
{
  //
  //get geometry
  //
  LogDebug("GsfTrackProducer") << "get geometry" << "\n";
  setup.get<TrackerDigiGeometryRecord>().get(theG);
  //
  //get magnetic field
  //
  LogDebug("GsfTrackProducer") << "get magnetic field" << "\n";
  setup.get<IdealMagneticFieldRecord>().get(theMF);  
  //
  // get the fitter from the ES
  //
  LogDebug("GsfTrackProducer") << "get the fitter from the ES" << "\n";
  std::string fitterName = conf_.getParameter<std::string>("Fitter");   
  setup.get<TrackingComponentsRecord>().get(fitterName,theFitter);
  //
  // get also the propagator
  //
  LogDebug("GsfTrackProducer") << "get also the propagator" << "\n";
  std::string propagatorName = conf_.getParameter<std::string>("Propagator");   
  setup.get<TrackingComponentsRecord>().get(propagatorName,thePropagator);
  //
  // get the builder
  //
  LogDebug("GsfTrackProducer") << "get also the TransientTrackingRecHitBuilder" << "\n";
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  setup.get<TransientRecHitRecord>().get(builderName,theBuilder);

  

}

void GsfTrackProducerBase::getFromEvt(edm::Event& theEvent,edm::Handle<TrackCandidateCollection>& theTCCollection)
{
  //
  //get the TrackCandidateCollection from the event
  //
  LogDebug("GsfTrackProducer") << 
    "get the TrackCandidateCollection from the event, source is " << src_<<"\n";
  theEvent.getByLabel(src_,theTCCollection );
}

void GsfTrackProducerBase::getFromEvt(edm::Event& theEvent,edm::Handle<reco::TrackCollection>& theTCollection)
{
  //
  //get the TrackCollection from the event
  //
  LogDebug("GsfTrackProducer") << 
    "get the TrackCollection from the event, source is " << src_<<"\n";
  theEvent.getByLabel(src_,theTCollection );
}

void GsfTrackProducerBase::putInEvt(edm::Event& evt,
				 std::auto_ptr<TrackingRecHitCollection>& selHits,
				 std::auto_ptr<reco::GsfTrackCollection>& selTracks,
				 std::auto_ptr<reco::GsfTrackExtraCollection>& selTrackExtras,
				 std::auto_ptr<std::vector<Trajectory> >&   selTrajectories,
				 AlgoProductCollection& algoResults)
{

  TrackingRecHitRefProd rHits = evt.getRefBeforePut<TrackingRecHitCollection>();
  reco::GsfTrackExtraRefProd rTrackExtras = evt.getRefBeforePut<reco::GsfTrackExtraCollection>();
  reco::GsfTrackRefProd rTracks = evt.getRefBeforePut<reco::GsfTrackCollection>();

  edm::Ref<reco::GsfTrackExtraCollection>::key_type idx = 0;
  edm::Ref<reco::GsfTrackExtraCollection>::key_type hidx = 0;
  for(AlgoProductCollection::iterator i=algoResults.begin(); i!=algoResults.end();i++){
    Trajectory * theTraj = (*i).first;
    if(trajectoryInEvent_) selTrajectories->push_back(*theTraj);
    const TrajectoryFitter::RecHitContainer& transHits = theTraj->recHits();

    reco::GsfTrack * theTrack = (*i).second;
    
    //     if( ) {
    reco::GsfTrack t = * theTrack;
    selTracks->push_back( t );
    
    //sets the outermost and innermost TSOSs
    TrajectoryStateOnSurface outertsos;
    TrajectoryStateOnSurface innertsos;
    unsigned int innerId, outerId;
    if (theTraj->direction() == alongMomentum) {
      outertsos = theTraj->lastMeasurement().updatedState();
      innertsos = theTraj->firstMeasurement().updatedState();
      outerId = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
      innerId = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
    } else { 
      outertsos = theTraj->firstMeasurement().updatedState();
      innertsos = theTraj->lastMeasurement().updatedState();
      outerId = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
      innerId = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
   }
    //build the GsfTrackExtra
    GlobalPoint v = outertsos.globalParameters().position();
    GlobalVector p = outertsos.globalParameters().momentum();
    math::XYZVector outmom( p.x(), p.y(), p.z() );
    math::XYZPoint  outpos( v.x(), v.y(), v.z() );
    std::vector<reco::GsfComponent5D> outerStates;
    outerStates.reserve(outertsos.components().size());
    fillStates(outertsos,outerStates);

    v = innertsos.globalParameters().position();
    p = innertsos.globalParameters().momentum();
    math::XYZVector inmom( p.x(), p.y(), p.z() );
    math::XYZPoint  inpos( v.x(), v.y(), v.z() );
    std::vector<reco::GsfComponent5D> innerStates;
    innerStates.reserve(innertsos.components().size());
    fillStates(innertsos,innerStates);

    reco::GsfTrackExtraRef teref= reco::GsfTrackExtraRef ( rTrackExtras, idx ++ );
    reco::GsfTrack & track = selTracks->back();
    track.setExtra( teref );
    selTrackExtras->push_back( reco::GsfTrackExtra (outpos, outmom, outertsos.curvilinearError(), 
						    outerStates, outertsos.localParameters().pzSign(),
						    outerId, true,
						    inpos, inmom, innertsos.curvilinearError(), 
						    innerStates, innertsos.localParameters().pzSign(),
						    innerId, true));

    reco::GsfTrackExtra & tx = selTrackExtras->back();
    size_t i = 0;
    for( TrajectoryFitter::RecHitContainer::const_iterator j = transHits.begin();
	 j != transHits.end(); j ++ ) {
      TrackingRecHit * hit = (**j).hit()->clone();
      selHits->push_back( hit );
      track.setHitPattern( * hit, i ++ );
      tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
    }
    delete theTrack;
    delete theTraj;
  }
  
  evt.put( selTracks );
  evt.put( selTrackExtras );
  evt.put( selHits );
  if(trajectoryInEvent_) evt.put(selTrajectories);
}

void
GsfTrackProducerBase::fillStates (TrajectoryStateOnSurface tsos,
				  std::vector<reco::GsfComponent5D>& states) const
{
//   std::cout << "in fill states" << std::endl;
//   if ( !tsos.isValid() ) {
//     std::cout << std::endl << std::endl << "invalid tsos" << std::endl;
//     return;
//   }
  reco::GsfComponent5D::ParameterVector pLocS;
  reco::GsfComponent5D::CovarianceMatrix cLocS;
  std::vector<TrajectoryStateOnSurface> components(tsos.components());
  for ( std::vector<TrajectoryStateOnSurface>::const_iterator i=components.begin();
	  i!=components.end(); ++i ) {
//     if ( !(*i).isValid() ) {
//       std::cout << std::endl << "invalid component" << std::endl;
//       continue;
//     }
    const AlgebraicVector& pLoc = i->localParameters().vector();
    for ( int j=0; j<reco::GsfTrackExtra::dimension; ++j )  pLocS(j) = pLoc[j];
    const AlgebraicSymMatrix& cLoc = i->localError().matrix();
    for ( int j1=0; j1<reco::GsfTrack::dimension; ++j1 )
      for ( int j2=0; j2<=j1; ++j2 )  cLocS(j1,j2) = cLoc[j1][j2];
    states.push_back(reco::GsfComponent5D(i->weight(),pLocS,cLocS));
  }
//   std::cout << "end fill states" << std::endl;
}
