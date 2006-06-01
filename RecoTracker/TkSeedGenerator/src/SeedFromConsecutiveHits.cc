#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
//#include "CommonReco/Propagators/interface/GtfPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
//#include "RecoTracker/TkSeedGenerator/interface/TrivialVertex.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "FWCore/Framework/interface/ESHandle.h"

using namespace std;
SeedFromConsecutiveHits::
SeedFromConsecutiveHits( const TrackingRecHit* outerHit, 
			 const TrackingRecHit* innerHit, 
			 const GlobalPoint& vertexPos,
			 const GlobalError& vertexErr,
			 const edm::EventSetup& iSetup,
			 const edm::ParameterSet& p
			 ) :p_(p)
{
  construct( outerHit,  innerHit, vertexPos, vertexErr,iSetup,p_);
}


void SeedFromConsecutiveHits::
construct( const TrackingRecHit* outerHit, 
	   const TrackingRecHit* innerHit, 
	   const GlobalPoint& vertexPos,
	   const GlobalError& vertexErr,
	   const edm::EventSetup& iSetup,
	   const edm::ParameterSet& p

) 
{
  typedef TrajectoryStateOnSurface     TSOS;
  typedef TrajectoryMeasurement        TM;

  // make a spiral
  //MP
  //  TrivialVertex vtx( vertexPos, vertexErr);
  //  FastHelix helix(outerHit.globalPosition(), innerHit.globalPosition(), 
  //		  vtx.position());
  //   FastHelix helix(outerHit.globalPosition(), innerHit.globalPosition(), 
  // 		  GlobalPoint(0.,0.,0.));

  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

  GlobalPoint inner = tracker->idToDet(innerHit->geographicalId())->surface().toGlobal(innerHit->localPosition());
  GlobalPoint outer = tracker->idToDet(outerHit->geographicalId())->surface().toGlobal(outerHit->localPosition());
  //  DetId outerDet(outerHit.geographicalId());
  FastHelix helix(outer, inner, 
		  GlobalPoint(0.,0.,0.),iSetup);

//   if ( helix.isValid()) {
    FreeTrajectoryState fts( helix.stateAtVertex().parameters(),
			     initialError( outerHit, innerHit, 
					   vertexPos, vertexErr));



//MP
//    GtfPropagator thePropagator( alongMomentum);
//    const MagneticField  *ppl;
    edm::ESHandle<MagneticField> pSetup;
    iSetup.get<IdealMagneticFieldRecord>().get(pSetup);
    AnalyticalPropagator  thePropagator(&(*pSetup), alongMomentum);


    KFUpdator     theUpdator;

    //   TSOS innerState = thePropagator.propagate(fts,tracker->idToDet(innerHit.geographicalId())->surface());
    const TSOS innerState = thePropagator.propagate(fts,tracker->idToDet(innerHit->geographicalId())->surface());
    if ( !innerState.isValid()) 
      edm::LogError("Propagation") << " SeedFromConsecutiveHits first propagation failed ";

				    

    //
    // get from the eventsetup
    //

    

    //    TkTransientTrackingRecHitBuilder TTTRHBuilder((tracker.product()));

    //   intrhit=TTRHBuilder.build(&(*tracker),innerHit->clone());

  //
  // get the transient builder
  //
    edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
    std::string builderName = p_.getParameter<std::string>("TTRHBuilder");   
    iSetup.get<TrackingComponentsRecord>().get(builderName,theBuilder);

    intrhit=theBuilder.product()->build(innerHit);

    const TSOS innerUpdated= theUpdator.update( innerState,*intrhit);			      

    TSOS outerState = 
      thePropagator.propagate( innerUpdated,
			       tracker->idToDet(outerHit->geographicalId())->surface());
 
    if ( !outerState.isValid()) 
     edm::LogError("Propagation") << " SeedFromConsecutiveHits first propagation failed ";
  
    outrhit=theBuilder.product()->build(outerHit);

    TSOS outerUpdated = theUpdator.update( outerState, *outrhit);
 

    //MP
    //what is the estimate value?
    theInnerMeas = TM( innerState, innerUpdated, intrhit, 0);
    theOuterMeas = TM( outerState, outerUpdated, outrhit, 0);
 


    _hits.push_back(innerHit->clone());
    _hits.push_back(outerHit->clone());
     PTraj=  transformer.persistentState(outerUpdated, outerHit->geographicalId().rawId());
}



CurvilinearTrajectoryError SeedFromConsecutiveHits::
initialError( const TrackingRecHit* outerHit,
	      const TrackingRecHit* innerHit,
	      const GlobalPoint& vertexPos,
	      const GlobalError& vertexErr) 
{
  AlgebraicSymMatrix C(5,1);

  float zErr = vertexErr.czz();
  float transverseErr = vertexErr.cxx(); // assume equal cxx cyy 
  C[3][3] = transverseErr;
  C[4][4] = zErr;

  return CurvilinearTrajectoryError(C);
}

