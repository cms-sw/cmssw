#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
//#include "CommonReco/Propagators/interface/GtfPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
//#include "RecoTracker/TkSeedGenerator/interface/TrivialVertex.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

SeedFromConsecutiveHits::
SeedFromConsecutiveHits( const TrackingRecHit& outerHit, 
			 const TrackingRecHit& innerHit, 
			 const GlobalPoint& vertexPos,
			 const GlobalError& vertexErr,
			 const edm::EventSetup& iSetup) 
{
  construct( outerHit,  innerHit, vertexPos, vertexErr,iSetup);
}


void SeedFromConsecutiveHits::
construct( const TrackingRecHit& outerHit, 
	   const TrackingRecHit& innerHit, 
	   const GlobalPoint& vertexPos,
	   const GlobalError& vertexErr,
	   const edm::EventSetup& iSetup) 
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
  edm::ESHandle<TrackingGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

  GlobalPoint inner = tracker->idToDet(innerHit.geographicalId())->surface().toGlobal(innerHit.localPosition());
  GlobalPoint outer = tracker->idToDet(outerHit.geographicalId())->surface().toGlobal(outerHit.localPosition());
  //  DetId outerDet(outerHit.geographicalId());
  FastHelix helix(outer, inner, 
		  GlobalPoint(0.,0.,0.),iSetup);

//   if ( helix.isValid()) {
    FreeTrajectoryState fts( helix.stateAtVertex().parameters(),
			     initialError( outerHit, innerHit, 
					   vertexPos, vertexErr));

//     cout << "SeedFromConsecutiveHits: Initial fts:" << fts << endl;

//    Helix h(outerHit.det().toGlobal(outerHit.localPosition()),
//  	  innerHit.det().toGlobal(innerHit.localPosition()),
//  	  vertexPos);
//    if (h.isValid()) {
//      FreeTrajectoryState fts = 
//        h.atVertex( sqrt( vertexErr.cxx()), 
//  		  sqrt( vertexErr.cyy()), 
//  		  sqrt( vertexErr.czz()), 10.);

//MP
//    GtfPropagator thePropagator( alongMomentum);
//    const MagneticField  *ppl;
    edm::ESHandle<MagneticField> pSetup;
    iSetup.get<IdealMagneticFieldRecord>().get(pSetup);
    AnalyticalPropagator  thePropagator(&(*pSetup), alongMomentum);


    KFUpdator     theUpdator;

    //   TSOS innerState = thePropagator.propagate(fts,tracker->idToDet(innerHit.geographicalId())->surface());
    const TSOS innerState = thePropagator.propagate(fts,tracker->idToDet(innerHit.geographicalId())->surface());
    if ( !innerState.isValid()) 
      edm::LogError("Propagation") << " SeedFromConsecutiveHits first propagation failed ";

				    



    intrhit=TTRHBuilder.build(&(*tracker),innerHit.clone());
    const TSOS innerUpdated= theUpdator.update( innerState,*intrhit);			      

    TSOS outerState = 
      thePropagator.propagate( innerUpdated,
			       tracker->idToDet(outerHit.geographicalId())->surface());

    if ( !outerState.isValid()) 
     edm::LogError("Propagation") << " SeedFromConsecutiveHits first propagation failed ";
  

    outrhit=TTRHBuilder.build(&(*tracker),outerHit.clone());

    TSOS outerUpdated = theUpdator.update( outerState, *outrhit);

    //       cout << "SeedFromConsecutiveHits: after second update :"
    //   	 << outerUpdated << endl;
    //MP
    //what is the estimate value?
    theInnerMeas = TM( innerState, innerUpdated, intrhit, 0);
    theOuterMeas = TM( outerState, outerUpdated, outrhit, 0);

}


// const FreeTrajectoryState& 
// SeedFromConsecutiveHits::freeTrajectoryState() const {
 
//   return *theOuterMeas.updatedState().freeState();
// }

// PropagationDirection SeedFromConsecutiveHits::direction() const {
//   return alongMomentum;
// }

// vector<SiPixelRecHit> SeedFromConsecutiveHits::recHits() const {
//   vector<SiPixelRecHit> result; result.reserve(2);
//   result.push_back( theInnerMeas.recHit());
//   result.push_back( theOuterMeas.recHit());
//   return result;
// }

// vector<TrajectoryMeasurement> SeedFromConsecutiveHits::measurements() const
// {
//   vector<TrajectoryMeasurement> result; result.reserve(2);
//   result.push_back( theInnerMeas);
//   result.push_back( theOuterMeas);
//   return result;
// }

// bool SeedFromConsecutiveHits::share( const BasicTrajectorySeed&) const {
//   return false;
// }

// BasicTrajectorySeed* SeedFromConsecutiveHits::clone() const { 
//   return new SeedFromConsecutiveHits(*this);
// }

CurvilinearTrajectoryError SeedFromConsecutiveHits::
initialError( const TrackingRecHit& outerHit,
	      const TrackingRecHit& innerHit,
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

