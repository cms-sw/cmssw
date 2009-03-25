#include "TrackingTools/GsfTracking/interface/GsfConstraintAtVertex.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
// #include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
// #include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
// #include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"
#include "TrackingTools/GsfTools/interface/GsfPropagatorAdapter.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/GsfTracking/interface/GsfMultiStateUpdator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

GsfConstraintAtVertex::GsfConstraintAtVertex(const edm::EventSetup& setup) 
{
  edm::ESHandle<TrackerGeometry> geometryHandle;
  setup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
  geometry_ = geometryHandle.product();

  edm::ESHandle<MagneticField> magFieldHandle;
  setup.get<IdealMagneticFieldRecord>().get(magFieldHandle);
  magField_ = magFieldHandle.product();

//   edm::ESHandle<Propagator> propagatorHandle;
//   setup.get<TrackingComponentsRecord>().get(propagatorName_,propagatorHandle);
//   propagator_ = propagatorHandle.product();
  gsfPropagator_ = new GsfPropagatorAdapter(AnalyticalPropagator(magField_,anyDirection));
  tipExtrapolator_  = new TransverseImpactPointExtrapolator(*gsfPropagator_);
}

GsfConstraintAtVertex::~GsfConstraintAtVertex () 
{
  delete tipExtrapolator_;
  delete gsfPropagator_;
}



TrajectoryStateOnSurface
GsfConstraintAtVertex::constrainAtVertex (const reco::GsfTrack& track,
				       const reco::BeamSpot& beamSpot) const
{
  using namespace std;
  //
  // Beamspot (global co-ordinates)
  //
  GlobalPoint bsPosGlobal(beamSpot.x0(),beamSpot.y0(),beamSpot.z0());
  GlobalError bsCovGlobal(beamSpot.covariance3D());
  //
  // Track on TIP plane
  //
  TrajectoryStateOnSurface innerState = 
    multiStateTransformer_.innerStateOnSurface(track,*geometry_,magField_);
  if ( !innerState.isValid() )  return TrajectoryStateOnSurface();
  TrajectoryStateOnSurface tipState = tipExtrapolator_->extrapolate(innerState,bsPosGlobal);
  if ( !tipState.isValid() )  return TrajectoryStateOnSurface();
  //
  // RecHit from beam spot
  //
  LocalError bsCovLocal = ErrorFrameTransformer().transform(bsCovGlobal,tipState.surface());
  TransientTrackingRecHit::RecHitPointer bsHit = 
    TRecHit2DPosConstraint::build(tipState.surface().toLocal(bsPosGlobal),
				  bsCovLocal,
				  &tipState.surface());
  //
  // update with constraint
  //
  TrajectoryStateOnSurface updatedState = gsfUpdator_.update(tipState,*bsHit);
  if ( !updatedState.isValid() ) {
    edm::LogWarning("GsfConstraintAtVertex") << " GSF update with vertex constraint failed";
    return TrajectoryStateOnSurface();
  }

  return updatedState;
}

