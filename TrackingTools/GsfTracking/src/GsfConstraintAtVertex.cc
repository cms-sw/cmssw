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
GsfConstraintAtVertex::constrainAtBeamSpot (const reco::GsfTrack& track,
					    const reco::BeamSpot& beamSpot) const
{
  //
  // Beamspot (global co-ordinates)
  //
  GlobalPoint bsPosGlobal(beamSpot.x0(),beamSpot.y0(),beamSpot.z0());
  GlobalError bsCovGlobal(beamSpot.rotatedCovariance3D());
  //
  return constrainAtPoint(track,bsPosGlobal,bsCovGlobal);
}

TrajectoryStateOnSurface
GsfConstraintAtVertex::constrainAtVertex (const reco::GsfTrack& track,
					  const reco::Vertex& vertex) const
{
  //
  // Beamspot (global co-ordinates)
  //
  GlobalPoint vtxPosGlobal(vertex.position().x(),vertex.position().y(),vertex.position().z());
  GlobalError vtxCovGlobal(vertex.covariance());
  //
  return constrainAtPoint(track,vtxPosGlobal,vtxCovGlobal);
}

TrajectoryStateOnSurface
GsfConstraintAtVertex::constrainAtPoint (const reco::GsfTrack& track,
					 const GlobalPoint& globalPosition,
					 const GlobalError& globalError) const
{
  //
  // Track on TIP plane
  //
  TrajectoryStateOnSurface innerState = 
    multiStateTransformer_.innerStateOnSurface(track,*geometry_,magField_);
  if ( !innerState.isValid() )  return TrajectoryStateOnSurface();
  TrajectoryStateOnSurface tipState = tipExtrapolator_->extrapolate(innerState,globalPosition);
  if ( !tipState.isValid() )  return TrajectoryStateOnSurface();
  //
  // RecHit from beam spot
  //
  LocalError bsCovLocal = ErrorFrameTransformer().transform(globalError,tipState.surface());
  TransientTrackingRecHit::RecHitPointer bsHit = 
    TRecHit2DPosConstraint::build(tipState.surface().toLocal(globalPosition),
				  bsCovLocal,&tipState.surface());
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

