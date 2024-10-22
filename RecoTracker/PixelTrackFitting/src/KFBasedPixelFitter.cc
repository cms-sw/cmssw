#include "RecoTracker/PixelTrackFitting/interface/KFBasedPixelFitter.h"

#include "FWCore/Framework/interface/Event.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/PixelTrackFitting/interface/CircleFromThreePoints.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

template <class T>
inline T sqr(T t) {
  return t * t;
}

KFBasedPixelFitter::MyBeamSpotHit::MyBeamSpotHit(const reco::BeamSpot &beamSpot, const GeomDet *geom)
    : TValidTrackingRecHit(*geom) {
  localPosition_ = LocalPoint(0., 0., 0.);
  //neglect XY differences and BS slope
  localError_ = LocalError(sqr(beamSpot.BeamWidthX()), 0.0, sqr(beamSpot.sigmaZ()));
}

AlgebraicVector KFBasedPixelFitter::MyBeamSpotHit::parameters() const {
  AlgebraicVector result(1);
  result[0] = localPosition().x();
  return result;
}
AlgebraicSymMatrix KFBasedPixelFitter::MyBeamSpotHit::parametersError() const {
  LocalError le = localPositionError();
  AlgebraicSymMatrix m(1);
  m[0][0] = le.xx();
  return m;
}
AlgebraicMatrix KFBasedPixelFitter::MyBeamSpotHit::projectionMatrix() const {
  AlgebraicMatrix matrix(1, 5, 0);
  matrix[0][3] = 1;
  return matrix;
}

KFBasedPixelFitter::KFBasedPixelFitter(const Propagator *propagator,
                                       const Propagator *opropagator,
                                       const TransientTrackingRecHitBuilder *ttrhBuilder,
                                       const TrackerGeometry *tracker,
                                       const MagneticField *field,
                                       const reco::BeamSpot *beamSpot)
    : thePropagator(propagator),
      theOPropagator(opropagator),
      theTTRHBuilder(ttrhBuilder),
      theTracker(tracker),
      theField(field),
      theBeamSpot(beamSpot) {}

std::unique_ptr<reco::Track> KFBasedPixelFitter::run(const std::vector<const TrackingRecHit *> &hits,
                                                     const TrackingRegion &region) const {
  std::unique_ptr<reco::Track> ret;

  int nhits = hits.size();
  if (nhits < 2)
    return ret;

  float ptMin = region.ptMin();

  const GlobalPoint &vertexPos = region.origin();
  GlobalError vertexErr(sqr(region.originRBound()), 0, sqr(region.originRBound()), 0, 0, sqr(region.originZBound()));

  std::vector<GlobalPoint> points(nhits);
  points[0] = theTracker->idToDet(hits[0]->geographicalId())->toGlobal(hits[0]->localPosition());
  points[1] = theTracker->idToDet(hits[1]->geographicalId())->toGlobal(hits[1]->localPosition());
  points[2] = theTracker->idToDet(hits[2]->geographicalId())->toGlobal(hits[2]->localPosition());

  //
  //initial Kinematics
  //
  GlobalVector initMom;
  int charge;
  float theta;
  CircleFromThreePoints circle(points[0], points[1], points[2]);
  if (circle.curvature() > 1.e-4) {
    float invPt = PixelRecoUtilities::inversePt(circle.curvature(), *theField);
    float valPt = 1.f / invPt;
    float chargeTmp = (points[1].x() - points[0].x()) * (points[2].y() - points[1].y()) -
                      (points[1].y() - points[0].y()) * (points[2].x() - points[1].x());
    charge = (chargeTmp > 0) ? -1 : 1;
    float valPhi = (charge > 0) ? std::atan2(circle.center().x(), -circle.center().y())
                                : std::atan2(-circle.center().x(), circle.center().y());
    theta = GlobalVector(points[1] - points[0]).theta();
    initMom = GlobalVector(valPt * cos(valPhi), valPt * sin(valPhi), valPt / tan(theta));
  } else {
    initMom = GlobalVector(points[1] - points[0]);
    initMom *= 10000. / initMom.perp();
    charge = 1;
    theta = initMom.theta();
  }
  GlobalTrajectoryParameters initialKine(vertexPos, initMom, TrackCharge(charge), theField);

  //
  // initial error
  //
  AlgebraicSymMatrix55 C = ROOT::Math::SMatrixIdentity();
  float sin2th = sqr(sin(theta));
  float minC00 = 1.0;
  C[0][0] = std::max(sin2th / sqr(ptMin), minC00);
  float zErr = vertexErr.czz();
  float transverseErr = vertexErr.cxx();  // assume equal cxx cyy
  C[3][3] = transverseErr;
  C[4][4] = zErr * sin2th + transverseErr * (1 - sin2th);
  CurvilinearTrajectoryError initialError(C);

  FreeTrajectoryState fts(initialKine, initialError);

  // get updator
  KFUpdator updator;

  // Now update initial state track using information from hits.
  TrajectoryStateOnSurface outerState;
  DetId outerDetId = 0;
  const TrackingRecHit *hit = nullptr;
  for (unsigned int iHit = 0; iHit < hits.size(); iHit++) {
    hit = hits[iHit];
    if (iHit == 0)
      outerState = thePropagator->propagate(fts, theTracker->idToDet(hit->geographicalId())->surface());
    outerDetId = hit->geographicalId();
    TrajectoryStateOnSurface state = thePropagator->propagate(outerState, theTracker->idToDet(outerDetId)->surface());
    if (!state.isValid())
      return ret;
    //    TransientTrackingRecHit::RecHitPointer recHit = (theTTRHBuilder->build(hit))->clone(state);
    TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(hit);
    outerState = updator.update(state, *recHit);
    if (!outerState.isValid())
      return ret;
  }

  TrajectoryStateOnSurface innerState = outerState;
  DetId innerDetId = 0;
  innerState.rescaleError(100000.);
  for (int iHit = 2; iHit >= 0; --iHit) {
    hit = hits[iHit];
    innerDetId = hit->geographicalId();
    TrajectoryStateOnSurface state = theOPropagator->propagate(innerState, theTracker->idToDet(innerDetId)->surface());
    if (!state.isValid())
      return ret;
    //  TransientTrackingRecHit::RecHitPointer recHit = (theTTRHBuilder->build(hit))->clone(state);
    TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(hit);
    innerState = updator.update(state, *recHit);
    if (!innerState.isValid())
      return ret;
  }

  // extrapolate to vertex
  auto impactPointState = TransverseImpactPointExtrapolator(theField).extrapolate(innerState, vertexPos);
  if (!impactPointState.isValid())
    return ret;

  //
  // optionally update impact point state with Bs constraint
  // using this potion makes sense if vertexPos (from TrackingRegion is centerewd at BeamSpot).
  //
  if (theBeamSpot) {
    MyBeamSpotGeomDet bsgd(Plane::build(impactPointState.surface().position(), impactPointState.surface().rotation()));
    MyBeamSpotHit bsrh(*theBeamSpot, &bsgd);
    impactPointState = updator.update(impactPointState, bsrh);  //update
    impactPointState =
        TransverseImpactPointExtrapolator(theField).extrapolate(impactPointState, vertexPos);  //reextrapolate
    if (!impactPointState.isValid())
      return ret;
  }

  int ndof = 2 * hits.size() - 5;
  GlobalPoint vv = impactPointState.globalPosition();
  math::XYZPoint pos(vv.x(), vv.y(), vv.z());
  GlobalVector pp = impactPointState.globalMomentum();
  math::XYZVector mom(pp.x(), pp.y(), pp.z());

  float chi2 = 0.;
  ret = std::make_unique<reco::Track>(
      chi2, ndof, pos, mom, impactPointState.charge(), impactPointState.curvilinearError());

  /*
    vv = outerState.globalPosition(); 
    pp = outerState.globalMomentum();
    math::XYZPoint  outerPosition( vv.x(), vv.y(), vv.z()); 
    math::XYZVector outerMomentum( pp.x(), pp.y(), pp.z());
    vv = innerState.globalPosition(); 
    pp = innerState.globalMomentum();
    math::XYZPoint  innerPosition( vv.x(), vv.y(), vv.z()); 
    math::XYZVector innerMomentum( pp.x(), pp.y(), pp.z());

    reco::TrackExtra extra( outerPosition, outerMomentum, true,
                      innerPosition, innerMomentum, true,
                      outerState.curvilinearError(), outerDetId,
                      innerState.curvilinearError(), innerDetId,  
                      anyDirection);
*/

  //  std::cout <<"TRACK CREATED" << std::endl;
  return ret;
}
