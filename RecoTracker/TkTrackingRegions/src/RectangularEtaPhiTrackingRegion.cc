#include <cmath>
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "OuterEstimator.h"

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRCheck.h"
#include "RecoTracker/TkTrackingRegions/interface/HitZCheck.h"
#include "RecoTracker/TkTrackingRegions/interface/HitEtaCheck.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/KalmanUpdators/interface/EtaPhiMeasurementEstimator.h"

#include <iostream>
#include <algorithm>
#include <cctype>

namespace {
  template <class T>
  T sqr(T t) {
    return t * t;
  }
}  // namespace

using namespace PixelRecoUtilities;
using namespace std;

RectangularEtaPhiTrackingRegion::UseMeasurementTracker RectangularEtaPhiTrackingRegion::stringToUseMeasurementTracker(
    const std::string& name) {
  std::string tmp = name;
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
  if (tmp == "never")
    return UseMeasurementTracker::kNever;
  if (tmp == "forsistrips")
    return UseMeasurementTracker::kForSiStrips;
  if (tmp == "always")
    return UseMeasurementTracker::kAlways;
  throw cms::Exception("Configuration") << "Got invalid string '" << name
                                        << "', valid values are 'Never', 'ForSiStrips', 'Always' (case insensitive)";
}

void RectangularEtaPhiTrackingRegion::initEtaRange(const GlobalVector& dir, const Margin& margin) {
  float eta = dir.eta();
  theEtaRange = Range(eta - margin.left(), eta + margin.right());
  theLambdaRange = Range(std::sinh(theEtaRange.min()), std::sinh(theEtaRange.max()));
  theMeanLambda = std::sinh(theEtaRange.mean());
}

std::unique_ptr<HitRZCompatibility> RectangularEtaPhiTrackingRegion::checkRZOld(const DetLayer* layer,
                                                                                const Hit& outerHit,
                                                                                const edm::EventSetup& iSetup,
                                                                                const DetLayer* outerlayer) const {
  bool isBarrel = (layer->location() == GeomDetEnumerators::barrel);
  GlobalPoint ohit = outerHit->globalPosition();
  float outerred_r = std::sqrt(sqr(ohit.x() - origin().x()) + sqr(ohit.y() - origin().y()));
  PixelRecoPointRZ outer(outerred_r, ohit.z());

  float zMinOrigin = origin().z() - originZBound();
  float zMaxOrigin = origin().z() + originZBound();

  if (!thePrecise) {
    float vcotMin = (outer.z() > zMaxOrigin) ? (outer.z() - zMaxOrigin) / (outer.r() + originRBound())
                                             : (outer.z() - zMaxOrigin) / (outer.r() - originRBound());
    float vcotMax = (outer.z() > zMinOrigin) ? (outer.z() - zMinOrigin) / (outer.r() - originRBound())
                                             : (outer.z() - zMinOrigin) / (outer.r() + originRBound());
    float cotRight = std::max(vcotMin, theLambdaRange.min());
    float cotLeft = std::min(vcotMax, theLambdaRange.max());
    return std::make_unique<HitEtaCheck>(isBarrel, outer, cotLeft, cotRight);
  }

  float outerZscatt = 0;
  float innerScatt = 0;
  //CHECK
  if (theUseMS) {
    MultipleScatteringParametrisation oSigma(layer, iSetup);
    float cotThetaOuter = theMeanLambda;
    float sinThetaOuterInv = std::sqrt(1.f + sqr(cotThetaOuter));
    outerZscatt = 3.f * oSigma(ptMin(), cotThetaOuter) * sinThetaOuterInv;
  }

  PixelRecoLineRZ boundL(outer, theLambdaRange.max());
  PixelRecoLineRZ boundR(outer, theLambdaRange.min());
  float zMinLine = boundL.zAtR(0.) - outerZscatt;
  float zMaxLine = boundR.zAtR(0.) + outerZscatt;
  PixelRecoPointRZ vtxL(0., max(zMinLine, zMinOrigin));
  PixelRecoPointRZ vtxR(0., min(zMaxLine, zMaxOrigin));
  PixelRecoPointRZ vtxMean(0., (vtxL.z() + vtxR.z()) * 0.5f);
  //CHECK

  if (theUseMS) {
    MultipleScatteringParametrisation iSigma(layer, iSetup);

    innerScatt =
        3.f * (outerlayer ? iSigma(ptMin(), vtxMean, outer, outerlayer->seqNum()) : iSigma(ptMin(), vtxMean, outer));

    // innerScatt = 3.f *iSigma( ptMin(), vtxMean, outer);
  }

  SimpleLineRZ leftLine(vtxL, outer);
  SimpleLineRZ rightLine(vtxR, outer);

  HitRZConstraint rzConstraint(leftLine, rightLine);
  auto cotTheta = std::abs(leftLine.cotLine() + rightLine.cotLine()) * 0.5f;

  // std::cout << "RectangularEtaPhiTrackingRegion " << outer.r()<<','<< outer.z() << " " << innerScatt << " " << cotTheta << " " <<  hitZErr <<  std::endl;

  if (isBarrel) {
    auto sinThetaInv = std::sqrt(1.f + sqr(cotTheta));
    auto corr = innerScatt * sinThetaInv;
    return std::make_unique<HitZCheck>(rzConstraint, HitZCheck::Margin(corr, corr));
  } else {
    auto cosThetaInv = std::sqrt(1.f + sqr(1.f / cotTheta));
    auto corr = innerScatt * cosThetaInv;
    return std::make_unique<HitRCheck>(rzConstraint, HitRCheck::Margin(corr, corr));
  }
}

std::unique_ptr<MeasurementEstimator> RectangularEtaPhiTrackingRegion::estimator(const BarrelDetLayer* layer,
                                                                                 const edm::EventSetup& iSetup) const {
  using Algo = HitZCheck;

  // det dimensions
  float halfLength = 0.5f * layer->surface().bounds().length();
  float halfThickness = 0.5f * layer->surface().bounds().thickness();
  float z0 = layer->position().z();
  float radius = layer->specificSurface().radius();

  // det ranges
  Range detRWindow(radius - halfThickness, radius + halfThickness);
  Range detZWindow(z0 - halfLength, z0 + halfLength);

  // z prediction, skip if not intersection
  HitZCheck zPrediction(rzConstraint());
  Range hitZWindow = zPrediction.range(detRWindow.min()).intersection(detZWindow);
  if (hitZWindow.empty())
    return nullptr;

  // phi prediction
  OuterHitPhiPrediction phiPrediction = phiWindow(iSetup);

  //
  // optional corrections for tolerance (mult.scatt, error, bending)
  //
  OuterHitPhiPrediction::Range phiRange;
  if (thePrecise) {
    auto invR = 1.f / radius;
    auto cotTheta = (hitZWindow.mean() - origin().z()) * invR;
    auto sinThetaInv = std::sqrt(1.f + sqr(cotTheta));
    MultipleScatteringParametrisation msSigma(layer, iSetup);
    auto scatt = 3.f * msSigma(ptMin(), cotTheta);
    auto bendR = longitudinalBendingCorrection(radius, ptMin(), iSetup);

    float hitErrRPhi = 0.;
    float hitErrZ = 0.;
    float corrPhi = (scatt + hitErrRPhi) * invR;
    float corrZ = scatt * sinThetaInv + bendR * std::abs(cotTheta) + hitErrZ;

    phiPrediction.setTolerance(corrPhi);
    zPrediction.setTolerance(HitZCheck::Margin(corrZ, corrZ));

    //
    // hit ranges in det
    //
    OuterHitPhiPrediction::Range phi1 = phiPrediction(detRWindow.min());
    OuterHitPhiPrediction::Range phi2 = phiPrediction(detRWindow.max());
    phiRange = Range(std::min(phi1.min(), phi2.min()), std::max(phi1.max(), phi2.max()));
    Range w1 = zPrediction.range(detRWindow.min());
    Range w2 = zPrediction.range(detRWindow.max());
    hitZWindow = Range(std::min(w1.min(), w2.min()), std::max(w1.max(), w2.max())).intersection(detZWindow);
  } else {
    phiRange = phiPrediction(detRWindow.mean());
  }

  return std::make_unique<OuterEstimator<Algo>>(OuterDetCompatibility(layer, phiRange, detRWindow, hitZWindow),
                                                OuterHitCompatibility<Algo>(phiPrediction, zPrediction),
                                                iSetup);
}

std::unique_ptr<MeasurementEstimator> RectangularEtaPhiTrackingRegion::estimator(const ForwardDetLayer* layer,
                                                                                 const edm::EventSetup& iSetup) const {
  using Algo = HitRCheck;
  // det dimensions, ranges
  float halfThickness = 0.5f * layer->surface().bounds().thickness();
  float zLayer = layer->position().z();
  Range detZWindow(zLayer - halfThickness, zLayer + halfThickness);
  Range detRWindow(layer->specificSurface().innerRadius(), layer->specificSurface().outerRadius());

  // r prediction, skip if not intersection
  HitRCheck rPrediction(rzConstraint());
  Range hitRWindow = rPrediction.range(zLayer).intersection(detRWindow);
  if (hitRWindow.empty())
    return nullptr;

  // phi prediction
  OuterHitPhiPrediction phiPrediction = phiWindow(iSetup);
  OuterHitPhiPrediction::Range phiRange = phiPrediction(detRWindow.max());

  //
  // optional corrections for tolerance (mult.scatt, error, bending)
  //
  if (thePrecise) {
    float cotTheta = (detZWindow.mean() - origin().z()) / hitRWindow.mean();
    float cosThetaInv = std::sqrt(1 + sqr(cotTheta)) / cotTheta;
    MultipleScatteringParametrisation msSigma(layer, iSetup);
    float scatt = 3.f * msSigma(ptMin(), cotTheta);
    float bendR = longitudinalBendingCorrection(hitRWindow.max(), ptMin(), iSetup);
    float hitErrRPhi = 0.;
    float hitErrR = 0.;
    float corrPhi = (scatt + hitErrRPhi) / detRWindow.min();
    float corrR = scatt * std::abs(cosThetaInv) + bendR + hitErrR;

    phiPrediction.setTolerance(corrPhi);
    rPrediction.setTolerance(HitRCheck::Margin(corrR, corrR));

    //
    // hit ranges in det
    //
    Range w1, w2;
    if (zLayer > 0) {
      w1 = rPrediction.range(detZWindow.min());
      w2 = rPrediction.range(detZWindow.max());
    } else {
      w1 = rPrediction.range(detZWindow.max());
      w2 = rPrediction.range(detZWindow.min());
    }
    hitRWindow = Range(w1.min(), w2.max()).intersection(detRWindow);
  }

  return std::make_unique<OuterEstimator<Algo>>(OuterDetCompatibility(layer, phiRange, hitRWindow, detZWindow),
                                                OuterHitCompatibility<Algo>(phiPrediction, rPrediction),
                                                iSetup);
}

OuterHitPhiPrediction RectangularEtaPhiTrackingRegion::phiWindow(const edm::EventSetup& iSetup) const {
  auto phi0 = phiDirection();
  return OuterHitPhiPrediction(
      OuterHitPhiPrediction::Range(phi0 - thePhiMargin.left(), phi0 + thePhiMargin.right()),
      OuterHitPhiPrediction::Range(curvature(invPtRange().min(), iSetup), curvature(invPtRange().max(), iSetup)),
      originRBound());
}

HitRZConstraint RectangularEtaPhiTrackingRegion::rzConstraint() const {
  HitRZConstraint::Point pLeft, pRight;
  float zMin = origin().z() - originZBound();
  float zMax = origin().z() + originZBound();
  float rMin = -originRBound();
  float rMax = originRBound();
  if (theEtaRange.max() > 0) {
    pRight = HitRZConstraint::Point(rMin, zMax);
  } else {
    pRight = HitRZConstraint::Point(rMax, zMax);
  }
  if (theEtaRange.min() > 0.) {
    pLeft = HitRZConstraint::Point(rMax, zMin);
  } else {
    pLeft = HitRZConstraint::Point(rMin, zMin);
  }
  return HitRZConstraint(pLeft, theLambdaRange.min(), pRight, theLambdaRange.max());
}

TrackingRegion::Hits RectangularEtaPhiTrackingRegion::hits(const edm::EventSetup& es,
                                                           const SeedingLayerSetsHits::SeedingLayer& layer) const {
  TrackingRegion::Hits result;

  //ESTIMATOR

  const DetLayer* detLayer = layer.detLayer();

  bool measurementMethod = false;
  if (theMeasurementTrackerUsage == UseMeasurementTracker::kAlways)
    measurementMethod = true;
  else if (theMeasurementTrackerUsage == UseMeasurementTracker::kForSiStrips &&
           GeomDetEnumerators::isTrackerStrip(detLayer->subDetector()))
    measurementMethod = true;

  if (measurementMethod) {
    edm::ESHandle<MagneticField> field;
    es.get<IdealMagneticFieldRecord>().get(field);
    const MagneticField* magField = field.product();

    const GlobalPoint vtx = origin();
    GlobalVector dir = direction();

    std::unique_ptr<MeasurementEstimator> est;
    if ((GeomDetEnumerators::isTrackerPixel(detLayer->subDetector()) &&
         GeomDetEnumerators::isBarrel(detLayer->subDetector())) ||
        (!theUseEtaPhi && detLayer->location() == GeomDetEnumerators::barrel)) {
      const BarrelDetLayer& bl = dynamic_cast<const BarrelDetLayer&>(*detLayer);
      est = estimator(&bl, es);
    } else if ((GeomDetEnumerators::isTrackerPixel(detLayer->subDetector()) &&
                GeomDetEnumerators::isEndcap(detLayer->subDetector())) ||
               (!theUseEtaPhi && detLayer->location() == GeomDetEnumerators::endcap)) {
      const ForwardDetLayer& fl = dynamic_cast<const ForwardDetLayer&>(*detLayer);
      est = estimator(&fl, es);
    }

    EtaPhiMeasurementEstimator etaPhiEstimator((theEtaRange.second - theEtaRange.first) * 0.5f,
                                               (thePhiMargin.left() + thePhiMargin.right()) * 0.5f);
    MeasurementEstimator* findDetAndHits = &etaPhiEstimator;
    if (est) {
      LogDebug("RectangularEtaPhiTrackingRegion") << "use pixel specific estimator.";
      findDetAndHits = est.get();
    } else {
      LogDebug("RectangularEtaPhiTrackingRegion") << "use generic etat phi estimator.";
    }

    // TSOS
    float phi = phiDirection();
    // std::cout << "dir " << direction().x()/direction().perp() <<','<< direction().y()/direction().perp() << " " << sin(phi) <<','<<cos(phi)<< std::endl;
    Surface::RotationType rot(sin(phi), -cos(phi), 0, 0, 0, -1, cos(phi), sin(phi), 0);

    Plane::PlanePointer surface = Plane::build(GlobalPoint(0., 0., 0.), rot);
    //TrajectoryStateOnSurface tsos(lpar, *surface, magField);

    FreeTrajectoryState fts(GlobalTrajectoryParameters(vtx, dir, 1, magField));
    TrajectoryStateOnSurface tsos(fts, *surface);

    // propagator
    StraightLinePropagator prop(magField, alongMomentum);

    LayerMeasurements lm(theMeasurementTracker->measurementTracker(), *theMeasurementTracker);

    auto hits = lm.recHits(*detLayer, tsos, prop, *findDetAndHits);

    result.reserve(hits.size());
    for (auto h : hits) {
      cache.emplace_back(h);
      result.emplace_back(h);
    }

    LogDebug("RectangularEtaPhiTrackingRegion")
        << " found " << hits.size() << " minus one measurements on layer: " << detLayer->subDetector();
    // std::cout << "RectangularEtaPhiTrackingRegion" <<" found "<< meas.size()<<" minus one measurements on layer: "<<detLayer->subDetector() << std::endl;

  } else {
    //
    // temporary solution (actually heavily used for Pixels....)
    //
    if (detLayer->location() == GeomDetEnumerators::barrel) {
      const BarrelDetLayer& bl = dynamic_cast<const BarrelDetLayer&>(*detLayer);
      auto est = estimator(&bl, es);
      if (!est)
        return result;
      using Algo = HitZCheck;
      auto const& hitComp = (reinterpret_cast<OuterEstimator<Algo> const&>(*est)).hitCompatibility();
      auto layerHits = layer.hits();
      result.reserve(layerHits.size());
      for (auto&& ih : layerHits) {
        if (hitComp(*ih))
          result.emplace_back(std::move(ih));
      }

    } else {
      const ForwardDetLayer& fl = dynamic_cast<const ForwardDetLayer&>(*detLayer);
      auto est = estimator(&fl, es);
      if (!est)
        return result;
      using Algo = HitRCheck;
      auto const& hitComp = (reinterpret_cast<OuterEstimator<Algo> const&>(*est)).hitCompatibility();
      auto layerHits = layer.hits();
      result.reserve(layerHits.size());
      for (auto&& ih : layerHits) {
        if (hitComp(*ih))
          result.emplace_back(std::move(ih));
      }
    }
  }

  // std::cout << "RectangularEtaPhiTrackingRegion hits "  << result.size() << std::endl;

  return result;
}

std::string RectangularEtaPhiTrackingRegion::print() const {
  std::ostringstream str;
  str << TrackingRegionBase::print() << " eta: " << theEtaRange << " phi:" << thePhiMargin << "precise: " << thePrecise;
  return str.str();
}
