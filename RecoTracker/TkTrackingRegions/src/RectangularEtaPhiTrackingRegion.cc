#include <cmath>
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/OuterEstimator.h"

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
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/KalmanUpdators/interface/EtaPhiMeasurementEstimator.h"

template <class T> T sqr( T t) {return t*t;}


using namespace PixelRecoUtilities;
using namespace std;
using namespace ctfseeding; 

void RectangularEtaPhiTrackingRegion::
    initEtaRange( const GlobalVector & dir, const Margin& margin)
{
  float eta = dir.eta();
  theEtaRange = Range(eta-margin.left(), eta+margin.right());
}

HitRZCompatibility* RectangularEtaPhiTrackingRegion::
checkRZOld(const DetLayer* layer, const TrackingRecHit *outerHit,const edm::EventSetup& iSetup) const
{
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

  bool isBarrel = (layer->location() == GeomDetEnumerators::barrel);
  GlobalPoint ohit =  tracker->idToDet(outerHit->geographicalId())->surface().toGlobal(outerHit->localPosition());
  float outerred_r = sqrt( sqr(ohit.x()-origin().x())+sqr(ohit.y()-origin().y()) );
  //PixelRecoPointRZ outer(ohit.perp(), ohit.z());
  PixelRecoPointRZ outer(outerred_r, ohit.z());
  
  float zMinOrigin = origin().z() - originZBound();
  float zMaxOrigin = origin().z() + originZBound();

  if (!thePrecise) {
    double vcotMin = (outer.z() > zMaxOrigin) ?
        (outer.z()-zMaxOrigin)/(outer.r()+originRBound())
      : (outer.z()-zMaxOrigin)/(outer.r()-originRBound());
    double vcotMax = (outer.z() > zMinOrigin) ?
        (outer.z()-zMinOrigin)/(outer.r()-originRBound())
      : (outer.z()-zMinOrigin)/(outer.r()+originRBound());
    float cotRight  = max(vcotMin,(double) sinh(theEtaRange.min()));
    float cotLeft = min(vcotMax, (double) sinh(theEtaRange.max()));
    return new HitEtaCheck( isBarrel, outer, cotLeft, cotRight);
  }
  float hitZErr = 0.;
  float hitRErr = 0.;

  PixelRecoPointRZ  outerL, outerR;
  if (layer->location() == GeomDetEnumerators::barrel) {
    outerL = PixelRecoPointRZ(outer.r(), outer.z()-hitZErr);
    outerR = PixelRecoPointRZ(outer.r(), outer.z()+hitZErr);
  } else if (outer.z() > 0) {
    outerL = PixelRecoPointRZ(outer.r()+hitRErr, outer.z());
    outerR = PixelRecoPointRZ(outer.r()-hitRErr, outer.z());
  } else {
    outerL = PixelRecoPointRZ(outer.r()-hitRErr, outer.z());
    outerR = PixelRecoPointRZ(outer.r()+hitRErr, outer.z());
  }
  //CHECK
  MultipleScatteringParametrisation oSigma(layer,iSetup);
  float cotThetaOuter = sinh(theEtaRange.mean());
  float sinThetaOuter = 1/sqrt(1+sqr(cotThetaOuter)); 
  float outerZscatt = 3*oSigma(ptMin(),cotThetaOuter) / sinThetaOuter;

  PixelRecoLineRZ boundL(outerL, sinh(theEtaRange.max()));
  PixelRecoLineRZ boundR(outerR, sinh(theEtaRange.min()));
  float zMinLine = boundL.zAtR(0.)-outerZscatt;
  float zMaxLine = boundR.zAtR(0.)+outerZscatt;
  PixelRecoPointRZ vtxL(0.,max(zMinLine, zMinOrigin));
  PixelRecoPointRZ vtxR(0.,min(zMaxLine, zMaxOrigin)); 
  PixelRecoPointRZ vtxMean(0.,(vtxL.z()+vtxR.z())/2.);
  //CHECK
  MultipleScatteringParametrisation iSigma(layer,iSetup);
  float innerScatt = 3 * iSigma(ptMin(),vtxMean, outer);
  
  SimpleLineRZ leftLine( vtxL, outerL);
  SimpleLineRZ rightLine( vtxR, outerR);

  HitRZConstraint rzConstraint(leftLine, rightLine);
  float cotTheta = fabs(leftLine.cotLine()+rightLine.cotLine())/2;

//  float bendR = longitudinalBendingCorrection(outer.r(),ptMin());

  if (isBarrel) {
    float sinTheta = 1/sqrt(1+sqr(cotTheta));
    float corrZ = innerScatt/sinTheta + hitZErr;
    return new HitZCheck(rzConstraint, HitZCheck::Margin(corrZ,corrZ));
  } else {
    float cosTheta = 1/sqrt(1+sqr(1/cotTheta));
    float corrR = innerScatt/cosTheta + hitRErr;
    return new HitRCheck( rzConstraint, HitRCheck::Margin(corrR,corrR));
  }
}

OuterEstimator *
  RectangularEtaPhiTrackingRegion::estimator(const BarrelDetLayer* layer,const edm::EventSetup& iSetup) const
{

  // det dimensions 
  float halfLength = layer->surface().bounds().length()/2;
  float halfThickness  = layer->surface().bounds().thickness()/2;
  float z0 = layer->position().z();
  float radius = layer->specificSurface().radius();

  // det ranges
  Range detRWindow (radius-halfThickness, radius+halfThickness);
  Range detZWindow(z0-halfLength,z0+halfLength);

  // z prediction, skip if not intersection
  HitZCheck zPrediction(rzConstraint());
  Range hitZWindow = zPrediction.range(detRWindow.min()).
                                               intersection(detZWindow);
  if (hitZWindow.empty()) return 0;

  // phi prediction
  OuterHitPhiPrediction phiPrediction = phiWindow(iSetup);

  //
  // optional corrections for tolerance (mult.scatt, error, bending)
  //
  OuterHitPhiPrediction::Range phiRange;
  if (thePrecise) {
    float cotTheta = (hitZWindow.mean()-origin().z()) / radius;
    float sinTheta = 1/sqrt(1+sqr(cotTheta));
    MultipleScatteringParametrisation msSigma(layer,iSetup);
    float scatt = 3 * msSigma(ptMin(), cotTheta);
    float bendR = longitudinalBendingCorrection(radius,ptMin(),iSetup);

    float hitErrRPhi = 0.;
    float hitErrZ = 0.;
    float corrPhi = (scatt+ hitErrRPhi)/radius;
    float corrZ = scatt/sinTheta + bendR*fabs(cotTheta) + hitErrZ;

    phiPrediction.setTolerance(OuterHitPhiPrediction::Margin(corrPhi,corrPhi));
    zPrediction.setTolerance(HitZCheck::Margin(corrZ,corrZ));

    //
    // hit ranges in det
    //
    OuterHitPhiPrediction::Range phi1 = phiPrediction(detRWindow.min());
    OuterHitPhiPrediction::Range phi2 = phiPrediction(detRWindow.max());
    phiRange = Range( min(phi1.min(),phi2.min()), max(phi1.max(),phi2.max()));
    Range w1 = zPrediction.range(detRWindow.min());
    Range w2 = zPrediction.range(detRWindow.max());
    hitZWindow = Range(
      min(w1.min(),w2.min()), max(w1.max(),w2.max())).intersection(detZWindow);
  }
  else {
    phiRange = phiPrediction(detRWindow.mean()); 
  }

  return new OuterEstimator(
			    OuterDetCompatibility( layer, phiRange, detRWindow, hitZWindow),
			    OuterHitCompatibility( phiPrediction, zPrediction ),
			    iSetup);
}

OuterEstimator *
RectangularEtaPhiTrackingRegion::estimator(const ForwardDetLayer* layer,const edm::EventSetup& iSetup) const
{

  // det dimensions, ranges
  float halfThickness  = layer->surface().bounds().thickness()/2;
  float zLayer = layer->position().z() ;
  Range detZWindow( zLayer-halfThickness, zLayer+halfThickness);
  Range detRWindow( layer->specificSurface().innerRadius(), 
                    layer->specificSurface().outerRadius());
  
  // r prediction, skip if not intersection
  HitRCheck rPrediction(rzConstraint());
  Range hitRWindow = rPrediction.range(zLayer).intersection(detRWindow);
  if (hitRWindow.empty()) return 0;

  // phi prediction
  OuterHitPhiPrediction phiPrediction = phiWindow(iSetup);
  OuterHitPhiPrediction::Range phiRange = phiPrediction(detRWindow.max());

  //
  // optional corrections for tolerance (mult.scatt, error, bending)
  //
  if (thePrecise) {
    float cotTheta = (detZWindow.mean()-origin().z())/hitRWindow.mean();
    float cosTheta = cotTheta/sqrt(1+sqr(cotTheta)); 
    MultipleScatteringParametrisation msSigma(layer,iSetup);
    float scatt = 3 * msSigma(ptMin(),cotTheta);
    float bendR = longitudinalBendingCorrection(hitRWindow.max(),ptMin(),iSetup);
    float hitErrRPhi = 0.;
    float hitErrR = 0.;
    float corrPhi = (scatt+hitErrRPhi)/detRWindow.min();
    float corrR   = scatt/fabs(cosTheta) + bendR + hitErrR;

    phiPrediction.setTolerance(OuterHitPhiPrediction::Margin(corrPhi,corrPhi));
    rPrediction.setTolerance(HitRCheck::Margin(corrR,corrR));

    //
    // hit ranges in det
    //
    Range w1,w2;
    if (zLayer > 0) {
      w1 = rPrediction.range(detZWindow.min());
      w2 = rPrediction.range(detZWindow.max());
    } else {
      w1 = rPrediction.range(detZWindow.max());
      w2 = rPrediction.range(detZWindow.min());
    }
    hitRWindow = Range(w1.min(),w2.max()).intersection(detRWindow);
  }

  return new OuterEstimator(
    OuterDetCompatibility( layer, phiRange, hitRWindow, detZWindow),
    OuterHitCompatibility( phiPrediction, rPrediction),iSetup );
}



OuterHitPhiPrediction 
    RectangularEtaPhiTrackingRegion::phiWindow(const edm::EventSetup& iSetup) const
{
  float phi0 = direction().phi();
  return OuterHitPhiPrediction( 
      OuterHitPhiPrediction::Range( phi0-thePhiMargin.left(),
                                    phi0+thePhiMargin.left()),
      OuterHitPhiPrediction::Range( curvature(invPtRange().min(),iSetup), 
                                    curvature(invPtRange().max(),iSetup)),
      originRBound());
}


HitRZConstraint
    RectangularEtaPhiTrackingRegion::rzConstraint() const
{
  HitRZConstraint::Point pLeft,pRight;
  float zMin = origin().z() - originZBound();
  float zMax = origin().z() + originZBound();
  float rMin = -originRBound();
  float rMax =  originRBound();
  if(theEtaRange.max() > 0) {
    pRight = HitRZConstraint::Point(rMin,zMax);
  } else { 
    pRight = HitRZConstraint::Point(rMax,zMax);
  } 
  if (theEtaRange.min() > 0.) {
    pLeft = HitRZConstraint::Point(rMax, zMin);
  } else {
    pLeft = HitRZConstraint::Point(rMin, zMin);
  } 
  return HitRZConstraint(pLeft, sinh(theEtaRange.min()),
                              pRight, sinh(theEtaRange.max()) );
}

TrackingRegion::Hits RectangularEtaPhiTrackingRegion::hits(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const  SeedingLayer* layer) const
{


  //ESTIMATOR
  TrackingRegion::Hits result;

  const DetLayer * detLayer = layer->detLayer();
  OuterEstimator * est = 0;

  bool measurementMethod = false;
  if ( theMeasurementTrackerUsage > 0.5) measurementMethod = true;
  if ( theMeasurementTrackerUsage > -0.5 &&
       !(detLayer->subDetector() == GeomDetEnumerators::PixelBarrel ||
         detLayer->subDetector() == GeomDetEnumerators::PixelEndcap) ) measurementMethod = true;

  if(measurementMethod) {
  edm::ESHandle<MagneticField> field;
  es.get<IdealMagneticFieldRecord>().get(field);
  const MagneticField * magField = field.product();

  const GlobalPoint vtx = origin();
  GlobalVector dir = direction();
  
  if (detLayer->subDetector() == GeomDetEnumerators::PixelBarrel || (!theUseEtaPhi  && detLayer->location() == GeomDetEnumerators::barrel)){
    const BarrelDetLayer& bl = dynamic_cast<const BarrelDetLayer&>(*detLayer);
    est = estimator(&bl,es);
  } else if (detLayer->subDetector() == GeomDetEnumerators::PixelEndcap || (!theUseEtaPhi  && detLayer->location() == GeomDetEnumerators::endcap)) {
    const ForwardDetLayer& fl = dynamic_cast<const ForwardDetLayer&>(*detLayer);
    est = estimator(&fl,es);
  }
  
  EtaPhiMeasurementEstimator etaPhiEstimator ((theEtaRange.second-theEtaRange.first)/2.,
					      (thePhiMargin.left()+thePhiMargin.right())/2.);
  MeasurementEstimator & findDetAndHits = etaPhiEstimator;
  if (est){
    LogDebug("RectangularEtaPhiTrackingRegion")<<"use pixel specific estimator.";
    findDetAndHits =*est;
  }
  else{
    LogDebug("RectangularEtaPhiTrackingRegion")<<"use generic etat phi estimator.";
  }
   
  // TSOS
  float phi = dir.phi();
  Surface::RotationType rot( sin(phi), -cos(phi),           0,
                             0,                0,          -1,
                             cos(phi),  sin(phi),           0);

  Plane::PlanePointer surface = Plane::build(GlobalPoint(0.,0.,0.), rot);
  //TrajectoryStateOnSurface tsos(lpar, *surface, magField);

  FreeTrajectoryState fts( GlobalTrajectoryParameters(vtx, dir, 1, magField) );
  TrajectoryStateOnSurface tsos(fts, *surface);

  // propagator
  StraightLinePropagator prop( magField, alongMomentum);

  edm::ESHandle<MeasurementTracker> measurementTrackerESH;
  es.get<CkfComponentsRecord>().get(theMeasurementTrackerName, measurementTrackerESH); 
  const MeasurementTracker * measurementTracker = measurementTrackerESH.product(); 
  measurementTracker->update(ev);

  LayerMeasurements lm(measurementTracker);
   
  vector<TrajectoryMeasurement> meas = lm.measurements(*detLayer, tsos, prop, findDetAndHits);
  typedef vector<TrajectoryMeasurement>::const_iterator IM;
  for (IM im = meas.begin(); im != meas.end(); im++) {
    TrajectoryMeasurement::ConstRecHitPointer ptrHit = im->recHit();
    if (ptrHit->isValid())  result.push_back( ptrHit );
  }

  LogDebug("RectangularEtaPhiTrackingRegion")<<" found "<< meas.size()<<" minus one measurements on layer: "<<detLayer->subDetector();

  } else {
  //
  // temporary solution 
  //
  if (detLayer->location() == GeomDetEnumerators::barrel) {
    const BarrelDetLayer& bl = dynamic_cast<const BarrelDetLayer&>(*detLayer);
    est = estimator(&bl,es);
  } else {
    const ForwardDetLayer& fl = dynamic_cast<const ForwardDetLayer&>(*detLayer);
    est = estimator(&fl,es);
  }
  if (!est) return result;

  TrackingRegion::Hits layerHits = layer->hits(ev,es);
  for (TrackingRegion::Hits::const_iterator ih= layerHits.begin(); ih != layerHits.end(); ih++) {
    if ( est->hitCompatibility()(ih->get()) ) {
       result.push_back( *ih );
    }
  }
  }
  
  delete est;
  return result;
}

std::string RectangularEtaPhiTrackingRegion::print() const {
  std::ostringstream str;
  str << TrackingRegionBase::print() 
      <<" eta: "<<theEtaRange<<" phi:"<<thePhiMargin
      << "precise: "<<thePrecise;
  return str.str();
}

