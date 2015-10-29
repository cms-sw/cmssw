#ifndef RectangularEtaPhiTrackingRegion_H
#define RectangularEtaPhiTrackingRegion_H

/** \class RectangularEtaPhiTrackingRegion
 * A concrete implementation of TrackingRegion. 
 * Apart of vertex constraint from TrackingRegion in this implementation
 * the region of interest is further constrainted in phi and eta 
 * around the direction of the region 
 */

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"
//#include "CommonDet/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZConstraint.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"



class OuterHitPhiPrediction;
class OuterEstimator;
class BarrelDetLayer;
class ForwardDetLayer;
class MeasurementTrackerEvent;

class RectangularEtaPhiTrackingRegion GCC11_FINAL : public TrackingRegion {
public:
  enum class UseMeasurementTracker {
    kNever = -1,
    kForSiStrips = 0,
    kAlways = 1
  };

  static UseMeasurementTracker intToUseMeasurementTracker(int value) {
    assert(value >= -1 && value <= 1);
    return static_cast<UseMeasurementTracker>(value);
  }

  static UseMeasurementTracker doubleToUseMeasurementTracker(double value) {
    // mimic the old behaviour
    if(value >  0.5) return UseMeasurementTracker::kAlways;
    if(value > -0.5) return UseMeasurementTracker::kForSiStrips;
    return UseMeasurementTracker::kNever;
  }

  static UseMeasurementTracker stringToUseMeasurementTracker(const std::string& name);

  RectangularEtaPhiTrackingRegion(RectangularEtaPhiTrackingRegion const & rh) :
    TrackingRegion(rh),
    theEtaRange(rh.theEtaRange),
    theLambdaRange(rh.theLambdaRange),
    thePhiMargin(rh.thePhiMargin),
    theMeanLambda(rh.theMeanLambda),
    theMeasurementTrackerUsage(rh.theMeasurementTrackerUsage),
    thePrecise(rh.thePrecise),
    theUseEtaPhi(rh.theUseEtaPhi),
    theMeasurementTracker(rh.theMeasurementTracker) {}
  
  RectangularEtaPhiTrackingRegion& operator=(RectangularEtaPhiTrackingRegion const &)=delete;
  RectangularEtaPhiTrackingRegion(RectangularEtaPhiTrackingRegion &&)=default;
  RectangularEtaPhiTrackingRegion& operator=(RectangularEtaPhiTrackingRegion &&)=default;

  typedef TkTrackingRegionsMargin<float> Margin;

 /** constructor (symmetric eta and phi margins). <BR>
  * dir        - the direction around which region is constructed <BR>
  *              the initial direction of the momentum of the particle 
  *              should be in the range <BR> 
  *              phi of dir +- deltaPhi <BR>
  *              eta of dir +- deltaEta <BR> 
  *              
  * vertexPos  - the position of the vertex (origin) of the of the region.<BR>
  *              It is a centre of cylinder constraind with rVertex, zVertex.
  *              The track of the particle should cross the cylinder <BR>
  *              WARNING: in the current implementaion the vertexPos is
  *              supposed to be placed on the beam line, i.e. to be of the form
  *              (0,0,float)
  *
  * ptMin      - minimal pt of interest <BR>
  * rVertex    - radius of the cylinder around beam line where the tracks
  *              of interest should point to. <BR>
  * zVertex    - half height of the cylinder around the beam line
  *              where the tracks of interest should point to.   <BR>
  * deltaEta   - allowed deviation of the initial direction of particle
  *              in eta in respect to direction of the region <BR>
  *  deltaPhi  - allowed deviation of the initial direction of particle
  *              in phi in respect to direction of the region 
  *  whereToUseMeasurementTracker: 1=everywhere, 0=outside pixles, -1=nowhere
  */
  RectangularEtaPhiTrackingRegion( const GlobalVector & dir, 
				   const GlobalPoint & vertexPos,
                                   float ptMin, float rVertex, float zVertex,
                                   float deltaEta, float deltaPhi,
                                   UseMeasurementTracker whereToUseMeasurementTracker = UseMeasurementTracker::kNever,
                                   bool precise = true,
                                   const MeasurementTrackerEvent *measurementTracker = nullptr,
				   bool etaPhiRegion=false) 
    : RectangularEtaPhiTrackingRegion(dir, vertexPos, Range( -1/ptMin, 1/ptMin), rVertex, zVertex,
                                      Margin(std::abs(deltaEta), std::abs(deltaEta)),
                                      Margin(std::abs(deltaPhi), std::abs(deltaPhi)),
                                      whereToUseMeasurementTracker, precise,
                                      measurementTracker, etaPhiRegion)
    {}
 
 /** constructor (asymmetrinc eta and phi margins). <BR>
  * non equal left-right eta and phi bounds around direction are
  * possible. The ranges are defined using \c Margin.
  * the meaning of other arguments is the same as in the case of 
  * symmetring bounds to direction of the region.
  */
  RectangularEtaPhiTrackingRegion( const GlobalVector & dir, 
		                       const GlobalPoint & vertexPos,
                                   float ptMin, float rVertex, float zVertex,
                                   Margin etaMargin,
                                   Margin phiMargin,
                                   UseMeasurementTracker whereToUseMeasurementTracker = UseMeasurementTracker::kNever,
				   bool precise = true, 
                                   const MeasurementTrackerEvent *measurementTracker = nullptr,
				   bool etaPhiRegion=false) 
    : RectangularEtaPhiTrackingRegion(dir, vertexPos, Range( -1/ptMin, 1/ptMin), rVertex, zVertex,
                                      etaMargin, phiMargin,
                                      whereToUseMeasurementTracker, precise,
                                      measurementTracker, etaPhiRegion)
    {}

 /** constructor (explicit pt range, asymmetrinc eta and phi margins). <BR>
  * the meaning of other arguments is the same as in the case of 
  * symmetring bounds to direction of the region.
  */
  RectangularEtaPhiTrackingRegion( const GlobalVector & dir, 
		                       const GlobalPoint & vertexPos,
                                   Range invPtRange, 
                                   float rVertex, float zVertex,
                                   Margin etaMargin,
                                   Margin phiMargin,
                                   UseMeasurementTracker whereToUseMeasurementTracker = UseMeasurementTracker::kNever,
                                   bool precise = true,
                                   const MeasurementTrackerEvent *measurementTracker = nullptr,
				   bool etaPhiRegion=false) 
    : TrackingRegionBase( dir, vertexPos, invPtRange, rVertex, zVertex),
    thePhiMargin( phiMargin), theMeasurementTrackerUsage(whereToUseMeasurementTracker), thePrecise(precise),theUseEtaPhi(etaPhiRegion),
    theMeasurementTracker(measurementTracker)
    { initEtaRange(dir, etaMargin); }


  /// allowed eta range [eta_min, eta_max] interval
  const Range & etaRange() const { return theEtaRange; }

  /// defined phi range around phi0, margin is [phi_left,phi_right]. 
  /// region is defined in a range: [phi0-phi_left, phi0+phi_right]
  const Margin & phiMargin() const { return thePhiMargin; }

  /// is precise error calculation switched on 
  bool  isPrecise() const { return thePrecise; }

  virtual TrackingRegion::Hits hits(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const SeedingLayerSetsHits::SeedingLayer& layer) const override;

  virtual HitRZCompatibility * checkRZ(const DetLayer* layer,  
				       const Hit &  outerHit,
				       const edm::EventSetup& iSetup,
				       const DetLayer* outerlayer=0,
				       float lr=0, float gz=0, float dr=0, float dz=0) const
  { return checkRZOld(layer,outerHit->hit(),iSetup); }

  virtual RectangularEtaPhiTrackingRegion* clone() const { 
    return new RectangularEtaPhiTrackingRegion(*this);
  }

  virtual std::string name() const { return "RectangularEtaPhiTrackingRegion"; }
  virtual std::string print() const;

private:
  HitRZCompatibility* checkRZOld(
      const DetLayer* layer, 
      const TrackingRecHit*  outerHit,
      const edm::EventSetup& iSetup) const;

  std::unique_ptr<OuterEstimator> estimator(const BarrelDetLayer* layer,const edm::EventSetup& iSetup) const dso_internal;
  std::unique_ptr<OuterEstimator> estimator(const ForwardDetLayer* layer,const edm::EventSetup& iSetup) const dso_internal;

  OuterHitPhiPrediction phiWindow(const edm::EventSetup& iSetup) const dso_internal;
  HitRZConstraint rzConstraint() const dso_internal;

  void  initEtaRange( const GlobalVector & dir, const Margin& margin);

private:

  Range theEtaRange;
  Range theLambdaRange;
  Margin thePhiMargin;
  float theMeanLambda;
  const UseMeasurementTracker theMeasurementTrackerUsage;
  bool thePrecise;
  bool theUseEtaPhi;
  const MeasurementTrackerEvent *theMeasurementTracker;



  using cacheHitPointer = mayown_ptr<BaseTrackerRecHit>;
  using cacheHits=std::vector<cacheHitPointer>;

  

  /*  wait... think! 
   *  done? questions?  think again, look where this region is constructed
   *  still question? study tracker code for the next couple of weeks, then we may discuss.  
   */
  mutable cacheHits cache;

};

#endif
