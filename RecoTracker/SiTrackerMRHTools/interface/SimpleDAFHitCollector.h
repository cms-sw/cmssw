#ifndef SiTrackerMRHTools_SimpleDAFHitCollector_h
#define SiTrackerMRHTools_SimpleDAFHitCollector_h
#include "RecoTracker/SiTrackerMRHTools/interface/MultiRecHitCollector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoTracker/SiTrackerMRHTools/interface//SiTrackerMultiRecHitUpdator.h"
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>
#include <vector>
#include <memory>

class Propagator;
class MeasurementEstimator;
//class SiTrackerMultiRecHitUpdator;
class StripRecHit1D;

class SimpleDAFHitCollector :public MultiRecHitCollector {
	public:
	explicit SimpleDAFHitCollector(const MeasurementTracker* measurementTracker,
				 const SiTrackerMultiRecHitUpdator* updator,
			         const MeasurementEstimator* est,
				 const Propagator* propagator, bool debug
				 ):MultiRecHitCollector(measurementTracker), theUpdator(updator), theEstimator(est), thePropagator(propagator), debug_(debug){
    theHitCloner = static_cast<TkTransientTrackingRecHitBuilder const *>(theUpdator->getBuilder())->cloner();
}
			

	virtual ~SimpleDAFHitCollector(){}
	
	//given a trajectory it returns a collection
        //of SiTrackerMultiRecHits and InvalidTransientRecHits.
        //For each measurement in the trajectory, measurements are looked for according to the 
        //MeasurementDet::fastMeasurements method only in the detector where the original measurement lays. 
        //If measurements are found a SiTrackerMultiRecHit is built.
	//All the components will lay on the same detector  
	
	virtual std::vector<TrajectoryMeasurement> recHits(const Trajectory&, const MeasurementTrackerEvent *theMTE) const override;

	const SiTrackerMultiRecHitUpdator* getUpdator() const {return theUpdator;}
	const MeasurementEstimator* getEstimator() const {return theEstimator;}
        const Propagator* getPropagator() const {return thePropagator;}

	void Debug( const std::vector<TrajectoryMeasurement> TM ) const;

	private:
	//TransientTrackingRecHit::ConstRecHitContainer buildMultiRecHits(const std::vector<TrajectoryMeasurementGroup>& measgroup) const;
	//void buildMultiRecHits(const std::vector<TrajectoryMeasurement>& measgroup, std::vector<TrajectoryMeasurement>& result) const;

        std::unique_ptr<TrackingRecHit> rightdimension( TrackingRecHit const & hit ) const{
          if( !hit.isValid() || ( hit.dimension()!=2) ) {
            return std::unique_ptr<TrackingRecHit>{hit.clone()};
          }
          auto const & thit = static_cast<BaseTrackerRecHit const&>(hit);
          auto const & clus = thit.firstClusterRef();
          if (clus.isPixel()) return std::unique_ptr<TrackingRecHit>{hit.clone()};
          else if (thit.isMatched()) {
            LogDebug("MultiRecHitCollector") << " SiStripMatchedRecHit2D to check!!!";
            return std::unique_ptr<TrackingRecHit>{hit.clone()};
          } else  if (thit.isProjected()) {
            edm::LogError("MultiRecHitCollector") << " ProjectedSiStripRecHit2D should not be present at this stage!!!";
            return std::unique_ptr<TrackingRecHit>{hit.clone()};
          } else return clone(thit);
       }
	
        std::unique_ptr<TrackingRecHit> clone(BaseTrackerRecHit const & hit2D ) const {
         auto const & detU = *hit2D.detUnit();
         //Use 2D SiStripRecHit in endcap
         bool endcap = detU.type().isEndcap();
         if (endcap) return std::unique_ptr<TrackingRecHit>{hit2D.clone()};
         return std::unique_ptr<TrackingRecHit>{
                   new SiStripRecHit1D(hit2D.localPosition(),
                                       LocalError(hit2D.localPositionError().xx(),0.f,std::numeric_limits<float>::max()),
                                       *hit2D.det(), hit2D.firstClusterRef()) };
 
        }

	private:
	const SiTrackerMultiRecHitUpdator* theUpdator;
	const MeasurementEstimator* theEstimator;
	//this actually is not used in the fastMeasurement method 	
	const Propagator* thePropagator; 
	TkClonerImpl theHitCloner;
	const bool debug_;


};


#endif 
