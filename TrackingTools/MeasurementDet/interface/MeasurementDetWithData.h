#ifndef MeasurementDetWithData_H
#define MeasurementDetWithData_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h" 

class MeasurementDetWithData {
public:
  MeasurementDetWithData() :
    det_(0), data_(0) {}
  
  MeasurementDetWithData(const MeasurementDet &det, const MeasurementTrackerEvent &data) :
    det_(&det), data_(&data) {}
  
  bool isValid() const { return det_ != 0; }
  bool isNull() const { return det_ == 0; }
  
  const MeasurementDet & mdet() const { return *det_; }

  // duplicate interface of MeasurementDet
  typedef MeasurementDet::TempMeasurements TempMeasurements;
  typedef MeasurementDet::RecHitContainer  RecHitContainer;

  using SimpleHitContainer = MeasurementDet::SimpleHitContainer;

        RecHitContainer recHits( const TrajectoryStateOnSurface &tsos ) const { 
            return mdet().recHits(tsos, data()); 
        }

        // use a MeasurementEstimator to filter the hits (same algo as below..)
        // default as above 
        bool recHits( const TrajectoryStateOnSurface& stateOnThisDet, const MeasurementEstimator & me, RecHitContainer & result, std::vector<float> & out) const {
            return mdet().recHits(stateOnThisDet, me, data(), result, out);
        }

  bool recHits(SimpleHitContainer & result,  
	       const TrajectoryStateOnSurface& stateOnThisDet, const MeasurementEstimator& me) const{
    return mdet().recHits(result,stateOnThisDet, me, data());
  }

  /** obsolete version in case the TrajectoryState on the surface of the
   *  Det is already available. The first TrajectoryStateOnSurface is on the surface of this 
   *  Det, and the second TrajectoryStateOnSurface is not used, as the propagator...
   * The stateOnThisDet should the result of <BR>
   *  prop.propagate( startingState, this->surface())
   */
  std::vector<TrajectoryMeasurement> 
  fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		    const TrajectoryStateOnSurface& tsos2, 
		    const Propagator& prop, 
		    const MeasurementEstimator& est) const {
    return mdet().fastMeasurements(stateOnThisDet, tsos2, prop, est, data());
  }
  
  // return false if missing (if inactive is true and one hit)
  bool measurements( const TrajectoryStateOnSurface& stateOnThisDet,
		     const MeasurementEstimator& est,
		     TempMeasurements & result) const {
    return mdet().measurements(stateOnThisDet, est, data(), result);
  }


  // forward methods which don't actually depend on data
  const GeomDet& fastGeomDet() const { return mdet().fastGeomDet(); }
  const GeomDet& geomDet() const { return mdet().geomDet(); } 
        const Surface& surface() const { return  mdet().geomDet().surface(); }
        const Surface::PositionType& position() const { return mdet().geomDet().position(); }

  // these instead potentially depend on the data
  bool isActive() const { return mdet().isActive(data()); }
  bool hasBadComponents(const TrajectoryStateOnSurface &tsos) const { return mdet().hasBadComponents(tsos, data()); }

    private:
        const MeasurementTrackerEvent & data() const { return *data_; }
  const MeasurementDet * det_;
  const MeasurementTrackerEvent * data_;
};

#endif
