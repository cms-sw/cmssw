#ifndef TrajectoryMeasurementGroup_H
#define TrajectoryMeasurementGroup_H

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetGroup.h"
#include <vector>
#include <utility>

/** A class that facilitates grouping of trajectory measurements
 *  according to the group of Dets from which they come.
 *  Functionally equivalent to 
 *  pair<vector<TrajectoryMeasurement>, vector<DetWithState> > 
 *  but with a more convenient interface.
 *  Used as a return type for the CompositeDet::groupedMeasurements method.
 */

class TrajectoryMeasurementGroup {
public:

  TrajectoryMeasurementGroup() {}
  TrajectoryMeasurementGroup( const std::vector<TrajectoryMeasurement>& meas,
			      const DetGroup& dg) : measurements_(meas), detGroup_(dg) {}

#if defined( __GXX_EXPERIMENTAL_CXX0X__)
  TrajectoryMeasurementGroup(std::vector<TrajectoryMeasurement>&& meas,
			     const DetGroup& dg) : measurements_(std::move(meas)), detGroup_(dg) {}
  TrajectoryMeasurementGroup(std::vector<TrajectoryMeasurement>&& meas,
			     DetGroup&& dg) : measurements_(std::move(meas)), detGroup_(std::move(dg)) {}

#endif

  const std::vector<TrajectoryMeasurement>& measurements() const {return measurements_;}
        std::vector<TrajectoryMeasurement>& measurements()       {return measurements_;}
  const DetGroup& detGroup() const {return detGroup_;}

private:

  std::vector<TrajectoryMeasurement> measurements_;
  DetGroup                           detGroup_;

};



#endif
