#ifndef Tracker_TrajMeasLessEstim_H
#define Tracker_TrajMeasLessEstim_H

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

/** Function object for ordering TrajectoryMeasurements according to the
 *  value of their compatibility estimate (typically Chi^2)
 */

class TrajMeasLessEstim {
public:
  bool operator()( const TrajectoryMeasurement& a, const TrajectoryMeasurement& b) {
    return a.estimate() < b.estimate();
  }
};

#endif // Tracker_TrajMeasLessEstim_H
