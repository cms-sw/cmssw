#ifndef TrajectoryFilter_H
#define TrajectoryFilter_H

#include <string>

namespace edm {
  class Event;
  class EventSetup;
  class ConsumesCollector;
}

class Trajectory;
class TempTrajectory;

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
//#include "RecoTracker/CkfPattern/interface/TempTrajectory.h"


/** An abstract base class for Filter<TempTrajectory>.
 *  Adds a name() method.
 *  This class is useful because the CkfTrajectoryBuilder
 *  uses TrajectoryFilters as stopping conditions.
 */

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

class TrajectoryFilter {
 public:
  
  //a type def while deciding what the record it
  typedef CkfComponentsRecord Record;

  static const bool qualityFilterIfNotContributing =true;
  static const bool toBeContinuedIfNotContributing =true;

  virtual ~TrajectoryFilter();
  virtual std::string name() const = 0;

  virtual void setEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  virtual bool operator()( TempTrajectory&t) const { return toBeContinued(t);}
  virtual bool operator()( Trajectory&t) const { return toBeContinued(t);}

  virtual bool qualityFilter( const TempTrajectory&) const = 0;
  virtual bool qualityFilter( const Trajectory&) const = 0;

  virtual bool toBeContinued( TempTrajectory&) const = 0;
  virtual bool toBeContinued( Trajectory&) const = 0;
};


#endif
