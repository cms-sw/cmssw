#ifndef GsfTrajectorySmootherESProducer_h_
#define GsfTrajectorySmootherESProducer_h_

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h" 
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include <boost/shared_ptr.hpp>

/** Provides a GSF smoother algorithm */

class  GsfTrajectorySmootherESProducer: public edm::ESProducer{
 public:
  GsfTrajectorySmootherESProducer(const edm::ParameterSet & p);
  virtual ~GsfTrajectorySmootherESProducer(); 
  boost::shared_ptr<TrajectorySmoother> produce(const TrajectoryFitterRecord &);
 private:
  edm::ParameterSet pset_;
};


#endif




