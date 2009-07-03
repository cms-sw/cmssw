#ifndef GsfTrajectoryFitterESProducer_h_
#define GsfTrajectoryFitterESProducer_h_

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h" 
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include <boost/shared_ptr.hpp>

/** Provides a GSF fitter algorithm */

class  GsfTrajectoryFitterESProducer: public edm::ESProducer{
 public:
  GsfTrajectoryFitterESProducer(const edm::ParameterSet & p);
  virtual ~GsfTrajectoryFitterESProducer(); 
  boost::shared_ptr<TrajectoryFitter> produce(const TrajectoryFitterRecord &);
 private:
  edm::ParameterSet pset_;
};


#endif




