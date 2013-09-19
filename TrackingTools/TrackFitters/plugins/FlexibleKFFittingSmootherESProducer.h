#ifndef TrackingTools_TrackFitters_FlexibleKFFittingSmootherESProducer_h
#define TrackingTools_TrackFitters_FlexibleKFFittingSmootherESProducer_h

/** \class FlexibleKFFittingSmootherESProducer
 *  ESProducer for the FlexibleKFFittingSmoother
 *
 *  \author mangano
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"
#include "TrackingTools/TrackFitters/interface/FlexibleKFFittingSmoother.h"
#include <boost/shared_ptr.hpp>

class  FlexibleKFFittingSmootherESProducer: public edm::ESProducer{
 public:
  FlexibleKFFittingSmootherESProducer(const edm::ParameterSet & p);
  virtual ~FlexibleKFFittingSmootherESProducer(); 
  boost::shared_ptr<TrajectoryFitter> produce(const TrajectoryFitterRecord &);
 private:
  boost::shared_ptr<TrajectoryFitter> _fitter;
  edm::ParameterSet pset_;
};


#endif




