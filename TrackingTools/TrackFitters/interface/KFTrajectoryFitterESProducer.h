#ifndef TrackingTools_ESProducers_KFUpdatorESProducer_h
#define TrackingTools_ESProducers_KFUpdatorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include <boost/shared_ptr.hpp>

class  KFTrajectoryFitterESProducer: public edm::ESProducer{
 public:
  KFTrajectoryFitterESProducer(const edm::ParameterSet & p);
  virtual ~KFTrajectoryFitterESProducer(); 
  boost::shared_ptr<TrajectoryFitter> produce(const TrackingComponentsRecord &);
 private:
  boost::shared_ptr<TrajectoryFitter> _fitter;
  edm::ParameterSet pset_;
};


#endif




