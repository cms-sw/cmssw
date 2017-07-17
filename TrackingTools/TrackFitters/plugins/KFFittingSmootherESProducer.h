#ifndef TrackingTools_TrackFitters_KFFittingSmootherESProducer_h
#define TrackingTools_TrackFitters_KFFittingSmootherESProducer_h

/** \class KFFittingSmootherESProducer
 *  ESProducer for the KFFittingSmoother
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"
#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include <memory>

class  KFFittingSmootherESProducer: public edm::ESProducer{
 public:
  KFFittingSmootherESProducer(const edm::ParameterSet & p);
  virtual ~KFFittingSmootherESProducer(); 
  std::shared_ptr<TrajectoryFitter> produce(const TrajectoryFitterRecord &);
 private:
  std::shared_ptr<TrajectoryFitter> _fitter;
  edm::ParameterSet pset_;
};


#endif




