#ifndef TrackingTools_TrackFitters_KFFittingSmootherESProducer_h
#define TrackingTools_TrackFitters_KFFittingSmootherESProducer_h

/** \class KFFittingSmootherESProducer
 *  ESProducer for the KFFittingSmoother
 *
 *  $Date: 2009/07/03 01:10:26 $
 *  $Revision: 1.3 $
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"
#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include <boost/shared_ptr.hpp>

class  KFFittingSmootherESProducer: public edm::ESProducer{
 public:
  KFFittingSmootherESProducer(const edm::ParameterSet & p);
  virtual ~KFFittingSmootherESProducer(); 
  boost::shared_ptr<TrajectoryFitter> produce(const TrajectoryFitterRecord &);
 private:
  boost::shared_ptr<TrajectoryFitter> _fitter;
  edm::ParameterSet pset_;
};


#endif




