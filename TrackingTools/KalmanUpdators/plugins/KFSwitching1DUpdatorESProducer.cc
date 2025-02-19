#include "TrackingTools/KalmanUpdators/interface/KFSwitching1DUpdatorESProducer.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

KFSwitching1DUpdatorESProducer::KFSwitching1DUpdatorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

KFSwitching1DUpdatorESProducer::~KFSwitching1DUpdatorESProducer() {}

boost::shared_ptr<TrajectoryStateUpdator> 
KFSwitching1DUpdatorESProducer::produce(const TrackingComponentsRecord & iRecord){ 
//   if (_updator){
//     delete _updator;
//     _updator = 0;
//   }
  
  _updator  = boost::shared_ptr<TrajectoryStateUpdator>(new KFSwitching1DUpdator(&pset_));
  return _updator;
}


