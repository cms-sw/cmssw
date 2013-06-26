#include "TrackingTools/KalmanUpdators/interface/TrackingRecHitPropagatorESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <string>
#include <memory>

using namespace edm;

TrackingRecHitPropagatorESProducer::TrackingRecHitPropagatorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

TrackingRecHitPropagatorESProducer::~TrackingRecHitPropagatorESProducer() {}

boost::shared_ptr<TrackingRecHitPropagator> 
TrackingRecHitPropagatorESProducer::produce(const TrackingComponentsRecord& iRecord){ 
   ESHandle<MagneticField> magfield;
   iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );	 
   theHitPropagator= boost::shared_ptr<TrackingRecHitPropagator>(new TrackingRecHitPropagator(magfield.product()));
   return theHitPropagator;
}


