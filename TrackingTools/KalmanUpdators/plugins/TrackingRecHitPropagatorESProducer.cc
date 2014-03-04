#include "TrackingTools/KalmanUpdators/interface/TrackingRecHitPropagatorESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include <FWCore/Utilities/interface/ESInputTag.h>

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
   std::string mfName = "";
   if (pset_.exists("SimpleMagneticField"))
     mfName = pset_.getParameter<std::string>("SimpleMagneticField");
   iRecord.getRecord<IdealMagneticFieldRecord>().get(mfName,magfield);
   //   edm::ESInputTag mfESInputTag(mfName);
   //   iRecord.getRecord<IdealMagneticFieldRecord>().get(mfESInputTag,magfield);
   theHitPropagator= boost::shared_ptr<TrackingRecHitPropagator>(new TrackingRecHitPropagator(magfield.product()));
   return theHitPropagator;
}


