#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagatorESProducer.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

SteppingHelixPropagatorESProducer::SteppingHelixPropagatorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

SteppingHelixPropagatorESProducer::~SteppingHelixPropagatorESProducer() {}

boost::shared_ptr<Propagator> 
SteppingHelixPropagatorESProducer::produce(const TrackingComponentsRecord & iRecord){ 
//   if (_propagator){
//     delete _propagator;
//     _propagator = 0;
//   }
  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );

  std::string pdir = pset_.getParameter<std::string>("PropagationDirection");

  PropagationDirection dir = alongMomentum;
  
  if (pdir == "oppositeToMomentum") dir = oppositeToMomentum;
  if (pdir == "alongMomentum") dir = alongMomentum;
  if (pdir == "anyDirection") dir = anyDirection;
  
  SteppingHelixPropagator* shProp = new SteppingHelixPropagator(&(*magfield), dir);

  bool haveX0Corr = pset_.getParameter<bool>("ApplyRadX0Correction");
  shProp->applyRadX0Correction(haveX0Corr);

  bool assumeNoMaterial = pset_.getParameter<bool>("AssumeNoMaterial");
  shProp->setMaterialMode(assumeNoMaterial);

  bool noErrorPropagation = pset_.getParameter<bool>("NoErrorPropagation");
  shProp->setNoErrorPropagation(noErrorPropagation);

  bool debugMode = pset_.getParameter<bool>("debug");
  shProp->setDebug(debugMode);

  bool useMagVolumes = pset_.getParameter<bool>("useMagVolumes");
  shProp->setUseMagVolumes(useMagVolumes);

  bool useMatVolumes = pset_.getParameter<bool>("useMatVolumes");
  shProp->setUseMatVolumes(useMatVolumes);

  bool useIsYokeFlag = pset_.getParameter<bool>("useIsYokeFlag");
  shProp->setUseIsYokeFlag(useIsYokeFlag);

  bool returnTangentPlane = pset_.getParameter<bool>("returnTangentPlane");
  shProp->setReturnTangentPlane(returnTangentPlane);

  bool sendLogWarning = pset_.getParameter<bool>("sendLogWarning");
  shProp->setSendLogWarning(sendLogWarning);

  bool useTuningForL2Speed = pset_.getParameter<bool>("useTuningForL2Speed");
  shProp->setUseTuningForL2Speed(useTuningForL2Speed);

  _propagator  = boost::shared_ptr<Propagator>(shProp);
  return _propagator;
}
