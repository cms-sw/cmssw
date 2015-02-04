#include "GeantPropagatorESProducer.h"
#include "TrackPropagation/Geant4e/interface/Geant4ePropagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

GeantPropagatorESProducer::GeantPropagatorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

GeantPropagatorESProducer::~GeantPropagatorESProducer() {}

boost::shared_ptr<Propagator> 
GeantPropagatorESProducer::produce(const TrackingComponentsRecord & iRecord){ 

  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );

  std::string pdir = pset_.getParameter<std::string>("PropagationDirection");
  std::string particleName = pset_.getParameter<std::string>("ParticleName");

  PropagationDirection dir = alongMomentum;
  
  if (pdir == "oppositeToMomentum") dir = oppositeToMomentum;
  if (pdir == "alongMomentum") dir = alongMomentum;
  if (pdir == "anyDirection") dir = anyDirection;
  
  _propagator  = boost::shared_ptr<Propagator>(new Geant4ePropagator(&(*magfield),particleName,dir));
  return _propagator;
}


