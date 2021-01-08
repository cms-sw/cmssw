#include "GeantPropagatorESProducer.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackPropagation/Geant4e/interface/Geant4ePropagator.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include <memory>
#include <string>

using namespace edm;

GeantPropagatorESProducer::GeantPropagatorESProducer(const edm::ParameterSet &p) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  plimit_ = pset_.getParameter<double>("PropagationPtotLimit");
  setWhatProduced(this, myname);
}

GeantPropagatorESProducer::~GeantPropagatorESProducer() {}

std::unique_ptr<Propagator> GeantPropagatorESProducer::produce(const TrackingComponentsRecord &iRecord) {
  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield);

  std::string pdir = pset_.getParameter<std::string>("PropagationDirection");
  std::string particleName = pset_.getParameter<std::string>("ParticleName");

  PropagationDirection dir = alongMomentum;

  if (pdir == "oppositeToMomentum")
    dir = oppositeToMomentum;
  else if (pdir == "alongMomentum")
    dir = alongMomentum;
  else if (pdir == "anyDirection")
    dir = anyDirection;

  return std::make_unique<Geant4ePropagator>(&(*magfield), particleName, dir, plimit_);
}
