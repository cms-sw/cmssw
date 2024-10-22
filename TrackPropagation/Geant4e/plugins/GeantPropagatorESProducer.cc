#include "GeantPropagatorESProducer.h"

#include "TrackPropagation/Geant4e/interface/Geant4ePropagator.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include <memory>
#include <string>

using namespace edm;

GeantPropagatorESProducer::GeantPropagatorESProducer(const edm::ParameterSet &p)
    : magFieldToken_(setWhatProduced(this, p.getParameter<std::string>("ComponentName"))
                         .consumesFrom<MagneticField, IdealMagneticFieldRecord>(edm::ESInputTag("", ""))) {
  pset_ = p;
  plimit_ = pset_.getParameter<double>("PropagationPtotLimit");
}

GeantPropagatorESProducer::~GeantPropagatorESProducer() {}

std::unique_ptr<Propagator> GeantPropagatorESProducer::produce(const TrackingComponentsRecord &iRecord) {
  std::string pdir = pset_.getParameter<std::string>("PropagationDirection");
  std::string particleName = pset_.getParameter<std::string>("ParticleName");

  PropagationDirection dir = alongMomentum;

  if (pdir == "oppositeToMomentum") {
    dir = oppositeToMomentum;
  } else if (pdir == "alongMomentum") {
    dir = alongMomentum;
  } else if (pdir == "anyDirection") {
    dir = anyDirection;
  }

  return std::make_unique<Geant4ePropagator>(&(iRecord.get(magFieldToken_)), particleName, dir, plimit_);
}
