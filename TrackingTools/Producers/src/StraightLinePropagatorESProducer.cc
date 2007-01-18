#include "TrackingTools/Producers/interface/StraightLinePropagatorESProducer.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

StraightLinePropagatorESProducer::StraightLinePropagatorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

StraightLinePropagatorESProducer::~StraightLinePropagatorESProducer() {}

boost::shared_ptr<Propagator> 
StraightLinePropagatorESProducer::produce(const TrackingComponentsRecord & iRecord){ 
//   if (_propagator){
//     delete _propagator;
//     _propagator = 0;
//   }
  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );
 std::string pdir = pset_.getParameter<std::string>("PropagationDirection");

  PropagationDirection dir = alongMomentum;

  if (pdir == "oppositeToMomentum, alongMomentum, anyDirection")
    if (pdir == "oppositeToMomentum") dir = oppositeToMomentum;
    if (pdir == "alongMomentum") dir = alongMomentum;
    if (pdir == "anyDirection") dir = anyDirection;
  _propagator  = boost::shared_ptr<Propagator>(new StraightLinePropagator(&(*magfield),dir));
  return _propagator;
}


