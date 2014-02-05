#include "TrackingTools/Producers/interface/AnalyticalPropagatorESProducer.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

AnalyticalPropagatorESProducer::AnalyticalPropagatorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

AnalyticalPropagatorESProducer::~AnalyticalPropagatorESProducer() {}

boost::shared_ptr<Propagator> 
AnalyticalPropagatorESProducer::produce(const TrackingComponentsRecord & iRecord){ 
//   if (_propagator){
//     delete _propagator;
//     _propagator = 0;
//   }
  ESHandle<MagneticField> magfield;
  if (pset_.exists("SimpleMagneticField"))
    iRecord.getRecord<IdealMagneticFieldRecord>().get(pset_.getParameter<std::string>("SimpleMagneticField"), magfield);
  else 
    iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield);

  std::string pdir = pset_.getParameter<std::string>("PropagationDirection");
  double dphiCut   = pset_.getParameter<double>("MaxDPhi");   

  PropagationDirection dir = alongMomentum;
  
  if (pdir == "oppositeToMomentum") dir = oppositeToMomentum;
  if (pdir == "alongMomentum") dir = alongMomentum;
  if (pdir == "anyDirection") dir = anyDirection;
  
  _propagator  = boost::shared_ptr<Propagator>(new AnalyticalPropagator(&(*magfield), dir,dphiCut));
  return _propagator;
}


