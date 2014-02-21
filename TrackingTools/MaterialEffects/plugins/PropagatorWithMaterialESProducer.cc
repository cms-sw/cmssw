#include "PropagatorWithMaterialESProducer.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include <FWCore/Utilities/interface/ESInputTag.h>

#include <string>
#include <memory>

using namespace edm;

PropagatorWithMaterialESProducer::PropagatorWithMaterialESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

PropagatorWithMaterialESProducer::~PropagatorWithMaterialESProducer() {}

boost::shared_ptr<Propagator> 
PropagatorWithMaterialESProducer::produce(const TrackingComponentsRecord & iRecord){ 
//   if (_propagator){
//     delete _propagator;
//     _propagator = 0;
//   }

  std::string pdir = pset_.getParameter<std::string>("PropagationDirection");
  double mass      = pset_.getParameter<double>("Mass");
  double maxDPhi   = pset_.getParameter<double>("MaxDPhi");
  bool useRK       = pset_.getParameter<bool>("useRungeKutta");
  bool useOldAnalPropLogic = pset_.existsAs<bool>("useOldAnalPropLogic") ? 
    pset_.getParameter<bool>("useOldAnalPropLogic") : true;
  double ptMin     = pset_.existsAs<double>("ptMin") ? pset_.getParameter<double>("ptMin") : -1.0;

  ESHandle<MagneticField> magfield;
  std::string mfName = "";
  if (pset_.exists("SimpleMagneticField"))
    mfName = pset_.getParameter<std::string>("SimpleMagneticField");
  iRecord.getRecord<IdealMagneticFieldRecord>().get(mfName,magfield);
  //  edm::ESInputTag mfESInputTag(mfName);
  //  iRecord.getRecord<IdealMagneticFieldRecord>().get(mfESInputTag,magfield);
  //fixme check that useRK is false when using SimpleMagneticField 

  PropagationDirection dir = alongMomentum;
  
  if (pdir == "oppositeToMomentum") dir = oppositeToMomentum;
  if (pdir == "alongMomentum") dir = alongMomentum;
  if (pdir == "anyDirection") dir = anyDirection;
  
  _propagator  = boost::shared_ptr<Propagator>(new PropagatorWithMaterial(dir, mass, &(*magfield),
									  maxDPhi,useRK,ptMin,
									  useOldAnalPropLogic));
  return _propagator;
}


