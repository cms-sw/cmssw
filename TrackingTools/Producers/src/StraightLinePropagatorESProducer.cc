#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include <string>
#include <memory>

class StraightLinePropagatorESProducer : public edm::ESProducer {
public:
  StraightLinePropagatorESProducer(const edm::ParameterSet &p);
  ~StraightLinePropagatorESProducer() override;
  std::unique_ptr<Propagator> produce(const TrackingComponentsRecord &);

private:
  edm::ParameterSet pset_;
};


using namespace edm;

StraightLinePropagatorESProducer::StraightLinePropagatorESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this, myname);
}

StraightLinePropagatorESProducer::~StraightLinePropagatorESProducer() {}

std::unique_ptr<Propagator> StraightLinePropagatorESProducer::produce(const TrackingComponentsRecord& iRecord) {
  //   if (_propagator){
  //     delete _propagator;
  //     _propagator = 0;
  //   }
  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield);
  std::string pdir = pset_.getParameter<std::string>("PropagationDirection");

  PropagationDirection dir = alongMomentum;

  if (pdir == "oppositeToMomentum")
    dir = oppositeToMomentum;
  else if (pdir == "anyDirection")
    dir = anyDirection;
  return std::make_unique<StraightLinePropagator>(&(*magfield), dir);
}

DEFINE_FWK_EVENTSETUP_MODULE(StraightLinePropagatorESProducer);
