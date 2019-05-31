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
  StraightLinePropagatorESProducer(const edm::ParameterSet& p);
  ~StraightLinePropagatorESProducer() override;
  std::unique_ptr<Propagator> produce(const TrackingComponentsRecord&);

private:
  const PropagationDirection dir_;
};

using namespace edm;

StraightLinePropagatorESProducer::StraightLinePropagatorESProducer(const edm::ParameterSet& p)
    : dir_{[](std::string const& pdir) {
        if (pdir == "oppositeToMomentum")
          return oppositeToMomentum;
        else if (pdir == "anyDirection")
          return anyDirection;
        return alongMomentum;
      }(p.getParameter<std::string>("PropagationDirection"))} {
  std::string myname = p.getParameter<std::string>("ComponentName");
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
  return std::make_unique<StraightLinePropagator>(&(*magfield), dir_);
}

DEFINE_FWK_EVENTSETUP_MODULE(StraightLinePropagatorESProducer);
