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
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magToken_;
};

using namespace edm;

StraightLinePropagatorESProducer::StraightLinePropagatorESProducer(const edm::ParameterSet& p)
    : dir_{[](std::string const& pdir) {
        if (pdir == "oppositeToMomentum")
          return oppositeToMomentum;
        else if (pdir == "anyDirection")
          return anyDirection;
        return alongMomentum;
      }(p.getParameter<std::string>("PropagationDirection"))},
      magToken_{setWhatProduced(this, p.getParameter<std::string>("ComponentName"))
                    .consumesFrom<MagneticField, IdealMagneticFieldRecord>()}

{}

StraightLinePropagatorESProducer::~StraightLinePropagatorESProducer() {}

std::unique_ptr<Propagator> StraightLinePropagatorESProducer::produce(const TrackingComponentsRecord& iRecord) {
  return std::make_unique<StraightLinePropagator>(&iRecord.get(magToken_), dir_);
}

DEFINE_FWK_EVENTSETUP_MODULE(StraightLinePropagatorESProducer);
