#ifndef FastTimingSimProducers_FastTimingCommon_FTLDigitizerBase_h
#define FastTimingSimProducers_FastTimingCommon_FTLDigitizerBase_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "SimFastTiming/FastTimingCommon/interface/FTLDigitizerTypes.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <string>

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

class PileUpEventPrincipal;

class FTLDigitizerBase {
public:
  FTLDigitizerBase(const edm::ParameterSet& config,
                   edm::ProducesCollector producesCollector,
                   edm::ConsumesCollector& iC)
      : inputSimHits_(config.getParameter<edm::InputTag>("inputSimHits")),
        digiCollection_(config.getParameter<std::string>("digiCollectionTag")),
        mySubDet_(FastTime),
        verbosity_(config.getUntrackedParameter<uint32_t>("verbosity", 0)),
        refSpeed_(0.1 * CLHEP::c_light),
        name_(config.getParameter<std::string>("digitizerName")) {
    iC.consumes<std::vector<PSimHit> >(inputSimHits_);
    producesCollector.produces<FTLDigiCollection>(digiCollection_);
  }

  virtual ~FTLDigitizerBase() {}

  /**
     @short handle SimHit accumulation
  */
  virtual void accumulate(edm::Event const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) = 0;
  virtual void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) = 0;
  virtual void accumulate(edm::Handle<edm::PSimHitContainer> const& hits,
                          int bxCrossing,
                          CLHEP::HepRandomEngine* hre) = 0;

  /**
     @short actions at the start/end of event
  */
  virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c) = 0;
  virtual void finalizeEvent(edm::Event& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre) = 0;

  /**
     @short actions at the start/end of run
  */
  virtual void beginRun(const edm::EventSetup& es) = 0;
  virtual void endRun() = 0;

  const std::string& name() const { return name_; }

protected:
  //input/output names
  const edm::InputTag inputSimHits_;
  const std::string digiCollection_;

  //subdetector id
  const ForwardSubdetector mySubDet_;

  //misc switches
  const uint32_t verbosity_;

  //reference speed to evaluate time of arrival at the sensititive detector, assuming the center of CMS
  const float refSpeed_;

private:
  std::string name_;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<FTLDigitizerBase*(
    const edm::ParameterSet&, edm::ProducesCollector, edm::ConsumesCollector&)>
    FTLDigitizerFactory;

#endif
