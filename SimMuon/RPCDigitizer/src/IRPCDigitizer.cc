#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimMuon/RPCDigitizer/src/IRPCDigitizer.h"
#include "SimMuon/RPCDigitizer/src/RPCSim.h"
#include "SimMuon/RPCDigitizer/src/RPCSimFactory.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

IRPCDigitizer::IRPCDigitizer(const edm::ParameterSet& config)
    : theRPCSim_{RPCSimFactory::get()->create(config.getParameter<std::string>("digiIRPCModel"),
                                              config.getParameter<edm::ParameterSet>("digiIRPCModelConfig"))} {
  theNoise_ = config.getParameter<bool>("doBkgNoise");
}

IRPCDigitizer::~IRPCDigitizer() = default;

void IRPCDigitizer::doAction(MixCollection<PSimHit>& simHits,
                             IRPCDigiCollection& rpcDigis,
                             RPCDigiSimLinks& rpcDigiSimLink,
                             CLHEP::HepRandomEngine* engine) {
  theRPCSim_->setRPCSimSetUp(theSimSetUp_);

  // arrange the hits by roll
  std::map<int, edm::PSimHitContainer> hitMap;
  for (MixCollection<PSimHit>::MixItr hitItr = simHits.begin(); hitItr != simHits.end(); ++hitItr) {
    hitMap[hitItr->detUnitId()].push_back(*hitItr);
  }

  if (!theGeometry_) {
    throw cms::Exception("Configuration")
        << "IRPCDigitizer requires the RPCGeometry \n which is not present in the configuration file.  You must add "
           "the service\n in the configuration file or remove the modules that require it.";
  }

  const std::vector<const RPCRoll*>& rpcRolls = theGeometry_->rolls();
  for (auto r = rpcRolls.begin(); r != rpcRolls.end(); r++) {
    RPCDetId id = (*r)->id();
    const edm::PSimHitContainer& rollSimHits = hitMap[id];

    if ((*r)->isIRPC()) {
      theRPCSim_->simulate(*r, rollSimHits, engine);

      if (theNoise_) {
        theRPCSim_->simulateNoise(*r, engine);
      }
    }

    theRPCSim_->fillDigis((*r)->id(), rpcDigis);
    if (rpcDigiSimLink.find((theRPCSim_->rpcDigiSimLinks()).detId()) == rpcDigiSimLink.end()) {
      rpcDigiSimLink.insert(theRPCSim_->rpcDigiSimLinks());
    }
  }
}

const RPCRoll* IRPCDigitizer::findDet(int detId) const {
  assert(theGeometry_ != nullptr);
  const GeomDetUnit* detUnit = theGeometry_->idToDetUnit(RPCDetId(detId));
  return dynamic_cast<const RPCRoll*>(detUnit);
}
