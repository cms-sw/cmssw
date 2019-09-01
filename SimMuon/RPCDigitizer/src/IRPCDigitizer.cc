#include "SimMuon/RPCDigitizer/src/IRPCDigitizer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimFactory.h"
#include "SimMuon/RPCDigitizer/src/RPCSim.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

// default constructor allocates default wire and strip digitizers

IRPCDigitizer::IRPCDigitizer(const edm::ParameterSet& config)
    : theRPCSim{RPCSimFactory::get()->create(config.getParameter<std::string>("digiIRPCModel"),
                                             config.getParameter<edm::ParameterSet>("digiIRPCModelConfig"))} {
  theNoise = config.getParameter<bool>("doBkgNoise");
}

IRPCDigitizer::~IRPCDigitizer() = default;

void IRPCDigitizer::doAction(MixCollection<PSimHit>& simHits,
                             RPCDigiCollection& rpcDigis,
                             RPCDigiSimLinks& rpcDigiSimLink,
                             CLHEP::HepRandomEngine* engine) {
  theRPCSim->setRPCSimSetUp(theSimSetUp);

  // arrange the hits by roll
  std::map<int, edm::PSimHitContainer> hitMap;
  for (MixCollection<PSimHit>::MixItr hitItr = simHits.begin(); hitItr != simHits.end(); ++hitItr) {
    hitMap[hitItr->detUnitId()].push_back(*hitItr);
  }

  if (!theGeometry) {
    throw cms::Exception("Configuration")
        << "IRPCDigitizer requires the RPCGeometry \n which is not present in the configuration file.  You must add "
           "the service\n in the configuration file or remove the modules that require it.";
  }

  const std::vector<const RPCRoll*>& rpcRolls = theGeometry->rolls();
  for (auto r = rpcRolls.begin(); r != rpcRolls.end(); r++) {
    RPCDetId id = (*r)->id();
    const edm::PSimHitContainer& rollSimHits = hitMap[id];

    if ((*r)->isIRPC()) {
      theRPCSim->simulate(*r, rollSimHits, engine);

      if (theNoise) {
        theRPCSim->simulateNoise(*r, engine);
      }
    }

    theRPCSim->fillDigis((*r)->id(), rpcDigis);
    rpcDigiSimLink.insert(theRPCSim->rpcDigiSimLinks());
  }
}

const RPCRoll* IRPCDigitizer::findDet(int detId) const {
  assert(theGeometry != nullptr);
  const GeomDetUnit* detUnit = theGeometry->idToDetUnit(RPCDetId(detId));
  return dynamic_cast<const RPCRoll*>(detUnit);
}
