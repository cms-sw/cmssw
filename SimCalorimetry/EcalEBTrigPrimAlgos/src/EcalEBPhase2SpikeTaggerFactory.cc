///
/// \class EcalEBPhase2SpikeTaggerFactory
///
/// This class implements the ECAL Phase 2 spike tagger algorithm factory.
/// Based on a type string and version it selects the appropriate algorithm.
///

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2SpikeTaggerFactory.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2SpikeTaggerLDV1.h"

EcalEBPhase2SpikeTaggerFactory::ReturnType EcalEBPhase2SpikeTaggerFactory::create(std::string const& algoType,
                                                                                  uint32_t version,
                                                                                  edm::ConsumesCollector& cc,
                                                                                  bool debug) {
  ReturnType algo;

  // factory
  if (algoType == "ld") {
    if (version >= 1) {
      edm::LogInfo("EcalEBPhase2SpikeTaggerFactory") << "Creating LD spike tagger algo version " << version;
      algo = std::make_unique<EcalEBPhase2SpikeTaggerLDV1>(cc, debug);
    } else {
      throw cms::Exception("Unknown algo version") << "No ECAL LD spike tagger algo to create for version " << version;
    }
  } else {
    throw cms::Exception("Unknown algorithm") << "Unknown ECAL spike tagger algo type '" << algoType << "'";
  }

  return algo;
}
