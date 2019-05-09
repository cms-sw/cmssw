#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimMuon/CSCDigitizer/src/CSCNeutronReader.h"

void CSCNeutronReader::addHits(std::map<int, edm::PSimHitContainer> &hitMap, CLHEP::HepRandomEngine *engine) {
  std::vector<int> chambersDone;

  std::map<int, edm::PSimHitContainer> signalHits = hitMap;
  for (std::map<int, edm::PSimHitContainer>::const_iterator signalHitItr = signalHits.begin();
       signalHitItr != signalHits.end();
       ++signalHitItr) {
    int chamberIndex = chamberId(signalHitItr->first);

    // see if this chamber has been done yet
    if (find(chambersDone.begin(), chambersDone.end(), chamberIndex) == chambersDone.end()) {
      edm::PSimHitContainer neutronHits;
      generateChamberNoise(chamberType(chamberIndex), chamberIndex, neutronHits, engine);

      // add these hits to the original map
      for (edm::PSimHitContainer::const_iterator neutronHitItr = neutronHits.begin();
           neutronHitItr != neutronHits.end();
           ++neutronHitItr) {
        uint32_t layerId = neutronHitItr->detUnitId();
        hitMap[layerId].push_back(*neutronHitItr);
      }
      // mark chamber as done
      chambersDone.push_back(chamberIndex);
    }
  }
}

int CSCNeutronReader::detId(int chamberIndex, int localDetId) {
  // add the layer bits
  return chamberIndex + localDetId;
}

int CSCNeutronReader::localDetId(int globalDetId) const { return CSCDetId(globalDetId).layer(); }

int CSCNeutronReader::chamberType(int globalDetId) const {
  CSCDetId id(globalDetId);
  return CSCChamberSpecs::whatChamberType(id.station(), id.ring());
}

int CSCNeutronReader::chamberId(int globalDetId) const { return CSCDetId(globalDetId).chamberId().rawId(); }
