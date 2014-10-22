#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"
#include "DQMServices/Core/interface/DQMStore.h"

CSCBaseValidation::CSCBaseValidation(const edm::InputTag & inputTag)
: theInputTag(inputTag),
  theSimHitMap(0),
  theCSCGeometry(0)
{
}

const CSCLayer * CSCBaseValidation::findLayer(int detId) const {
  assert(theCSCGeometry != 0);
  const GeomDetUnit* detUnit = theCSCGeometry->idToDetUnit(CSCDetId(detId));
  return dynamic_cast<const CSCLayer *>(detUnit);
}
