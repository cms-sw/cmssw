#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"
#include "DQMServices/Core/interface/DQMStore.h"

CSCBaseValidation::CSCBaseValidation(DQMStore* dbe, const edm::InputTag & inputTag)
: dbe_(dbe),
  theInputTag(inputTag),
  theSimHitMap(0),
  theCSCGeometry(0)
{
}


const CSCLayer * CSCBaseValidation::findLayer(int detId) const {
  assert(theCSCGeometry != 0);
  const GeomDetUnit* detUnit = theCSCGeometry->idToDetUnit(CSCDetId(detId));
  return dynamic_cast<const CSCLayer *>(detUnit);
}

