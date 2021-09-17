#include "DQMServices/Core/interface/DQMStore.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

CSCBaseValidation::CSCBaseValidation(const edm::ParameterSet &ps)
    : doSim_(ps.getParameter<bool>("doSim")), theSimHitMap(nullptr), theCSCGeometry(nullptr) {}

const CSCLayer *CSCBaseValidation::findLayer(int detId) const {
  assert(theCSCGeometry != nullptr);
  const GeomDetUnit *detUnit = theCSCGeometry->idToDetUnit(CSCDetId(detId));
  return dynamic_cast<const CSCLayer *>(detUnit);
}
