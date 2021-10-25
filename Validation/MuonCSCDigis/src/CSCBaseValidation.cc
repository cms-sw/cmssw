#include "DQMServices/Core/interface/DQMStore.h"
#include "Validation/MuonCSCDigis/interface/CSCBaseValidation.h"

CSCBaseValidation::CSCBaseValidation(const edm::ParameterSet &ps)
    : doSim_(ps.getParameter<bool>("doSim")), theSimHitMap(nullptr), theCSCGeometry(nullptr) {
  const auto &simTrack = ps.getParameter<edm::ParameterSet>("simTrack");
  simTrackMinPt_ = simTrack.getParameter<double>("minPt");
  simTrackMinEta_ = simTrack.getParameter<double>("minEta");
  simTrackMaxEta_ = simTrack.getParameter<double>("maxEta");
}

const CSCLayer *CSCBaseValidation::findLayer(int detId) const {
  assert(theCSCGeometry != nullptr);
  const GeomDetUnit *detUnit = theCSCGeometry->idToDetUnit(CSCDetId(detId));
  return dynamic_cast<const CSCLayer *>(detUnit);
}

bool CSCBaseValidation::isSimTrackGood(const SimTrack &t) const {
  // SimTrack selection
  if (t.noVertex())
    return false;
  if (t.noGenpart())
    return false;
  // only muons
  if (std::abs(t.type()) != 13)
    return false;
  // pt selection
  if (t.momentum().pt() < simTrackMinPt_)
    return false;
  // eta selection
  const float eta(std::abs(t.momentum().eta()));
  if (eta > simTrackMaxEta_ || eta < simTrackMinEta_)
    return false;
  return true;
}
