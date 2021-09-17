#include "TrackingTools/RecoGeometry/interface/GlobalDetLayerGeometry.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/Utilities/interface/typelookup.h"

const DetLayer* GlobalDetLayerGeometry::idToLayer(const DetId& detId) const {
  if (detId.det() == DetId::Tracker)
    return tracker_->idToLayer(detId);
  else if (detId.det() == DetId::Muon)
    return muon_->idToLayer(detId);
  else if (mtd_ != nullptr && detId.det() == DetId::Forward && detId.subdetId() == FastTime)
    return mtd_->idToLayer(detId);
  else {
    throw cms::Exception("DetLayers")
        << "Error: called GlobalDetLayerGeometry::idToLayer() for a detId which is neither Tracker nor Muon "
        << (mtd_ == nullptr ? "" : "nor MTD ") << " det rawId " << detId.rawId() << " det " << detId.det()
        << " subdetId " << detId.subdetId();
  }
}

TYPELOOKUP_DATA_REG(GlobalDetLayerGeometry);
