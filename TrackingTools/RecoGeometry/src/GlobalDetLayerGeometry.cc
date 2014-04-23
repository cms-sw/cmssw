#include "TrackingTools/RecoGeometry/interface/GlobalDetLayerGeometry.h"
#include "FWCore/Utilities/interface/typelookup.h"

const DetLayer* 
GlobalDetLayerGeometry::idToLayer(const DetId& detId) const{
  if(detId.det() ==1) return tracker_->idToLayer(detId);
  else if(detId.det() ==2) return muon_->idToLayer(detId);
  else{
    throw cms::Exception("DetLayers") 
      << "Error: called GlobalDetLayerGeometry::idToLayer() for a detId which is neither Tracker nor Muon " << detId;      
  }

}


TYPELOOKUP_DATA_REG(GlobalDetLayerGeometry);
