#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "FWCore/Utilities/interface/Exception.h"

void ClusterTPAssociation::checkKeyProductID(const edm::ProductID& id) const {
  if(std::find(std::begin(keyProductIDs_), std::end(keyProductIDs_), id) == std::end(keyProductIDs_)) {
    auto e = cms::Exception("InvalidReference");
    e << "ClusterTPAssociation has OmniClusterRefs with ProductIDs ";
    for(size_t i=0; i<keyProductIDs_.size(); ++i) {
      e << keyProductIDs_[i];
      if(i < keyProductIDs_.size()-1) {
	e << ",";
      }
    }
    e << " but got OmniClusterRef/ProductID with ID " << id << ". This is typically caused by a configuration error.";
    throw e;
  }
}

void ClusterTPAssociation::checkMappedProductID(const edm::ProductID& id) const {
  if(id != mappedProductId_) {
    throw cms::Exception("InvalidReference") << "ClusterTPAssociation has TrackingParticles with ProductID " << mappedProductId_ << ", but got TrackingParticleRef/Handle/ProductID with ID " << id << ". This is typically caused by a configuration error.";
  }
}
