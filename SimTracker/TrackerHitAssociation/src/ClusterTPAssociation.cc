#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "FWCore/Utilities/interface/Exception.h"

void ClusterTPAssociation::checkMappedProductID(const edm::ProductID& id) const {
  if(id != mappedProductId_) {
    throw cms::Exception("InvalidReference") << "ClusterTPAssociation has TrackingParticles with ProductID " << mappedProductId_ << ", but got TrackingParticleRef/Handle/ProductID with ID " << id << ". This is typically caused by a configuration error.";
  }
}
