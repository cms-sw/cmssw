/*! \class   TTStubAssociationMap
 *  \brief   Stores association of Truth Particles (TP) to L1 Track-Trigger Stubs
 *
 *  \details Contains two maps. One associates each stub to its principle TP.
 *           (i.e. Not to all TP that contributed to it). 
 *           The other associates each TP to a vector of all stubs 
 *           it contributed to. The two maps are therefore not
 *           forward-backward symmetric.
 *
 *           (The template structure is used to accomodate types
 *           other than PixelDigis, in case they are needed in future). 
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *  (tidy up: Ian Tomalin, 2020)
 */

#ifndef L1_TRACK_TRIGGER_STUB_ASSOCIATION_FORMAT_H
#define L1_TRACK_TRIGGER_STUB_ASSOCIATION_FORMAT_H

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
//#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"  /// NOTE: this is needed even if it seems not
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"

// Templated aliases
template <typename T>
using MapStubToTP = std::map<TTStubRefT<T>, TrackingParticlePtr>;
template <typename T>
using MapTPToVecStub = std::map<TrackingParticlePtr, std::vector<TTStubRefT<T>>>;

template <typename T>
class TTStubAssociationMap {
public:
  /// Constructors
  TTStubAssociationMap();

  /// Destructor
  ~TTStubAssociationMap();

  /// Get/set stub <-> truth association maps

  const MapStubToTP<T>& getTTStubToTrackingParticleMap() const { return stubToTrackingParticleMap_; }
  const MapTPToVecStub<T>& getTrackingParticleToTTStubsMap() const { return trackingParticleToStubVectorMap_; }

  void setTTStubToTrackingParticleMap(const MapStubToTP<T>& aMap) { stubToTrackingParticleMap_ = aMap; }
  void setTrackingParticleToTTStubsMap(const MapTPToVecStub<T>& aMap) { trackingParticleToStubVectorMap_ = aMap; }

  /// Set cluster <-> truth association object.
  void setTTClusterAssociationMap(edm::RefProd<TTClusterAssociationMap<T>> aCluAssoMap) {
    theClusterAssociationMap_ = aCluAssoMap;
  }

  /// Get principle TP associated to a stub. (Non-NULL if isGenuine() below is true).
  /// (N.B. There is no function returning all TP associated to a stub).
  /// (P.S. As this function only returns principle TP, it is not used when constructing
  ///  the TTTrackAssociationMap).
  const TrackingParticlePtr& findTrackingParticlePtr(TTStubRefT<T> aStub) const;

  /// Get all stubs associated to a TP.
  /// (Even if the TP just contributes to one cluster in stub,
  /// and even if their are other such TP, it is still listed here).
  const std::vector<TTStubRefT<T>>& findTTStubRefs(TrackingParticlePtr aTrackingParticle) const;

  ///--- Get quality of stub based on truth info.
  /// (N.B. Both genuine & combinatoric stubs contribute to "genuine" L1 tracks
  ///  associated by TTTrackAssociationMap).
  /// (exactly 1 of following 3 functions is always true)

  /// If both clusters are unknown, the stub is "unknown".
  /// If only one cluster is unknown, the stub is combinatoric.
  /// If both clusters are genuine, and are associated to the same (main) TrackingParticle,
  /// the stub is "genuine".
  /// If both clusters are genuine, but are associated to different (main) TrackingParticles,
  /// the stub is "combinatoric".
  /// If one cluster is combinatoric and the other is genuine/combinatoric, and they both share exactly
  /// one TrackingParticle in common, then the stub is "genuine". (The clusters can have other
  /// TrackingParticles besides the shared one, as long as these are not shared). If instead the clusters
  /// share 0 or ≥2 TrackingParticles in common, then the stub is "combinatoric".

  bool isGenuine(TTStubRefT<T> aStub) const;
  bool isCombinatoric(TTStubRefT<T> aStub) const;
  bool isUnknown(TTStubRefT<T> aStub) const;

private:
  /// Data members
  MapStubToTP<T> stubToTrackingParticleMap_;
  MapTPToVecStub<T> trackingParticleToStubVectorMap_;
  edm::RefProd<TTClusterAssociationMap<T>> theClusterAssociationMap_;

  // Allow functions to return reference to null.
  static const TrackingParticlePtr nullTrackingParticlePtr_;
  static const std::vector<TTStubRefT<T>> nullVecStubRef_;

};  /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

// Static constant data members.
template <typename T>
const TrackingParticlePtr TTStubAssociationMap<T>::nullTrackingParticlePtr_;
template <typename T>
const std::vector<TTStubRefT<T>> TTStubAssociationMap<T>::nullVecStubRef_;

/// Default Constructor
/// NOTE: to be used with setSomething(...) methods
template <typename T>
TTStubAssociationMap<T>::TTStubAssociationMap() {
  /// Set default data members
}

/// Destructor
template <typename T>
TTStubAssociationMap<T>::~TTStubAssociationMap() {}

/// Operations
template <typename T>
const TrackingParticlePtr& TTStubAssociationMap<T>::findTrackingParticlePtr(TTStubRefT<T> aStub) const {
  if (stubToTrackingParticleMap_.find(aStub) != stubToTrackingParticleMap_.end()) {
    return stubToTrackingParticleMap_.find(aStub)->second;
  } else {
    return nullTrackingParticlePtr_;
  }
}

template <typename T>
const std::vector<TTStubRefT<T>>& TTStubAssociationMap<T>::findTTStubRefs(TrackingParticlePtr aTrackingParticle) const {
  if (trackingParticleToStubVectorMap_.find(aTrackingParticle) != trackingParticleToStubVectorMap_.end()) {
    return trackingParticleToStubVectorMap_.find(aTrackingParticle)->second;
  } else {
    return nullVecStubRef_;
  }
}

/// MC truth
template <typename T>
bool TTStubAssociationMap<T>::isGenuine(TTStubRefT<T> aStub) const {
  /// Check if there is a SimTrack
  if ((this->findTrackingParticlePtr(aStub)).isNull())
    return false;

  return true;
}

template <typename T>
bool TTStubAssociationMap<T>::isCombinatoric(TTStubRefT<T> aStub) const {
  /// Defined by exclusion
  if (this->isGenuine(aStub))
    return false;

  if (this->isUnknown(aStub))
    return false;

  return true;
}

template <typename T>
bool TTStubAssociationMap<T>::isUnknown(TTStubRefT<T> aStub) const {
  /// UNKNOWN means that both clusters are unknown

  /// Sanity check
  if (theClusterAssociationMap_.isNull()) {
    return true;
  }

  if (theClusterAssociationMap_->isUnknown(aStub->clusterRef(0)) &&
      theClusterAssociationMap_->isUnknown(aStub->clusterRef(1)))
    return true;

  return false;
}

#endif
