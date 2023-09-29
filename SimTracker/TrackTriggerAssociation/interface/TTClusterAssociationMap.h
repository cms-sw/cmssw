/*! \class   TTClusterAssociationMap
 *  \brief   Stores association of Truth Particles (TP) to L1 Track-Trigger Clusters
 *
 *  \details Contains two maps. One associates each cluster to a vector
 *           of all TPs that made its hits. The other associates each TP 
 *           to a vector of all clusters it contributed to.
 *
 *           (The template structure is used to accomodate types
 *           other than PixelDigis, in case they are needed in future).
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *  (tidy up: Ian Tomalin, 2020)
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_ASSOCIATION_FORMAT_H
#define L1_TRACK_TRIGGER_CLUSTER_ASSOCIATION_FORMAT_H

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"  /// NOTE: this is needed even if it seems not
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

// Templated aliases
template <typename T>
using MapClusToVecTP = std::map<TTClusterRefT<T>, std::vector<TrackingParticlePtr>>;
template <typename T>
using MapTPToVecClus = std::map<TrackingParticlePtr, std::vector<TTClusterRefT<T>>>;

template <typename T>
class TTClusterAssociationMap {
public:
  /// Constructors
  TTClusterAssociationMap();

  /// Destructor
  ~TTClusterAssociationMap();

  /// Get/set cluster <-> truth association maps

  const MapClusToVecTP<T>& getTTClusterToTrackingParticlesMap() const { return clusterToTrackingParticleVectorMap_; }
  const MapTPToVecClus<T>& getTrackingParticleToTTClustersMap() const { return trackingParticleToClusterVectorMap_; }

  void setTTClusterToTrackingParticlesMap(const MapClusToVecTP<T>& aMap) { clusterToTrackingParticleVectorMap_ = aMap; }
  void setTrackingParticleToTTClustersMap(const MapTPToVecClus<T>& aMap) { trackingParticleToClusterVectorMap_ = aMap; }

  /// Get all TPs associated to a cluster
  const std::vector<TrackingParticlePtr>& findTrackingParticlePtrs(TTClusterRefT<T> aCluster) const;

  /// Get main TP associated to a cluster. (Non-NULL if isGenuine() below is true).
  const TrackingParticlePtr& findTrackingParticlePtr(TTClusterRefT<T> aCluster) const;

  // Get all clusters associated to TP.
  const std::vector<TTClusterRefT<T>>& findTTClusterRefs(TrackingParticlePtr aTrackingParticle) const;

  ///--- Get quality of L1 cluster based on truth info.
  /// (exactly 1 of following 3 functions is always true)

  /// Cluster "genuine": i.e. cluster associated to exactly 1 TP.
  /// (If other TPs are associated, but have in total < 1% of Pt of main TP,
  ///  or if they are null, then they are neglected here).
  bool isGenuine(TTClusterRefT<T> aCluster) const;
  /// Cluster "unknown": i.e. not associated with any TP.
  bool isUnknown(TTClusterRefT<T> aCluster) const;
  /// Cluster is not "genuine" or "unknown".
  bool isCombinatoric(TTClusterRefT<T> aCluster) const;

private:
  /// Data members
  MapClusToVecTP<T> clusterToTrackingParticleVectorMap_;
  MapTPToVecClus<T> trackingParticleToClusterVectorMap_;

  int nclus;

  // Allow functions to return reference to null.
  static const TrackingParticlePtr nullTrackingParticlePtr_;
  static const std::vector<TrackingParticlePtr> nullVecTrackingParticlePtr_;
  static const std::vector<TTClusterRefT<T>> nullVecClusterRef_;

};  /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

// Static constant data members.
template <typename T>
const TrackingParticlePtr TTClusterAssociationMap<T>::nullTrackingParticlePtr_;
template <typename T>
const std::vector<TrackingParticlePtr> TTClusterAssociationMap<T>::nullVecTrackingParticlePtr_;
template <typename T>
const std::vector<TTClusterRefT<T>> TTClusterAssociationMap<T>::nullVecClusterRef_;

/// Default Constructor
/// NOTE: to be used with setSomething(...) methods
template <typename T>
TTClusterAssociationMap<T>::TTClusterAssociationMap() {
  /// Set default data members
  nclus = 0;
}

/// Destructor
template <typename T>
TTClusterAssociationMap<T>::~TTClusterAssociationMap() {}

/// Operations
template <typename T>
const std::vector<TTClusterRefT<T>>& TTClusterAssociationMap<T>::findTTClusterRefs(
    TrackingParticlePtr aTrackingParticle) const {
  if (trackingParticleToClusterVectorMap_.find(aTrackingParticle) != trackingParticleToClusterVectorMap_.end()) {
    return trackingParticleToClusterVectorMap_.find(aTrackingParticle)->second;
  } else {
    return nullVecClusterRef_;
  }
}

template <typename T>
const std::vector<TrackingParticlePtr>& TTClusterAssociationMap<T>::findTrackingParticlePtrs(
    TTClusterRefT<T> aCluster) const {
  if (clusterToTrackingParticleVectorMap_.find(aCluster) != clusterToTrackingParticleVectorMap_.end()) {
    return clusterToTrackingParticleVectorMap_.find(aCluster)->second;
  } else {
    return nullVecTrackingParticlePtr_;
  }
}

template <typename T>
const TrackingParticlePtr& TTClusterAssociationMap<T>::findTrackingParticlePtr(TTClusterRefT<T> aCluster) const {
  if (this->isGenuine(aCluster)) {
    return this->findTrackingParticlePtrs(aCluster).at(0);
  } else {
    return nullTrackingParticlePtr_;
  }
}

/// MC truth
/// Table to define Genuine, Combinatoric and Unknown
///
/// N = number of NULL TP pointers
/// D = number of GOOD TP pointers different from each other
///
/// OLD DEFINITION
///
/// N / D--> | 0 | 1 | >1
/// ----------------------
/// 0        | U | G | C
/// ----------------------
/// >0       | U | C | C
///

/// NEW DEFINITION SV 060617
///
/// N / D--> | 0 | 1 | >1 (with 1 TP getting >99% of the total pT) | >1
/// -------------------------------------------------------------------
/// 0        | U | G | G                                           | C
/// -------------------------------------------------------------------
/// >0       | U | G | G                                           | C
///

template <typename T>
bool TTClusterAssociationMap<T>::isGenuine(TTClusterRefT<T> aCluster) const {
  /// Get the TrackingParticles
  const std::vector<TrackingParticlePtr>& theseTrackingParticles = this->findTrackingParticlePtrs(aCluster);

  /// If the vector is empty, then the cluster is UNKNOWN
  if (theseTrackingParticles.empty())
    return false;

  /// If we are here, it means there are some TrackingParticles
  unsigned int goodDifferentTPs = 0;
  std::vector<const TrackingParticle*> tpAddressVector;

  std::vector<float> tp_mom;

  float tp_tot = 0;

  /// Loop over the TrackingParticles
  for (const auto& tp : theseTrackingParticles) {
    /// Get the TrackingParticle
    const TrackingParticlePtr& curTP = tp;

    /// Count the NULL TrackingParticles
    if (curTP.isNull()) {
      tp_mom.push_back(0);
    } else {
      tp_mom.push_back(curTP.get()->p4().pt());
      tp_tot += curTP.get()->p4().pt();
    }
  }

  if (tp_tot == 0)
    return false;

  for (unsigned int itp = 0; itp < theseTrackingParticles.size(); itp++) {
    /// Get the TrackingParticle
    TrackingParticlePtr curTP = theseTrackingParticles.at(itp);

    /// Count the NULL TrackingParticles
    if (tp_mom.at(itp) > 0.01 * tp_tot) {
      /// Store the pointers (addresses) of the TrackingParticle
      /// to be able to count how many different there are
      tpAddressVector.push_back(curTP.get());
    }
  }

  /// Count how many different TrackingParticle there are
  std::sort(tpAddressVector.begin(), tpAddressVector.end());
  tpAddressVector.erase(std::unique(tpAddressVector.begin(), tpAddressVector.end()), tpAddressVector.end());
  goodDifferentTPs = tpAddressVector.size();

  return (goodDifferentTPs == 1);
}

template <typename T>
bool TTClusterAssociationMap<T>::isUnknown(TTClusterRefT<T> aCluster) const {
  /// Get the TrackingParticles
  const std::vector<TrackingParticlePtr>& theseTrackingParticles = this->findTrackingParticlePtrs(aCluster);

  /// If the vector is empty, then the cluster is UNKNOWN
  if (theseTrackingParticles.empty())
    return true;

  /// If we are here, it means there are some TrackingParticles
  unsigned int goodDifferentTPs = 0;
  std::vector<const TrackingParticle*> tpAddressVector;

  /// Loop over the TrackingParticles
  for (unsigned int itp = 0; itp < theseTrackingParticles.size(); itp++) {
    /// Get the TrackingParticle
    TrackingParticlePtr curTP = theseTrackingParticles.at(itp);

    /// Count the non-NULL TrackingParticles
    if (!curTP.isNull()) {
      /// Store the pointers (addresses) of the TrackingParticle
      /// to be able to count how many different there are
      tpAddressVector.push_back(curTP.get());
    }
  }

  /// Count how many different TrackingParticle there are
  std::sort(tpAddressVector.begin(), tpAddressVector.end());
  tpAddressVector.erase(std::unique(tpAddressVector.begin(), tpAddressVector.end()), tpAddressVector.end());
  goodDifferentTPs = tpAddressVector.size();

  /// UNKNOWN means no good TP is found
  return (goodDifferentTPs == 0);
}

template <typename T>
bool TTClusterAssociationMap<T>::isCombinatoric(TTClusterRefT<T> aCluster) const {
  /// Get the TrackingParticles
  const std::vector<TrackingParticlePtr>& theseTrackingParticles = this->findTrackingParticlePtrs(aCluster);

  /// If the vector is empty, then the cluster is UNKNOWN
  if (theseTrackingParticles.empty())
    return false;

  bool genuineClu = this->isGenuine(aCluster);
  bool unknownClu = this->isUnknown(aCluster);

  if (genuineClu || unknownClu)
    return false;

  return true;

  /// If we are here, it means there are some TrackingParticles
  unsigned int goodDifferentTPs = 0;
  std::vector<const TrackingParticle*> tpAddressVector;

  /// Loop over the TrackingParticles
  for (unsigned int itp = 0; itp < theseTrackingParticles.size(); itp++) {
    /// Get the TrackingParticle
    TrackingParticlePtr curTP = theseTrackingParticles.at(itp);

    /// Count the NULL TrackingParticles
    if (!curTP.isNull()) {
      /// Store the pointers (addresses) of the TrackingParticle
      /// to be able to count how many different there are
      tpAddressVector.push_back(curTP.get());
    }
  }

  /// Count how many different TrackingParticle there are
  std::sort(tpAddressVector.begin(), tpAddressVector.end());
  tpAddressVector.erase(std::unique(tpAddressVector.begin(), tpAddressVector.end()), tpAddressVector.end());
  goodDifferentTPs = tpAddressVector.size();

  return (goodDifferentTPs > 1);
}

#endif
