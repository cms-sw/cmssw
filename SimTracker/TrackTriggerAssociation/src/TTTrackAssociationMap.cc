/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"

/// Operations
template <>
const TrackingParticlePtr& TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::findTrackingParticlePtr(
    TTTrackPtr aTrack) const {
  if (trackToTrackingParticleMap_.find(aTrack) != trackToTrackingParticleMap_.end()) {
    return trackToTrackingParticleMap_.find(aTrack)->second;
  } else {
    return nullTrackingParticlePtr_;
  }
}

template <>
const std::vector<TTTrackPtr>& TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::findTTTrackPtrs(
    TrackingParticlePtr aTrackingParticle) const {
  if (trackingParticleToTrackVectorMap_.find(aTrackingParticle) != trackingParticleToTrackVectorMap_.end()) {
    return trackingParticleToTrackVectorMap_.find(aTrackingParticle)->second;
  } else {
    return nullVecTTTrackPtr_;
  }
}

/// MC truth
template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isLooselyGenuine(TTTrackPtr aTrack) const {
  /// Check if there is a TrackingParticle
  if ((this->findTrackingParticlePtr(aTrack)).isNull())
    return false;

  return true;
}

/// MC truth
template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isGenuine(TTTrackPtr aTrack) const {
  /// Check if there is an associated TrackingParticle
  if ((this->findTrackingParticlePtr(aTrack)).isNull())
    return false;

  /// Get all the stubs from this track & associated TrackingParticle
  const std::vector<TTStubRef>& TRK_Stubs = aTrack->getStubRefs();
  const std::vector<TTStubRef>& TP_Stubs =
      theStubAssociationMap_->findTTStubRefs(this->findTrackingParticlePtr(aTrack));

  bool one2SStub = false;
  for (unsigned int js = 0; js < TRK_Stubs.size(); js++) {
    /// We want that all the stubs of the track are included in the container of
    /// all the stubs produced by this particular TrackingParticle which we
    /// already know is one of the TrackingParticles that released hits
    /// in this track we are evaluating right now
    /// Now modifying to allow one and only one false 2S stub in the track  idr 06/19
    if (std::find(TP_Stubs.begin(), TP_Stubs.end(), TRK_Stubs.at(js)) == TP_Stubs.end()) {
      if (!AllowOneFalse2SStub || TRK_Stubs.at(js)->moduleTypePS() || one2SStub)  // Has to be first false 2S stub
      {
        return false;
      } else {
        one2SStub = true;
      }
    }
  }

  return true;
}

template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isCombinatoric(TTTrackPtr aTrack) const {
  /// Defined by exclusion
  if (this->isLooselyGenuine(aTrack))
    return false;

  if (this->isUnknown(aTrack))
    return false;

  return true;
}

template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isUnknown(TTTrackPtr aTrack) const {
  /// UNKNOWN means that >= 2 stubs are unknown
  int unknownstubs = 0;

  const std::vector<TTStubRef>& theseStubs = aTrack->getStubRefs();
  for (unsigned int i = 0; i < theseStubs.size(); i++) {
    if (theStubAssociationMap_->isUnknown(theseStubs.at(i)) == false) {
      ++unknownstubs;
      if (unknownstubs >= 2)
        return false;
    }
  }

  return true;
}

template <>
void TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::setAllowOneFalse2SStub(bool allowFalse2SStub) {
  AllowOneFalse2SStub = allowFalse2SStub;
}

template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::getAllowOneFalse2SStub() {
  return AllowOneFalse2SStub;
}
