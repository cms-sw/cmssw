/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

#include "SimDataFormats/Associations/interface/TTTrackAssociationMap.h"

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

/// MC truth association according to user specified requirements
template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isGenuine(
    TTTrackPtr aTrack,
    TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::MatchCrit matchCrit,
    unsigned int minShareStubs) const {
  /// Find associated TrackingParticle in map.
  /// If none, then more than one stub on track is incorrect.
  const TrackingParticlePtr& assocTP = this->findTrackingParticlePtr(aTrack);
  if (assocTP.isNull())
    return false;

  /// Get all stubs on this track
  const std::vector<TTStubRef>& TRK_Stubs = aTrack->getStubRefs();
  const unsigned int nStubs = TRK_Stubs.size();

  // If track has fewer than the required number of shared stubs, then it fails.
  if (nStubs < minShareStubs)
    return false;

  /// In this case, can't allow any incorrect stubs on track.
  if (nStubs == minShareStubs)
    matchCrit = MatchCrit::allCorrect;

  /// At most one stub on track is incorrect, since track was in map.
  if (matchCrit == MatchCrit::allowOneFalse2SorPS)
    return true;

  /// Get all stubs on associated TrackingParticle
  const std::vector<TTStubRef>& TP_Stubs = theStubAssociationMap_->findTTStubRefs(assocTP);

  /// Check if any stubs are incorrect.
  for (const TTStubRef& trkStub : TRK_Stubs) {
    if (std::find(TP_Stubs.begin(), TP_Stubs.end(), trkStub) == TP_Stubs.end()) {
      /// This stub is incorrect. It wasn't produced by the TP.
      if (matchCrit == MatchCrit::allCorrect)
        return false;
      if (matchCrit == MatchCrit::allowOneFalse2S && trkStub->moduleTypePS())
        return false;
    }
  }

  return true;
}

/// MC truth association allowing one incorrect stub on track
/// Same as calling isGenuine(trk, allowOneFalse2SorPS, 0).
template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isLooselyGenuine(TTTrackPtr aTrack) const {
  /// Check if there is an associated TrackingParticle in map.
  /// If none, then more than one stub on track is incorrect.
  const TrackingParticlePtr& assocTP = this->findTrackingParticlePtr(aTrack);
  if (assocTP.isNull())
    return false;

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
