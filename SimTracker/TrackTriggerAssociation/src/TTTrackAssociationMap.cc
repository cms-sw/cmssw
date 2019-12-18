/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"

/// Operations
template <>
edm::Ptr<TrackingParticle> TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::findTrackingParticlePtr(
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > aTrack) const {
  if (trackToTrackingParticleMap.find(aTrack) != trackToTrackingParticleMap.end()) {
    return trackToTrackingParticleMap.find(aTrack)->second;
  }

  /// Default: return NULL
  edm::Ptr<TrackingParticle>* temp = new edm::Ptr<TrackingParticle>();
  return *temp;
}

template <>
std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > > TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::findTTTrackPtrs(
    edm::Ptr<TrackingParticle> aTrackingParticle) const {
  if (trackingParticleToTrackVectorMap.find(aTrackingParticle) != trackingParticleToTrackVectorMap.end()) {
    return trackingParticleToTrackVectorMap.find(aTrackingParticle)->second;
  }

  /// Default: return empty vector
  std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > > tempVec;
  tempVec.clear();
  return tempVec;
}

/// MC truth
template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isLooselyGenuine(
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > aTrack) const {
  /// Check if there is a TrackingParticle
  if ((this->findTrackingParticlePtr(aTrack)).isNull())
    return false;

  return true;
}

/// MC truth
template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isGenuine(edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > aTrack) const {
  /// Check if there is a TrackingParticle
  if ((this->findTrackingParticlePtr(aTrack)).isNull())
    return false;

  /// Get all the stubs from this TrackingParticle
  std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >
      TP_Stubs = theStubAssociationMap->findTTStubRefs(this->findTrackingParticlePtr(aTrack));
  std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >
      TRK_Stubs = aTrack->getStubRefs();

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
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isCombinatoric(
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > aTrack) const {
  /// Defined by exclusion
  if (this->isLooselyGenuine(aTrack))
    return false;

  if (this->isUnknown(aTrack))
    return false;

  return true;
}

template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isUnknown(edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > aTrack) const {
  /// UNKNOWN means that more than 2 stubs are unknown
  int unknownstubs = 0;

  std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >
      theseStubs = aTrack->getStubRefs();
  for (unsigned int i = 0; i < theseStubs.size(); i++) {
    if (theStubAssociationMap->isUnknown(theseStubs.at(i)) == false) {
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
