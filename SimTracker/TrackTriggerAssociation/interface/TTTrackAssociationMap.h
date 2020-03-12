/*! \class   TTTrackAssociationMap
 *  \brief   Class to store the MC truth of L1 Track Trigger tracks
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#ifndef L1_TRACK_TRIGGER_TRACK_ASSOCIATION_FORMAT_H
#define L1_TRACK_TRIGGER_TRACK_ASSOCIATION_FORMAT_H

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"  /// NOTE: this is needed even if it seems not
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

template <typename T>
class TTTrackAssociationMap {
public:
  /// Constructors
  TTTrackAssociationMap();

  /// Destructor
  ~TTTrackAssociationMap();

  /// Data members:   getABC( ... )
  /// Helper methods: findABC( ... )

  /// Maps
  std::map<edm::Ptr<TTTrack<T> >, edm::Ptr<TrackingParticle> > getTTTrackToTrackingParticleMap() const {
    return trackToTrackingParticleMap;
  }
  std::map<edm::Ptr<TrackingParticle>, std::vector<edm::Ptr<TTTrack<T> > > > getTrackingParticleToTTTracksMap() const {
    return trackingParticleToTrackVectorMap;
  }

  void setTTTrackToTrackingParticleMap(std::map<edm::Ptr<TTTrack<T> >, edm::Ptr<TrackingParticle> > aMap) {
    trackToTrackingParticleMap = aMap;
  }
  void setTrackingParticleToTTTracksMap(
      std::map<edm::Ptr<TrackingParticle>, std::vector<edm::Ptr<TTTrack<T> > > > aMap) {
    trackingParticleToTrackVectorMap = aMap;
  }
  void setTTStubAssociationMap(edm::RefProd<TTStubAssociationMap<T> > aStubAssoMap) {
    theStubAssociationMap = aStubAssoMap;
  }

  /// Operations
  edm::Ptr<TrackingParticle> findTrackingParticlePtr(edm::Ptr<TTTrack<T> > aTrack) const;
  std::vector<edm::Ptr<TTTrack<T> > > findTTTrackPtrs(edm::Ptr<TrackingParticle> aTrackingParticle) const;

  /// MC Truth methods
  bool isGenuine(edm::Ptr<TTTrack<T> > aTrack) const;
  bool isLooselyGenuine(edm::Ptr<TTTrack<T> > aTrack) const;
  bool isCombinatoric(edm::Ptr<TTTrack<T> > aTrack) const;
  bool isUnknown(edm::Ptr<TTTrack<T> > aTrack) const;

  void setAllowOneFalse2SStub(bool allowFalse2SStub);
  bool getAllowOneFalse2SStub();

private:
  /// Data members
  std::map<edm::Ptr<TTTrack<T> >, edm::Ptr<TrackingParticle> > trackToTrackingParticleMap;
  std::map<edm::Ptr<TrackingParticle>, std::vector<edm::Ptr<TTTrack<T> > > > trackingParticleToTrackVectorMap;
  edm::RefProd<TTStubAssociationMap<T> > theStubAssociationMap;

  bool AllowOneFalse2SStub;

};  /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Default Constructor
/// NOTE: to be used with setSomething(...) methods
template <typename T>
TTTrackAssociationMap<T>::TTTrackAssociationMap() {
  /// Set default data members
  trackToTrackingParticleMap.clear();
  trackingParticleToTrackVectorMap.clear();
}

/// Destructor
template <typename T>
TTTrackAssociationMap<T>::~TTTrackAssociationMap() {}

/// Operations
template <>
edm::Ptr<TrackingParticle> TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::findTrackingParticlePtr(
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > aTrack) const;

template <>
std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > > TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::findTTTrackPtrs(
    edm::Ptr<TrackingParticle> aTrackingParticle) const;

/// MC truth
template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isLooselyGenuine(
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > aTrack) const;

/// MC truth
template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isGenuine(edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > aTrack) const;

template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isCombinatoric(
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > aTrack) const;

template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isUnknown(edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > aTrack) const;

#endif
