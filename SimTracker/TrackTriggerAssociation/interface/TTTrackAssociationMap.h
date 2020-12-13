/*! \class   TTTrackAssociationMap
 *  \brief   Stores association of Truth Particles (TP) to L1 Track-Trigger Tracks
 * 
 *  \details Contains two maps. One associates each L1 track its principle TP.
 *           (i.e. Not to all TP that contributed to it). 
 *           The other associates each TP to a vector of all L1 tracks 
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

#ifndef L1_TRACK_TRIGGER_TRACK_ASSOCIATION_FORMAT_H
#define L1_TRACK_TRIGGER_TRACK_ASSOCIATION_FORMAT_H

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
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

// Templated aliases
template<typename T> using MapL1TrackToTP = std::map<TTTrackPtrT<T>, TrackingParticlePtr >;
template<typename T> using MapTPToVecL1Track = std::map<TrackingParticlePtr, std::vector<TTTrackPtrT<T>> >;

template <typename T>
class TTTrackAssociationMap {
public:
  /// Constructors
  TTTrackAssociationMap();

  /// Destructor
  ~TTTrackAssociationMap();

  /// Get/set stub <-> truth association maps

  const MapL1TrackToTP<T>& getTTTrackToTrackingParticleMap() const {
    return trackToTrackingParticleMap;
  }
  const MapTPToVecL1Track<T>& getTrackingParticleToTTTracksMap() const {
    return trackingParticleToTrackVectorMap;
  }

  void setTTTrackToTrackingParticleMap(const MapL1TrackToTP<T>& aMap) {
    trackToTrackingParticleMap = aMap;
  }
  void setTrackingParticleToTTTracksMap(const MapTPToVecL1Track<T>& aMap) {
    trackingParticleToTrackVectorMap = aMap;
  }

  /// Set stub <-> truth association object.
  void setTTStubAssociationMap(edm::RefProd<TTStubAssociationMap<T> > aStubAssoMap) {
    theStubAssociationMap = aStubAssoMap;
  }

  /// Get principle TP associated to a L1 track. (Non-NULL if isLooselyGenuine() below is true).
  /// (N.B. There is no function returning all TP associated to a L1 track).
  TrackingParticlePtr findTrackingParticlePtr(TTTrackPtrT<T> aTrack) const;

  /// Get all L1 tracks associated to a TP.
  /// (Even if the TP just contributes to one cluster in one stub, 
  /// and even if their are other such TP, it is still listed here).
  std::vector<TTTrackPtrT<T> > findTTTrackPtrs(TrackingParticlePtr aTrackingParticle) const;

  ///--- Get quality of L1 track based on truth info.
  /// (N.B. "genuine" tracks are used for official L1 track efficiency measurements).

  /// Exactly one (i.e. not 0 or >= 2) unique TP contributes to every stub on the track.
  /// (even if it is not the principle TP in a stub, or contributes to only one cluster
  /// in the stub, it still counts).
  /// N.B. If cfg param getAllowOneFalse2SStub() is true, then one incorrect stub in 
  /// a 2S module is alowed
  /// ISSUE: a track with 4 stubs can be accepted if only 3 of its stubs are correct!
  /// ISSUE: isLooselyGenuine() must also be true. So if 2 TPs match track, one with
  /// an incorrect PS stub, both isGenuine() & isLooselyGenuine() will be false!
  bool isGenuine(TTTrackPtrT<T> aTrack) const;
  /// Same criteria as for "genuine" track, except that one incorrect stub in either
  /// PS or 2S module is allowed, irrespective of value of cfg param getAllowOneFalse2SStub().
  bool isLooselyGenuine(TTTrackPtrT<T> aTrack) const;
  /// More than one stub on track is "unknown".
  bool isUnknown(TTTrackPtrT<T> aTrack) const;
  /// Both isLooselyGenuine() & isUnknown() are false.
  bool isCombinatoric(TTTrackPtrT<T> aTrack) const;

  // Cfg param allowing one incorrect 2S stub in "genuine" tracks.
  void setAllowOneFalse2SStub(bool allowFalse2SStub);
  bool getAllowOneFalse2SStub();

private:
  /// Data members
  MapL1TrackToTP<T> trackToTrackingParticleMap;
  MapTPToVecL1Track<T> trackingParticleToTrackVectorMap;
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
TrackingParticlePtr TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::findTrackingParticlePtr(
    TTTrackPtr aTrack) const;

template <>
std::vector<TTTrackPtr> TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::findTTTrackPtrs(
    TrackingParticlePtr aTrackingParticle) const;

/// MC truth
template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isLooselyGenuine(TTTrackPtr aTrack) const;

/// MC truth
template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isGenuine(TTTrackPtr aTrack) const;

template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isCombinatoric(TTTrackPtr aTrack) const;

template <>
bool TTTrackAssociationMap<Ref_Phase2TrackerDigi_>::isUnknown(TTTrackPtr aTrack) const;

#endif
