#include <iostream>
#include <fstream>

#include "QuickTrackAssociatorByHitsImpl.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "SimTracker/TrackAssociation/interface/trackHitsToClusterRefs.h"

//
// Use the unnamed namespace for utility functions only used in this file
//
namespace {
  //
  // All of these functions are pretty straightforward but the implementation is type dependent.
  // The templated methods call these for the type specific parts and the compiler will resolve
  // the type and call the correct overload.
  //

  template <class T_element>
  size_t collectionSize(const edm::RefToBaseVector<T_element>& collection) {
    return collection.size();
  }

  template <class T_element>
  size_t collectionSize(const edm::Handle<T_element>& hCollection) {
    return hCollection->size();
  }

  template <class T_element>
  size_t collectionSize(const edm::RefVector<T_element>& collection) {
    return collection.size();
  }

  const reco::Track* getTrackAt(const edm::RefToBaseVector<reco::Track>& trackCollection, size_t index) {
    return &*trackCollection[index];  // pretty obscure dereference
  }

  const reco::Track* getTrackAt(const edm::Handle<edm::View<reco::Track> >& pTrackCollection, size_t index) {
    return &(*pTrackCollection.product())[index];
  }

  const TrackingParticle* getTrackingParticleAt(const edm::Handle<TrackingParticleCollection>& pCollection,
                                                size_t index) {
    return &(*pCollection.product())[index];
  }

  const TrackingParticle* getTrackingParticleAt(const edm::RefVector<TrackingParticleCollection>& collection,
                                                size_t index) {
    return &*collection[index];
  }

  edm::RefToBase<reco::Track> getRefToTrackAt(const edm::RefToBaseVector<reco::Track>& trackCollection, size_t index) {
    return trackCollection[index];
  }

  edm::RefToBase<reco::Track> getRefToTrackAt(const edm::Handle<edm::View<reco::Track> >& pTrackCollection,
                                              size_t index) {
    return edm::RefToBase<reco::Track>(pTrackCollection, index);
  }

  edm::Ref<TrackingParticleCollection> getRefToTrackingParticleAt(
      const edm::Handle<TrackingParticleCollection>& pCollection, size_t index) {
    return edm::Ref<TrackingParticleCollection>(pCollection, index);
  }

  edm::Ref<TrackingParticleCollection> getRefToTrackingParticleAt(
      const edm::RefVector<TrackingParticleCollection>& collection, size_t index) {
    return collection[index];
  }

  void fillKeys(edm::IndexSet& keys, const edm::RefVector<TrackingParticleCollection>& collection) {
    keys.reserve(collection.size());
    for (const auto& ref : collection) {
      keys.insert(ref.key());
    }
  }

  template <typename Coll>
  void checkClusterMapProductID(const ClusterTPAssociation& clusterToTPMap, const Coll& collection) {
    clusterToTPMap.checkMappedProductID(collection.id());
  }

  template <typename Coll>
  void checkClusterMapProductID(const TrackerHitAssociator& hitAssociator, const Coll& collection) {}

}  // namespace

QuickTrackAssociatorByHitsImpl::QuickTrackAssociatorByHitsImpl(edm::EDProductGetter const& productGetter,
                                                               std::unique_ptr<const TrackerHitAssociator> hitAssoc,
                                                               const ClusterTPAssociation* clusterToTPMap,

                                                               bool absoluteNumberOfHits,
                                                               double qualitySimToReco,
                                                               double puritySimToReco,
                                                               double pixelHitWeight,
                                                               double cutRecoToSim,
                                                               bool threeHitTracksAreSpecial,
                                                               SimToRecoDenomType simToRecoDenominator)
    : productGetter_(&productGetter),
      hitAssociator_(std::move(hitAssoc)),
      clusterToTPMap_(clusterToTPMap),
      qualitySimToReco_(qualitySimToReco),
      puritySimToReco_(puritySimToReco),
      pixelHitWeight_(pixelHitWeight),
      cutRecoToSim_(cutRecoToSim),
      simToRecoDenominator_(simToRecoDenominator),
      threeHitTracksAreSpecial_(threeHitTracksAreSpecial),
      absoluteNumberOfHits_(absoluteNumberOfHits) {}

reco::RecoToSimCollection QuickTrackAssociatorByHitsImpl::associateRecoToSim(
    const edm::Handle<edm::View<reco::Track> >& trackCollectionHandle,
    const edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle) const {
  // Only pass the one that was successfully created to the templated method.
  if (not clusterToTPMap_)
    return associateRecoToSimImplementation(
        trackCollectionHandle, trackingParticleCollectionHandle, nullptr, *hitAssociator_);
  else
    return associateRecoToSimImplementation(
        trackCollectionHandle, trackingParticleCollectionHandle, nullptr, *clusterToTPMap_);
}

reco::SimToRecoCollection QuickTrackAssociatorByHitsImpl::associateSimToReco(
    const edm::Handle<edm::View<reco::Track> >& trackCollectionHandle,
    const edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle) const {
  // Only pass the one that was successfully created to the templated method.
  if (not clusterToTPMap_)
    return associateSimToRecoImplementation(
        trackCollectionHandle, trackingParticleCollectionHandle, nullptr, *hitAssociator_);
  else
    return associateSimToRecoImplementation(
        trackCollectionHandle, trackingParticleCollectionHandle, nullptr, *clusterToTPMap_);
}

reco::RecoToSimCollection QuickTrackAssociatorByHitsImpl::associateRecoToSim(
    const edm::RefToBaseVector<reco::Track>& trackCollection,
    const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection) const {
  // Only pass the one that was successfully created to the templated method.
  if (not clusterToTPMap_)
    return associateRecoToSimImplementation(trackCollection, trackingParticleCollection, nullptr, *hitAssociator_);
  else {
    TrackingParticleRefKeySet tpKeys;
    fillKeys(tpKeys, trackingParticleCollection);
    return associateRecoToSimImplementation(trackCollection, trackingParticleCollection, &tpKeys, *clusterToTPMap_);
  }
}

reco::SimToRecoCollection QuickTrackAssociatorByHitsImpl::associateSimToReco(
    const edm::RefToBaseVector<reco::Track>& trackCollection,
    const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection) const {
  // Only pass the one that was successfully created to the templated method.
  if (not clusterToTPMap_)
    return associateSimToRecoImplementation(trackCollection, trackingParticleCollection, nullptr, *hitAssociator_);
  else {
    TrackingParticleRefKeySet tpKeys;
    fillKeys(tpKeys, trackingParticleCollection);
    return associateSimToRecoImplementation(trackCollection, trackingParticleCollection, &tpKeys, *clusterToTPMap_);
  }
}

template <class T_TrackCollection, class T_TrackingParticleCollection, class T_hitOrClusterAssociator>
reco::RecoToSimCollection QuickTrackAssociatorByHitsImpl::associateRecoToSimImplementation(
    const T_TrackCollection& trackCollection,
    const T_TrackingParticleCollection& trackingParticleCollection,
    const TrackingParticleRefKeySet* trackingParticleKeys,
    T_hitOrClusterAssociator hitOrClusterAssociator) const {
  reco::RecoToSimCollection returnValue(productGetter_);
  if (::collectionSize(trackingParticleCollection) == 0)
    return returnValue;

  checkClusterMapProductID(hitOrClusterAssociator, trackingParticleCollection);

  size_t collectionSize = ::collectionSize(trackCollection);  // Delegate away type specific part

  for (size_t i = 0; i < collectionSize; ++i) {
    const reco::Track* pTrack = ::getTrackAt(
        trackCollection, i);  // Get a normal pointer for ease of use. This part is type specific so delegate.

    // The return of this function has first as the index and second as the number of associated hits
    std::vector<std::pair<edm::Ref<TrackingParticleCollection>, double> > trackingParticleQualityPairs =
        associateTrack(hitOrClusterAssociator,
                       trackingParticleCollection,
                       trackingParticleKeys,
                       pTrack->recHitsBegin(),
                       pTrack->recHitsEnd());

    // int nt = 0;
    for (auto iTrackingParticleQualityPair = trackingParticleQualityPairs.begin();
         iTrackingParticleQualityPair != trackingParticleQualityPairs.end();
         ++iTrackingParticleQualityPair) {
      const edm::Ref<TrackingParticleCollection>& trackingParticleRef = iTrackingParticleQualityPair->first;
      double numberOfSharedClusters = iTrackingParticleQualityPair->second;
      double numberOfValidTrackClusters = weightedNumberOfTrackClusters(*pTrack, hitOrClusterAssociator);

      if (numberOfSharedClusters == 0.0)
        continue;  // No point in continuing if there was no association

      //if electron subtract double counting
      if (abs(trackingParticleRef->pdgId()) == 11 &&
          (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1) {
        numberOfSharedClusters -=
            getDoubleCount(hitOrClusterAssociator, pTrack->recHitsBegin(), pTrack->recHitsEnd(), trackingParticleRef);
      }

      double quality;
      if (absoluteNumberOfHits_)
        quality = numberOfSharedClusters;
      else if (numberOfValidTrackClusters != 0.0)
        quality = numberOfSharedClusters / numberOfValidTrackClusters;
      else
        quality = 0;
      if (quality > cutRecoToSim_ &&
          !(threeHitTracksAreSpecial_ && pTrack->numberOfValidHits() == 3 && numberOfSharedClusters < 3.0)) {
        // Getting the RefToBase is dependent on the type of trackCollection, so delegate that to an overload.
        returnValue.insert(::getRefToTrackAt(trackCollection, i), std::make_pair(trackingParticleRef, quality));
      }
    }
  }
  returnValue.post_insert();
  return returnValue;
}

template <class T_TrackCollection, class T_TrackingParticleCollection, class T_hitOrClusterAssociator>
reco::SimToRecoCollection QuickTrackAssociatorByHitsImpl::associateSimToRecoImplementation(
    const T_TrackCollection& trackCollection,
    const T_TrackingParticleCollection& trackingParticleCollection,
    const TrackingParticleRefKeySet* trackingParticleKeys,
    T_hitOrClusterAssociator hitOrClusterAssociator) const {
  reco::SimToRecoCollection returnValue(productGetter_);
  if (::collectionSize(trackingParticleCollection) == 0)
    return returnValue;

  checkClusterMapProductID(hitOrClusterAssociator, trackingParticleCollection);

  size_t collectionSize = ::collectionSize(trackCollection);  // Delegate away type specific part

  for (size_t i = 0; i < collectionSize; ++i) {
    const reco::Track* pTrack = ::getTrackAt(
        trackCollection, i);  // Get a normal pointer for ease of use. This part is type specific so delegate.

    // The return of this function has first as an edm:Ref to the associated TrackingParticle, and second as the number of associated hits
    std::vector<std::pair<edm::Ref<TrackingParticleCollection>, double> > trackingParticleQualityPairs =
        associateTrack(hitOrClusterAssociator,
                       trackingParticleCollection,
                       trackingParticleKeys,
                       pTrack->recHitsBegin(),
                       pTrack->recHitsEnd());

    // int nt = 0;
    for (auto iTrackingParticleQualityPair = trackingParticleQualityPairs.begin();
         iTrackingParticleQualityPair != trackingParticleQualityPairs.end();
         ++iTrackingParticleQualityPair) {
      const edm::Ref<TrackingParticleCollection>& trackingParticleRef = iTrackingParticleQualityPair->first;
      double numberOfSharedClusters = iTrackingParticleQualityPair->second;
      double numberOfValidTrackClusters = weightedNumberOfTrackClusters(*pTrack, hitOrClusterAssociator);
      size_t numberOfSimulatedHits = 0;  // Set a few lines below, but only if required.

      if (numberOfSharedClusters == 0.0)
        continue;  // No point in continuing if there was no association

      if (simToRecoDenominator_ == denomsim ||
          (numberOfSharedClusters < 3.0 &&
           threeHitTracksAreSpecial_))  // the numberOfSimulatedHits is not always required, so can skip counting in some circumstances
      {
        // Note that in the standard TrackAssociatorByHits, all of the hits in associatedTrackingParticleHits are checked for
        // various things.  I'm not sure what these checks are for but they depend on the UseGrouping and UseSplitting settings.
        // This associator works as though both UseGrouping and UseSplitting were set to true, i.e. just counts the number of
        // hits in the tracker.
        numberOfSimulatedHits = trackingParticleRef->numberOfTrackerHits();
      }

      //if electron subtract double counting
      if (abs(trackingParticleRef->pdgId()) == 11 &&
          (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1) {
        numberOfSharedClusters -=
            getDoubleCount(hitOrClusterAssociator, pTrack->recHitsBegin(), pTrack->recHitsEnd(), trackingParticleRef);
      }

      double purity = numberOfSharedClusters / numberOfValidTrackClusters;
      double quality;
      if (absoluteNumberOfHits_)
        quality = numberOfSharedClusters;
      else if (simToRecoDenominator_ == denomsim && numberOfSimulatedHits != 0)
        quality = numberOfSharedClusters / static_cast<double>(numberOfSimulatedHits);
      else if (simToRecoDenominator_ == denomreco && numberOfValidTrackClusters != 0)
        quality = purity;
      else
        quality = 0;

      if (quality > qualitySimToReco_ &&
          !(threeHitTracksAreSpecial_ && numberOfSimulatedHits == 3 && numberOfSharedClusters < 3.0) &&
          (absoluteNumberOfHits_ || (purity > puritySimToReco_))) {
        // Getting the RefToBase is dependent on the type of trackCollection, so delegate that to an overload.
        returnValue.insert(trackingParticleRef, std::make_pair(::getRefToTrackAt(trackCollection, i), quality));
      }
    }
  }
  returnValue.post_insert();
  return returnValue;
}

template <typename T_TPCollection, typename iter>
std::vector<std::pair<edm::Ref<TrackingParticleCollection>, double> > QuickTrackAssociatorByHitsImpl::associateTrack(
    const TrackerHitAssociator& hitAssociator,
    const T_TPCollection& trackingParticles,
    const TrackingParticleRefKeySet* trackingParticleKeys,
    iter begin,
    iter end) const {
  // The pairs in this vector have a Ref to the associated TrackingParticle as "first" and the weighted number of associated hits as "second"
  std::vector<std::pair<edm::Ref<TrackingParticleCollection>, double> > returnValue;

  // The pairs in this vector have first as the sim track identifiers, and second the number of reco hits associated to that sim track.
  // Most reco hits will probably have come from the same sim track, so the number of entries in this vector should be fewer than the
  // number of reco hits.  The pair::second entries should add up to the total number of reco hits though.
  std::vector<std::pair<SimTrackIdentifiers, double> > hitIdentifiers =
      getAllSimTrackIdentifiers(hitAssociator, begin, end);

  // Loop over the TrackingParticles
  size_t collectionSize = ::collectionSize(trackingParticles);

  for (size_t i = 0; i < collectionSize; ++i) {
    const TrackingParticle* pTrackingParticle = getTrackingParticleAt(trackingParticles, i);

    // Historically there was a requirement that pTrackingParticle->numberOfHits() > 0
    // However, in TrackingTruthAccumulator, the numberOfHits is calculated from a subset
    // of the SimHits of the SimTracks of a TrackingParticle (essentially limiting the
    // processType and particleType to those of the "first" hit, and particleType to the pdgId of the SimTrack).
    // But, here the association between tracks and TrackingParticles is done with *all* the hits of
    // TrackingParticle, so we should not rely on the numberOfHits() calculated with a subset of SimHits.

    double numberOfAssociatedHits = 0;
    // Loop over all of the sim track identifiers and see if any of them are part of this TrackingParticle. If they are, add
    // the number of reco hits associated to that sim track to the total number of associated hits.
    for (const auto& identifierCountPair : hitIdentifiers) {
      if (trackingParticleContainsIdentifier(pTrackingParticle, identifierCountPair.first))
        numberOfAssociatedHits += identifierCountPair.second;
    }

    if (numberOfAssociatedHits > 0) {
      returnValue.push_back(std::make_pair(getRefToTrackingParticleAt(trackingParticles, i), numberOfAssociatedHits));
    }
  }

  return returnValue;
}

template <typename T_TPCollection, typename iter>
std::vector<std::pair<edm::Ref<TrackingParticleCollection>, double> > QuickTrackAssociatorByHitsImpl::associateTrack(
    const ClusterTPAssociation& clusterToTPMap,
    const T_TPCollection& trackingParticles,
    const TrackingParticleRefKeySet* trackingParticleKeys,
    iter begin,
    iter end) const {
  // Note that the trackingParticles parameter is not actually required since all the information is in clusterToTPMap,
  // but the method signature has to match the other overload because it is called from a templated method.

  // Note further, that we can't completely ignore the
  // trackingParticles parameter, in case it is a subset of those
  // TrackingParticles used to construct clusterToTPMap (via the
  // TrackingParticleRefVector overloads). The trackingParticles
  // parameter is still ignored since looping over it on every call
  // would be expensive, but the keys of the TrackingParticleRefs are
  // cached to an IndexSet (trackingParticleKeys) which is used
  // as a fast search structure.

  // The pairs in this vector have a Ref to the associated TrackingParticle as "first" and the weighted number of associated clusters as "second"
  // Note: typedef edm::Ref<TrackingParticleCollection> TrackingParticleRef;
  std::vector<std::pair<edm::Ref<TrackingParticleCollection>, double> > returnValue;
  if (clusterToTPMap.empty())
    return returnValue;

  // The pairs in this vector have first as the TP, and second the number of reco clusters associated to that TP.
  // Most reco clusters will probably have come from the same sim track (i.e TP), so the number of entries in this
  // vector should be fewer than the number of clusters. The pair::second entries should add up to the total
  // number of reco clusters though.
  std::vector<OmniClusterRef> oClusters = track_associator::hitsToClusterRefs(begin, end);

  std::map<TrackingParticleRef, double> lmap;
  for (std::vector<OmniClusterRef>::const_iterator it = oClusters.begin(); it != oClusters.end(); ++it) {
    auto range = clusterToTPMap.equal_range(*it);
    const double weight = it->isPixel() ? pixelHitWeight_ : 1.0;
    if (range.first != range.second) {
      for (auto ip = range.first; ip != range.second; ++ip) {
        const TrackingParticleRef trackingParticle = (ip->second);

        if (trackingParticleKeys && !trackingParticleKeys->has(trackingParticle.key()))
          continue;

        // Historically there was a requirement that pTrackingParticle->numberOfHits() > 0
        // However, in TrackingTruthAccumulator, the numberOfHits is calculated from a subset
        // of the SimHits of the SimTracks of a TrackingParticle (essentially limiting the
        // processType and particleType to those of the "first" hit, and particleType to the pdgId of the SimTrack).
        // But, here the association between tracks and TrackingParticles is done with *all* the hits of
        // TrackingParticle, so we should not rely on the numberOfHits() calculated with a subset of SimHits.

        /* Alternative implementation to avoid the use of lmap... memory slightly improved but slightly slower...
				 std::pair<edm::Ref<TrackingParticleCollection>,size_t> tpIntPair(trackingParticle, 1);
				 auto tp_range = std::equal_range(returnValue.begin(), returnValue.end(), tpIntPair, tpIntPairGreater);
				 if ((tp_range.second-tp_range.first)>1) {
				 edm::LogError("TrackAssociator") << ">>> Error in counting TPs!" << " file: " << __FILE__ << " line: " << __LINE__;
				 }
				 if(tp_range.first != tp_range.second) {
				 tp_range.first->second++;
				 } else {
				 returnValue.push_back(tpIntPair);
				 std::sort(returnValue.begin(), returnValue.end(), tpIntPairGreater);
				 }
				 */
        auto jpos = lmap.find(trackingParticle);
        if (jpos != lmap.end())
          jpos->second += weight;
        else
          lmap.insert(std::make_pair(trackingParticle, weight));
      }
    }
  }
  // now copy the map to returnValue
  for (auto ip = lmap.begin(); ip != lmap.end(); ++ip) {
    returnValue.push_back(std::make_pair(ip->first, ip->second));
  }
  return returnValue;
}

template <typename iter>
std::vector<std::pair<QuickTrackAssociatorByHitsImpl::SimTrackIdentifiers, double> >
QuickTrackAssociatorByHitsImpl::getAllSimTrackIdentifiers(const TrackerHitAssociator& hitAssociator,
                                                          iter begin,
                                                          iter end) const {
  // The pairs in this vector have first as the sim track identifiers, and second the number of reco hits associated to that sim track.
  std::vector<std::pair<SimTrackIdentifiers, double> > returnValue;

  std::vector<SimTrackIdentifiers> simTrackIdentifiers;
  // Loop over all of the rec hits in the track
  //iter tRHIterBeginEnd = getTRHIterBeginEnd( pTrack );
  for (iter iRecHit = begin; iRecHit != end; ++iRecHit) {
    if (track_associator::getHitFromIter(iRecHit)->isValid()) {
      simTrackIdentifiers.clear();

      // Get the identifiers for the sim track that this hit came from. There should only be one entry unless clusters
      // have merged (as far as I know).
      hitAssociator.associateHitId(*(track_associator::getHitFromIter(iRecHit)),
                                   simTrackIdentifiers);  // This call fills simTrackIdentifiers

      const auto subdetId = track_associator::getHitFromIter(iRecHit)->geographicalId().subdetId();
      const double weight = (subdetId == PixelSubdetector::PixelBarrel || subdetId == PixelSubdetector::PixelEndcap)
                                ? pixelHitWeight_
                                : 1.0;

      // Loop over each identifier, and add it to the return value only if it's not already in there
      for (std::vector<SimTrackIdentifiers>::const_iterator iIdentifier = simTrackIdentifiers.begin();
           iIdentifier != simTrackIdentifiers.end();
           ++iIdentifier) {
        std::vector<std::pair<SimTrackIdentifiers, double> >::iterator iIdentifierCountPair;
        for (auto iIdentifierCountPair = returnValue.begin(); iIdentifierCountPair != returnValue.end();
             ++iIdentifierCountPair) {
          if (iIdentifierCountPair->first.first == iIdentifier->first &&
              iIdentifierCountPair->first.second == iIdentifier->second) {
            // This sim track identifier is already in the list, so increment the count of how many hits it relates to.
            iIdentifierCountPair->second += weight;
            break;
          }
        }
        if (iIdentifierCountPair == returnValue.end())
          returnValue.push_back(std::make_pair(*iIdentifier, 1.0));
        // This identifier wasn't found, so add it
      }
    }
  }
  return returnValue;
}

bool QuickTrackAssociatorByHitsImpl::trackingParticleContainsIdentifier(const TrackingParticle* pTrackingParticle,
                                                                        const SimTrackIdentifiers& identifier) const {
  // Loop over all of the g4 tracks in the tracking particle
  for (std::vector<SimTrack>::const_iterator iSimTrack = pTrackingParticle->g4Track_begin();
       iSimTrack != pTrackingParticle->g4Track_end();
       ++iSimTrack) {
    // And see if the sim track identifiers match
    if (iSimTrack->eventId() == identifier.second && iSimTrack->trackId() == identifier.first) {
      return true;
    }
  }

  // If control has made it this far then none of the identifiers were found in
  // any of the g4 tracks, so return false.
  return false;
}

template <typename iter>
double QuickTrackAssociatorByHitsImpl::getDoubleCount(const TrackerHitAssociator& hitAssociator,
                                                      iter startIterator,
                                                      iter endIterator,
                                                      TrackingParticleRef associatedTrackingParticle) const {
  // This method is largely copied from the standard TrackAssociatorByHits. Once I've tested how much difference
  // it makes I'll go through and comment it properly.

  // FIXME: It may be that this piece is not fully correct for
  // counting how many times a single *cluster* is matched to many
  // SimTracks of a single TrackingParticle (see comments in
  // getDoubleCount(ClusterTPAssociation) overload). To be verified
  // some time.

  double doubleCount = 0.0;
  std::vector<SimHitIdpr> SimTrackIdsDC;

  for (iter iHit = startIterator; iHit != endIterator; iHit++) {
    int idcount = 0;

    SimTrackIdsDC.clear();
    hitAssociator.associateHitId(*(track_associator::getHitFromIter(iHit)), SimTrackIdsDC);
    if (SimTrackIdsDC.size() > 1) {
      for (TrackingParticle::g4t_iterator g4T = associatedTrackingParticle->g4Track_begin();
           g4T != associatedTrackingParticle->g4Track_end();
           ++g4T) {
        if (find(SimTrackIdsDC.begin(),
                 SimTrackIdsDC.end(),
                 SimHitIdpr((*g4T).trackId(), SimTrackIdsDC.begin()->second)) != SimTrackIdsDC.end()) {
          idcount++;
        }
      }
    }
    if (idcount > 1) {
      const auto subdetId = track_associator::getHitFromIter(iHit)->geographicalId().subdetId();
      const double weight = (subdetId == PixelSubdetector::PixelBarrel || subdetId == PixelSubdetector::PixelEndcap)
                                ? pixelHitWeight_
                                : 1.0;
      doubleCount += weight * (idcount - 1);
    }
  }

  return doubleCount;
}

template <typename iter>
double QuickTrackAssociatorByHitsImpl::getDoubleCount(const ClusterTPAssociation& clusterToTPList,
                                                      iter startIterator,
                                                      iter endIterator,
                                                      TrackingParticleRef associatedTrackingParticle) const {
  // This code here was written by Subir Sarkar. I'm just splitting it off into a
  // separate method. - Grimes 01/May/2014

  // The point here is that the electron TrackingParticles may contain
  // multiple SimTracks (from the bremsstrahling), and (historically)
  // the each matched hit/cluster has been multiplied by "how many
  // SimTracks from the TrackingParticle" it contains charge from.
  // Here the amount of this double counting is calculated, so that it
  // can be subtracted by the calling code.
  //
  // Note that recently (hence "historically" in the paragraph above)
  // the ClusterTPAssociationProducer was changed to remove the
  // duplicate cluster->TP associations (hence making this function
  // obsolete), but there is more recent proof that there is some
  // duplication left (to be investigated).

  double doubleCount = 0;
  std::vector<SimHitIdpr> SimTrackIdsDC;

  for (iter iHit = startIterator; iHit != endIterator; iHit++) {
    std::vector<OmniClusterRef> oClusters =
        track_associator::hitsToClusterRefs(iHit, iHit + 1);  //only for the cluster being checked
    for (std::vector<OmniClusterRef>::const_iterator it = oClusters.begin(); it != oClusters.end(); ++it) {
      int idcount = 0;

      auto range = clusterToTPList.equal_range(*it);
      if (range.first != range.second) {
        for (auto ip = range.first; ip != range.second; ++ip) {
          const TrackingParticleRef trackingParticle = (ip->second);
          if (associatedTrackingParticle == trackingParticle) {
            idcount++;
          }
        }
      }

      if (idcount > 1) {
        const auto subdetId = track_associator::getHitFromIter(iHit)->geographicalId().subdetId();
        const double weight = (subdetId == PixelSubdetector::PixelBarrel || subdetId == PixelSubdetector::PixelEndcap)
                                  ? pixelHitWeight_
                                  : 1.0;
        doubleCount += weight * (idcount - 1);
      }
    }
  }

  return doubleCount;
}

reco::RecoToSimCollectionSeed QuickTrackAssociatorByHitsImpl::associateRecoToSim(
    const edm::Handle<edm::View<TrajectorySeed> >& pSeedCollectionHandle_,
    const edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle) const {
  edm::LogVerbatim("TrackAssociator") << "Starting TrackAssociatorByHitsImpl::associateRecoToSim - #seeds="
                                      << pSeedCollectionHandle_->size()
                                      << " #TPs=" << trackingParticleCollectionHandle->size();

  reco::RecoToSimCollectionSeed returnValue(productGetter_);

  size_t collectionSize = pSeedCollectionHandle_->size();

  for (size_t i = 0; i < collectionSize; ++i) {
    const TrajectorySeed* pSeed = &(*pSeedCollectionHandle_)[i];

    // The return of this function has first as the index and second as the number of associated hits
    std::vector<std::pair<edm::Ref<TrackingParticleCollection>, double> > trackingParticleQualityPairs =
        (clusterToTPMap_) ? associateTrack(*clusterToTPMap_,
                                           trackingParticleCollectionHandle,
                                           nullptr,
                                           pSeed->recHits().begin(),
                                           pSeed->recHits().end())
                          : associateTrack(*hitAssociator_,
                                           trackingParticleCollectionHandle,
                                           nullptr,
                                           pSeed->recHits().begin(),
                                           pSeed->recHits().end());
    for (auto iTrackingParticleQualityPair = trackingParticleQualityPairs.begin();
         iTrackingParticleQualityPair != trackingParticleQualityPairs.end();
         ++iTrackingParticleQualityPair) {
      const edm::Ref<TrackingParticleCollection>& trackingParticleRef = iTrackingParticleQualityPair->first;
      double numberOfSharedClusters = iTrackingParticleQualityPair->second;
      double numberOfValidTrackClusters = clusterToTPMap_ ? weightedNumberOfTrackClusters(*pSeed, *clusterToTPMap_)
                                                          : weightedNumberOfTrackClusters(*pSeed, *hitAssociator_);

      if (numberOfSharedClusters == 0.0)
        continue;  // No point in continuing if there was no association

      //if electron subtract double counting
      if (abs(trackingParticleRef->pdgId()) == 11 &&
          (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1) {
        if (clusterToTPMap_)
          numberOfSharedClusters -=
              getDoubleCount(*clusterToTPMap_, pSeed->recHits().begin(), pSeed->recHits().end(), trackingParticleRef);
        else
          numberOfSharedClusters -=
              getDoubleCount(*hitAssociator_, pSeed->recHits().begin(), pSeed->recHits().end(), trackingParticleRef);
      }

      double quality;
      if (absoluteNumberOfHits_)
        quality = numberOfSharedClusters;
      else if (numberOfValidTrackClusters != 0.0)
        quality = numberOfSharedClusters / numberOfValidTrackClusters;
      else
        quality = 0;

      if (quality > cutRecoToSim_ &&
          !(threeHitTracksAreSpecial_ && pSeed->nHits() == 3 && numberOfSharedClusters < 3.0)) {
        returnValue.insert(edm::RefToBase<TrajectorySeed>(pSeedCollectionHandle_, i),
                           std::make_pair(trackingParticleRef, quality));
      }
    }
  }

  LogTrace("TrackAssociator") << "% of Assoc Seeds="
                              << ((double)returnValue.size()) / ((double)pSeedCollectionHandle_->size());
  returnValue.post_insert();
  return returnValue;
}

reco::SimToRecoCollectionSeed QuickTrackAssociatorByHitsImpl::associateSimToReco(
    const edm::Handle<edm::View<TrajectorySeed> >& pSeedCollectionHandle_,
    const edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle) const {
  edm::LogVerbatim("TrackAssociator") << "Starting TrackAssociatorByHitsImpl::associateSimToReco - #seeds="
                                      << pSeedCollectionHandle_->size()
                                      << " #TPs=" << trackingParticleCollectionHandle->size();

  reco::SimToRecoCollectionSeed returnValue(productGetter_);
  if (trackingParticleCollectionHandle->empty())
    return returnValue;

  if (clusterToTPMap_) {
    checkClusterMapProductID(*clusterToTPMap_, trackingParticleCollectionHandle);
  }

  size_t collectionSize = pSeedCollectionHandle_->size();

  for (size_t i = 0; i < collectionSize; ++i) {
    const TrajectorySeed* pSeed = &(*pSeedCollectionHandle_)[i];

    // The return of this function has first as an edm:Ref to the associated TrackingParticle, and second as the number of associated hits
    std::vector<std::pair<edm::Ref<TrackingParticleCollection>, double> > trackingParticleQualityPairs =
        (clusterToTPMap_) ? associateTrack(*clusterToTPMap_,
                                           trackingParticleCollectionHandle,
                                           nullptr,
                                           pSeed->recHits().begin(),
                                           pSeed->recHits().end())
                          : associateTrack(*hitAssociator_,
                                           trackingParticleCollectionHandle,
                                           nullptr,
                                           pSeed->recHits().begin(),
                                           pSeed->recHits().end());
    for (auto iTrackingParticleQualityPair = trackingParticleQualityPairs.begin();
         iTrackingParticleQualityPair != trackingParticleQualityPairs.end();
         ++iTrackingParticleQualityPair) {
      const edm::Ref<TrackingParticleCollection>& trackingParticleRef = iTrackingParticleQualityPair->first;
      double numberOfSharedClusters = iTrackingParticleQualityPair->second;
      double numberOfValidTrackClusters = clusterToTPMap_ ? weightedNumberOfTrackClusters(*pSeed, *clusterToTPMap_)
                                                          : weightedNumberOfTrackClusters(*pSeed, *hitAssociator_);
      size_t numberOfSimulatedHits = 0;  // Set a few lines below, but only if required.

      if (numberOfSharedClusters == 0.0)
        continue;  // No point in continuing if there was no association

      //if electron subtract double counting
      if (abs(trackingParticleRef->pdgId()) == 11 &&
          (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1) {
        if (clusterToTPMap_)
          numberOfSharedClusters -=
              getDoubleCount(*clusterToTPMap_, pSeed->recHits().begin(), pSeed->recHits().end(), trackingParticleRef);
        else
          numberOfSharedClusters -=
              getDoubleCount(*hitAssociator_, pSeed->recHits().begin(), pSeed->recHits().end(), trackingParticleRef);
      }

      if (simToRecoDenominator_ == denomsim ||
          (numberOfSharedClusters < 3.0 &&
           threeHitTracksAreSpecial_))  // the numberOfSimulatedHits is not always required, so can skip counting in some circumstances
      {
        // Note that in the standard TrackAssociatorByHits, all of the hits in associatedTrackingParticleHits are checked for
        // various things.  I'm not sure what these checks are for but they depend on the UseGrouping and UseSplitting settings.
        // This associator works as though both UseGrouping and UseSplitting were set to true, i.e. just counts the number of
        // hits in the tracker.
        numberOfSimulatedHits = trackingParticleRef->numberOfTrackerHits();
      }

      double purity = numberOfSharedClusters / numberOfValidTrackClusters;
      double quality;
      if (absoluteNumberOfHits_)
        quality = numberOfSharedClusters;
      else if (simToRecoDenominator_ == denomsim && numberOfSimulatedHits != 0)
        quality = numberOfSharedClusters / static_cast<double>(numberOfSimulatedHits);
      else if (simToRecoDenominator_ == denomreco && numberOfValidTrackClusters != 0.0)
        quality = purity;
      else
        quality = 0;

      if (quality > qualitySimToReco_ &&
          !(threeHitTracksAreSpecial_ && numberOfSimulatedHits == 3 && numberOfSharedClusters < 3.0) &&
          (absoluteNumberOfHits_ || (purity > puritySimToReco_))) {
        returnValue.insert(trackingParticleRef,
                           std::make_pair(edm::RefToBase<TrajectorySeed>(pSeedCollectionHandle_, i), quality));
      }
    }
  }

  LogTrace("TrackAssociator") << "% of Assoc TPs="
                              << ((double)returnValue.size()) / ((double)trackingParticleCollectionHandle->size());
  returnValue.post_insert();
  return returnValue;
}

// count hits
double QuickTrackAssociatorByHitsImpl::weightedNumberOfTrackClusters(const reco::Track& track,
                                                                     const TrackerHitAssociator&) const {
  const reco::HitPattern& p = track.hitPattern();
  const auto pixelHits = p.numberOfValidPixelHits();
  const auto otherHits = p.numberOfValidHits() - pixelHits;
  return pixelHits * pixelHitWeight_ + otherHits;
}

double QuickTrackAssociatorByHitsImpl::weightedNumberOfTrackClusters(const TrajectorySeed& seed,
                                                                     const TrackerHitAssociator&) const {
  double sum = 0.0;
  for (auto iHit = seed.recHits().begin(); iHit != seed.recHits().end(); ++iHit) {
    const auto subdetId = track_associator::getHitFromIter(iHit)->geographicalId().subdetId();
    const double weight = (subdetId == PixelSubdetector::PixelBarrel || subdetId == PixelSubdetector::PixelEndcap)
                              ? pixelHitWeight_
                              : 1.0;
    sum += weight;
  }
  return sum;
}

// count clusters
double QuickTrackAssociatorByHitsImpl::weightedNumberOfTrackClusters(const reco::Track& track,
                                                                     const ClusterTPAssociation&) const {
  return weightedNumberOfTrackClusters(track.recHitsBegin(), track.recHitsEnd());
}
double QuickTrackAssociatorByHitsImpl::weightedNumberOfTrackClusters(const TrajectorySeed& seed,
                                                                     const ClusterTPAssociation&) const {
  const auto& hitRange = seed.recHits();
  return weightedNumberOfTrackClusters(hitRange.begin(), hitRange.end());
}

template <typename iter>
double QuickTrackAssociatorByHitsImpl::weightedNumberOfTrackClusters(iter begin, iter end) const {
  double weightedClusters = 0.0;
  for (iter iRecHit = begin; iRecHit != end; ++iRecHit) {
    const auto subdetId = track_associator::getHitFromIter(iRecHit)->geographicalId().subdetId();
    const double weight = (subdetId == PixelSubdetector::PixelBarrel || subdetId == PixelSubdetector::PixelEndcap)
                              ? pixelHitWeight_
                              : 1.0;
    LogTrace("QuickTrackAssociatorByHitsImpl")
        << "  detId: " << track_associator::getHitFromIter(iRecHit)->geographicalId().rawId();
    LogTrace("QuickTrackAssociatorByHitsImpl") << "  weight: " << weight;
    std::vector<OmniClusterRef> oClusters =
        track_associator::hitsToClusterRefs(iRecHit, iRecHit + 1);  //only for the cluster being checked
    for (std::vector<OmniClusterRef>::const_iterator it = oClusters.begin(); it != oClusters.end(); ++it) {
      weightedClusters += weight;
    }
  }
  LogTrace("QuickTrackAssociatorByHitsImpl") << "  total weighted clusters: " << weightedClusters;
  return weightedClusters;
}
