/*! \brief   
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *  
 */

#include "SimTracker/TrackTriggerAssociation/plugins/TTTrackAssociator.h"

/// Implement the producer
template <>
void TTTrackAssociator<Ref_Phase2TrackerDigi_>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  /// Exit if real data
  if (iEvent.isRealData())
    return;

  /// Get the Stub and Cluster MC truth
  edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> ttClusterAssociationMapHandle;
  iEvent.getByToken(ttClusterTruthToken_, ttClusterAssociationMapHandle);
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> ttStubAssociationMapHandle;
  iEvent.getByToken(ttStubTruthToken_, ttStubAssociationMapHandle);

  int ncont1 = 0;

  /// Loop over InputTags to handle multiple collections
  for (const auto& iTag : ttTracksTokens_) {
    /// Prepare output
    auto associationMapForOutput = std::make_unique<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>();

    /// Get the Tracks already stored away
    edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> TTTrackHandle;
    iEvent.getByToken(iTag, TTTrackHandle);

    /// Prepare the necessary maps
    std::map<TTTrackPtr, TrackingParticlePtr> trackToTrackingParticleMap;
    std::map<TrackingParticlePtr, std::vector<TTTrackPtr>> trackingParticleToTrackVectorMap;

    // Start the loop on tracks

    for (unsigned int jTrk = 0; jTrk < TTTrackHandle->size(); jTrk++) {
      /// Make the pointer to be put in the map
      TTTrackPtr tempTrackPtr(TTTrackHandle, jTrk);

      /// Get all the stubs of the TTTrack (theseStubs)
      const std::vector<TTStubRef>& theseStubs = tempTrackPtr->getStubRefs();

      /// Auxiliary map to relate TP addresses and TP edm::Ptr
      std::map<const TrackingParticle*, TrackingParticlePtr> auxMap;
      int mayCombinUnknown = 0;

      /// Fill the map associating each TP to a vector of L1 tracks.
      /// Do this using the association map of the clusters inside each stub,
      /// as stub associator misses stub --> all TP map (FIX).
      for (const TTStubRef& stub : theseStubs) {
        for (unsigned int ic = 0; ic < 2; ic++) {
          const std::vector<TrackingParticlePtr>& tempTPs =
              ttClusterAssociationMapHandle->findTrackingParticlePtrs(stub->clusterRef(ic));
          for (const TrackingParticlePtr& testTP : tempTPs)  // List of TPs linked to stub clusters
          {
            if (testTP.isNull())  // No TP linked to this cluster
              continue;

            /// Prepare the maps wrt TrackingParticle
            if (trackingParticleToTrackVectorMap.find(testTP) == trackingParticleToTrackVectorMap.end()) {
              std::vector<TTTrackPtr> trackVector;
              trackingParticleToTrackVectorMap.emplace(testTP, trackVector);
            }
            trackingParticleToTrackVectorMap.find(testTP)->second.push_back(tempTrackPtr);  /// Fill the auxiliary map

            /// Fill the other auxiliary map
            if (auxMap.find(testTP.get()) == auxMap.end()) {
              auxMap.emplace(testTP.get(), testTP);
            }
          }
        }  /// End of loop over the clusters

        /// Check if the stub is unknown
        if (ttStubAssociationMapHandle->isUnknown(stub))
          ++mayCombinUnknown;

      }  /// End of loop over the stubs

      /// If there are >= 2 unknown stubs, go to the next track
      /// as this track may be COMBINATORIC or UNKNOWN
      /// (One unknown is allowed, if in 2S module).
      if (mayCombinUnknown >= 2)
        continue;

      /// If we are here, all the stubs on track are either combinatoric or genuine
      /// and there is no more than one fake stub in the track
      /// Loop over all the TrackingParticle which have been found in the track
      /// (stored in auxMap), to check if any are present in all stubs on Track.

      std::vector<const TrackingParticle*> tpInAllStubs;

      for (const auto& auxPair : auxMap) {
        /// Get all associated stubs of this TrackingParticle
        const std::vector<TTStubRef>& tempStubs = ttStubAssociationMapHandle->findTTStubRefs(auxPair.second);

        // Count stubs on track that are not related to this TP
        int nnotfound = 0;
        for (const TTStubRef& stub : theseStubs) {
          /// We want that all the stubs of the track are included in the container of
          /// all the stubs produced by this particular TrackingParticle which we
          /// already know is one of the TrackingParticles that released hits
          /// in this track we are evaluating right now
          if (std::find(tempStubs.begin(), tempStubs.end(), stub) == tempStubs.end()) {
            ++nnotfound;
          }
        }

        /// If this TP does not appear in all stubs (allowing one wrong stub)
        /// then try next TP.
        if (nnotfound > 1)
          continue;

        /// If we are here, it means that the TrackingParticle
        /// generates hits in all stubs (allowing one incorrect one) of the current track
        /// so put it into the vector
        tpInAllStubs.push_back(auxPair.first);
      }

      /// Count how many TrackingParticles were associated to all stubs on this track.
      /// FIX: Could avoid this by using std::set for tpInAllStubs?
      std::sort(tpInAllStubs.begin(), tpInAllStubs.end());
      tpInAllStubs.erase(std::unique(tpInAllStubs.begin(), tpInAllStubs.end()), tpInAllStubs.end());
      unsigned int nTPs = tpInAllStubs.size();

      /// If only one TP associated to all stubs (allowing one incorrect) on track: GENUINE or LOOSELY_GENUINE.
      /// If 0 or >= 2 TP: COMBINATORIC
      /// WARNING: This means if one TP matches all stubs, and another matches all STUBS except
      /// one, then the trackToTrackingParticleMap will not be filled.
      /// WARNING: This also means that trackToTrackingParticleMap will be filled if
      /// one TP matches all stubs, except for an incorrect one in either PS or 2S modules.
      if (nTPs != 1)
        continue;

      /// Here, the track may only be GENUINE/LOOSELY_GENUINE
      /// CHECK: Surely if one incorrect PS stub, it can also be COMBINATORIC?
      /// Fill the map associating track to its principle TP.
      trackToTrackingParticleMap.emplace(tempTrackPtr, auxMap.find(tpInAllStubs.at(0))->second);

    }  /// End of loop over Tracks

    /// Remove duplicates from the only output map that needs it.
    /// (Map gets multiple entries per track if it has several stubs belonging to same TP).
    for (auto& p : trackingParticleToTrackVectorMap) {
      /// Get the vector of edm::Ptr< TTTrack >
      /// (CHECK: Couldn't this be done by reference, to save CPU?)
      std::vector<TTTrackPtr>& tempVector = p.second;

      /// Sort and remove duplicates
      std::sort(tempVector.begin(), tempVector.end());
      tempVector.erase(std::unique(tempVector.begin(), tempVector.end()), tempVector.end());
    }

    /// Also, create the pointer to the TTClusterAssociationMap
    edm::RefProd<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> theStubAssoMap(ttStubAssociationMapHandle);

    /// Put the maps in the association object
    associationMapForOutput->setTTTrackToTrackingParticleMap(trackToTrackingParticleMap);
    associationMapForOutput->setTrackingParticleToTTTracksMap(trackingParticleToTrackVectorMap);
    associationMapForOutput->setTTStubAssociationMap(theStubAssoMap);
    associationMapForOutput->setAllowOneFalse2SStub(TTTrackAllowOneFalse2SStub);

    /// Put output in the event
    iEvent.put(std::move(associationMapForOutput), ttTracksInputTags_.at(ncont1).instance());

    ++ncont1;
  }  /// End of loop over InputTags
}
