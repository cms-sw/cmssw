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
  edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> TTClusterAssociationMapHandle;
  iEvent.getByToken(TTClusterTruthToken, TTClusterAssociationMapHandle);
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> TTStubAssociationMapHandle;
  iEvent.getByToken(TTStubTruthToken, TTStubAssociationMapHandle);

  int ncont1 = 0;

  /// Loop over InputTags to handle multiple collections
  for (auto iTag = TTTracksTokens.begin(); iTag != TTTracksTokens.end(); iTag++) {
    /// Prepare output
    auto associationMapForOutput = std::make_unique<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>();

    /// Get the Tracks already stored away
    edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> TTTrackHandle;
    iEvent.getByToken(*iTag, TTTrackHandle);

    /// Prepare the necessary maps
    std::map<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>, edm::Ptr<TrackingParticle>> trackToTrackingParticleMap;
    std::map<edm::Ptr<TrackingParticle>, std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>>>
        trackingParticleToTrackVectorMap;
    trackToTrackingParticleMap.clear();
    trackingParticleToTrackVectorMap.clear();

    // Start the loop on tracks

    unsigned int j = 0;  /// Counter needed to build the edm::Ptr to the TTTrack
    typename std::vector<TTTrack<Ref_Phase2TrackerDigi_>>::const_iterator inputIter;
    for (inputIter = TTTrackHandle->begin(); inputIter != TTTrackHandle->end(); ++inputIter) {
      /// Make the pointer to be put in the map
      edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>> tempTrackPtr(TTTrackHandle, j++);

      /// Get the stubs of the TTTrack (theseStubs)
      std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
          theseStubs = tempTrackPtr->getStubRefs();

      /// Auxiliary map to store TP addresses and TP edm::Ptr
      std::map<const TrackingParticle*, edm::Ptr<TrackingParticle>> auxMap;
      auxMap.clear();
      int mayCombinUnknown = 0;

      /// Fill the inclusive map which is careless of the stub classification
      for (unsigned int is = 0; is < theseStubs.size(); is++) {
        //        std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > > > theseClusters = theseStubs.at(is)->getClusterRefs();
        for (unsigned int ic = 0; ic < 2; ic++) {
          std::vector<edm::Ptr<TrackingParticle>> tempTPs =
              TTClusterAssociationMapHandle->findTrackingParticlePtrs(theseStubs.at(is)->clusterRef(ic));
          for (unsigned int itp = 0; itp < tempTPs.size(); itp++)  // List of TPs linked to stub clusters
          {
            edm::Ptr<TrackingParticle> testTP = tempTPs.at(itp);

            if (testTP.isNull())  // No TP linked to this cluster
              continue;

            /// Prepare the maps wrt TrackingParticle
            if (trackingParticleToTrackVectorMap.find(testTP) == trackingParticleToTrackVectorMap.end()) {
              std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>> trackVector;
              trackVector.clear();
              trackingParticleToTrackVectorMap.insert(std::make_pair(testTP, trackVector));
            }
            trackingParticleToTrackVectorMap.find(testTP)->second.push_back(tempTrackPtr);  /// Fill the auxiliary map

            /// Fill the other auxiliary map
            if (auxMap.find(testTP.get()) == auxMap.end()) {
              auxMap.insert(std::make_pair(testTP.get(), testTP));
            }
          }
        }  /// End of loop over the clusters

        /// Check if the stub is unknown
        if (TTStubAssociationMapHandle->isUnknown(theseStubs.at(is)))
          ++mayCombinUnknown;

      }  /// End of loop over the stubs

      /// If there more than 2 unknown stubs, go to the next track
      /// as this track may be COMBINATORIC or UNKNOWN
      if (mayCombinUnknown >= 2)
        continue;

      /// If we are here, all the stubs are either combinatoric or genuine
      /// and there is no more than one fake stub in the track
      /// Loop over all the TrackingParticle which have been found in the track at some point
      /// (stored in auxMap)

      std::vector<const TrackingParticle*> tpInAllStubs;

      std::map<const TrackingParticle*, edm::Ptr<TrackingParticle>>::const_iterator iterAuxMap;
      for (iterAuxMap = auxMap.begin(); iterAuxMap != auxMap.end(); ++iterAuxMap) {
        /// Get all the stubs from this TrackingParticle
        std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
            tempStubs = TTStubAssociationMapHandle->findTTStubRefs(iterAuxMap->second);

        int nnotfound = 0;
        //bool allFound = true;
        /// Loop over the stubs
        //        for ( unsigned int js = 0; js < theseStubs.size() && allFound; js++ )
        for (unsigned int js = 0; js < theseStubs.size(); js++) {
          /// We want that all the stubs of the track are included in the container of
          /// all the stubs produced by this particular TrackingParticle which we
          /// already know is one of the TrackingParticles that released hits
          /// in this track we are evaluating right now
          if (std::find(tempStubs.begin(), tempStubs.end(), theseStubs.at(js)) == tempStubs.end()) {
            //  allFound = false;
            ++nnotfound;
          }
        }

        /// If the TrackingParticle does not appear in all stubs but one
        /// then go to the next track
        if (nnotfound > 1)
          //if (!allFound)
          continue;

        /// If we are here, it means that the TrackingParticle
        /// generates hits in all stubs but one of the current track
        /// so put it into the vector
        tpInAllStubs.push_back(iterAuxMap->first);
      }

      /// Count how many TrackingParticles we do have
      std::sort(tpInAllStubs.begin(), tpInAllStubs.end());
      tpInAllStubs.erase(std::unique(tpInAllStubs.begin(), tpInAllStubs.end()), tpInAllStubs.end());
      unsigned int nTPs = tpInAllStubs.size();

      /// If only one TrackingParticle, GENUINE
      /// if different than one, COMBINATORIC
      if (nTPs != 1)
        continue;

      /// Here, the track may only be GENUINE
      /// Fill the map
      trackToTrackingParticleMap.insert(std::make_pair(tempTrackPtr, auxMap.find(tpInAllStubs.at(0))->second));

    }  /// End of loop over Tracks

    /// Clean the only map that needs cleaning
    /// Prepare the output map wrt TrackingParticle
    typename std::map<edm::Ptr<TrackingParticle>, std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>>>::iterator
        iterMapToClean;
    for (iterMapToClean = trackingParticleToTrackVectorMap.begin();
         iterMapToClean != trackingParticleToTrackVectorMap.end();
         ++iterMapToClean) {
      /// Get the vector of edm::Ptr< TTTrack >
      std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>> tempVector = iterMapToClean->second;

      /// Sort and remove duplicates
      std::sort(tempVector.begin(), tempVector.end());
      tempVector.erase(std::unique(tempVector.begin(), tempVector.end()), tempVector.end());

      iterMapToClean->second = tempVector;
    }

    /// Also, create the pointer to the TTClusterAssociationMap
    edm::RefProd<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> theStubAssoMap(TTStubAssociationMapHandle);

    /// Put the maps in the association object
    associationMapForOutput->setTTTrackToTrackingParticleMap(trackToTrackingParticleMap);
    associationMapForOutput->setTrackingParticleToTTTracksMap(trackingParticleToTrackVectorMap);
    associationMapForOutput->setTTStubAssociationMap(theStubAssoMap);
    associationMapForOutput->setAllowOneFalse2SStub(TTTrackAllowOneFalse2SStub);

    /// Put output in the event
    iEvent.put(std::move(associationMapForOutput), TTTracksInputTags.at(ncont1).instance());

    ++ncont1;
  }  /// End of loop over InputTags
}
