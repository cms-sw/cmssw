/** \class TrackerCleaner
 *
 * 
 * \author Stefan Wayand;
 *         Christian Veelken, LLR
 *
 * 
 *
 * 
 *
 */

#ifndef TauAnalysis_MCEmbeddingTools_TrackerCleaner_H
#define TauAnalysis_MCEmbeddingTools_TrackerCleaner_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

#include <string>
#include <iostream>
#include <map>

template <typename T>
class TrackerCleaner : public edm::stream::EDProducer<> {
public:
  explicit TrackerCleaner(const edm::ParameterSet&);
  ~TrackerCleaner() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<edm::View<pat::Muon> > mu_input_;
  typedef edmNew::DetSetVector<T> TrackClusterCollection;

  std::map<std::string, edm::EDGetTokenT<TrackClusterCollection> > inputs_;

  bool match_rechit_type(const TrackingRecHit& murechit);
};

template <typename T>
TrackerCleaner<T>::TrackerCleaner(const edm::ParameterSet& iConfig)
    : mu_input_(consumes<edm::View<pat::Muon> >(iConfig.getParameter<edm::InputTag>("MuonCollection")))

{
  std::vector<edm::InputTag> inCollections = iConfig.getParameter<std::vector<edm::InputTag> >("oldCollection");
  for (const auto& inCollection : inCollections) {
    inputs_[inCollection.instance()] = consumes<TrackClusterCollection>(inCollection);
    produces<TrackClusterCollection>(inCollection.instance());
  }
}

template <typename T>
TrackerCleaner<T>::~TrackerCleaner() {
  // nothing to be done yet...
}

template <typename T>
void TrackerCleaner<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<edm::View<pat::Muon> > muonHandle;
  iEvent.getByToken(mu_input_, muonHandle);
  edm::View<pat::Muon> muons = *muonHandle;

  for (auto input_ : inputs_) {
    edm::Handle<TrackClusterCollection> inputClusters;
    iEvent.getByToken(input_.second, inputClusters);

    std::vector<bool> vetodClusters;

    vetodClusters.resize(inputClusters->dataSize(), false);

    for (edm::View<pat::Muon>::const_iterator iMuon = muons.begin(); iMuon != muons.end(); ++iMuon) {
      if (!iMuon->isGlobalMuon())
        continue;
      const reco::Track* mutrack = iMuon->globalTrack().get();
      //  reco::Track *mutrack = new reco::Track(*(iMuon->innerTrack() ));
      for (trackingRecHit_iterator hitIt = mutrack->recHitsBegin(); hitIt != mutrack->recHitsEnd(); ++hitIt) {
        const TrackingRecHit& murechit = **hitIt;
        if (!(murechit).isValid())
          continue;

        if (match_rechit_type(murechit)) {
          auto& thit = reinterpret_cast<BaseTrackerRecHit const&>(murechit);
          auto const& cluster = thit.firstClusterRef();
          vetodClusters[cluster.key()] = true;
        }
      }
    }
    std::unique_ptr<TrackClusterCollection> output(new TrackClusterCollection());

    int idx = 0;
    for (typename TrackClusterCollection::const_iterator clustSet = inputClusters->begin();
         clustSet != inputClusters->end();
         ++clustSet) {
      DetId detIdObject(clustSet->detId());
      typename TrackClusterCollection::FastFiller spc(*output, detIdObject);
      for (typename edmNew::DetSet<T>::const_iterator clustIt = clustSet->begin(); clustIt != clustSet->end();
           ++clustIt) {
        idx++;
        if (vetodClusters[idx - 1])
          continue;
        //if (!vetodClusters[idx-1]) continue; for inverted selction
        spc.push_back(*clustIt);
      }
    }
    iEvent.put(std::move(output), input_.first);
  }
}
#endif
