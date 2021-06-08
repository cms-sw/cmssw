/** \class MuonDetCleaner
 *
 * Clean collections of hits in muon detectors (CSC, DT and RPC)
 * for original Zmumu event and "embedded" simulated tau decay products
 * 
 * \author Christian Veelken, LLR
 *
 * 
 *
 * 
 *
 * Clean Up from STefan Wayand, KIT
 * 
 */
#ifndef TauAnalysis_MCEmbeddingTools_MuonDetCleaner_H
#define TauAnalysis_MCEmbeddingTools_MuonDetCleaner_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <vector>
#include <map>

template <typename T1, typename T2>
class MuonDetCleaner : public edm::stream::EDProducer<> {
public:
  explicit MuonDetCleaner(const edm::ParameterSet&);
  ~MuonDetCleaner() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  typedef edm::RangeMap<T1, edm::OwnVector<T2> > RecHitCollection;
  void fillVetoHits(const TrackingRecHit&, std::vector<uint32_t>*);

  uint32_t getRawDetId(const T2&);
  bool checkrecHit(const TrackingRecHit&);

  const edm::EDGetTokenT<edm::View<pat::Muon> > mu_input_;

  std::map<std::string, edm::EDGetTokenT<RecHitCollection> > inputs_;
};

template <typename T1, typename T2>
MuonDetCleaner<T1, T2>::MuonDetCleaner(const edm::ParameterSet& iConfig)
    : mu_input_(consumes<edm::View<pat::Muon> >(iConfig.getParameter<edm::InputTag>("MuonCollection"))) {
  std::vector<edm::InputTag> inCollections = iConfig.getParameter<std::vector<edm::InputTag> >("oldCollection");
  for (const auto& inCollection : inCollections) {
    inputs_[inCollection.instance()] = consumes<RecHitCollection>(inCollection);
    produces<RecHitCollection>(inCollection.instance());
  }
}

template <typename T1, typename T2>
MuonDetCleaner<T1, T2>::~MuonDetCleaner() {
  // nothing to be done yet...
}

template <typename T1, typename T2>
void MuonDetCleaner<T1, T2>::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  std::map<T1, std::vector<T2> > recHits_output;  // This data format is easyer to handle
  std::vector<uint32_t> vetoHits;

  // First fill the veto RecHits colletion with the Hits from the input muons
  edm::Handle<edm::View<pat::Muon> > muonHandle;
  iEvent.getByToken(mu_input_, muonHandle);
  edm::View<pat::Muon> muons = *muonHandle;
  for (edm::View<pat::Muon>::const_iterator iMuon = muons.begin(); iMuon != muons.end(); ++iMuon) {
    const reco::Track* track = nullptr;
    if (iMuon->isGlobalMuon())
      track = iMuon->outerTrack().get();
    else if (iMuon->isStandAloneMuon())
      track = iMuon->outerTrack().get();
    else if (iMuon->isRPCMuon())
      track = iMuon->innerTrack().get();  // To add, try to access the rpc track
    else if (iMuon->isTrackerMuon())
      track = iMuon->innerTrack().get();
    else {
      edm::LogError("TauEmbedding") << "The imput muon: " << (*iMuon)
                                    << " must be either global or does or be tracker muon";
      assert(0);
    }

    for (trackingRecHit_iterator hitIt = track->recHitsBegin(); hitIt != track->recHitsEnd(); ++hitIt) {
      const TrackingRecHit& murechit = **hitIt;  // Base class for all rechits
      if (!(murechit).isValid())
        continue;
      if (!checkrecHit(murechit))
        continue;                         // Check if the hit belongs to a specifc detector section
      fillVetoHits(murechit, &vetoHits);  // Go back to the very basic rechits
    }
  }

  // Now this can also handle different instance
  for (auto input_ : inputs_) {
    // Second read in the RecHit Colltection which is to be replaced, without the vetoRecHits
    typedef edm::Handle<RecHitCollection> RecHitCollectionHandle;
    RecHitCollectionHandle RecHitinput;
    iEvent.getByToken(input_.second, RecHitinput);
    for (typename RecHitCollection::const_iterator recHit = RecHitinput->begin(); recHit != RecHitinput->end();
         ++recHit) {  // loop over the basic rec hit collection (DT CSC or RPC)
      //if (find(vetoHits.begin(),vetoHits.end(),getRawDetId(*recHit)) == vetoHits.end()) continue; // For the invertec selcetion
      if (find(vetoHits.begin(), vetoHits.end(), getRawDetId(*recHit)) != vetoHits.end())
        continue;  // If the hit is not in the
      T1 detId(getRawDetId(*recHit));
      recHits_output[detId].push_back(*recHit);
    }

    // Last step savet the output in the CMSSW Data Format
    std::unique_ptr<RecHitCollection> output(new RecHitCollection());
    for (typename std::map<T1, std::vector<T2> >::const_iterator recHit = recHits_output.begin();
         recHit != recHits_output.end();
         ++recHit) {
      output->put(recHit->first, recHit->second.begin(), recHit->second.end());
    }
    output->post_insert();
    iEvent.put(std::move(output), input_.first);
  }
}

template <typename T1, typename T2>
void MuonDetCleaner<T1, T2>::fillVetoHits(const TrackingRecHit& rh, std::vector<uint32_t>* HitsList) {
  std::vector<const TrackingRecHit*> rh_components = rh.recHits();
  if (rh_components.empty()) {
    HitsList->push_back(rh.rawId());
  } else {
    for (std::vector<const TrackingRecHit*>::const_iterator rh_component = rh_components.begin();
         rh_component != rh_components.end();
         ++rh_component) {
      fillVetoHits(**rh_component, HitsList);
    }
  }
}

#endif
