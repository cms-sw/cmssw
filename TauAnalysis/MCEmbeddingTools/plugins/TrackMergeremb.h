/** \class TrackMergeremb
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
#ifndef TauAnalysis_MCEmbeddingTools_TrackMergeremb_H
#define TauAnalysis_MCEmbeddingTools_TrackMergeremb_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/MuonReco/interface/MuonQuality.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/MuonReco/interface/MuonToMuonMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

#include <string>
#include <iostream>
#include <map>

template <typename T1>
class TrackMergeremb : public edm::stream::EDProducer<> {
public:
  explicit TrackMergeremb(const edm::ParameterSet&);
  ~TrackMergeremb() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  typedef T1 TrackCollectionemb;

  void willproduce(std::string instance, std::string alias);
  void willconsume(const edm::ParameterSet& iConfig);
  void merg_and_put(edm::Event&, std::string, std::vector<edm::EDGetTokenT<TrackCollectionemb> >&);

  std::map<std::string, std::vector<edm::EDGetTokenT<TrackCollectionemb> > > inputs_;
  std::map<std::string, std::vector<edm::EDGetTokenT<edm::ValueMap<reco::MuonQuality> > > > inputs_qual_;

  //typedef edm::ValueMap<reco::TrackRef> TrackToTrackMapnew;
  typedef edm::ValueMap<reco::TrackRefVector> TrackToTrackMapnew;

  edm::EDGetTokenT<TrackToTrackMapnew> inputs_fixtrackrefs_;
  edm::EDGetTokenT<reco::TrackCollection> inputs_fixtrackcol_;

  edm::EDGetTokenT<reco::MuonToMuonMap> inputs_fixmurefs_;
  edm::EDGetTokenT<reco::MuonCollection> inputs_fixmucol_;
};

template <typename T1>
TrackMergeremb<T1>::TrackMergeremb(const edm::ParameterSet& iConfig) {
  std::string alias(iConfig.getParameter<std::string>("@module_label"));
  std::vector<edm::InputTag> inCollections = iConfig.getParameter<std::vector<edm::InputTag> >("mergCollections");
  for (const auto& inCollection : inCollections) {
    inputs_[inCollection.instance()].push_back(consumes<TrackCollectionemb>(inCollection));
  }
  willconsume(iConfig);
  for (const auto& toproduce : inputs_) {
    willproduce(toproduce.first, alias);
  }
}

template <typename T1>
TrackMergeremb<T1>::~TrackMergeremb() {
  // nothing to be done yet...
}

template <typename T1>
void TrackMergeremb<T1>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  for (auto input_ : inputs_) {
    merg_and_put(iEvent, input_.first, input_.second);

  }  // end instance
}
#endif
