/** \class CollectionMerger
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
#ifndef TauAnalysis_MCEmbeddingTools_CollectionMerger_H
#define TauAnalysis_MCEmbeddingTools_CollectionMerger_H

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
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

//#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"
#include <string>
#include <iostream>
#include <map>

template <typename T1, typename T2>
class CollectionMerger : public edm::stream::EDProducer<> {
public:
  explicit CollectionMerger(const edm::ParameterSet &);
  ~CollectionMerger() override;

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  typedef T1 MergeCollection;
  typedef T2 BaseHit;
  std::map<std::string, std::vector<edm::EDGetTokenT<MergeCollection> > > inputs_;

  void fill_output_obj(std::unique_ptr<MergeCollection> &output,
                       std::vector<edm::Handle<MergeCollection> > &inputCollections);
  void fill_output_obj_tracker(std::unique_ptr<MergeCollection> &output,
                               std::vector<edm::Handle<MergeCollection> > &inputCollections,
                               bool print_pixel = false);
  void fill_output_obj_calo(std::unique_ptr<MergeCollection> &output,
                            std::vector<edm::Handle<MergeCollection> > &inputCollections);
  void fill_output_obj_muonchamber(std::unique_ptr<MergeCollection> &output,
                                   std::vector<edm::Handle<MergeCollection> > &inputCollections);
};

template <typename T1, typename T2>
CollectionMerger<T1, T2>::CollectionMerger(const edm::ParameterSet &iConfig) {
  std::vector<edm::InputTag> inCollections = iConfig.getParameter<std::vector<edm::InputTag> >("mergCollections");
  for (auto const &inCollection : inCollections) {
    inputs_[inCollection.instance()].push_back(consumes<MergeCollection>(inCollection));
  }
  for (const auto &toproduce : inputs_) {
    //  std::cout<<toproduce.first<<"\t"<<toproduce.second.size()<<std::endl;
    produces<MergeCollection>(toproduce.first);
  }
}

template <typename T1, typename T2>
CollectionMerger<T1, T2>::~CollectionMerger() {
  // nothing to be done yet...
}

template <typename T1, typename T2>
void CollectionMerger<T1, T2>::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  for (auto input_ : inputs_) {
    std::unique_ptr<MergeCollection> output(new MergeCollection());
    std::vector<edm::Handle<MergeCollection> > inputCollections;
    inputCollections.resize(input_.second.size());
    for (unsigned id = 0; id < input_.second.size(); id++) {
      iEvent.getByToken(input_.second[id], inputCollections[id]);
    }
    fill_output_obj(output, inputCollections);
    iEvent.put(std::move(output), input_.first);
  }
}
#endif
