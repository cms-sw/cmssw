/** \class CollectionMerger
 *
 * \author Stefan Wayand;
 *         Christian Veelken, LLR
 *
 */
#ifndef TauAnalysis_MCEmbeddingTools_CollectionMerger_H
#define TauAnalysis_MCEmbeddingTools_CollectionMerger_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

// #include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"
#include <iostream>
#include <map>
#include <string>

template <typename T1, typename T2>
class CollectionMerger : public edm::stream::EDProducer<> {
public:
  explicit CollectionMerger(const edm::ParameterSet &);
  ~CollectionMerger() override;

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  typedef T1 MergeCollection;
  typedef T2 BaseHit;
  std::map<std::string, std::vector<edm::EDGetTokenT<MergeCollection>>> inputs_;

  edm::EDGetTokenT<reco::SuperClusterCollection> inputs_scEB_, inputs_scEE_, inputs_SC_;

  typedef edm::ValueMap<reco::TrackRefVector> TrackToTrackMapnew;
  edm::EDGetTokenT<TrackToTrackMapnew> inputs_fixtrackrefs_;
  edm::EDGetTokenT<reco::TrackCollection> inputs_fixtrackcol_;

  void fill_output_obj(edm::Event &,
                       std::unique_ptr<MergeCollection> &output,
                       std::vector<edm::Handle<MergeCollection>> &inputCollections);
  void fill_output_obj_tracker(std::unique_ptr<MergeCollection> &output,
                               std::vector<edm::Handle<MergeCollection>> &inputCollections,
                               bool print_pixel = false);
  void fill_output_obj_calo(std::unique_ptr<MergeCollection> &output,
                            std::vector<edm::Handle<MergeCollection>> &inputCollections);
  void fill_output_obj_muonchamber(std::unique_ptr<MergeCollection> &output,
                                   std::vector<edm::Handle<MergeCollection>> &inputCollections);
  // seed merger
  void fill_output_obj_seed(edm::Event &,
                            std::unique_ptr<MergeCollection> &output,
                            std::vector<edm::Handle<MergeCollection>> &inputCollections);

  void willproduce(std::string instance, std::string alias);
  void willconsume(const edm::ParameterSet &iConfig);
};

#endif
