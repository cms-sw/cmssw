#include "TauAnalysis/MCEmbeddingTools/plugins/CollectionMerger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/DTRecHit/interface/DTSLRecCluster.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include "DataFormats/EgammaCandidates/interface/Conversion.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/RangeMap.h"

typedef CollectionMerger<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> PixelColMerger;
typedef CollectionMerger<edmNew::DetSetVector<SiStripCluster>, SiStripCluster> StripColMerger;

typedef CollectionMerger<std::vector<reco::ElectronSeed>, reco::ElectronSeed> ElectronSeedColMerger;
typedef CollectionMerger<std::vector<reco::ElectronSeed>, reco::ElectronSeed> EcalDrivenElectronSeedColMerger;
typedef CollectionMerger<edm::SortedCollection<EcalRecHit>, EcalRecHit> EcalRecHitColMerger;
typedef CollectionMerger<edm::SortedCollection<HBHERecHit>, HBHERecHit> HBHERecHitColMerger;
typedef CollectionMerger<edm::SortedCollection<HFRecHit>, HFRecHit> HFRecHitColMerger;
typedef CollectionMerger<edm::SortedCollection<HORecHit>, HORecHit> HORecHitColMerger;
typedef CollectionMerger<edm::SortedCollection<CastorRecHit>, CastorRecHit> CastorRecHitColMerger;
typedef CollectionMerger<edm::SortedCollection<ZDCRecHit>, ZDCRecHit> ZDCRecHitColMerger;

typedef CollectionMerger<edm::RangeMap<DTLayerId, edm::OwnVector<DTRecHit1DPair>>, DTRecHit1DPair> DTRecHitColMerger;
typedef CollectionMerger<edm::RangeMap<CSCDetId, edm::OwnVector<CSCRecHit2D>>, CSCRecHit2D> CSCRecHitColMerger;
typedef CollectionMerger<edm::RangeMap<RPCDetId, edm::OwnVector<RPCRecHit>>, RPCRecHit> RPCRecHitColMerger;

template <typename T1, typename T2>
void CollectionMerger<T1, T2>::willconsume(const edm::ParameterSet &iConfig) {}

template <typename T1, typename T2>
void CollectionMerger<T1, T2>::willproduce(std::string instance, std::string alias) {
  produces<MergeCollection>(instance);
}

// -------- Here Tracker Merger -----------
template <typename T1, typename T2>
void CollectionMerger<T1, T2>::fill_output_obj_tracker(std::unique_ptr<MergeCollection> &output,
                                                       std::vector<edm::Handle<MergeCollection>> &inputCollections,
                                                       bool print_pixel) {
  std::map<uint32_t, std::vector<BaseHit>> output_map;
  // First merge the collections with the help of the output map
  for (auto const &inputCollection : inputCollections) {
    for (typename MergeCollection::const_iterator clustSet = inputCollection->begin();
         clustSet != inputCollection->end();
         ++clustSet) {
      DetId detIdObject(clustSet->detId());
      for (typename edmNew::DetSet<BaseHit>::const_iterator clustIt = clustSet->begin(); clustIt != clustSet->end();
           ++clustIt) {
        output_map[detIdObject.rawId()].push_back(*clustIt);
      }
    }
  }
  // Now save it into the standard CMSSW format, with the standard Filler
  for (typename std::map<uint32_t, std::vector<BaseHit>>::const_iterator outHits = output_map.begin();
       outHits != output_map.end();
       ++outHits) {
    DetId detIdObject(outHits->first);
    typename MergeCollection::FastFiller spc(*output, detIdObject);
    for (const auto &Hit : outHits->second) {
      spc.push_back(Hit);
    }
  }
}

template <typename T1, typename T2>
void CollectionMerger<T1, T2>::fill_output_obj_calo(std::unique_ptr<MergeCollection> &output,
                                                    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  std::map<uint32_t, BaseHit> output_map;
  // First merge the two collections again
  for (auto const &inputCollection : inputCollections) {
    for (typename MergeCollection::const_iterator recHit = inputCollection->begin(); recHit != inputCollection->end();
         ++recHit) {
      DetId detIdObject(recHit->detid().rawId());
      T2 *akt_calo_obj = &output_map[detIdObject.rawId()];
      float new_energy = akt_calo_obj->energy() + recHit->energy();
      T2 newRecHit(*recHit);
      newRecHit.setEnergy(new_energy);
      *akt_calo_obj = newRecHit;
    }
  }
  // Now save it into the standard CMSSW format
  for (typename std::map<uint32_t, BaseHit>::const_iterator outHits = output_map.begin(); outHits != output_map.end();
       ++outHits) {
    output->push_back(outHits->second);
  }
  output->sort();  // Do a sort for this collection
}

// -------- Here Muon Chamber Merger -----------
template <typename T1, typename T2>
void CollectionMerger<T1, T2>::fill_output_obj_muonchamber(
    std::unique_ptr<MergeCollection> &output, std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  std::map<uint32_t, std::vector<BaseHit>> output_map;
  // First merge the collections with the help of the output map
  for (auto const &inputCollection : inputCollections) {
    for (typename MergeCollection::const_iterator recHit = inputCollection->begin(); recHit != inputCollection->end();
         ++recHit) {
      DetId detIdObject(recHit->geographicalId());
      output_map[detIdObject].push_back(*recHit);
    }
  }
  // Now save it into the standard CMSSW format, with the standard Filler
  for (typename std::map<uint32_t, std::vector<BaseHit>>::const_iterator outHits = output_map.begin();
       outHits != output_map.end();
       ++outHits) {
    output->put((typename T1::id_iterator::value_type)outHits->first,
                outHits->second.begin(),
                outHits->second.end());  // The DTLayerId misses the automatic type cast
  }
}

// -------- Here Electron Seed Merger -----------
// TODO: move seed merger to TrackMerger class
template <>
void CollectionMerger<std::vector<reco::ElectronSeed>, reco::ElectronSeed>::willconsume(
    const edm::ParameterSet &iConfig) {
  // track refs for trackerdriven seeds
  inputs_fixtrackrefs_ = consumes<TrackToTrackMapnew>(edm::InputTag("generalTracks"));
  inputs_fixtrackcol_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));

  // sc refs for ecaldriven seeds
  inputs_scEB_ = consumes<reco::SuperClusterCollection>(
      edm::InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel"));
  inputs_scEE_ = consumes<reco::SuperClusterCollection>(
      edm::InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower"));
}

template <>
void CollectionMerger<std::vector<reco::ElectronSeed>, reco::ElectronSeed>::willproduce(std::string instance,
                                                                                        std::string alias) {
  produces<MergeCollection>(instance);
}

template <typename T1, typename T2>
void CollectionMerger<T1, T2>::fill_output_obj_seed(edm::Event &iEvent,
                                                    std::unique_ptr<MergeCollection> &output,
                                                    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  // track to track map for trackerdriven seed fix
  edm::Handle<TrackToTrackMapnew> track_ref_map;
  iEvent.getByToken(inputs_fixtrackrefs_, track_ref_map);

  edm::Handle<reco::TrackCollection> track_new_col;
  iEvent.getByToken(inputs_fixtrackcol_, track_new_col);
  std::map<reco::TrackRef, reco::TrackRef>
      simple_track_to_track_map;  // I didn't find a more elegant way, so just build a good old fassion std::map
  for (unsigned abc = 0; abc < track_new_col->size(); ++abc) {
    reco::TrackRef trackRef(track_new_col, abc);
    simple_track_to_track_map[((*track_ref_map)[trackRef])[0]] = trackRef;
  }

  // ECAL barrel supercluster collection
  edm::Handle<reco::SuperClusterCollection> scEB;
  iEvent.getByToken(inputs_scEB_, scEB);
  auto bScEB = scEB->cbegin();
  auto eScEB = scEB->cend();

  // ECAL endcap supercluster collection
  edm::Handle<reco::SuperClusterCollection> scEE;
  iEvent.getByToken(inputs_scEE_, scEE);
  auto bScEE = scEE->cbegin();
  auto eScEE = scEE->cend();

  // First merge the two collections again
  for (auto const &inputCollection : inputCollections) {
    for (typename MergeCollection::const_iterator seed = inputCollection->begin(); seed != inputCollection->end();
         ++seed) {
      T2 newSeed(*seed);

      // fix reference to supercluster if seed is ecal driven
      if (seed->isEcalDriven()) {
        const reco::ElectronSeed::CaloClusterRef &seedCcRef(seed->caloCluster());
        if (seedCcRef.isNonnull()) {
          // find corresponding supercluster in ECAL barrel and endcap
          float dx, dy, dz, dr;
          float drMin = std::numeric_limits<float>::infinity();
          reco::ElectronSeed::CaloClusterRef ccRefMin;
          for (auto sc = bScEB; sc != eScEB; ++sc) {
            const reco::SuperClusterRef &scRef(reco::SuperClusterRef(scEB, std::distance(bScEB, sc)));
            dx = fabs(scRef->x() - seedCcRef->x());
            dy = fabs(scRef->y() - seedCcRef->y());
            dz = fabs(scRef->z() - seedCcRef->z());
            dr = sqrt(dx * dx + dy * dy + dz * dz);
            if (dr < drMin) {
              drMin = dr;
              ccRefMin = reco::ElectronSeed::CaloClusterRef(scRef);
            }
          }
          for (auto sc = bScEE; sc != eScEE; ++sc) {
            const reco::SuperClusterRef &scRef(reco::SuperClusterRef(scEE, std::distance(bScEE, sc)));
            dx = fabs(scRef->x() - seedCcRef->x());
            dy = fabs(scRef->y() - seedCcRef->y());
            dz = fabs(scRef->z() - seedCcRef->z());
            dr = sqrt(dx * dx + dy * dy + dz * dz);
            if (dr < drMin) {
              drMin = dr;
              ccRefMin = reco::ElectronSeed::CaloClusterRef(scRef);
            }
          }
          // set new calocluster if new ref was found
          if (drMin < 10.0) {
            newSeed.setCaloCluster(ccRefMin);
          }
        }
      }
      if (seed->isTrackerDriven()) {
        const reco::ElectronSeed::CtfTrackRef &ctfTrackRef(seed->ctfTrack());
        if (ctfTrackRef.isNonnull()) {
          newSeed.setCtfTrack(simple_track_to_track_map[ctfTrackRef]);
        }
      }
      output->push_back(newSeed);
    }
  }
}

// Here some overloaded functions, which are needed such that the right merger function is called for the indivudal Collections
template <typename T1, typename T2>
void CollectionMerger<T1, T2>::fill_output_obj(edm::Event &iEvent,
                                               std::unique_ptr<MergeCollection> &output,
                                               std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  assert(0);  // CV: make sure general function never gets called;
              //     always use template specializations
}

// Start with the Tracker collections
template <>
void CollectionMerger<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_tracker(output, inputCollections, true);
}

template <>
void CollectionMerger<edmNew::DetSetVector<SiStripCluster>, SiStripCluster>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_tracker(output, inputCollections);
}

// Next are the Calo entries
template <>
void CollectionMerger<edm::SortedCollection<EcalRecHit>, EcalRecHit>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_calo(output, inputCollections);
}

template <>
void CollectionMerger<edm::SortedCollection<HBHERecHit>, HBHERecHit>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_calo(output, inputCollections);
}

template <>
void CollectionMerger<edm::SortedCollection<HFRecHit>, HFRecHit>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_calo(output, inputCollections);
}

template <>
void CollectionMerger<edm::SortedCollection<HORecHit>, HORecHit>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_calo(output, inputCollections);
}

template <>
void CollectionMerger<edm::SortedCollection<CastorRecHit>, CastorRecHit>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_calo(output, inputCollections);
}

template <>
void CollectionMerger<edm::SortedCollection<ZDCRecHit>, ZDCRecHit>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_calo(output, inputCollections);
}

// Here the Muon Chamber
template <>
void CollectionMerger<edm::RangeMap<DTLayerId, edm::OwnVector<DTRecHit1DPair>>, DTRecHit1DPair>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_muonchamber(output, inputCollections);
}

template <>
void CollectionMerger<edm::RangeMap<CSCDetId, edm::OwnVector<CSCRecHit2D>>, CSCRecHit2D>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_muonchamber(output, inputCollections);
}

template <>
void CollectionMerger<edm::RangeMap<RPCDetId, edm::OwnVector<RPCRecHit>>, RPCRecHit>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_muonchamber(output, inputCollections);
}

// Here Electron Seeds
template <>
void CollectionMerger<std::vector<reco::ElectronSeed>, reco::ElectronSeed>::fill_output_obj(
    edm::Event &iEvent,
    std::unique_ptr<MergeCollection> &output,
    std::vector<edm::Handle<MergeCollection>> &inputCollections) {
  fill_output_obj_seed(iEvent, output, inputCollections);
}

DEFINE_FWK_MODULE(PixelColMerger);
DEFINE_FWK_MODULE(StripColMerger);

DEFINE_FWK_MODULE(ElectronSeedColMerger);
DEFINE_FWK_MODULE(EcalDrivenElectronSeedColMerger);

DEFINE_FWK_MODULE(EcalRecHitColMerger);
DEFINE_FWK_MODULE(HBHERecHitColMerger);
DEFINE_FWK_MODULE(HFRecHitColMerger);
DEFINE_FWK_MODULE(HORecHitColMerger);
DEFINE_FWK_MODULE(CastorRecHitColMerger);
DEFINE_FWK_MODULE(ZDCRecHitColMerger);

DEFINE_FWK_MODULE(DTRecHitColMerger);
DEFINE_FWK_MODULE(CSCRecHitColMerger);
DEFINE_FWK_MODULE(RPCRecHitColMerger);
