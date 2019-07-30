#ifndef SimMuon_GEMDigitizer_GEMPadDigiClusterProducer_h
#define SimMuon_GEMDigitizer_GEMPadDigiClusterProducer_h

/**
 *  \class ME0PadDigiClusterProducer
 *
 *  Produces GEM pad clusters from at most 8 adjacent GEM pads.
 *  Clusters are used downstream in the CSC local trigger to build
 *  GEM-CSC triggers and in the muon trigger to build EMTF tracks
 *
 *  \author Sven Dildick (TAMU)
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"

#include <string>
#include <map>
#include <vector>

class GEMPadDigiClusterProducer : public edm::stream::EDProducer<> {
public:
  explicit GEMPadDigiClusterProducer(const edm::ParameterSet& ps);

  ~GEMPadDigiClusterProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void buildClusters(const GEMPadDigiCollection& pads, GEMPadDigiClusterCollection& out_clusters);

  /// Name of input digi Collection
  edm::EDGetTokenT<GEMPadDigiCollection> pad_token_;
  edm::InputTag pads_;

  unsigned int maxClusters_;
  unsigned int maxClusterSize_;

  const GEMGeometry* geometry_;
};

GEMPadDigiClusterProducer::GEMPadDigiClusterProducer(const edm::ParameterSet& ps) : geometry_(nullptr) {
  pads_ = ps.getParameter<edm::InputTag>("InputCollection");
  maxClusters_ = ps.getParameter<unsigned int>("maxClusters");
  maxClusterSize_ = ps.getParameter<unsigned int>("maxClusterSize");

  pad_token_ = consumes<GEMPadDigiCollection>(pads_);

  produces<GEMPadDigiClusterCollection>();
  consumes<GEMPadDigiCollection>(pads_);
}

GEMPadDigiClusterProducer::~GEMPadDigiClusterProducer() {}

void GEMPadDigiClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputCollection", edm::InputTag("simMuonGEMPadDigis"));
  desc.add<unsigned int>("maxClusters", 8);
  desc.add<unsigned int>("maxClusterSize", 8);

  descriptions.add("simMuonGEMPadDigiClustersDef", desc);
}

void GEMPadDigiClusterProducer::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {
  edm::ESHandle<GEMGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  geometry_ = &*hGeom;
}

void GEMPadDigiClusterProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Handle<GEMPadDigiCollection> hpads;
  e.getByToken(pad_token_, hpads);

  // Create empty output
  std::unique_ptr<GEMPadDigiClusterCollection> pClusters(new GEMPadDigiClusterCollection());

  // build the clusters
  buildClusters(*(hpads.product()), *pClusters);

  // store them in the event
  e.put(std::move(pClusters));
}

void GEMPadDigiClusterProducer::buildClusters(const GEMPadDigiCollection& det_pads,
                                              GEMPadDigiClusterCollection& out_clusters) {
  // construct clusters
  for (const auto& ch : geometry_->chambers()) {
    // proto collection
    std::vector<std::pair<GEMDetId, GEMPadDigiCluster> > proto_clusters;

    for (const auto& part : ch->etaPartitions()) {
      auto pads = det_pads.get(part->id());
      std::vector<uint16_t> cl;
      int startBX = 99;
      for (auto d = pads.first; d != pads.second; ++d) {
        if (cl.empty()) {
          cl.push_back((*d).pad());
        } else {
          if ((*d).bx() == startBX and            // same bunch crossing
              (*d).pad() == cl.back() + 1         // pad difference is 1
              and cl.size() < maxClusterSize_) {  // max 8 in cluster
            cl.push_back((*d).pad());
          } else {
            // put the current cluster in the proto collection
            GEMPadDigiCluster pad_cluster(cl, startBX);
            proto_clusters.emplace_back(part->id(), pad_cluster);

            // start a new cluster
            cl.clear();
            cl.push_back((*d).pad());
          }
        }
        startBX = (*d).bx();
      }
      // put the last cluster in the proto collection
      if (pads.first != pads.second) {
        GEMPadDigiCluster pad_cluster(cl, startBX);
        proto_clusters.emplace_back(part->id(), pad_cluster);
      }
    }  // end of partition loop

    // cluster selection: pick first maxClusters_ for now
    unsigned loopMax = std::min(maxClusters_, unsigned(proto_clusters.size()));
    for (unsigned int i = 0; i < loopMax; i++) {
      const auto& detid(proto_clusters[i].first);
      const auto& cluster(proto_clusters[i].second);
      out_clusters.insertDigi(detid, cluster);
    }
  }  // end of chamber loop
}

DEFINE_FWK_MODULE(GEMPadDigiClusterProducer);
#endif
