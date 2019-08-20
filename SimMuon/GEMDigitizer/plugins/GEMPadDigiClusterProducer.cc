#ifndef SimMuon_GEMDigitizer_GEMPadDigiClusterProducer_h
#define SimMuon_GEMDigitizer_GEMPadDigiClusterProducer_h

/**
 *  \class ME0PadDigiClusterProducer
 *
 *  Produces GEM pad clusters from at most 8 adjacent GEM pads.
 *  Clusters are used downstream in the CSC local trigger to build
 *  GEM-CSC triggers and in the muon trigger to build EMTF tracks
 *
 *  Based on documentation provided by the GEM firmware architects
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
  typedef std::map<GEMDetId, GEMPadDigiCluster> GEMPadDigiClusterContainer;

  explicit GEMPadDigiClusterProducer(const edm::ParameterSet& ps);

  ~GEMPadDigiClusterProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /**
   *
   *************************************
   ** Light Cluster Packing Algorithm **
   *************************************

   Based on: https://github.com/cms-gem-daq-project/OptoHybridv3/raw/master/doc/OH_modules.docx
   (Andrew Peck, Thomas Lenzi, Evaldas Juska)

   In the current version of the algorithm, cluster finding is segmented
   into two separate halves of the GE1/1 chambers. Thus, each one of the
   trigger fibers can transmit clusters only from the half of the chamber
   that it corresponds to. For GE2/1, there are four separate quarts of
   the GE2/1 chamber.

   This has the downside of being unable to transmit more than 4 clusters
   when they occur within that side of the chamber, so there will be a
   slightly higher rate of cluster overflow. For GE2/1 each OH can transmit
   up to 5 clusters.

   The benefit, however, is in terms of (1) latency and (2) resource usage.

   The burden of finding clusters on  of the chamber is significantly less,
   and allows the cluster packer to operate in a simple, pipelined architecture
   which returns up to 4 (or 5) clusters per half-chamber per bunch crossing.

   This faster architecture allows the mechanism to operate with only a
   single copy of the cluster finding priority encoder and cluster truncator
   (instead of two multiplexed copies), so the total resource usage of
   these stages is approximately half.

   Further, a second step of cluster merging that is required in the full
   algorithm is avoided, which reduces latency by an additional bunch
   crossing and significantly reduces resource usage as well.

   The sorting of the clusters favors lower eta partitions and lower pad numbers
  */

  void buildClusters(const GEMPadDigiCollection& pads, GEMPadDigiClusterContainer& out_clusters) const;
  void selectClusters(GEMPadDigiClusterContainer& in, GEMPadDigiClusterCollection& out) const;

  /// Name of input digi Collection
  edm::EDGetTokenT<GEMPadDigiCollection> pad_token_;
  edm::InputTag pads_;

  unsigned int maxClustersOHGE11_;
  unsigned int maxClustersOHGE21_;
  unsigned int nOHGE11_;
  unsigned int nOHGE21_;
  unsigned int maxClusterSize_;

  const GEMGeometry* geometry_;
};

GEMPadDigiClusterProducer::GEMPadDigiClusterProducer(const edm::ParameterSet& ps) : geometry_(nullptr) {
  pads_ = ps.getParameter<edm::InputTag>("InputCollection");
  maxClustersOHGE11_ = ps.getParameter<unsigned int>("maxClustersOHGE11");
  maxClustersOHGE21_ = ps.getParameter<unsigned int>("maxClustersOHGE21");
  nOHGE11_ = ps.getParameter<unsigned int>("nOHGE11");
  nOHGE21_ = ps.getParameter<unsigned int>("nOHGE21");
  maxClusterSize_ = ps.getParameter<unsigned int>("maxClusterSize");

  pad_token_ = consumes<GEMPadDigiCollection>(pads_);

  produces<GEMPadDigiClusterCollection>();
  consumes<GEMPadDigiCollection>(pads_);
}

GEMPadDigiClusterProducer::~GEMPadDigiClusterProducer() {}

void GEMPadDigiClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputCollection", edm::InputTag("simMuonGEMPadDigis"));
  desc.add<unsigned int>("maxClustersOHGE11", 4);
  desc.add<unsigned int>("maxClustersOHGE21", 5);
  desc.add<unsigned int>("nOHGE11", 2);
  desc.add<unsigned int>("nOHGE21", 4);
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

  // build the proto clusters
  GEMPadDigiClusterContainer proto_clusters;
  buildClusters(*(hpads.product()), proto_clusters);

  // select the clusters from proto clusters
  selectClusters(proto_clusters, *pClusters);

  // store them in the event
  e.put(std::move(pClusters));
}

void GEMPadDigiClusterProducer::buildClusters(const GEMPadDigiCollection& det_pads,
                                              GEMPadDigiClusterContainer& proto_clusters) const {
  // clear the container
  proto_clusters.clear();

  // construct clusters
  for (const auto& part : geometry_->etaPartitions()) {
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
          proto_clusters.emplace(part->id(), pad_cluster);

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
      proto_clusters.emplace(part->id(), pad_cluster);
    }
  }  // end of partition loop
}

void GEMPadDigiClusterProducer::selectClusters(GEMPadDigiClusterContainer& proto_clusters,
                                               GEMPadDigiClusterCollection& out_clusters) const {
  // construct clusters
  for (const auto& ch : geometry_->chambers()) {
    const int station = ch->id().station();
    const bool isGE11 = (station == 1);
    const unsigned nOH = isGE11 ? nOHGE11_ : nOHGE21_;
    const unsigned maxClustersOH = isGE11 ? maxClustersOHGE11_ : maxClustersOHGE21_;
    const unsigned nPartOH = ch->nEtaPartitions() / nOH;

    // loop over all the optohybrids
    for (unsigned int iOH = 0; iOH < nOH; iOH++) {
      // all clusters for a set of eta partitions
      GEMPadDigiClusterContainer temp_clusters;

      // loop over the 4 or 2 eta partitions for this optohybrid
      for (unsigned iPart = 0; iPart < nPartOH; iPart++) {
        // get the clusters for this eta partition
        const GEMDetId& partId = ch->etaPartition(iPart + iOH * nPartOH)->id();
        temp_clusters.emplace(partId, proto_clusters[partId]);
      }

      // cluster selection: pick first maxClusters for now for each OH
      unsigned loopMax = std::min(maxClustersOH, unsigned(temp_clusters.size()));
      unsigned nClusters = 0;
      for (const auto& p : temp_clusters) {
        if (nClusters > loopMax)
          break;
        const auto& detid(p.first);
        const auto& cluster(p.second);
        // now add the clusters to the output collection
        out_clusters.insertDigi(detid, cluster);
        nClusters++;
      }
    }
  }  // end of chamber loop
}

DEFINE_FWK_MODULE(GEMPadDigiClusterProducer);

#endif
