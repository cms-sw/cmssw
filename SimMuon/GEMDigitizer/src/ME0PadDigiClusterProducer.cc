#include "SimMuon/GEMDigitizer/interface/ME0PadDigiClusterProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <map>
#include <vector>


ME0PadDigiClusterProducer::ME0PadDigiClusterProducer(const edm::ParameterSet& ps)
: geometry_(nullptr)
{
  pads_ = ps.getParameter<edm::InputTag>("InputCollection");
  maxClusters_ = ps.getParameter<unsigned int>("maxClusters");
  maxClusterSize_ = ps.getParameter<unsigned int>("maxClusterSize");

  pad_token_ = consumes<ME0PadDigiCollection>(pads_);

  produces<ME0PadDigiClusterCollection>();
  consumes<ME0PadDigiCollection>(pads_);
}


ME0PadDigiClusterProducer::~ME0PadDigiClusterProducer()
{}


void ME0PadDigiClusterProducer::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup)
{
  edm::ESHandle<ME0Geometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  geometry_ = &*hGeom;
}


void ME0PadDigiClusterProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  edm::Handle<ME0PadDigiCollection> hpads;
  e.getByToken(pad_token_, hpads);

  // Create empty output
  std::unique_ptr<ME0PadDigiClusterCollection> pClusters(new ME0PadDigiClusterCollection());

  // build the clusters
  buildClusters(*(hpads.product()), *pClusters);

  // store them in the event
  e.put(std::move(pClusters));
}


void ME0PadDigiClusterProducer::buildClusters(const ME0PadDigiCollection &det_pads,
                                              ME0PadDigiClusterCollection &out_clusters)
{
  // construct clusters
  for (const auto& ch: geometry_->layers()) {

    // proto collection
    std::vector<std::pair<ME0DetId, ME0PadDigiCluster> > proto_clusters;

    for (const auto& part: ch->etaPartitions()) {
      auto pads = det_pads.get(part->id());
      std::vector<uint16_t> cl;
      int startBX = 99;
      for (auto d = pads.first; d != pads.second; ++d) {
        if (cl.empty()) {
          cl.push_back((*d).pad());
        }
        else {
          if ((*d).bx() == startBX and // same bunch crossing
              (*d).pad() == cl.back() + 1 // pad difference is 1
              and cl.size()<maxClusterSize_) { // max 8 in cluster
            cl.push_back((*d).pad());
          }
          else {
            // put the current cluster in the proto collection
            ME0PadDigiCluster pad_cluster(cl, startBX);
            proto_clusters.emplace_back(part->id(), pad_cluster);

            // start a new cluster
            cl.clear();
            cl.push_back((*d).pad());
          }
        }
        startBX = (*d).bx();
      }
      // put the last cluster in the proto collection
      if (pads.first != pads.second){
        ME0PadDigiCluster pad_cluster(cl, startBX);
        proto_clusters.emplace_back(part->id(), pad_cluster);
      }
    } // end of partition loop

    // cluster selection: pick first maxClusters_ for now
    unsigned loopMax=std::min(maxClusters_,unsigned(proto_clusters.size()));
    for ( unsigned int i=0; i<loopMax; i++) {
      const auto& detid(proto_clusters[i].first);
      const auto& cluster(proto_clusters[i].second);
      out_clusters.insertDigi(detid, cluster);
    }
  } // end of chamber loop
}
