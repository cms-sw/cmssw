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


void ME0PadDigiClusterProducer::buildClusters(const ME0PadDigiCollection &det_pads, ME0PadDigiClusterCollection &out_clusters)
{
  for (const auto& ch: geometry_->chambers()) {
    for (const auto& part: ch->etaPartitions()) {
      auto pads = det_pads.get(part->id());
      std::vector<uint16_t> cl;
      int startBX = 99;
      for (auto d = pads.first; d != pads.second; ++d) {
        if (cl.empty()) {
          cl.push_back((*d).pad());
        }
        else {
          if ((*d).bx() == startBX and (*d).pad() == cl.back() + 1) {
            cl.push_back((*d).pad());
          }
          else {
            ME0PadDigiCluster pad_cluster(cl, startBX);
            out_clusters.insertDigi(part->id(), pad_cluster);
            cl.clear();
            cl.push_back((*d).pad());
          }
        }
        startBX = (*d).bx();
      }
      if (pads.first != pads.second){
        ME0PadDigiCluster pad_cluster(cl, startBX);
        out_clusters.insertDigi(part->id(), pad_cluster);
      }
    }
  }
}
