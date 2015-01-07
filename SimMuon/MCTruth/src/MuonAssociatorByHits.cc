#include "SimMuon/MCTruth/interface/MuonAssociatorByHits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "SimMuon/MCTruth/interface/TrackerMuonHitExtractor.h"
#include <sstream>

using namespace reco;
using namespace std;

MuonAssociatorByHits::MuonAssociatorByHits (const edm::ParameterSet& conf, edm::ConsumesCollector && iC) :  
  helper_(conf),
  conf_(conf)
{
  //hack for consumes
  RPCHitAssociator rpctruth(conf,std::move(iC));
  DTHitAssociator dttruth(conf,std::move(iC));
  CSCHitAssociator muonTruth(conf,std::move(iC));
  TrackerHitAssociator trackertruth(conf,std::move(iC));
}

//compatibility constructor - argh
MuonAssociatorByHits::MuonAssociatorByHits (const edm::ParameterSet& conf) :  
  helper_(conf),
  conf_(conf)
{
}



MuonAssociatorByHits::~MuonAssociatorByHits()
{
}

RecoToSimCollection  
MuonAssociatorByHits::associateRecoToSim( const edm::RefToBaseVector<reco::Track>& tC,
					  const edm::RefVector<TrackingParticleCollection>& TPCollectionH,
    					  const edm::Event * e, const edm::EventSetup * setup) const{
  RecoToSimCollection  outputCollection;

  MuonAssociatorByHitsHelper::TrackHitsCollection tH;
  for (auto it = tC.begin(), ed = tC.end(); it != ed; ++it) {
    tH.push_back(std::make_pair((*it)->recHitsBegin(), (*it)->recHitsEnd()));
  }

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  setup->get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

  bool printRtS(true);

  // Tracker hit association  
  TrackerHitAssociator trackertruth(*e, conf_);
  // CSC hit association
  CSCHitAssociator csctruth(*e,*setup,conf_);
  // DT hit association
  printRtS = false;
  DTHitAssociator dttruth(*e,*setup,conf_,printRtS);  
  // RPC hit association
  RPCHitAssociator rpctruth(*e,*setup,conf_);
   
  MuonAssociatorByHitsHelper::Resources resources = {tTopo, &trackertruth, &csctruth, &dttruth, &rpctruth};

  auto bareAssoc = helper_.associateRecoToSimIndices(tH, TPCollectionH, resources);
  for (auto it = bareAssoc.begin(), ed = bareAssoc.end(); it != ed; ++it) {
    for (auto itma = it->second.begin(), edma = it->second.end(); itma != edma; ++itma) {
        outputCollection.insert(tC[it->first], std::make_pair(edm::Ref<TrackingParticleCollection>(TPCollectionH, itma->idx), itma->quality));
    }
  }

  outputCollection.post_insert(); // perhaps not even necessary
  return outputCollection;
}

SimToRecoCollection  
MuonAssociatorByHits::associateSimToReco( const edm::RefToBaseVector<reco::Track>& tC, 
					  const edm::RefVector<TrackingParticleCollection>& TPCollectionH,
					  const edm::Event * e, const edm::EventSetup * setup) const{

  SimToRecoCollection  outputCollection;
  MuonAssociatorByHitsHelper::TrackHitsCollection tH;
  for (auto it = tC.begin(), ed = tC.end(); it != ed; ++it) {
    tH.push_back(std::make_pair((*it)->recHitsBegin(), (*it)->recHitsEnd()));
  }

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  setup->get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

  // Tracker hit association  
  TrackerHitAssociator trackertruth(*e, conf_);
  // CSC hit association
  CSCHitAssociator csctruth(*e,*setup,conf_);
  // DT hit association
  bool printRtS = false;
  DTHitAssociator dttruth(*e,*setup,conf_,printRtS);  
  // RPC hit association
  RPCHitAssociator rpctruth(*e,*setup,conf_);
   
  MuonAssociatorByHitsHelper::Resources resources = {tTopo, &trackertruth, &csctruth, &dttruth, &rpctruth};
  
  auto bareAssoc = helper_.associateSimToRecoIndices(tH, TPCollectionH, resources);
  for (auto it = bareAssoc.begin(), ed = bareAssoc.end(); it != ed; ++it) {
    for (auto itma = it->second.begin(), edma = it->second.end(); itma != edma; ++itma) {
        outputCollection.insert(edm::Ref<TrackingParticleCollection>(TPCollectionH, it->first),
                                std::make_pair(tC[itma->idx], itma->quality));
    }
  }

  outputCollection.post_insert(); // perhaps not even necessary
  return outputCollection;
}
