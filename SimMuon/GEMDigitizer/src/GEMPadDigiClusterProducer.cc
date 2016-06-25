#include "SimMuon/GEMDigitizer/interface/GEMPadDigiClusterProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <map>
#include <vector>


GEMPadDigiClusterProducer::GEMPadDigiClusterProducer(const edm::ParameterSet& ps)
: geometry_(nullptr)
{
  pads_ = ps.getParameter<edm::InputTag>("InputCollection");
  maxClusters_ = ps.getParameter<unsigned int>("maxClusters");
  maxClusterSize_ = ps.getParameter<unsigned int>("maxClusterSize");

  pad_token_ = consumes<GEMPadDigiCollection>(pads_);

  produces<GEMPadDigiClusterCollection>();
  consumes<GEMPadDigiCollection>(pads_);
}


GEMPadDigiClusterProducer::~GEMPadDigiClusterProducer()
{}


void GEMPadDigiClusterProducer::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  geometry_ = &*hGeom;
}


void GEMPadDigiClusterProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  edm::Handle<GEMPadDigiCollection> hpads;
  e.getByToken(pad_token_, hpads);

  // Create empty output
  std::unique_ptr<GEMPadDigiClusterCollection> pClusters(new GEMPadDigiClusterCollection());

  // build the clusters
  buildClusters(*(hpads.product()), *pClusters);

  for (auto pad_range_it = pClusters->begin(); pad_range_it != pClusters->end(); ++pad_range_it)
  {
    auto id = (*pad_range_it).first;
    //    auto roll = geometry_->etaPartition(id);
    
    auto pads_range = (*pad_range_it).second;
    for (auto p = pads_range.first; p != pads_range.second; ++p)
    {  
      std::cout<<id <<" paddigi(pad,bx) "<<*p << std::endl;
    }
  }
  // store them in the event
  e.put(std::move(pClusters));
}


void GEMPadDigiClusterProducer::buildClusters(const GEMPadDigiCollection &det_pads, GEMPadDigiClusterCollection &out_clusters)
{
  auto etaPartitions = geometry_->etaPartitions();
  std::cout << "Build pad digi clusters"  << std::endl;
  for(auto p: etaPartitions)
  {
    auto pads = det_pads.get(p->id());
    //unsigned int clusters = 0;
    std::vector<GEMPadDigi> cl;//(maxClusterSize_);
    for (auto d = pads.first; d != pads.second; ++d)
    {
      std::cout << "Check status of " << *d << " " << GEMDetId(p->id()) << std::endl;
      if (cl.size() == 0) {
	cl.push_back(*d);
	std::cout << "\tadd first pad to cluster " << *d << std::endl;
      }
      else {
	std::cout << "\tCheck " << (*d).bx() << " " << cl.back().bx() << " " << (*d).pad() << " " << cl.back().pad() + 1 << std::endl; 
	if ((*d).bx() == cl.back().bx() and (*d).pad() == cl.back().pad() + 1) {
	  std::cout << "\t\tsuccess" << std::endl;
	  cl.push_back(*d);
	  std::cout << "\tadd pad to cluster " << *d << std::endl;
	}
	else {
	  std::cout << "\t\tfailed" << std::endl;
	  GEMPadDigiCluster pad_cluster(cl.front().pad(), cl.back().pad(), cl.front().bx());
	  out_clusters.insertDigi(p->id(), pad_cluster);
	  std::cout << "\tdefine new cluster " << pad_cluster << std::endl;
	  cl.clear();
	  cl.push_back(*d);
	  std::cout << "\tadd pad to cluster " << *d << std::endl;
	}
      }
      std::cout << std::endl;
    }
    if (pads.first != pads.second){
      GEMPadDigiCluster pad_cluster(cl.front().pad(), cl.back().pad(), cl.front().bx());
      out_clusters.insertDigi(p->id(), pad_cluster);
      std::cout << "\tdefine new cluster " << pad_cluster << std::endl;
    } 
  }
}

      // auto pad = *d;     
      // if (cl.size() != 0 and (clusterBX != pad.bx() or cl.back()+1 != pad.pad())){
      // 	GEMPadDigiCluster pad_cluster(cl.front(), cl.back(), clusterBX);
      // 	out_clusters.insertDigi(p->id(), pad_cluster);
      // 	std::cout << "\tdefine new cluster " << pad_cluster << std::endl;  	
      // 	cl.clear();
      // 	clusters++;
      // 	clusterBX = pad.bx();
      // 	if (clusters == 8) break;
      // }
      // cl.push_back(pad.pad());
      // std::cout << GEMDetId(p->id()) << " " << pad << " current cluster size "<< cl.size()<< " " << clusterBX << " " << cl.back() << std::endl;
    //   std::cout << "current cluster size "<< cl.size() << std::endl;
    //   std::cout << "current cluster BX "<< clusterBX << std::endl;

    //   auto pad = *d;
    //   std::cout << GEMDetId(p->id()) << " " << pad << " " << pad.bx() << " " << cl.back() << " " << pad.pad() << std::endl;
    //   if (cl.size()==0 or (clusterBX == pad.bx() and cl.back()+1 == pad.pad())) {
    // 	std::cout << "\tadd pad to cluster" << std::endl;
    // 	cl.push_back(pad.pad());
    //     clusterBX = pad.bx();
    // 	//if (cl.size() == maxClusterSize_) break;
    //   }
    //   else {
    // 	std::cout << "\tdefine new cluster " << std::endl;
    // 	std::cout << "\tfirst " << cl.front() << " back " << cl.back() << " BX " << clusterBX << std::endl;
    // 	GEMPadDigiCluster pad_cluster(cl.front(), cl.back(), clusterBX);
    // 	out_clusters.insertDigi(p->id(), pad_cluster);
    // 	cl.clear();
    // 	clusters++;
    //   }
    //   std::cout << std::endl;
    // }
    // if (clusters == maxClusters_) break;
