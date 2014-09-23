// -*- C++ -*-
//
// Package:    HGCalRecHitValidation
// Class:      HGCalRecHitValidation
// 
/**\class HGCalRecHitValidation HGCalRecHitValidation.cc Validaion/HGCalValidation/plugins/HGCalRecHitValidation.cc
   Description: Validates SimHits of High Granularity Calorimeter
   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Raman Khurana
////      and Kalyanmoy Chatterjee
//         Created:  Sunday, 17th Augst 2014 11:30:15 GMT
// $Id$

// system include files
#include "Validation/HGCalValidation/plugins/HGCalRecHitValidation.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "TVector3.h"
#include <cmath>


HGCalRecHitValidation::HGCalRecHitValidation(const edm::ParameterSet& iConfig){
  dbe_             = edm::Service<DQMStore>().operator->();
  nameDetector_    = iConfig.getParameter<std::string>("DetectorName");
  HGCRecHitSource_ = iConfig.getParameter<std::string>("RecHitSource");
  verbosity_       = iConfig.getUntrackedParameter<int>("Verbosity",0);
}


HGCalRecHitValidation::~HGCalRecHitValidation() { }

void HGCalRecHitValidation::analyze(const edm::Event& iEvent, 
				    const edm::EventSetup& iSetup) {
  OccupancyMap_plus.clear();
  OccupancyMap_minus.clear();

  edm::ESHandle<HGCalGeometry> geom;
  iSetup.get<IdealGeometryRecord>().get(nameDetector_,geom);
  const HGCalGeometry& geom0 = *geom;

  //get the HGC RecHit Collection
  edm::Handle<HGCRecHitCollection> theRecHitContainers;
  iEvent.getByLabel("HGCalRecHit", HGCRecHitSource_, theRecHitContainers);
  if (theRecHitContainers.isValid()) {
    if (verbosity_>0) std::cout << nameDetector_ << " with " << theRecHitContainers->size()
				<< " HGCRecHit element(s)"<< std::endl;

    for(HGCRecHitCollection::const_iterator it = theRecHitContainers->begin();
	it !=theRecHitContainers->end(); ++it) {
    
      DetId detId = it->id();

      int layer   = (detId.subdetId() == HGCEE) ? (HGCEEDetId(detId).layer()) : (HGCHEDetId(detId).layer());
      layer = layer;

      GlobalPoint global1 = geom0.getPosition(detId);
      float globalx = global1.x();
      float globaly = global1.y();
      float globalz = global1.z();
      
      double energy = it->energy();
      
      HitsInfo   hinfo;
      hinfo.energy = energy;
      hinfo.x      = globalx;
      hinfo.y      = globaly;
      hinfo.z      = globalz;
      hinfo.layer  = layer;
      hinfo.phi    = global1.phi();
      hinfo.eta    = global1.eta();
      
      if (verbosity_>1) std::cout << " --------------------------   gx = "
				  << globalx << " gy = "  << globaly   << " gz = "
				  << globalz << " phi = " << hinfo.phi << " eta = "
				  << hinfo.eta  << std::endl;
      
      FillHitsInfo(hinfo);
      
      double eta = hinfo.eta;
      if (eta > 0)  fillOccupancyMap(OccupancyMap_plus, layer -1);
      else          fillOccupancyMap(OccupancyMap_minus, layer -1);
      
      
    }// loop over hits ends here
    
    FillHitsInfo();
  }
  else if(verbosity_ > 0)
    std::cout << "HGCRecHitCollection Handle does not exist !!!" <<std::endl;
}
void HGCalRecHitValidation::fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer){
  if (OccupancyMap.find(layer) != OccupancyMap.end()) OccupancyMap[layer]++;
  else                                                OccupancyMap[layer] = 1;
}

void HGCalRecHitValidation::FillHitsInfo() { 

  for (auto itr = OccupancyMap_plus.begin() ; itr != OccupancyMap_plus.end(); ++itr) {
    int layer      = (*itr).first;
    int occupancy  = (*itr).second;
    HitOccupancy_Plus_.at(layer)->Fill(occupancy);
  }

  for (auto itr = OccupancyMap_minus.begin() ; itr != OccupancyMap_minus.end(); ++itr) {
    int layer      = (*itr).first;
    int occupancy  = (*itr).second;
    HitOccupancy_Minus_.at(layer)->Fill(occupancy);
  }
  
}

void HGCalRecHitValidation::FillHitsInfo(HitsInfo& hits) {

  unsigned int ilayer = hits.layer -1;
  energy_.at(ilayer)->Fill(hits.energy);

  EtaPhi_Plus_.at(ilayer)->Fill(hits.eta , hits.phi);
  EtaPhi_Minus_.at(ilayer)->Fill(hits.eta, hits.phi);

}

void HGCalRecHitValidation::beginJob() {}

void HGCalRecHitValidation::endJob() { }

void HGCalRecHitValidation::beginRun(edm::Run const& iRun, 
				     edm::EventSetup const& iSetup) {

  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get( pDD );
  const DDCompactView & cview = *pDD;
  hgcons_ = new HGCalDDDConstants(cview, nameDetector_);
  
  if (dbe_) {
    layers_ = hgcons_->layers(true);
    dbe_->setCurrentFolder("HGCalRecHitsV/"+nameDetector_);
    std::ostringstream histoname;

    for (unsigned int ilayer = 0; ilayer < layers_; ilayer++ ) {
      histoname.str(""); histoname << "HitOccupancy_Plus_layer_" << ilayer;
      HitOccupancy_Plus_.push_back(dbe_->book1D( histoname.str().c_str(), "RecHitOccupancy_Plus", 2000, 0, 10000));
      histoname.str(""); histoname << "HitOccupancy_Minus_layer_" << ilayer;
      HitOccupancy_Minus_.push_back(dbe_->book1D( histoname.str().c_str(), "RecHitOccupancy_Minus", 2000, 0, 10000));

      histoname.str(""); histoname << "EtaPhi_Plus_" << "layer_" << ilayer;
      EtaPhi_Plus_.push_back(dbe_->book2D(histoname.str().c_str(), "Occupancy", 100, 0, 2.5, 72, -3.15, 3.15));
      histoname.str(""); histoname << "EtaPhi_Minus_" << "layer_" << ilayer;
      EtaPhi_Minus_.push_back(dbe_->book2D(histoname.str().c_str(), "Occupancy", 100, -2.5, 0, 72, -3.15, 3.15));
      
      histoname.str(""); histoname << "energy_layer_" << ilayer; 
      energy_.push_back(dbe_->book1D(histoname.str().c_str(),"energy_",5000,0,0.002));
    }//loop over layers ends here 

    histoname.str(""); histoname << "SUMOfRecHitOccupancy_Plus";
    MeanHitOccupancy_Plus_= dbe_->book1D( histoname.str().c_str(), "SUMOfRecHitOccupancy_Plus", layers_, -0.5, layers_-0.5);
    histoname.str(""); histoname << "SUMOfRecHitOccupancy_Minus";
    MeanHitOccupancy_Minus_ = dbe_->book1D( histoname.str().c_str(), "SUMOfRecHitOccupancy_Minus", layers_, -0.5,layers_-0.5);
  }
}

void HGCalRecHitValidation::endRun(edm::Run const&, edm::EventSetup const&) {
  for(int ilayer=0; ilayer < (int)layers_; ++ilayer) {
    double meanVal = HitOccupancy_Plus_.at(ilayer)->getMean();
    MeanHitOccupancy_Plus_->setBinContent(ilayer+1, meanVal);
    meanVal = HitOccupancy_Minus_.at(ilayer)->getMean();
    MeanHitOccupancy_Minus_->setBinContent(ilayer+1, meanVal);
  }
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalRecHitValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalRecHitValidation);
