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
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/FCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "TVector3.h"
#include <cmath>


HGCalRecHitValidation::HGCalRecHitValidation(const edm::ParameterSet& iConfig){
  dbe_             = edm::Service<DQMStore>().operator->();
  nameDetector_    = iConfig.getParameter<std::string>("DetectorName");
  recHitSource_    = iConfig.getParameter<edm::InputTag>("RecHitSource");
  verbosity_       = iConfig.getUntrackedParameter<int>("Verbosity",0);
}


HGCalRecHitValidation::~HGCalRecHitValidation() { }

void HGCalRecHitValidation::analyze(const edm::Event& iEvent, 
				    const edm::EventSetup& iSetup) {
  OccupancyMap_plus.clear();
  OccupancyMap_minus.clear();

  bool ok(true);
  unsigned int ntot(0), nused(0);
  if (nameDetector_ == "HCal") {
    edm::ESHandle<CaloGeometry> geom;
    iSetup.get<CaloGeometryRecord>().get(geom);
    if (!geom.isValid()) edm::LogWarning("HGCalValidation") << "Cannot get valid HGCalGeometry Object for " << nameDetector_;
    const CaloGeometry* geom0 = geom.product();

    edm::Handle<HBHERecHitCollection> hbhecoll;
    iEvent.getByLabel(recHitSource_, hbhecoll);
    if (hbhecoll.isValid()) {
      if (verbosity_>0) 
	edm::LogInfo("HGCalValidation") << nameDetector_ << " with " 
					<< hbhecoll->size() << " element(s)";
      for (HBHERecHitCollection::const_iterator it=hbhecoll->begin(); 
	   it != hbhecoll->end(); ++it) {
	DetId detId = it->id();
	ntot++;
	if (detId.subdetId() == HcalEndcap) {
	  nused++;
	  int   layer = HcalDetId(detId).depth();
	  recHitValidation(detId, layer, geom0, it);
	}
      }
    } else {
      ok = false;
      edm::LogWarning("HGCalValidation") << "HBHERecHitCollection Handle does not exist !!!";
    }
  } else {
    edm::ESHandle<HGCalGeometry> geom;
    iSetup.get<IdealGeometryRecord>().get(nameDetector_, geom);
    if (!geom.isValid()) edm::LogWarning("HGCalValidation") << "Cannot get valid HGCalGeometry Object for " << nameDetector_;
    const HGCalGeometry* geom0 = geom.product();

    edm::Handle<HGCRecHitCollection> theRecHitContainers;
    iEvent.getByLabel(recHitSource_, theRecHitContainers);
    if (theRecHitContainers.isValid()) {
      if (verbosity_>0) 
	edm::LogInfo("HGCalValidation") << nameDetector_ << " with " 
					<< theRecHitContainers->size()
					<< " element(s)";
      for (HGCRecHitCollection::const_iterator it=theRecHitContainers->begin();
	   it !=theRecHitContainers->end(); ++it) {
	ntot++; nused++;
	DetId detId = it->id();
	int layer   = (detId.subdetId() == HGCEE) ? (HGCEEDetId(detId).layer()) : (HGCHEDetId(detId).layer());
	recHitValidation(detId, layer, geom0, it);
      }
    } else {
      ok = false;
      edm::LogWarning("HGCalValidation") << "HGCRecHitCollection Handle does not exist !!!";
    }
  }
  if (ok) fillHitsInfo();
  edm::LogWarning("HGCalValidation") << "Event " << iEvent.id().event()
				     << " with " << ntot << " total and "
				     << nused << " used recHits";
}

template<class T1, class T2>
void HGCalRecHitValidation::recHitValidation(DetId & detId, int layer, 
					     const T1* geom, T2 it) {

  GlobalPoint global = geom->getPosition(detId);
  double      energy = it->energy();

  float globalx = global.x();
  float globaly = global.y();
  float globalz = global.z();
      
  HitsInfo   hinfo;
  hinfo.energy = energy;
  hinfo.x      = globalx;
  hinfo.y      = globaly;
  hinfo.z      = globalz;
  hinfo.layer  = layer;
  hinfo.phi    = global.phi();
  hinfo.eta    = global.eta();
      
  if (verbosity_>1) 
    edm::LogInfo("HGCalValidation") << " --------------------------   gx = "
				    << globalx << " gy = "  << globaly   
				    << " gz = " << globalz << " phi = " 
				    << hinfo.phi << " eta = " << hinfo.eta;
      
  fillHitsInfo(hinfo);
      
  if (hinfo.eta > 0)  fillOccupancyMap(OccupancyMap_plus, layer -1);
  else                fillOccupancyMap(OccupancyMap_minus, layer -1);
      
}      

void HGCalRecHitValidation::fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer){
  if (OccupancyMap.find(layer) != OccupancyMap.end()) OccupancyMap[layer]++;
  else                                                OccupancyMap[layer] = 1;
}

void HGCalRecHitValidation::fillHitsInfo() { 

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

void HGCalRecHitValidation::fillHitsInfo(HitsInfo& hits) {

  unsigned int ilayer = hits.layer -1;
  energy_.at(ilayer)->Fill(hits.energy);

  EtaPhi_Plus_.at(ilayer)->Fill(hits.eta , hits.phi);
  EtaPhi_Minus_.at(ilayer)->Fill(hits.eta, hits.phi);

}

void HGCalRecHitValidation::beginJob() {}

void HGCalRecHitValidation::endJob() { }

void HGCalRecHitValidation::beginRun(edm::Run const& iRun, 
				     edm::EventSetup const& iSetup) {

  if (nameDetector_ == "HCal") {
    edm::ESHandle<HcalDDDRecConstants> pHRNDC;
    iSetup.get<HcalRecNumberingRecord>().get( pHRNDC );
    const HcalDDDRecConstants *hcons   = &(*pHRNDC);
    layers_ = hcons->getMaxDepth(1);
  } else {
    edm::ESTransientHandle<DDCompactView> pDD;
    iSetup.get<IdealGeometryRecord>().get( pDD );
    const DDCompactView & cview = *pDD;
    HGCalDDDConstants   *hgcons_ = new HGCalDDDConstants(cview, nameDetector_);
    layers_ = hgcons_->layers(true);
  }

  if (dbe_) {
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

  for (int ilayer=0; ilayer < (int)layers_; ++ilayer) {
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
