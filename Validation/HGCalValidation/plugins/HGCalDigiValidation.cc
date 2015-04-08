// -*- C++ -*-
//
// Package:    HGCalDigiValidation
// Class:      HGCalDigiValidation
// 
/**\class HGCalDigiValidation HGCalDigiValidation.cc Validaion/HGCalValidation/plugins/HGCalDigiValidation.cc
 Description: Validates SimHits of High Granularity Calorimeter
 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Raman Khurana
////      and Kalyanmoy Chatterjee
//         Created:  Fri, 31 Jan 2014 18:35:18 GMT
// $Id$

// system include files
#include "Validation/HGCalValidation/plugins/HGCalDigiValidation.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/FCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <cmath>

HGCalDigiValidation::HGCalDigiValidation(const edm::ParameterSet& iConfig){
  dbe_           = edm::Service<DQMStore>().operator->();
  nameDetector_  = iConfig.getParameter<std::string>("DetectorName");
  digiSource_    = iConfig.getParameter<edm::InputTag>("DigiSource");
  verbosity_     = iConfig.getUntrackedParameter<int>("Verbosity",0);
  SampleIndx_    = iConfig.getUntrackedParameter<int>("SampleIndx",5);
}


HGCalDigiValidation::~HGCalDigiValidation() { }

void HGCalDigiValidation::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup) { 
  OccupancyMap_plus_.clear();
  OccupancyMap_minus_.clear();
  
  const HGCalGeometry* geom0(0);
  const CaloGeometry*  geom1(0);
  if (nameDetector_ == "HCal") {
    edm::ESHandle<CaloGeometry> geom;
    iSetup.get<CaloGeometryRecord>().get(geom);
    if (!geom.isValid()) edm::LogWarning("HGCalValidation") << "Cannot get valid HGCalGeometry Object for " << nameDetector_;
    geom1 = geom.product();
  } else {
    edm::ESHandle<HGCalGeometry> geom;
    iSetup.get<IdealGeometryRecord>().get(nameDetector_, geom);
    if (!geom.isValid()) edm::LogWarning("HGCalValidation") << "Cannot get valid HGCalGeometry Object for " << nameDetector_;
    geom0 = geom.product();
  }

  unsigned int ntot(0), nused(0);
  if (nameDetector_ == "HGCalEESensitive") {
    //HGCalEE
    edm::Handle<HGCEEDigiCollection> theHGCEEDigiContainers;
    iEvent.getByLabel(digiSource_, theHGCEEDigiContainers);
    if (theHGCEEDigiContainers.isValid()) {
      if (verbosity_>0) 
	edm::LogInfo("HGCalValidation") << nameDetector_ << " with " 
					<< theHGCEEDigiContainers->size() 
					<< " element(s)";
      
      for (HGCEEDigiCollection::const_iterator it =theHGCEEDigiContainers->begin();
	   it !=theHGCEEDigiContainers->end(); ++it) {
	ntot++; nused++;
	HGCEEDetId detId     = (it->id());
	int        layer     = detId.layer();
	HGCSample  hgcSample = it->sample(SampleIndx_);
	uint16_t   gain      = hgcSample.gain();
	uint16_t   adc       = hgcSample.adc();
	double     charge    = adc*gain;
	digiValidation(detId, geom0, layer, adc, charge);
      }
      fillDigiInfo();
    } else {
      edm::LogWarning("HGCalValidation") << "HGCEEDigiCollection handle does not exist !!!";
    }
  } else if ((nameDetector_ == "HGCalHESiliconSensitive") || 
	     (nameDetector_ == "HGCalHEScintillatorSensitive")) {
    //HGCalHE
    edm::Handle<HGCHEDigiCollection> theHGCHEDigiContainers;
    iEvent.getByLabel(digiSource_, theHGCHEDigiContainers);
    if (theHGCHEDigiContainers.isValid()) {
      if (verbosity_>0) 
	edm::LogInfo("HGCalValidation") << nameDetector_ << " with " 
					<< theHGCHEDigiContainers->size()
					<< " element(s)";
      
      for (HGCHEDigiCollection::const_iterator it =theHGCHEDigiContainers->begin();
	   it !=theHGCHEDigiContainers->end(); ++it) {
	ntot++; nused++;
	HGCHEDetId detId     = (it->id());
	int        layer     = detId.layer();
	HGCSample  hgcSample = it->sample(SampleIndx_);
	uint16_t   gain      = hgcSample.gain();
	uint16_t   adc       = hgcSample.adc();
	double     charge    = adc*gain;
	digiValidation(detId, geom0, layer, adc, charge);
      }
      fillDigiInfo();
    } else {
      edm::LogWarning("HGCalValidation") << "HGCHEDigiCollection handle does not exist !!!";
    }
  } else if (nameDetector_ == "HCal") {
    //HE
    edm::Handle<edm::SortedCollection<HcalUpgradeDataFrame> >  theHEDigiContainers;
    iEvent.getByLabel(digiSource_, theHEDigiContainers);
    if (theHEDigiContainers.isValid()) {
      if (verbosity_>0) 
	edm::LogInfo("HGCalValidation") << nameDetector_ << " with " 
					<< theHEDigiContainers->size()
					<< " element(s)";
      edm::ESHandle<HcalDbService> conditions;
      iSetup.get<HcalDbRecord > ().get(conditions);

      for (edm::SortedCollection<HcalUpgradeDataFrame>::const_iterator it =theHEDigiContainers->begin();
	   it !=theHEDigiContainers->end(); ++it) {
	HcalDetId detId  = (it->id());
	ntot++;
	if (detId.subdet() == HcalEndcap) {
	  nused++;
	  HcalCalibrations calibrations = conditions->getHcalCalibrations(detId);
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder(detId);
	  const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
	  HcalCoderDb coder(*channelCoder, *shape);
	  CaloSamples tool;
	  coder.adc2fC(*it, tool);
	  int       layer  = detId.depth();
	  uint16_t  adc    = (*it)[SampleIndx_].adc();
	  int       capid  = (*it)[SampleIndx_].capid();
	  double    charge = (tool[SampleIndx_] - calibrations.pedestal(capid));
	  digiValidation(detId, geom1, layer, adc, charge);
	}
      }
      fillDigiInfo();
    } else {
      edm::LogWarning("HGCalValidation") << "HGCHEDigiCollection handle does not exist !!!";
    }
  } else {
    edm::LogWarning("HGCalValidation") << "invalid detector name !! " 
				       << nameDetector_;
  }
  edm::LogInfo("HGCalValidation") << "Event " << iEvent.id().event()
				  << " with " << ntot << " total and "
				  << nused << " used digis";
}

template<class T1, class T2>
void HGCalDigiValidation::digiValidation(const T1& detId, const T2* geom, 
					 int layer, uint16_t adc, double charge) {
  
  //  std::vector<double> charges;
  if (verbosity_>1) edm::LogInfo("HGCalValidation") << detId;
  DetId id1 = DetId(detId.rawId());
  GlobalPoint global1 = geom->getPosition(id1);
  
  if (verbosity_>1) 
    edm::LogInfo("HGCalValidation") << " adc = "         <<  adc
				    << " charge = "      <<  charge;
  
  digiInfo   hinfo; 
  hinfo.x       =  global1.x();
  hinfo.y       =  global1.y();
  hinfo.z       =  global1.z();
  hinfo.adc     =  adc;
  hinfo.charge  =  charge; //charges[0];
  hinfo.layer   =  layer;
  
  if (verbosity_>1) 
    edm::LogInfo("HGCalValidation") << "gx =  "  << hinfo.x
				    << " gy = "  << hinfo.y
				    << " gz = "  << hinfo.z;
  
  fillDigiInfo(hinfo);

  if (global1.eta() > 0)  fillOccupancyMap(OccupancyMap_plus_, layer -1);
  else                    fillOccupancyMap(OccupancyMap_minus_, layer -1);
  
}

void HGCalDigiValidation::fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer) {
  if (OccupancyMap.find(layer) != OccupancyMap.end()) OccupancyMap[layer] ++;
  else                                                OccupancyMap[layer] = 1;
}

void HGCalDigiValidation::fillDigiInfo(digiInfo& hinfo) {
  int ilayer = hinfo.layer -1;
  charge_.at(ilayer)->Fill(hinfo.charge);
  DigiOccupancy_XY_.at(ilayer)->Fill(hinfo.x, hinfo.y);
  ADC_.at(ilayer)->Fill(hinfo.adc);
}

void HGCalDigiValidation::fillDigiInfo() {
  if(!dbe_) return;
  for (auto itr = OccupancyMap_plus_.begin(); 
       itr != OccupancyMap_plus_.end(); ++itr) {
    int layer = (*itr).first;
    int occupancy = (*itr).second;
    DigiOccupancy_Plus_.at(layer)->Fill(occupancy);
  }
  for (auto itr = OccupancyMap_minus_.begin(); 
       itr != OccupancyMap_minus_.end(); ++itr) {
    int layer = (*itr).first;
    int occupancy = (*itr).second;
    DigiOccupancy_Minus_.at(layer)->Fill(occupancy);
  }
}

void HGCalDigiValidation::beginJob() { }

void HGCalDigiValidation::endJob() { }

void HGCalDigiValidation::beginRun(edm::Run const& iRun, 
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
    HGCalDDDConstants  *hgcons_ = new HGCalDDDConstants(cview, nameDetector_);
    layers_ = hgcons_->layers(true);
  }
  
  if (dbe_) {
    dbe_->setCurrentFolder("HGCalDigiV/"+nameDetector_);
    
    if (verbosity_>0) 
      edm::LogInfo("HGCalValidation") << "current DQM directory:  "
				      << "HGCalDigiV/" << nameDetector_ 
				      << "  layer = "<< layers_;
    std::ostringstream histoname;
    for (int ilayer = 0; ilayer < layers_; ilayer++ ) {
      histoname.str(""); histoname << "charge_"<< "layer_" << ilayer;
      charge_.push_back(dbe_->book1D(histoname.str().c_str(),"charge_",1000,-25,25));
      
      histoname.str(""); histoname << "ADC_" << "layer_" << ilayer;
      ADC_.push_back(dbe_->book1D(histoname.str().c_str(), "DigiOccupancy", 1000, 0, 1000));
      
      histoname.str(""); histoname << "DigiOccupancy_XY_" << "layer_" << ilayer;
      DigiOccupancy_XY_.push_back(dbe_->book2D(histoname.str().c_str(), "DigiOccupancy", 1000, -500, 500, 1000, -500, 500));
      
      histoname.str(""); histoname << "DigiOccupancy_Plus_" << "layer_" << ilayer;
      DigiOccupancy_Plus_.push_back(dbe_->book1D(histoname.str().c_str(), "DigiOccupancy +z", 1000, 0, 1000));
      histoname.str(""); histoname << "DigiOccupancy_Minus_" << "layer_" << ilayer;
      DigiOccupancy_Minus_.push_back(dbe_->book1D(histoname.str().c_str(), "DigiOccupancy -z", 1000, 0, 1000));
    }

    histoname.str(""); histoname << "SUMOfDigiOccupancy_Plus";
    //    MeanDigiOccupancy_Plus_.push_back(dbe_->book1D( histoname.str().c_str(), "SUMOfDigiOccupancy_Plus", 
    MeanDigiOccupancy_Plus_ = dbe_->book1D( histoname.str().c_str(), "SUMOfDigiOccupancy_Plus", layers_, -0.5, layers_-0.5);
    histoname.str(""); histoname << "SUMOfRecDigiOccupancy_Minus";
    MeanDigiOccupancy_Minus_ = dbe_->book1D( histoname.str().c_str(), "SUMOfDigiOccupancy_Minus", layers_, -0.5,layers_-0.5);
  }
  
}

// ------------ method called when ending the processing of a run  ------------

void HGCalDigiValidation::endRun(edm::Run const&, edm::EventSetup const&) { 
  if (!dbe_) return;
  for(int ilayer=0; ilayer < (int)layers_; ++ilayer) {
    double meanVal = DigiOccupancy_Plus_.at(ilayer)->getMean();
    MeanDigiOccupancy_Plus_->setBinContent(ilayer+1, meanVal);
    meanVal = DigiOccupancy_Minus_.at(ilayer)->getMean();
    MeanDigiOccupancy_Minus_->setBinContent(ilayer+1, meanVal);
  }
}


// ------------ method called when starting to processes a luminosity block  ------------
/*
void HGCalDigiValidation::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void HGCalDigiValidation::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalDigiValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalDigiValidation);
