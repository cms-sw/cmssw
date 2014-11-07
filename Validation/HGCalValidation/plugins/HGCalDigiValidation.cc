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
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
/////
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
////
#include "FWCore/Utilities/interface/InputTag.h" // included to header also, have to remove from here

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <cmath>

HGCalDigiValidation::HGCalDigiValidation(const edm::ParameterSet& iConfig){
  dbe_           = edm::Service<DQMStore>().operator->();
  nameDetector_  = iConfig.getParameter<std::string>("DetectorName");
  DigiSource_    = iConfig.getParameter<std::string>("DigiSource");
  verbosity_     = iConfig.getUntrackedParameter<int>("Verbosity",0);
  SampleIndx_    = iConfig.getUntrackedParameter<int>("SampleIndx",5);
}


HGCalDigiValidation::~HGCalDigiValidation() { }

void HGCalDigiValidation::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup) { 
  OccupancyMap_plus_.clear();
  OccupancyMap_minus_.clear();

  edm::ESHandle<HGCalGeometry> geom;
  iSetup.get<IdealGeometryRecord>().get(nameDetector_, geom);
  const HGCalGeometry& geom0 = *geom;

  if (!geom.isValid())
    std::cout << "Cannot get valid HGCalGeometry Object for " << nameDetector_ << std::endl;

  if (nameDetector_ == "HGCalEESensitive") {
    //HGCalEE
    edm::Handle<HGCEEDigiCollection> theHGCEEDigiContainers;

    if (theHGCEEDigiContainers.isValid()) {
      if (verbosity_>0) std::cout << nameDetector_ << " with " 
				  << theHGCEEDigiContainers->size() 
				  << " element(s)" << std::endl;
      
      iEvent.getByLabel("mix", DigiSource_, theHGCEEDigiContainers);
      for(HGCEEDigiCollection::const_iterator it =theHGCEEDigiContainers->begin();
	  it !=theHGCEEDigiContainers->end(); ++it) {
	HGCEEDetId detId = it->id();
	HGCDigiValidation(detId, geom0, it);
      }
      FillDigiInfo();
    } else if (verbosity_>0) {
      std::cout << "HGCEEDigiCollection handle does not exist !!!"  << std::endl;
    }
  } else if ((nameDetector_ == "HGCalHESiliconSensitive") || 
	     (nameDetector_ == "HGCalHEScintillatorSensitive")){
    //HGCalHE
    edm::Handle<HGCHEDigiCollection> theHGCHEDigiContainers;
    if (theHGCHEDigiContainers.isValid()) {
      if (verbosity_>0) std::cout << nameDetector_ << " with " 
				  << theHGCHEDigiContainers->size()
                                  << " element(s)" << std::endl;
      
      iEvent.getByLabel("mix", DigiSource_, theHGCHEDigiContainers);
      for(HGCHEDigiCollection::const_iterator it =theHGCHEDigiContainers->begin();
	  it !=theHGCHEDigiContainers->end(); ++it) {
	HGCHEDetId detId = it->id();
	HGCDigiValidation(detId, geom0, it);
      }
      FillDigiInfo();
    } else if (verbosity_>0) {
      std::cout << "HGCHEDigiCollection handle does not exist !!!"  << std::endl;
    }
  } else {
    std::cout << "invalid detector name !!" << std::endl;
  }
}


template<class T1, class T2>
void HGCalDigiValidation::HGCDigiValidation(T1 detId, const HGCalGeometry& geom0, const T2 it) {
  
  //  std::vector<double> charges;
  int    cell        =   detId.cell();
  int    subsector   =   detId.subsector();
  int    sector      =   detId.sector();
  int    layer       =   detId.layer();
  int    zside       =   detId.zside();
  ForwardSubdetector subdet = detId.subdet();
  
  DetId id1= ((subdet == HGCEE) ? (DetId)(HGCEEDetId(subdet,zside,layer,sector,subsector,cell))
                                : (DetId)(HGCHEDetId(subdet,zside,layer,sector,subsector,cell)));
  GlobalPoint global1 = geom0.getPosition(id1);
  
  std::cout<< " >>>>>>>> nameDetector : " << nameDetector_ << " >>>>>>>> subsector: " << subsector << std::endl;
  if (verbosity_>1) std::cout << "cell = "        <<  cell
			      << " subsector = "  <<  subsector
			      << " sector = "     <<  sector
			      << " layer = "      <<  layer
			      << " zside = "      <<  zside << std::endl;
  
  HGCSample   hgcSample  =  it->sample(SampleIndx_);
  uint16_t    gain       =  hgcSample.gain();
  uint16_t    adc        =  hgcSample.adc();
  int         charge     =  adc*gain;

  if (verbosity_>1) std::cout << "nSample = "      <<  it->size()
			      << " SampleIndx = "  <<  SampleIndx_
			      << " gain = "        <<  gain
			      << " adc = "         <<  adc
			      << " charge = "      <<  charge << std::endl;
  
  digiInfo   hinfo; 
  hinfo.x       =  global1.x();
  hinfo.y       =  global1.y();
  hinfo.z       =  global1.z();
  hinfo.adc     =  adc;
  hinfo.charge  =  charge; //charges[0];
  hinfo.layer   =  layer;
  
  if (verbosity_>1) std::cout << "gx =  "  << hinfo.x
			      << " gy = "   << hinfo.y
			      << " gz = "   << hinfo.z << std::endl;
  
  FillDigiInfo(hinfo);

  if (global1.eta() > 0)  fillOccupancyMap(OccupancyMap_plus_, layer -1);
  else                    fillOccupancyMap(OccupancyMap_minus_, layer -1);

}

void HGCalDigiValidation::fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer) {
  if (OccupancyMap.find(layer) != OccupancyMap.end()) OccupancyMap[layer] ++;
  else                                                OccupancyMap[layer] = 1;
}

void HGCalDigiValidation::FillDigiInfo(digiInfo&   hinfo) {
  int ilayer = hinfo.layer -1;
  charge_.at(ilayer)->Fill(hinfo.charge);
  DigiOccupancy_XY_.at(ilayer)->Fill(hinfo.x, hinfo.y);
  ADC_.at(ilayer)->Fill(hinfo.adc);
}

void HGCalDigiValidation::FillDigiInfo() {
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
  
  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get( pDD );
  const DDCompactView & cview = *pDD;
  hgcons_ = new HGCalDDDConstants(cview, nameDetector_);
  
  if (dbe_) {
    layers_ = hgcons_->layers(true);
    dbe_->setCurrentFolder("HGCalDigiV/"+nameDetector_);
    
    if (verbosity_>0) std::cout << "current DQM directory:  "
				<< "HGCalDigiV/" << nameDetector_ 
				<< "  layer = "<< layers_ << std::endl;
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
