#include <iostream>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

class HGCalDigiTester : public edm::EDAnalyzer {
public:
  explicit HGCalDigiTester(const edm::ParameterSet& );
  ~HGCalDigiTester();

  
  virtual void analyze(const edm::Event&, const edm::EventSetup& );

private:
  // ----------member data ---------------------------
};

HGCalDigiTester::HGCalDigiTester(const edm::ParameterSet& ) {}


HGCalDigiTester::~HGCalDigiTester() {}

void HGCalDigiTester::analyze(const edm::Event& iEvent, 
			      const edm::EventSetup& iSetup ) {

  std::string name;
  edm::ESHandle<HGCalGeometry> geom;

  name = "HGCalEESensitive";
  iSetup.get<IdealGeometryRecord>().get(name,geom);
  if (geom.isValid()) {
    edm::Handle<HGCEEDigiCollection> digis;
    iEvent.getByLabel(edm::InputTag("mix","HGCDigisEE"),digis); 
    if (digis.isValid()) {
      std::cout << "HGCEE with " << (*digis).size() << " elements" 
		<< std::endl;
      for (unsigned int k=0; k < (*digis).size(); ++k) {
	HGCEEDetId  id = (*digis)[k].id();
	GlobalPoint global = (*geom).getPosition(id);
	HGCEEDetId  idc    = (HGCEEDetId)((*geom).getClosestCell(global));
	std::cout << "HGCalDigiTester:ID " << id << " global (" << global.x() 
		  << ", " << global.y() << ", " << global.z() << ") new ID " 
		  << idc << std::endl;
      }
    } else {
      std::cout << "No valid collection for HGCEE" << std::endl;
    }
  } else {
    std::cout << "Cannot get valid HGCalGeometry Object for " << name 
	      << std::endl;
  }

  name = "HGCalHESiliconSensitive";
  iSetup.get<IdealGeometryRecord>().get(name,geom);
  if (geom.isValid()) {
    edm::Handle<HGCHEDigiCollection> digis;
    iEvent.getByLabel(edm::InputTag("mix","HGCDigisHEfront"),digis); 
    if (digis.isValid()) {
      std::cout << "HGCHEfront with " << (*digis).size() << " elements" 
		<< std::endl;
      for (unsigned int k=0; k < (*digis).size(); ++k) {
	HGCHEDetId  id = (*digis)[k].id();
	GlobalPoint global = (*geom).getPosition(id);
	HGCHEDetId  idc    = (HGCHEDetId)((*geom).getClosestCell(global));
	std::cout << "HGCalDigiTester:ID " << id << " global (" << global.x() 
		  << ", " << global.y() << ", " << global.z() << ") new ID " 
		  << idc << std::endl;
      }
    } else {
      std::cout << "No valid collection for HGCHEfront" << std::endl;
    }
  } else {
    std::cout << "Cannot get valid HGCalGeometry Object for " << name 
	      << std::endl;
  }

  name = "HGCalHEScintillatorSensitive";
  iSetup.get<IdealGeometryRecord>().get(name,geom);
  if (geom.isValid()) {
    edm::Handle<HGCHEDigiCollection> digis;
    iEvent.getByLabel(edm::InputTag("mix","HGCDigisHEback"),digis); 
    if (digis.isValid()) {
      std::cout << "HGCHEback with " << (*digis).size() << " elements" 
		<< std::endl;
      for (unsigned int k=0; k < (*digis).size(); ++k) {
	HGCHEDetId  id = (*digis)[k].id();
	GlobalPoint global = (*geom).getPosition(id);
	HGCHEDetId  idc    = (HGCHEDetId)((*geom).getClosestCell(global));
	std::cout << "HGCalDigiTester:ID " << id << " global (" << global.x()
		  << ", " << global.y() << ", " << global.z() << ") new ID "
		  << idc << std::endl;
      }
    } else {
      std::cout << "No valid collection for HGCHEback" << std::endl;
    }
  } else {
    std::cout << "Cannot get valid HGCalGeometry Object for " << name 
	      << std::endl;
  }

}


//define this as a plug-in
DEFINE_FWK_MODULE(HGCalDigiTester);
