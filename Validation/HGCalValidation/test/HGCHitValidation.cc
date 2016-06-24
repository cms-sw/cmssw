/// -*- C++ -*-
//
// Package:    HGCHitValidation
// Class:      HGCHitValidation
// 
/**\class HGCHitValidation HGCHitValidation.cc MyHGCAnalyzer/HGCHitValidation/src/HGCHitValidation.cc

 Description: [one line class summary]

 Implementation:
 	[Notes on implementation]
*/
//
// Original Author:  "Maksat Haytmyradov"
//         Created:  Fri March 19 13:32:26 CDT 2016
// $Id$
//
//



#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include <TH1.h>
#include <TH2.h>

#include <cmath>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

//#define DebugLog

class HGCHitValidation : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:

  explicit HGCHitValidation( const edm::ParameterSet& );
  ~HGCHitValidation();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  typedef std::tuple<float,float,float,float> HGCHitTuple;

  virtual void beginJob();
  virtual void endJob() {}
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

private:
  //HGC Geometry
  std::vector<const HGCalDDDConstants*> hgcCons_;
  std::vector<const HGCalGeometry*>     hgcGeometry_;
  const HcalDDDSimConstants*            hcCons_;
  const HcalDDDRecConstants*            hcConr_;
  const CaloSubdetectorGeometry*        hcGeometry_;
  std::vector<std::string>              geometrySource_;
  std::vector<int>                      ietaExcludeBH_;

  edm::InputTag eeSimHitSource, fhSimHitSource, bhSimHitSource;
  edm::EDGetTokenT<std::vector<PCaloHit>> eeSimHitToken_;
  edm::EDGetTokenT<std::vector<PCaloHit>> fhSimHitToken_;
  edm::EDGetTokenT<std::vector<PCaloHit>> bhSimHitToken_;
  edm::EDGetTokenT<HGCeeRecHitCollection> eeRecHitToken_;
  edm::EDGetTokenT<HGChefRecHitCollection> fhRecHitToken_;
  edm::EDGetTokenT<HBHERecHitCollection> bhRecHitToken_;

  //histogram related stuff
  TH2F *heedzVsZ, *heedyVsY, *heedxVsX;
  TH2F *hefdzVsZ, *hefdyVsY, *hefdxVsX;
  TH2F *hebdzVsZ, *hebdPhiVsPhi, *hebdEtaVsEta;
	
  TH2F *heeRecVsSimZ, *heeRecVsSimY, *heeRecVsSimX;
  TH2F *hefRecVsSimZ, *hefRecVsSimY, *hefRecVsSimX;
  TH2F *hebRecVsSimZ, *hebRecVsSimY, *hebRecVsSimX;

  TH2F *heeEnSimRec, *hefEnSimRec, *hebEnSimRec;

  TH1F *hebEnRec, *hebEnSim;
  TH1F *hefEnRec, *hefEnSim;
  TH1F *heeEnRec, *heeEnSim;

};


HGCHitValidation::HGCHitValidation( const edm::ParameterSet &cfg ) {

  usesResource("TFileService");
  geometrySource_ = cfg.getUntrackedParameter< std::vector<std::string> >("geometrySource");
  eeSimHitToken_  = consumes<std::vector<PCaloHit>>(cfg.getParameter<edm::InputTag>("eeSimHitSource"));
  fhSimHitToken_  = consumes<std::vector<PCaloHit>>(cfg.getParameter<edm::InputTag>("fhSimHitSource"));
  bhSimHitToken_  = consumes<std::vector<PCaloHit>>(cfg.getParameter<edm::InputTag>("bhSimHitSource"));
  eeRecHitToken_  = consumes<HGCeeRecHitCollection>(cfg.getParameter<edm::InputTag>("eeRecHitSource"));
  fhRecHitToken_  = consumes<HGChefRecHitCollection>(cfg.getParameter<edm::InputTag>("fhRecHitSource"));
  bhRecHitToken_  = consumes<HBHERecHitCollection>(cfg.getParameter<edm::InputTag>("bhRecHitSource"));
  ietaExcludeBH_  = cfg.getParameter<std::vector<int> >("ietaExcludeBH");
#ifdef DebugLog
  std::cout << "Exclude the following " << ietaExcludeBH_.size()
	    << " ieta values from BH plots";
  for (unsigned int k=0; k<ietaExcludeBH_.size(); ++k) 
    std::cout << " " << ietaExcludeBH_[k];
  std::cout << std::endl;
#endif
}

HGCHitValidation::~HGCHitValidation() { }

void HGCHitValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void HGCHitValidation::beginJob() {

  //initiating fileservice
  edm::Service<TFileService> fs;

  //initiating histograms
  heedzVsZ = fs->make<TH2F>("heedzVsZ","",720000,-360,360,100,-0.1,0.1);
  heedyVsY = fs->make<TH2F>("heedyVsY","",400,-200,200,100,-0.02,0.02);
  heedxVsX = fs->make<TH2F>("heedxVsX","",400,-200,200,100,-0.02,0.02);

  heeRecVsSimZ = fs->make<TH2F>("heeRecVsSimZ","",7200,-360,360,7200,-360,360);
  heeRecVsSimY = fs->make<TH2F>("heeRecVsSimY","",400,-200,200,400,-200,200);
  heeRecVsSimX = fs->make<TH2F>("heeRecVsSimX","",400,-200,200,400,-200,200);

  hefdzVsZ = fs->make<TH2F>("hefdzVsZ","",820000,-410,410,100,-0.1,0.1);
  hefdyVsY = fs->make<TH2F>("hefdyVsY","",400,-200,200,100,-0.02,0.02);
  hefdxVsX = fs->make<TH2F>("hefdxVsX","",400,-200,200,100,-0.02,0.02);

  hefRecVsSimZ = fs->make<TH2F>("hefRecVsSimZ","",8200,-410,410,8200,-410,410);
  hefRecVsSimY = fs->make<TH2F>("hefRecVsSimY","",400,-200,200,400,-200,200);
  hefRecVsSimX = fs->make<TH2F>("hefRecVsSimX","",400,-200,200,400,-200,200);

  hebdzVsZ = fs->make<TH2F>("hebdzVsZ","",1080,-540,540,100,-1.0,1.0);
  hebdPhiVsPhi = fs->make<TH2F>("hebdPhiVsPhi","",M_PI*100,-0.5,M_PI+0.5,200,-0.2,0.2);
  hebdEtaVsEta = fs->make<TH2F>("hebdEtaVsEta","",1000,-5,5,200,-0.1,0.1);

  hebRecVsSimZ = fs->make<TH2F>("hebRecVsSimZ","",1080,-540,540,1080,-540,540);
  hebRecVsSimY = fs->make<TH2F>("hebRecVsSimY","",400,-200,200,400,-200,200);
  hebRecVsSimX = fs->make<TH2F>("hebRecVsSimX","",400,-200,200,400,-200,200);

  heeEnRec = fs->make<TH1F>("heeEnRec","",1000,0,10);
  heeEnSim = fs->make<TH1F>("heeEnSim","",1000,0,0.01);
  heeEnSimRec = fs->make<TH2F>("heeEnSimRec","",1000,0,0.01,100,0,0.01);

  hefEnRec = fs->make<TH1F>("hefEnRec","",1000,0,10);
  hefEnSim = fs->make<TH1F>("hefEnSim","",1000,0,0.01);
  hefEnSimRec = fs->make<TH2F>("hefEnSimRec","",1000,0,0.01,100,0,0.01);

  hebEnRec = fs->make<TH1F>("hebEnRec","",1000,0,15);
  hebEnSim = fs->make<TH1F>("hebEnSim","",1000,0,0.01);
  hebEnSimRec = fs->make<TH2F>("hebEnSimRec","",1000,0,0.01,100,0,4);

}

void HGCHitValidation::beginRun(edm::Run const& iRun,
				edm::EventSetup const& iSetup) {
  //initiating hgc Geometry
  for (size_t i=0; i<geometrySource_.size(); i++) {
    if (geometrySource_[i].find("Hcal") != std::string::npos) {
      edm::ESHandle<HcalDDDSimConstants> pHSNDC;
      iSetup.get<HcalSimNumberingRecord>().get(pHSNDC);
      if (pHSNDC.isValid()) {
        hcCons_ = pHSNDC.product();
        hgcCons_.push_back(0);
      } else {
        edm::LogWarning("HGCalValid") << "Cannot initiate HcalDDDSimConstants: "
                                      << geometrySource_[i] << std::endl;
      }
      edm::ESHandle<HcalDDDRecConstants> pHRNDC;
      iSetup.get<HcalRecNumberingRecord>().get(pHRNDC);
      if (pHRNDC.isValid()) {
        hcConr_ = pHRNDC.product();
      } else {
        edm::LogWarning("HGCalValid") << "Cannot initiate HcalDDDRecConstants: "
                                      << geometrySource_[i] << std::endl;
      }
      edm::ESHandle<CaloGeometry> caloG;
      iSetup.get<CaloGeometryRecord>().get(caloG);
      if (caloG.isValid()) {
	const CaloGeometry* geo = caloG.product();
	hcGeometry_ = geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);
	hgcGeometry_.push_back(0);
      } else {
        edm::LogWarning("HGCalValid") << "Cannot initiate HcalGeometry for "
                                      << geometrySource_[i] << std::endl;
      }
   } else {
      edm::ESHandle<HGCalDDDConstants> hgcCons;
      iSetup.get<IdealGeometryRecord>().get(geometrySource_[i],hgcCons);
      if (hgcCons.isValid()) {
        hgcCons_.push_back(hgcCons.product());
      } else {
        edm::LogWarning("HGCalValid") << "Cannot initiate HGCalDDDConstants for "
                                      << geometrySource_[i] << std::endl;
      }
      edm::ESHandle<HGCalGeometry> hgcGeom;
      iSetup.get<IdealGeometryRecord>().get(geometrySource_[i],hgcGeom);	
      if(hgcGeom.isValid()) {
	hgcGeometry_.push_back(hgcGeom.product());	
      } else {
	edm::LogWarning("HGCalValid") << "Cannot initiate HGCalGeometry for "
				      << geometrySource_[i] << std::endl;
      }
    }
  }
}


void HGCHitValidation::analyze( const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::map<unsigned int, HGCHitTuple> eeHitRefs, fhHitRefs, bhHitRefs;

  //declare topology and DDD constants
  const HGCalTopology &heeTopo=hgcGeometry_[0]->topology();
  const HGCalTopology &hefTopo=hgcGeometry_[1]->topology();

  //Accesing ee simhits
  edm::Handle<std::vector<PCaloHit>> eeSimHits;
  iEvent.getByToken(eeSimHitToken_, eeSimHits);

  if (eeSimHits.isValid()) {
    for (std::vector<PCaloHit>::const_iterator simHit = eeSimHits->begin(); simHit != eeSimHits->end(); ++simHit) {
      int subdet, zside, layer, wafer, celltype, cell;
      HGCalTestNumbering::unpackHexagonIndex(simHit->id(), subdet, zside, layer, wafer, celltype, cell);
      std::pair<float, float> xy = hgcCons_[0]->locateCell(cell,layer,wafer,false);
      float zp = hgcCons_[0]->waferZ(layer,false);
      if (zside < 0) zp = -zp;
      float xp = (zp<0) ? -xy.first/10 : xy.first/10;
      float yp = xy.second/10.0;

      //skip this hit if after ganging it is not valid
      std::pair<int,int> recoLayerCell=hgcCons_[0]->simToReco(cell,layer,wafer,heeTopo.detectorType());
      cell  = recoLayerCell.first;
      layer = recoLayerCell.second;

      //skip this hit if after ganging it is not valid
      if (layer<0 || cell<0) {
      } else {
	
	//assign the RECO DetId
	HGCalDetId id = HGCalDetId((ForwardSubdetector)(subdet),zside,layer,celltype,wafer,cell);
	float energy = simHit->energy();

	float energySum(0);
	if (eeHitRefs.count(id.rawId()) != 0) energySum = std::get<0>(eeHitRefs[id.rawId()]);
	energySum += energy;
	eeHitRefs[id.rawId()] = std::make_tuple(energySum,xp,yp,zp);
	heeEnSim->Fill(energy);
      }
    }
#ifdef DebugLog
    for (std::map<unsigned int,HGCHitTuple>::iterator itr=eeHitRefs.begin();
	 itr != eeHitRefs.end(); ++itr) {
      int idx = std::distance(eeHitRefs.begin(),itr);
      std::cout << "EEHit[" << idx << "] " << std::hex << itr->first 
		<< std::dec << "; Energy " << std::get<0>(itr->second) 
		<< "; Position (" << std::get<1>(itr->second) << ", "
		<< std::get<2>(itr->second) <<", " << std::get<3>(itr->second)
		<< ")" << std::endl;
    }
#endif
  } else {   
    edm::LogWarning("HGCalValid") << "No EE SimHit Found " << std::endl;
  }

  //Accesing fh simhits
  edm::Handle<std::vector<PCaloHit>> fhSimHits;
  iEvent.getByToken(fhSimHitToken_, fhSimHits);
  if (fhSimHits.isValid()) {
    for (std::vector<PCaloHit>::const_iterator simHit = fhSimHits->begin(); 
	 simHit != fhSimHits->end();++simHit) {
      int subdet, zside, layer, wafer, celltype, cell;
      HGCalTestNumbering::unpackHexagonIndex(simHit->id(), subdet, zside, layer, wafer, celltype, cell);
      std::pair<float, float> xy = hgcCons_[1]->locateCell(cell,layer,wafer,false);
      float zp = hgcCons_[1]->waferZ(layer,false);
      if (zside < 0) zp = -zp;
      float xp = (zp<0) ? -xy.first/10 : xy.first/10;
      float yp = xy.second/10.0;

      //skip this hit if after ganging it is not valid
      std::pair<int,int> recoLayerCell = hgcCons_[1]->simToReco(cell,layer,wafer,hefTopo.detectorType());
      cell  = recoLayerCell.first;
      layer = recoLayerCell.second;
      //skip this hit if after ganging it is not valid
      if(layer<0 || cell<0) {
      } else {
	//assign the RECO DetId
	HGCalDetId id = HGCalDetId((ForwardSubdetector)(subdet),zside,layer,celltype,wafer,cell);

	float energy = simHit->energy();
	float energySum(0);
	if (fhHitRefs.count(id.rawId()) != 0) energySum = std::get<0>(fhHitRefs[id.rawId()]);
	energySum += energy;
	fhHitRefs[id.rawId()] = std::make_tuple(energySum,xp,yp,zp);
	hefEnSim->Fill(energy);
      }
    }
#ifdef DebugLog
    for (std::map<unsigned int,HGCHitTuple>::iterator itr=fhHitRefs.begin();
	 itr != fhHitRefs.end(); ++itr) {
      int idx = std::distance(fhHitRefs.begin(),itr);
      std::cout << "FHHit[" << idx << "] " << std::hex << itr->first 
		<< std::dec << "; Energy " << std::get<0>(itr->second) 
		<< "; Position (" << std::get<1>(itr->second) << ", "
		<< std::get<2>(itr->second) <<", " << std::get<3>(itr->second)
		<< ")" << std::endl;
    }
#endif
  } else {
    edm::LogWarning("HGCalValid") << "No FH SimHit Found " << std::endl;
  }
	
  //Accessing bh simhits
  edm::Handle<std::vector<PCaloHit>> bhSimHits;
  iEvent.getByToken(bhSimHitToken_, bhSimHits);
  if (bhSimHits.isValid()) {
    for (std::vector<PCaloHit>::const_iterator simHit = bhSimHits->begin();
	 simHit != bhSimHits->end(); ++simHit) {
      int subdet, z, depth, eta, phi, lay;
      HcalTestNumbering::unpackHcalIndex(simHit->id(), subdet, z, depth, eta, phi, lay);
      HcalCellType::HcalCell cell = hcCons_->cell(subdet, z, lay, eta, phi);

      double zp  = cell.rz/10; 

      if (subdet == static_cast<int>(HcalEndcap)) {
	HcalDDDRecConstants::HcalID idx = hcConr_->getHCID(subdet,eta,phi,lay,depth);
	int sign = (z==0)?(-1):(1);
	zp      *= sign;
	HcalDetId id = HcalDetId(HcalEndcap,sign*idx.eta,idx.phi,idx.depth);  

	float energy = simHit->energy();
	float energySum(0);
	if (bhHitRefs.count(id.rawId()) != 0) energySum = std::get<0>(bhHitRefs[id.rawId()]);
	energySum += energy;
	hebEnSim->Fill(energy);
	if (std::find(ietaExcludeBH_.begin(),ietaExcludeBH_.end(),idx.eta) ==
	    ietaExcludeBH_.end()) {
	  bhHitRefs[id.rawId()] = std::make_tuple(energySum,cell.eta,cell.phi,zp);
#ifdef DebugLog
	  std::cout << "Accept " << id << std::endl;
	} else {
	  std::cout << "Reject " << id << std::endl;
#endif
	}
      }
    }
#ifdef DebugLog
    for (std::map<unsigned int,HGCHitTuple>::iterator itr=bhHitRefs.begin();
	 itr != bhHitRefs.end(); ++itr) {
      int idx = std::distance(bhHitRefs.begin(),itr);
      std::cout << "BHHit[" << idx << "] " << std::hex << itr->first 
		<< std::dec << "; Energy " << std::get<0>(itr->second) 
		<< "; Position (" << std::get<1>(itr->second) << ", "
		<< std::get<2>(itr->second) <<", " << std::get<3>(itr->second)
		<< ")" << std::endl;
    }
#endif
  } else {
    edm::LogWarning("HGCalValid") << "No BH SimHit Found " << std::endl;
  }

  //accessing EE Rechit information
  edm::Handle<HGCeeRecHitCollection> eeRecHit;	
  iEvent.getByToken(eeRecHitToken_, eeRecHit);
  if (eeRecHit.isValid()) {
    const HGCeeRecHitCollection* theHits = (eeRecHit.product());
    for (auto it = theHits->begin(); it != theHits->end(); ++it) {
      double energy = it->energy(); 
      heeEnRec->Fill(energy);
      std::map<unsigned int, HGCHitTuple>::const_iterator itr = eeHitRefs.find(it->id().rawId());
      if (itr != eeHitRefs.end()) {
	GlobalPoint xyz = hgcGeometry_[0]->getPosition(it->id());
	heeRecVsSimX->Fill(std::get<1>(itr->second),xyz.x());
	heeRecVsSimY->Fill(std::get<2>(itr->second),xyz.y());
	heeRecVsSimZ->Fill(std::get<3>(itr->second),xyz.z());
	heedxVsX->Fill(std::get<1>(itr->second),(xyz.x()-std::get<1>(itr->second)));
	heedyVsY->Fill(std::get<2>(itr->second),(xyz.y()-std::get<2>(itr->second)));
	heedzVsZ->Fill(std::get<3>(itr->second),(xyz.z()-std::get<3>(itr->second)));
	heeEnSimRec->Fill(std::get<0>(itr->second),energy);
#ifdef DebugLog
	std::cout << "EEHit: " << std::hex << it->id().rawId() << std::dec
		  << " Sim (" << std::get<0>(itr->second) << ", "
		  << std::get<1>(itr->second) << ", " 
		  << std::get<2>(itr->second) << ", " 
		  << std::get<3>(itr->second) << ") Rec (" << energy << ", " 
		  << xyz.x() << ", " << xyz.y() << ", " << xyz.z() << ")\n";
#endif
      }
    }
  } else {
    edm::LogWarning("HGCalValid") << "No EE RecHit Found " << std::endl;
  }

  //accessing FH Rechit information
  edm::Handle<HGChefRecHitCollection> fhRecHit;
  iEvent.getByToken(fhRecHitToken_, fhRecHit);
  if (fhRecHit.isValid()) {
    const HGChefRecHitCollection* theHits = (fhRecHit.product());			
    for (auto it = theHits->begin(); it!=theHits->end(); ++it) {
      double energy = it->energy(); 
      hefEnRec->Fill(energy);
      std::map<unsigned int, HGCHitTuple>::const_iterator itr = fhHitRefs.find(it->id().rawId());
      if (itr != fhHitRefs.end()) {
	GlobalPoint xyz = hgcGeometry_[1]->getPosition(it->id());

	hefRecVsSimX->Fill(std::get<1>(itr->second),xyz.x());
        hefRecVsSimY->Fill(std::get<2>(itr->second),xyz.y());
        hefRecVsSimZ->Fill(std::get<3>(itr->second),xyz.z());
        hefdxVsX->Fill(std::get<1>(itr->second),(xyz.x()-std::get<1>(itr->second)));
        hefdyVsY->Fill(std::get<2>(itr->second),(xyz.y()-std::get<2>(itr->second)));
        hefdzVsZ->Fill(std::get<3>(itr->second),(xyz.z()-std::get<3>(itr->second)));
	hefEnSimRec->Fill(std::get<0>(itr->second),energy);
#ifdef DebugLog
	std::cout << "FHHit: " << std::hex << it->id().rawId() << std::dec
		  << " Sim (" << std::get<0>(itr->second) << ", "
		  << std::get<1>(itr->second) << ", " 
		  << std::get<2>(itr->second) << ", " 
		  << std::get<3>(itr->second) << ") Rec (" << energy << "," 
		  << xyz.x() << ", " << xyz.y() << ", " << xyz.z() << ")\n";
#endif
      }
    }
  } else {
    edm::LogWarning("HGCalValid") << "No FH RecHit Found " << std::endl;
  }


  //accessing BH Rechit information
  edm::Handle<HBHERecHitCollection> bhRecHit;
  iEvent.getByToken(bhRecHitToken_, bhRecHit);
  if (bhRecHit.isValid()) {
    const HBHERecHitCollection* theHits = (bhRecHit.product());
    
    for (auto it = theHits->begin(); it!=theHits->end(); ++it) {
      DetId id = it->id();
      if (id.subdetId() == (int)(HcalEndcap)) {
	double energy = it->energy();
	hebEnRec->Fill(energy);
	GlobalPoint xyz = hcGeometry_->getGeometry(id)->getPosition();

	std::map<unsigned int, HGCHitTuple>::const_iterator itr = bhHitRefs.find(id.rawId());
	if (itr != bhHitRefs.end()) {
	  float ang3 = xyz.phi().value(); // returns the phi in radians
	  double fac = sinh(std::get<1>(itr->second));
	  double pT  = std::get<3>(itr->second) / fac;
	  double xp = pT * cos(std::get<2>(itr->second));
	  double yp = pT * sin(std::get<2>(itr->second));
	  hebRecVsSimX->Fill(xp,xyz.x());
	  hebRecVsSimY->Fill(yp,xyz.y());
	  hebRecVsSimZ->Fill(std::get<3>(itr->second),xyz.z());
	  hebdEtaVsEta->Fill(std::get<1>(itr->second),(xyz.eta()-std::get<1>(itr->second)));
	  hebdPhiVsPhi->Fill(std::get<2>(itr->second),(ang3-std::get<2>(itr->second)));
	  hebdzVsZ->Fill(std::get<3>(itr->second),(xyz.z()-std::get<3>(itr->second)));
	  hebEnSimRec->Fill(std::get<0>(itr->second),energy);

#ifdef DebugLog
	  std::cout << "BHHit: " << std::hex << id.rawId() << std::dec
		    << " Sim (" << std::get<0>(itr->second) << ", "
		    << std::get<1>(itr->second) << ", " 
		    << std::get<2>(itr->second) << ", " 
		    << std::get<3>(itr->second) << ") Rec (" << energy << ", " 
		    << xyz.x() << ", " << xyz.y() << ", " << xyz.z() << ")\n";
#endif
	}
      }
    }
  } else {
    edm::LogWarning("HGCalValid") << "No BH RecHit Found " << std::endl;
  }
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCHitValidation);






