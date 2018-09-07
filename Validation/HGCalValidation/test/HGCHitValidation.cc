/// -*- C++ -*-
//
// Package:    HGCHitValidation
// Class:      HGCHitValidation
// 
/**\class HGCHitValidation HGCHitValidation.cc Validation/HGCalValidation/test/HGCHitValidation.cc

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
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
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
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include <TH1.h>
#include <TH2.h>
#include <TTree.h>
#include <TVector3.h>
#include <TSystem.h>
#include <TFile.h>

#include <cmath>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

class HGCHitValidation : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:

  explicit HGCHitValidation( const edm::ParameterSet& );
  ~HGCHitValidation() override { }
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  typedef std::tuple<float,float,float,float> HGCHitTuple;

  virtual void beginJob() override;
  virtual void endJob() override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  void analyzeHGCalSimHit(edm::Handle<std::vector<PCaloHit>> const& simHits,
			  int idet, std::map<unsigned int, HGCHitTuple>&);
  template<class T1>
  void analyzeHGCalRecHit(T1 const & theHits, 
			  std::map<unsigned int, HGCHitTuple> const& hitRefs);

private:
  //HGC Geometry
  std::vector<const HGCalDDDConstants*> hgcCons_;
  std::vector<const HGCalGeometry*>     hgcGeometry_;
  const HcalDDDSimConstants*            hcCons_;
  const HcalDDDRecConstants*            hcConr_;
  const CaloSubdetectorGeometry*        hcGeometry_;
  std::vector<std::string>              geometrySource_;
  std::vector<int>                      ietaExcludeBH_;
  bool                                  ifHCAL_, ifHcalG_;

  edm::InputTag eeSimHitSource, fhSimHitSource, bhSimHitSource;
  edm::EDGetTokenT<std::vector<PCaloHit>> eeSimHitToken_;
  edm::EDGetTokenT<std::vector<PCaloHit>> fhSimHitToken_;
  edm::EDGetTokenT<std::vector<PCaloHit>> bhSimHitToken_;
  edm::EDGetTokenT<HGCeeRecHitCollection> eeRecHitToken_;
  edm::EDGetTokenT<HGChefRecHitCollection> fhRecHitToken_;
  edm::EDGetTokenT<HGChebRecHitCollection> bhRecHitTokeng_;
  edm::EDGetTokenT<HBHERecHitCollection> bhRecHitTokenh_;

  TTree* hgcHits;
  std::vector<float>  *heeRecX, *heeRecY, *heeRecZ, *heeRecEnergy;
  std::vector<float>  *hefRecX, *hefRecY, *hefRecZ, *hefRecEnergy;
  std::vector<float>  *hebRecX, *hebRecY, *hebRecZ, *hebRecEnergy;
  std::vector<float>  *heeSimX, *heeSimY, *heeSimZ, *heeSimEnergy;
  std::vector<float>  *hefSimX, *hefSimY, *hefSimZ, *hefSimEnergy;
  std::vector<float>  *hebSimX, *hebSimY, *hebSimZ, *hebSimEnergy;
  std::vector<float>  *hebSimEta, *hebRecEta, *hebSimPhi, *hebRecPhi;
  std::vector<unsigned int> *heeDetID, *hefDetID, *hebDetID;
};


HGCHitValidation::HGCHitValidation( const edm::ParameterSet &cfg ) : ifHcalG_(false) {

  usesResource(TFileService::kSharedResource);

  geometrySource_ = cfg.getUntrackedParameter< std::vector<std::string> >("geometrySource");
  edm::InputTag eeSimHitSource, fhSimHitSource, bhSimHitSource;
  eeSimHitSource  = cfg.getParameter<edm::InputTag>("eeSimHitSource");
  fhSimHitSource  = cfg.getParameter<edm::InputTag>("fhSimHitSource");
  bhSimHitSource  = cfg.getParameter<edm::InputTag>("bhSimHitSource");
  eeSimHitToken_  = consumes<std::vector<PCaloHit>>(eeSimHitSource);
  fhSimHitToken_  = consumes<std::vector<PCaloHit>>(fhSimHitSource);
  bhSimHitToken_  = consumes<std::vector<PCaloHit>>(bhSimHitSource);
  edm::InputTag eeRecHitSource, fhRecHitSource, bhRecHitSource;
  eeRecHitSource  = cfg.getParameter<edm::InputTag>("eeRecHitSource");
  fhRecHitSource  = cfg.getParameter<edm::InputTag>("fhRecHitSource");
  bhRecHitSource  = cfg.getParameter<edm::InputTag>("bhRecHitSource");
  eeRecHitToken_  = consumes<HGCeeRecHitCollection>(eeRecHitSource);
  fhRecHitToken_  = consumes<HGChefRecHitCollection>(fhRecHitSource);
  ietaExcludeBH_  = cfg.getParameter<std::vector<int> >("ietaExcludeBH");
  ifHCAL_         = cfg.getParameter<bool>("ifHCAL");
  if (ifHCAL_) bhRecHitTokenh_ = consumes<HBHERecHitCollection>(bhRecHitSource);
  else         bhRecHitTokeng_ = consumes<HGChebRecHitCollection>(bhRecHitSource);
  hgcHits  = 0;
  heeRecX  = heeRecY  = heeRecZ  = heeRecEnergy = 0;
  hefRecX  = hefRecY  = hefRecZ  = hefRecEnergy = 0;
  hebRecX  = hebRecY  = hebRecZ  = hebRecEnergy = 0;
  heeSimX  = heeSimY  = heeSimZ  = heeSimEnergy = 0;
  hefSimX  = hefSimY  = hefSimZ  = hefSimEnergy = 0;
  hebSimX  = hebSimY  = hebSimZ  = hebSimEnergy = 0;
  hebSimEta= hebRecEta= hebSimPhi= hebRecPhi    = 0;
  heeDetID = hefDetID = hebDetID = 0;

  edm::LogVerbatim("HGCalValid") << "Use " << geometrySource_.size()
				 << " Geometry sources and HCAL flag "
				 << ifHCAL_;
  for (auto const& s : geometrySource_)
    edm::LogVerbatim("HGCalValid") << "  " << s;
  edm::LogVerbatim("HGCalValid") << "SimHit labels: " << eeSimHitSource << "  "
				 << fhSimHitSource << "  " << bhSimHitSource;
  edm::LogVerbatim("HGCalValid") << "RecHit labels: " << eeRecHitSource << "  "
				 << fhRecHitSource << "  " << bhRecHitSource;
  edm::LogVerbatim("HGCalValid") << "Exclude the following " 
				 << ietaExcludeBH_.size()
				 << " ieta values from BH plots";
  for (unsigned int k=0; k<ietaExcludeBH_.size(); ++k) 
    edm::LogVerbatim("HGCalValid") << " [" << k << "] " << ietaExcludeBH_[k];
}

void HGCHitValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  std::vector<std::string> names = {"HGCalEESensitive",
				    "HGCalHESiliconSensitive",
				    "Hcal"};
  std::vector<int> etas;
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<std::string>>("geometrySource",names);
  desc.add<edm::InputTag>("eeSimHitSource",edm::InputTag("g4SimHits","HGCHitsEE"));
  desc.add<edm::InputTag>("fhSimHitSource",edm::InputTag("g4SimHits","HGCHitsHEfront"));
  desc.add<edm::InputTag>("bhSimHitSource",edm::InputTag("g4SimHits","HcalHits"));
  desc.add<edm::InputTag>("eeRecHitSource",edm::InputTag("HGCalRecHit","HGCEERecHits"));
  desc.add<edm::InputTag>("fhRecHitSource",edm::InputTag("HGCalRecHit","HGCHEFRecHits"));
  desc.add<edm::InputTag>("bhRecHitSource",edm::InputTag("HGCalRecHit","HGCHEBRecHits"));
  desc.add<std::vector<int> >("ietaExcludeBH",etas);
  desc.add<bool>("ifHCAL",false);
  descriptions.add("hgcHitAnalysis",desc);
}

void HGCHitValidation::beginJob() {

  //initiating fileservice
  edm::Service<TFileService> fs;
  hgcHits = fs->make < TTree > ("hgcHits","Hit Collection");
  hgcHits->Branch("heeRecX", &heeRecX);
  hgcHits->Branch("heeRecY", &heeRecY);
  hgcHits->Branch("heeRecZ", &heeRecZ);
  hgcHits->Branch("heeRecEnergy", &heeRecEnergy);
  hgcHits->Branch("hefRecX", &hefRecX);
  hgcHits->Branch("hefRecY", &hefRecY);
  hgcHits->Branch("hefRecZ", &hefRecZ);
  hgcHits->Branch("hefRecEnergy", &hefRecEnergy);
  hgcHits->Branch("hebRecX", &hebRecX);
  hgcHits->Branch("hebRecY", &hebRecY);
  hgcHits->Branch("hebRecZ", &hebRecZ);
  hgcHits->Branch("hebRecEta", &hebRecEta);
  hgcHits->Branch("hebRecPhi", &hebRecPhi);
  hgcHits->Branch("hebRecEnergy", &hebRecEnergy);

  hgcHits->Branch("heeSimX", &heeSimX);
  hgcHits->Branch("heeSimY", &heeSimY);
  hgcHits->Branch("heeSimZ", &heeSimZ);
  hgcHits->Branch("heeSimEnergy", &heeSimEnergy);
  hgcHits->Branch("hefSimX", &hefSimX);
  hgcHits->Branch("hefSimY", &hefSimY);
  hgcHits->Branch("hefSimZ", &hefSimZ);
  hgcHits->Branch("hefSimEnergy", &hefSimEnergy);
  hgcHits->Branch("hebSimX", &hebSimX);
  hgcHits->Branch("hebSimY", &hebSimY);
  hgcHits->Branch("hebSimZ", &hebSimZ);
  hgcHits->Branch("hebSimEta", &hebSimEta);
  hgcHits->Branch("hebSimPhi", &hebSimPhi);
  hgcHits->Branch("hebSimEnergy", &hebSimEnergy);

  hgcHits->Branch("heeDetID", &heeDetID);
  hgcHits->Branch("hefDetID", &hefDetID);
  hgcHits->Branch("hebDetID", &hebDetID);
}

void HGCHitValidation::beginRun(edm::Run const& iRun,
				edm::EventSetup const& iSetup) {
  //initiating hgc Geometry
  for (size_t i=0; i<geometrySource_.size(); i++) {
    if (geometrySource_[i].find("Hcal") != std::string::npos) {
      edm::LogVerbatim("HGCalValid") << "Tries to initialize HcalGeometry "
				     << " and HcalDDDSimConstants for " << i;
      ifHcalG_ = true;
      edm::ESHandle<HcalDDDSimConstants> pHSNDC;
      iSetup.get<HcalSimNumberingRecord>().get(pHSNDC);
      if (pHSNDC.isValid()) {
        hcCons_ = pHSNDC.product();
        hgcCons_.push_back(nullptr);
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
	hgcGeometry_.push_back(nullptr);
      } else {
        edm::LogWarning("HGCalValid") << "Cannot initiate HcalGeometry for "
                                      << geometrySource_[i] << std::endl;
      }
   } else {
      edm::LogVerbatim("HGCalValid") << "Tries to initialize HGCalGeometry "
				     << " and HGCalDDDConstants for " << i;
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

  //Accesing ee simhits
  edm::Handle<std::vector<PCaloHit>> eeSimHits;
  iEvent.getByToken(eeSimHitToken_, eeSimHits);

  if (eeSimHits.isValid()) {
    analyzeHGCalSimHit(eeSimHits, 0, eeHitRefs);
    for (std::map<unsigned int,HGCHitTuple>::iterator itr=eeHitRefs.begin();
	 itr != eeHitRefs.end(); ++itr) {
      int idx = std::distance(eeHitRefs.begin(),itr);
      edm::LogVerbatim("HGCalValid") << "EEHit[" << idx << "] " << std::hex 
				     << itr->first << std::dec << "; Energy "
				     << std::get<0>(itr->second) <<"; Position"
				     << " (" << std::get<1>(itr->second) <<", "
				     << std::get<2>(itr->second) <<", " 
				     << std::get<3>(itr->second) << ")";
    }
  } else {   
    edm::LogWarning("HGCalValid") << "No EE SimHit Found " << std::endl;
  }

  //Accesing fh simhits
  edm::Handle<std::vector<PCaloHit>> fhSimHits;
  iEvent.getByToken(fhSimHitToken_, fhSimHits);
  if (fhSimHits.isValid()) {
    analyzeHGCalSimHit(fhSimHits, 1, fhHitRefs);
    for (std::map<unsigned int,HGCHitTuple>::iterator itr=fhHitRefs.begin();
	 itr != fhHitRefs.end(); ++itr) {
      int idx = std::distance(fhHitRefs.begin(),itr);
      edm::LogVerbatim("HGCalValid") << "FHHit[" << idx << "] " << std::hex 
				     << itr->first << std::dec << "; Energy "
				     << std::get<0>(itr->second) <<"; Position"
				     << " ("  << std::get<1>(itr->second)<<", "
				     << std::get<2>(itr->second) <<", " 
				     << std::get<3>(itr->second) << ")";
    }
  } else {
    edm::LogWarning("HGCalValid") << "No FH SimHit Found " << std::endl;
  }
	
  //Accessing bh simhits
  edm::Handle<std::vector<PCaloHit>> bhSimHits;
  iEvent.getByToken(bhSimHitToken_, bhSimHits);
  if (bhSimHits.isValid()) {
    if (ifHcalG_) {
      for (std::vector<PCaloHit>::const_iterator simHit = bhSimHits->begin();
	   simHit != bhSimHits->end(); ++simHit) {
	int subdet, z, depth, eta, phi, lay;
	HcalTestNumbering::unpackHcalIndex(simHit->id(), subdet, z, depth, eta, phi, lay);
	
	if (subdet == static_cast<int>(HcalEndcap)) {
	  HcalCellType::HcalCell cell = hcCons_->cell(subdet, z, lay, eta, phi);
	  double zp  = cell.rz/10; 

	  HcalDDDRecConstants::HcalID idx = hcConr_->getHCID(subdet,eta,phi,lay,depth);
	  int sign = (z==0)?(-1):(1);
	  zp      *= sign;
	  HcalDetId id = HcalDetId(HcalEndcap,sign*idx.eta,idx.phi,idx.depth);  

	  float energy = simHit->energy();
	  float energySum(0);
	  if (bhHitRefs.count(id.rawId()) != 0) energySum = std::get<0>(bhHitRefs[id.rawId()]);
	  energySum += energy;
	  if (std::find(ietaExcludeBH_.begin(),ietaExcludeBH_.end(),idx.eta) == ietaExcludeBH_.end()) {

	    bhHitRefs[id.rawId()] = std::make_tuple(energySum,cell.eta,cell.phi,zp);
	    edm::LogVerbatim("HGCalValid") << "Accept " << id << std::endl;
	  } else {
	    edm::LogVerbatim("HGCalValid") << "Rejected cell " << idx.eta
					   << "," << id << std::endl;
	  }
	}
      }
    } else {
      analyzeHGCalSimHit(bhSimHits, 2, bhHitRefs);
    }
    for (std::map<unsigned int,HGCHitTuple>::iterator itr=bhHitRefs.begin();
	 itr != bhHitRefs.end(); ++itr) {
      int idx = std::distance(bhHitRefs.begin(),itr);
      edm::LogVerbatim("HGCalValid") << "BHHit[" << idx << "] " << std::hex 
				     << itr->first << std::dec << "; Energy " 
				     << std::get<0>(itr->second) <<"; Position"
				     << " (" << std::get<1>(itr->second) <<", "
				     << std::get<2>(itr->second) <<", " 
				     << std::get<3>(itr->second) << ")";
    }
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
      std::map<unsigned int, HGCHitTuple>::const_iterator itr = eeHitRefs.find(it->id().rawId());
      if (itr != eeHitRefs.end()) {
	GlobalPoint xyz = hgcGeometry_[0]->getPosition(it->id());
	
	heeRecX->push_back(xyz.x());
        heeRecY->push_back(xyz.y());
        heeRecZ->push_back(xyz.z());
        heeRecEnergy->push_back(energy);

        heeSimX->push_back(std::get<1>(itr->second));
        heeSimY->push_back(std::get<2>(itr->second));
        heeSimZ->push_back(std::get<3>(itr->second));
        heeSimEnergy->push_back(std::get<0>(itr->second));

	heeDetID->push_back(itr->first);
	edm::LogVerbatim("HGCalValid") << "EEHit: " << std::hex 
				       << it->id().rawId() << std::dec 
				       << " Sim (" << std::get<0>(itr->second)
				       << ", " << std::get<1>(itr->second) 
				       << ", " << std::get<2>(itr->second) 
				       << ", " << std::get<3>(itr->second) 
				       << ") Rec (" << energy << ", " 
				       << xyz.x() << ", " << xyz.y() << ", "
				       << xyz.z();
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
      std::map<unsigned int, HGCHitTuple>::const_iterator itr = fhHitRefs.find(it->id().rawId());
      if (itr != fhHitRefs.end()) {
	GlobalPoint xyz = hgcGeometry_[1]->getPosition(it->id());
	
	hefRecX->push_back(xyz.x());
        hefRecY->push_back(xyz.y());
        hefRecZ->push_back(xyz.z());
        hefRecEnergy->push_back(energy);

        hefSimX->push_back(std::get<1>(itr->second));
        hefSimY->push_back(std::get<2>(itr->second));
        hefSimZ->push_back(std::get<3>(itr->second));
        hefSimEnergy->push_back(std::get<0>(itr->second));

	hefDetID->push_back(itr->first);
	edm::LogVerbatim("HGCalValid") << "FHHit: " << std::hex 
				       << it->id().rawId() << std::dec 
				       << " Sim (" << std::get<0>(itr->second)
				       << ", " << std::get<1>(itr->second)
				       << ", " << std::get<2>(itr->second)
				       << ", " << std::get<3>(itr->second)
				       << ") Rec (" << energy << ","
				       << xyz.x() << ", " << xyz.y() << ", "
				       << xyz.z();
      }
    }
  } else {
    edm::LogWarning("HGCalValid") << "No FH RecHit Found " << std::endl;
  }

  //accessing BH Rechit information
  if (ifHCAL_) {
    edm::Handle<HBHERecHitCollection> bhRecHit;
    iEvent.getByToken(bhRecHitTokenh_, bhRecHit);
    if (bhRecHit.isValid()) {
      const HBHERecHitCollection* theHits = (bhRecHit.product());
      analyzeHGCalRecHit(theHits, bhHitRefs);
    } else {
      edm::LogWarning("HGCalValid") << "No BH RecHit Found " << std::endl;
    }
  } else {
    edm::Handle<HGChebRecHitCollection> bhRecHit;
    iEvent.getByToken(bhRecHitTokeng_, bhRecHit);
    if (bhRecHit.isValid()) {
      const HGChebRecHitCollection* theHits = (bhRecHit.product());
      analyzeHGCalRecHit(theHits, bhHitRefs);
    } else {
      edm::LogWarning("HGCalValid") << "No BH RecHit Found " << std::endl;
    }
  }

  hgcHits->Fill();

  heeRecX->clear(); heeRecY->clear(); heeRecZ->clear(); heeRecEnergy->clear();
  hefRecX->clear(); hefRecY->clear(); hefRecZ->clear(); hefRecEnergy->clear();
  hebRecX->clear(); hebRecY->clear(); hebRecZ->clear(); hebRecEnergy->clear();
  heeSimX->clear(); heeSimY->clear(); heeSimZ->clear(); heeSimEnergy->clear();
  hefSimX->clear(); hefSimY->clear(); hefSimZ->clear(); hefSimEnergy->clear();
  hebSimX->clear(); hebSimY->clear(); hebSimZ->clear(); hebSimEnergy->clear();
  hebSimEta->clear(); hebRecEta->clear();
  hebSimPhi->clear(); hebRecPhi->clear();
  heeDetID->clear(); hefDetID->clear(); hebDetID->clear();
}

void HGCHitValidation::endJob() {
  hgcHits->GetDirectory()->cd();
  hgcHits->Write();
}

void HGCHitValidation::analyzeHGCalSimHit(edm::Handle<std::vector<PCaloHit>> const& simHits,
					  int idet, 
					  std::map<unsigned int, HGCHitTuple>& hitRefs) {

  const HGCalTopology &hTopo=hgcGeometry_[idet]->topology();
  for (auto const & simHit : *simHits) {
    unsigned int id  = simHit.id();
    std::pair<float,float> xy;
    bool ok(true);
    int subdet(0), zside, layer, wafer, celltype, cell, wafer2(0), cell2(0);
    if ((hgcCons_[idet]->geomMode() == HGCalGeometryMode::Hexagon8) ||
	(hgcCons_[idet]->geomMode() == HGCalGeometryMode::Hexagon8Full)) {
      HGCSiliconDetId detId = HGCSiliconDetId(id);
      subdet   = (int)(detId.det());
      cell     = detId.cellU();
      cell2    = detId.cellV();
      wafer    = detId.waferU();
      wafer2   = detId.waferV();
      celltype = detId.type();
      layer    = detId.layer();
      zside    = detId.zside();
      xy       = hgcCons_[idet]->locateCell(layer,wafer,wafer2,cell,cell2,
					    false,true);
    } else if (hgcCons_[idet]->geomMode() == HGCalGeometryMode::Trapezoid) {
      HGCScintillatorDetId detId = HGCScintillatorDetId(id);
      subdet   = (int)(detId.det());
      cell     = detId.ietaAbs();
      wafer    = detId.iphi();
      celltype = detId.type();
      layer    = detId.layer();
      zside    = detId.zside();
      xy       = hgcCons_[idet]->locateCellTrap(layer,wafer,cell,false);
    } else {
      HGCalTestNumbering::unpackHexagonIndex(simHit.id(), subdet, zside, 
					     layer, wafer, celltype, cell);
      xy       = hgcCons_[idet]->locateCell(cell,layer,wafer,false);

      //skip this hit if after ganging it is not valid
      std::pair<int,int> recoLayerCell=hgcCons_[idet]->simToReco(cell,layer,wafer,hTopo.detectorType());
      ok       = !(recoLayerCell.second<0 || recoLayerCell.first<0);
      id       = HGCalDetId((ForwardSubdetector)(subdet),zside,layer,celltype,
			    wafer,cell).rawId();
    }

    edm::LogVerbatim("HGCalValid") << "SimHit: " << std::hex << id << std::dec
				   << " (" << subdet << ":" << zside << ":"
				   << layer << ":" << celltype << ":" << wafer
				   << ":" << wafer2 << ":" << cell << ":"
				   << cell2 << ") Flag " << ok;

    if (ok) {
      float zp = hgcCons_[idet]->waferZ(layer,false);
      if (zside < 0) zp = -zp;
      float xp = (zside<0) ? -xy.first/10 : xy.first/10;
      float yp = xy.second/10.0;

      float energySum(simHit.energy());
      if (hitRefs.count(id) != 0) energySum += std::get<0>(hitRefs[id]);
      hitRefs[id] = std::make_tuple(energySum,xp,yp,zp);
      edm::LogVerbatim("HGCalValid") << "Position (" << xp << ", " << yp 
				     << ", " << zp << ") " << " Energy "
				     << simHit.energy() << ":" << energySum;
    }
  }
}

template<class T1>
void HGCHitValidation::analyzeHGCalRecHit(T1 const & theHits, 
					  std::map<unsigned int, HGCHitTuple> const& hitRefs) {

  for (auto it = theHits->begin(); it!=theHits->end(); ++it) {
    DetId id = it->id();
    bool  ok = (ifHCAL_) ? (id.subdetId() == (int)(HcalEndcap)) : true;
    if (ok) {
      double energy = it->energy();
      GlobalPoint xyz = (hcGeometry_ ?
			 (hcGeometry_->getGeometry(id)->getPosition()) :
			 (hgcGeometry_[2]->getPosition(id)));
	
      std::map<unsigned int, HGCHitTuple>::const_iterator itr = hitRefs.find(id.rawId());
      if (itr != hitRefs.end()) {
	float ang3 = xyz.phi().value(); // returns the phi in radians
	double fac = sinh(std::get<1>(itr->second));
	double pT  = std::get<3>(itr->second) / fac;
	double xp = pT * cos(std::get<2>(itr->second));
	double yp = pT * sin(std::get<2>(itr->second));

	hebRecX->push_back(xyz.x());
	hebRecY->push_back(xyz.y());
	hebRecZ->push_back(xyz.z());
	hebRecEnergy->push_back(energy);

	hebSimX->push_back(xp);
	hebSimY->push_back(yp);
	hebSimZ->push_back(std::get<3>(itr->second));
	hebSimEnergy->push_back(std::get<0>(itr->second));

	hebSimEta->push_back(std::get<1>(itr->second));
	hebRecEta->push_back(xyz.eta());
	hebSimPhi->push_back(std::get<2>(itr->second));
	hebRecPhi->push_back(ang3);

	hebDetID->push_back(itr->first);
	edm::LogVerbatim("HGCalValid") << "BHHit: " << std::hex << id.rawId() 
				       << std::dec << " Sim (" 
				       << std::get<0>(itr->second) << ", "
				       << std::get<1>(itr->second) << ", " 
				       << std::get<2>(itr->second) << ", " 
				       << std::get<3>(itr->second) << ") Rec ("
				       << energy << ", " << xyz.eta() << ", " 
				       << ang3 << ", " << xyz.z() << ")";
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCHitValidation);



