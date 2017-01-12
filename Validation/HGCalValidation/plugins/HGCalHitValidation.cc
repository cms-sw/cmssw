//// -*- C++ -*-
//
// Package:    HGCalHitValidation
// Class:      HGCalHitValidation
// 
/**\class HGCalHitValidation HGCalHitValidation.cc Validation/HGCalValidation/plugins/HGCalHitValidation.cc

 Description: [one line class summary]

 Implementation:
 	[Notes on implementation]
*/

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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include <cmath>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

//#define EDM_ML_DEBUG

class HGCalHitValidation : public DQMEDAnalyzer {

public:

  explicit HGCalHitValidation( const edm::ParameterSet& );
  ~HGCalHitValidation();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  typedef std::tuple<float,float,float,float> HGCHitTuple;

  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void analyzeHGCalSimHit(edm::Handle<std::vector<PCaloHit>> const& simHits,
			  int idet, MonitorElement *hist,
			  std::map<unsigned int, HGCHitTuple>&);
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
  bool                                  ifHCAL_;

  edm::InputTag eeSimHitSource, fhSimHitSource, bhSimHitSource;
  edm::EDGetTokenT<std::vector<PCaloHit>> eeSimHitToken_;
  edm::EDGetTokenT<std::vector<PCaloHit>> fhSimHitToken_;
  edm::EDGetTokenT<std::vector<PCaloHit>> bhSimHitToken_;
  edm::EDGetTokenT<HGCeeRecHitCollection> eeRecHitToken_;
  edm::EDGetTokenT<HGChefRecHitCollection> fhRecHitToken_;
  edm::EDGetTokenT<HGChebRecHitCollection> bhRecHitTokeng_;
  edm::EDGetTokenT<HBHERecHitCollection> bhRecHitTokenh_;

  //histogram related stuff
  MonitorElement *heedzVsZ, *heedyVsY, *heedxVsX;
  MonitorElement *hefdzVsZ, *hefdyVsY, *hefdxVsX;
  MonitorElement *hebdzVsZ, *hebdPhiVsPhi, *hebdEtaVsEta;
	
  MonitorElement *heeRecVsSimZ, *heeRecVsSimY, *heeRecVsSimX;
  MonitorElement *hefRecVsSimZ, *hefRecVsSimY, *hefRecVsSimX;
  MonitorElement *hebRecVsSimZ, *hebRecVsSimY, *hebRecVsSimX;
  MonitorElement *heeEnSimRec, *hefEnSimRec, *hebEnSimRec;

  MonitorElement *hebEnRec, *hebEnSim;
  MonitorElement *hefEnRec, *hefEnSim;
  MonitorElement *heeEnRec, *heeEnSim;

};


HGCalHitValidation::HGCalHitValidation(const edm::ParameterSet &cfg) {

  geometrySource_ = cfg.getUntrackedParameter< std::vector<std::string> >("geometrySource");
  eeSimHitToken_  = consumes<std::vector<PCaloHit>>(cfg.getParameter<edm::InputTag>("eeSimHitSource"));
  fhSimHitToken_  = consumes<std::vector<PCaloHit>>(cfg.getParameter<edm::InputTag>("fhSimHitSource"));
  bhSimHitToken_  = consumes<std::vector<PCaloHit>>(cfg.getParameter<edm::InputTag>("bhSimHitSource"));
  eeRecHitToken_  = consumes<HGCeeRecHitCollection>(cfg.getParameter<edm::InputTag>("eeRecHitSource"));
  fhRecHitToken_  = consumes<HGChefRecHitCollection>(cfg.getParameter<edm::InputTag>("fhRecHitSource"));
  ietaExcludeBH_  = cfg.getParameter<std::vector<int> >("ietaExcludeBH");
  ifHCAL_         = cfg.getParameter<bool>("ifHCAL");
  if (ifHCAL_) 
    bhRecHitTokenh_ = consumes<HBHERecHitCollection>(cfg.getParameter<edm::InputTag>("bhRecHitSource"));
  else
    bhRecHitTokeng_ = consumes<HGChebRecHitCollection>(cfg.getParameter<edm::InputTag>("bhRecHitSource"));

#ifdef EDM_ML_DEBUG
  edm::LogInfo("HGCalValid") << "Exclude the following " 
			     << ietaExcludeBH_.size() 
			     << " ieta values from BH plots (BH "
			     << ifHCAL_ << ") ";
  for (unsigned int k=0; k<ietaExcludeBH_.size(); ++k) 
    edm::LogInfo("HGCalValid") << " [" << k << "] " << ietaExcludeBH_[k];
#endif
}

HGCalHitValidation::~HGCalHitValidation() { }

void HGCalHitValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void HGCalHitValidation::bookHistograms(DQMStore::IBooker& iB, 
					edm::Run const&, 
					edm::EventSetup const&) {

  iB.setCurrentFolder("HGCalSimHitsV/HitValidation");

  //initiating histograms
  heedzVsZ     = iB.book2D("heedzVsZ","",7200,-360,360,100,-0.1,0.1);
  heedyVsY     = iB.book2D("heedyVsY","",400,-200,200,100,-0.02,0.02);
  heedxVsX     = iB.book2D("heedxVsX","",400,-200,200,100,-0.02,0.02);
  heeRecVsSimZ = iB.book2D("heeRecVsSimZ","",7200,-360,360,7200,-360,360);
  heeRecVsSimY = iB.book2D("heeRecVsSimY","",400,-200,200,400,-200,200);
  heeRecVsSimX = iB.book2D("heeRecVsSimX","",400,-200,200,400,-200,200);

  hefdzVsZ     = iB.book2D("hefdzVsZ","",8200,-410,410,100,-0.1,0.1);
  hefdyVsY     = iB.book2D("hefdyVsY","",400,-200,200,100,-0.02,0.02);
  hefdxVsX     = iB.book2D("hefdxVsX","",400,-200,200,100,-0.02,0.02);
  hefRecVsSimZ = iB.book2D("hefRecVsSimZ","",8200,-410,410,8200,-410,410);
  hefRecVsSimY = iB.book2D("hefRecVsSimY","",400,-200,200,400,-200,200);
  hefRecVsSimX = iB.book2D("hefRecVsSimX","",400,-200,200,400,-200,200);

  hebdzVsZ     = iB.book2D("hebdzVsZ","",1080,-540,540,100,-1.0,1.0);
  hebdPhiVsPhi = iB.book2D("hebdPhiVsPhi","",M_PI*100,-0.5,M_PI+0.5,200,-0.2,0.2);
  hebdEtaVsEta = iB.book2D("hebdEtaVsEta","",1000,-5,5,200,-0.1,0.1);
  hebRecVsSimZ = iB.book2D("hebRecVsSimZ","",1080,-540,540,1080,-540,540);
  hebRecVsSimY = iB.book2D("hebRecVsSimY","",400,-200,200,400,-200,200);
  hebRecVsSimX = iB.book2D("hebRecVsSimX","",400,-200,200,400,-200,200);

  heeEnRec     = iB.book1D("heeEnRec","",1000,0,10);
  heeEnSim     = iB.book1D("heeEnSim","",1000,0,0.01);
  heeEnSimRec  = iB.book2D("heeEnSimRec","",200,0,0.002,200,0,0.2);

  hefEnRec     = iB.book1D("hefEnRec","",1000,0,10);
  hefEnSim     = iB.book1D("hefEnSim","",1000,0,0.01);
  hefEnSimRec  = iB.book2D("hefEnSimRec","",200,0,0.001,200,0,0.5);

  hebEnRec     = iB.book1D("hebEnRec","",1000,0,15);
  hebEnSim     = iB.book1D("hebEnSim","",1000,0,0.01);
  hebEnSimRec  = iB.book2D("hebEnSimRec","",200,0,0.02,200,0,4);

}

void HGCalHitValidation::dqmBeginRun(edm::Run const& iRun,
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

void HGCalHitValidation::analyze( const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::map<unsigned int, HGCHitTuple> eeHitRefs, fhHitRefs, bhHitRefs;

  //Accesing ee simhits
  edm::Handle<std::vector<PCaloHit>> eeSimHits;
  iEvent.getByToken(eeSimHitToken_, eeSimHits);

  if (eeSimHits.isValid()) {
    analyzeHGCalSimHit(eeSimHits, 0, heeEnSim, eeHitRefs);
#ifdef EDM_ML_DEBUG
    for (std::map<unsigned int,HGCHitTuple>::iterator itr=eeHitRefs.begin();
	 itr != eeHitRefs.end(); ++itr) {
      int idx = std::distance(eeHitRefs.begin(),itr);
      edm::LogInfo("HGCalValid") << "EEHit[" << idx << "] " << std::hex 
				 << itr->first << std::dec << "; Energy " 
				 << std::get<0>(itr->second) 
				 << "; Position (" << std::get<1>(itr->second) 
				 << ", " << std::get<2>(itr->second) <<", " 
				 << std::get<3>(itr->second) << ")" <<std::endl;
    }
#endif
  } else {   
    edm::LogWarning("HGCalValid") << "No EE SimHit Found " << std::endl;
  }

  //Accesing fh simhits
  edm::Handle<std::vector<PCaloHit>> fhSimHits;
  iEvent.getByToken(fhSimHitToken_, fhSimHits);
  if (fhSimHits.isValid()) {
    analyzeHGCalSimHit(fhSimHits, 1, hefEnSim, fhHitRefs);
#ifdef EDM_ML_DEBUG
    for (std::map<unsigned int,HGCHitTuple>::iterator itr=fhHitRefs.begin();
	 itr != fhHitRefs.end(); ++itr) {
      int idx = std::distance(fhHitRefs.begin(),itr);
      edm::LogInfo("HGCalValid") << "FHHit[" << idx << "] " << std::hex 
				 << itr->first << std::dec << "; Energy "
				 << std::get<0>(itr->second) << "; Position (" 
				 << std::get<1>(itr->second) << ", "
				 << std::get<2>(itr->second) <<", " 
				 << std::get<3>(itr->second) << ")" <<std::endl;
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

      if (subdet == static_cast<int>(HcalEndcap)) {
	HcalCellType::HcalCell cell = hcCons_->cell(subdet, z, lay, eta, phi);
	double zp  = cell.rz/10; 

	HcalDDDRecConstants::HcalID idx = hcConr_->getHCID(subdet,eta,phi,lay,depth);
	int sign = (z==0)?(-1):(1);
	zp      *= sign;
	HcalDetId id = HcalDetId(HcalEndcap,sign*idx.eta,idx.phi,idx.depth);  

	float energy = simHit->energy();
	float energySum(energy);
	if (bhHitRefs.count(id.rawId()) != 0) energySum += std::get<0>(bhHitRefs[id.rawId()]);
	hebEnSim->Fill(energy);
	if (std::find(ietaExcludeBH_.begin(),ietaExcludeBH_.end(),idx.eta) ==
	    ietaExcludeBH_.end()) {
	  bhHitRefs[id.rawId()] = std::make_tuple(energySum,cell.eta,cell.phi,zp);
#ifdef EDM_ML_DEBUG
	  edm::LogInfo("HGCalValid") << "Accept " << id << std::endl;
	} else {
	  edm::LogInfo("HGCalValid") << "Reject " << id << std::endl;
#endif
	}
      }
    }
#ifdef EDM_ML_DEBUG
    for (std::map<unsigned int,HGCHitTuple>::iterator itr=bhHitRefs.begin();
	 itr != bhHitRefs.end(); ++itr) {
      int idx = std::distance(bhHitRefs.begin(),itr);
      edm::LogInfo("HGCalValid") << "BHHit[" << idx << "] " << std::hex 
				 << itr->first << std::dec << "; Energy " 
				 << std::get<0>(itr->second) << "; Position (" 
				 << std::get<1>(itr->second) << ", "
				 << std::get<2>(itr->second) <<", " 
				 << std::get<3>(itr->second) << ")" <<std::endl;
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
#ifdef EDM_ML_DEBUG
	edm::LogInfo("HGCalValid") << "EEHit: " << std::hex << it->id().rawId() 
				   << std::dec << " Sim (" 
				   << std::get<0>(itr->second) << ", "
				   << std::get<1>(itr->second) << ", " 
				   << std::get<2>(itr->second) << ", " 
				   << std::get<3>(itr->second) << ") Rec (" 
				   << energy << ", "  << xyz.x() << ", " 
				   << xyz.y() << ", " << xyz.z() << ")\n";
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
#ifdef EDM_ML_DEBUG
	edm::LogInfo("HGCalValid") << "FHHit: " << std::hex << it->id().rawId()
				   << std::dec << " Sim (" 
				   << std::get<0>(itr->second) << ", "
				   << std::get<1>(itr->second) << ", " 
				   << std::get<2>(itr->second) << ", " 
				   << std::get<3>(itr->second) << ") Rec (" 
				   << energy << "," << xyz.x() << ", " 
				   << xyz.y() << ", " << xyz.z() << ")\n";
#endif
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
}

void HGCalHitValidation::analyzeHGCalSimHit(edm::Handle<std::vector<PCaloHit>> const& simHits,
					    int idet, MonitorElement *hist,
					    std::map<unsigned int, HGCHitTuple>& hitRefs) {

  const HGCalTopology &hTopo=hgcGeometry_[idet]->topology();
  for (std::vector<PCaloHit>::const_iterator simHit = simHits->begin(); simHit != simHits->end(); ++simHit) {
    int subdet, zside, layer, wafer, celltype, cell;
    HGCalTestNumbering::unpackHexagonIndex(simHit->id(), subdet, zside, layer, wafer, celltype, cell);
    std::pair<float, float> xy = hgcCons_[idet]->locateCell(cell,layer,wafer,false);
    float zp = hgcCons_[idet]->waferZ(layer,false);
    if (zside < 0) zp = -zp;
    float xp = (zp<0) ? -xy.first/10 : xy.first/10;
    float yp = xy.second/10.0;

    //skip this hit if after ganging it is not valid
    std::pair<int,int> recoLayerCell=hgcCons_[idet]->simToReco(cell,layer,wafer,hTopo.detectorType());
    cell  = recoLayerCell.first;
    layer = recoLayerCell.second;

    //skip this hit if after ganging it is not valid
    if (layer<0 || cell<0) {
    } else {
      //assign the RECO DetId
      HGCalDetId id = HGCalDetId((ForwardSubdetector)(subdet),zside,layer,celltype,wafer,cell);
      float energy = simHit->energy();

      float energySum(energy);
      if (hitRefs.count(id.rawId()) != 0) energySum += std::get<0>(hitRefs[id.rawId()]);
      hitRefs[id.rawId()] = std::make_tuple(energySum,xp,yp,zp);
      hist->Fill(energy);
    }
  }
}

template<class T1>
void HGCalHitValidation::analyzeHGCalRecHit(T1 const & theHits, 
					    std::map<unsigned int, HGCHitTuple> const& hitRefs) {
  for (auto it = theHits->begin(); it!=theHits->end(); ++it) {
    DetId id = it->id();
    if (id.subdetId() == (int)(HcalEndcap)) {
      double energy = it->energy();
      hebEnRec->Fill(energy);
      GlobalPoint xyz = hcGeometry_->getGeometry(id)->getPosition();

      std::map<unsigned int, HGCHitTuple>::const_iterator itr = hitRefs.find(id.rawId());
      if (itr != hitRefs.end()) {
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

#ifdef EDM_ML_DEBUG
	edm::LogInfo("HGCalValid") << "BHHit: " << std::hex << id.rawId()
				   << std::dec << " Sim (" 
				   << std::get<0>(itr->second) << ", "
				   << std::get<1>(itr->second) << ", " 
				   << std::get<2>(itr->second) << ", " 
				   << std::get<3>(itr->second) << ") Rec ("
				   << energy << ", "  << xyz.x() << ", " 
				   << xyz.y() << ", " << xyz.z() << ")\n";
#endif
      }
    }
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalHitValidation);
