// system include files
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Transform3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include "TH2D.h"

class HGCGeometryTest : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {
  
public:

  explicit HGCGeometryTest(const edm::ParameterSet&);
  ~HGCGeometryTest() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:

  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  
private:

  void analyzeHits (int, const std::string&, const std::vector<PCaloHit>&);
  
  // ----------member data ---------------------------
  std::vector<std::string>              nameDetectors_, caloHitSources_;
  std::vector<const HGCalDDDConstants*> hgcons_;
  const HcalDDDRecConstants*            hcons_;
  int                                   verbosity_;
  std::vector<bool>                     heRebuild_;
  std::vector<edm::EDGetTokenT<edm::PCaloHitContainer> > tok_hits_;
  std::vector<int>                      layers_;
  
  //histogram related stuff
  std::vector<TH2D*>                   h_RZ_, h_EtaPhi_;
};

HGCGeometryTest::HGCGeometryTest(const edm::ParameterSet& iConfig) :
  nameDetectors_(iConfig.getParameter<std::vector<std::string> >("DetectorNames")),
  caloHitSources_(iConfig.getParameter<std::vector<std::string> >("CaloHitSources")),
  verbosity_(iConfig.getUntrackedParameter<int>("Verbosity",0)) {

  usesResource(TFileService::kSharedResource);

  for (auto const& name : nameDetectors_) {
    if (name == "HCal") heRebuild_.emplace_back(true);
    else                heRebuild_.emplace_back(false);
  }
  for (auto const& source : caloHitSources_) {
    tok_hits_.emplace_back(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits",source)));
  }
}

void HGCGeometryTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> names   = {"HGCalEESensitive",
				      "HGCalHESiliconSensitive",
				      "HGCalHEScintillatorSensitive"};
  std::vector<std::string> sources = {"HGCHitsEE","HGCHitsHEfront",
				      "HGCHitsHEback"};
  desc.add<std::vector<std::string> >("DetectorNames", names);
  desc.add<std::vector<std::string> >("CaloHitSources",sources);
  desc.addUntracked<int>("Verbosity",0);
  descriptions.add("hgcalGeometryTest",desc);
}

void HGCGeometryTest::analyze(const edm::Event& iEvent, 
			      const edm::EventSetup& iSetup) {


  //Now the hits
  for (unsigned int k=0; k<tok_hits_.size(); ++k) {
    edm::Handle<edm::PCaloHitContainer> theCaloHitContainers;
    iEvent.getByToken(tok_hits_[k], theCaloHitContainers);
    if (theCaloHitContainers.isValid()) {
      if (verbosity_>0) 
	edm::LogVerbatim("HGCalValidation") << " PcalohitItr = " 
					    << theCaloHitContainers->size();
      std::vector<PCaloHit>               caloHits;
      if (heRebuild_[k]) {
	for (auto const& hit : *(theCaloHitContainers.product()) ) {
	  unsigned int id = hit.id();
	  HcalDetId hid = HcalHitRelabeller::relabel(id,hcons_);
	  if (hid.subdet()!=int(HcalEndcap)) {
	    caloHits.emplace_back(hit);
	    caloHits.back().setID(hid.rawId());
	    if (verbosity_>0)
	      edm::LogVerbatim("HGCalValidation") << "Hit[" << caloHits.size()
						  << "] " << hid;
	  }
	}
      } else {
	caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), 
			theCaloHitContainers->end());
      }
      analyzeHits(k,nameDetectors_[k],caloHits);
    } else if (verbosity_>0) {
      edm::LogVerbatim("HGCalValidation") << "PCaloHitContainer does not "
					  << "exist for " << nameDetectors_[k];
    }
  }
}

void HGCGeometryTest::analyzeHits(int ih, std::string const& name,
				  std::vector<PCaloHit> const& hits) {
  
 
  if (verbosity_ > 0) 
    edm::LogVerbatim("HGCalValidation") << name << " with " << hits.size() 
					<< " PcaloHit elements";
  unsigned int nused(0);
  for (auto const& hit : hits) {
    double energy      = hit.energy();
    uint32_t id        = hit.id();
    int                     cell, sector, sector2(0), layer, zside;
    int                     subdet(0), cell2(0), type(0);
    HepGeom::Point3D<float> gcoord;
    if (heRebuild_[ih]) { 
      HcalDetId detId  = HcalDetId(id);
      subdet           = detId.subdet();
      cell             = detId.ietaAbs();
      sector           = detId.iphi();
      layer            = detId.depth();
      zside            = detId.zside();
      std::pair<double,double> etaphi = hcons_->getEtaPhi(subdet,zside*cell,
							  sector);
      double                   rz = hcons_->getRZ(subdet,zside*cell,layer);
      if (verbosity_>2) 
	edm::LogVerbatim("HGCalValidation") << "i/p " << subdet << ":" 
					    << zside << ":" << cell << ":" 
					    << sector << ":" << layer <<" o/p "
					    << etaphi.first << ":" 
					    << etaphi.second << ":" << rz;
      gcoord = HepGeom::Point3D<float>(rz*cos(etaphi.second)/cosh(etaphi.first),
				       rz*sin(etaphi.second)/cosh(etaphi.first),
				       rz*tanh(etaphi.first));
    } else {
      std::pair<float,float> xy;
      if ((hgcons_[ih]->geomMode() == HGCalGeometryMode::Hexagon8) ||
	  (hgcons_[ih]->geomMode() == HGCalGeometryMode::Hexagon8Full)) {
	HGCSiliconDetId detId = HGCSiliconDetId(id);
	subdet   = (int)(detId.det());
	cell     = detId.cellU();
	cell2    = detId.cellV();
	sector   = detId.waferU();
	sector2  = detId.waferV();
	type     = detId.type();
	layer    = detId.layer();
	zside    = detId.zside();
	xy       = hgcons_[ih]->locateCell(layer,sector,sector2,cell,cell2,
					   false,true);
      } else if (hgcons_[ih]->geomMode() == HGCalGeometryMode::Trapezoid) {
	HGCScintillatorDetId detId = HGCScintillatorDetId(id);
	subdet   = (int)(detId.det());
	sector   = detId.ietaAbs();
	cell     = detId.iphi();
	type     = detId.type();
	layer    = detId.layer();
	zside    = detId.zside();
	xy       = hgcons_[ih]->locateCellTrap(layer,sector,cell,false);
      } else {
	HGCalTestNumbering::unpackHexagonIndex(id, subdet, zside, layer, 
					       sector, type, cell);
	xy       = hgcons_[ih]->locateCell(cell,layer,sector,false);
      }
      double zp  = hgcons_[ih]->waferZ(layer,false);
      if (zside < 0) zp = -zp;
      double xp  = (zp < 0) ? -xy.first : xy.first;
      gcoord = HepGeom::Point3D<float>(xp,xy.second,zp);
      if (verbosity_>2) 
	edm::LogVerbatim("HGCalValidation") << "i/p " << subdet << ":" 
					    << zside << ":" << layer << ":" 
					    << sector << ":" << sector2 << ":"
					    << cell << ":" << cell2 << " o/p "
					    << xy.first << ":" << xy.second
					    << ":" << zp;
    }
    nused++;
    if (verbosity_>1) 
      edm::LogVerbatim("HGCalValidation") << "Detector " << name
					  << " zside = " << zside
					  << " layer = " << layer
					  << " type = "  << type
					  << " wafer = " << sector << ":" << sector2
					  << " cell = "  << cell << ":" << cell2
					  << " positon = " << gcoord
					  << " energy = "    << energy;
    //Fill in histograms
    h_RZ_[0]   ->Fill(std::abs(gcoord.z()),gcoord.rho());
    h_RZ_[ih+1]->Fill(std::abs(gcoord.z()),gcoord.rho());
    h_EtaPhi_[0]   ->Fill(std::abs(gcoord.eta()),gcoord.phi());
    h_EtaPhi_[ih+1]->Fill(std::abs(gcoord.eta()),gcoord.phi());
  }
  if (verbosity_>0) 
    edm::LogVerbatim("HGCalValidation") << name << " with " << nused
					<< " detector elements being hit";
}

// ------------ method called when starting to processes a run  ------------
void HGCGeometryTest::beginRun(const edm::Run&, 
			       const edm::EventSetup& iSetup) {
  for (unsigned int k=0; k<nameDetectors_.size(); ++k) {
    if (heRebuild_[k]) {
      edm::ESHandle<HcalDDDRecConstants> pHRNDC;
      iSetup.get<HcalRecNumberingRecord>().get( pHRNDC );
      hcons_  = &(*pHRNDC);
      layers_.emplace_back(hcons_->getMaxDepth(1));
      hgcons_.emplace_back(nullptr);
    } else {
      edm::ESHandle<HGCalDDDConstants>  pHGDC;
      iSetup.get<IdealGeometryRecord>().get(nameDetectors_[k], pHGDC);
      hgcons_.emplace_back(&(*pHGDC));
      layers_.emplace_back(hgcons_.back()->layers(false));
    }
    if (verbosity_>0) 
      edm::LogVerbatim("HGCalValidation") << nameDetectors_[k] 
					  << " defined with "
					  << layers_.back() << " Layers";
  }
}

void HGCGeometryTest::beginJob() {

  edm::Service<TFileService> fs;
    
  std::ostringstream name, title;
  for (unsigned int ih=0; ih <= nameDetectors_.size(); ++ih) {
    name.str(""); title.str("");
    if (ih == 0) {
      name << "RZ_AllDetectors";
      title << "R vs Z for All Detectors";
    } else {
      name  << "RZ_" << nameDetectors_[ih-1];
      title << "R vs Z for " << nameDetectors_[ih-1];
    }
    h_RZ_.emplace_back(fs->make<TH2D>(name.str().c_str(), 
				      title.str().c_str(),
				      300,3000.0,6000.0,300,0.0,3000.0));
    name.str(""); title.str("");
    if (ih == 0) {
      name << "EtaPhi_AllDetectors";
      title << "#phi vs #eta for All Detectors";
    } else {
      name  << "EtaPhi_" << nameDetectors_[ih-1];
      title << "#phi vs #eta for " << nameDetectors_[ih-1];
    }
    h_EtaPhi_.emplace_back(fs->make<TH2D>(name.str().c_str(), 
					  title.str().c_str(),
					  200,1.0,3.0,200,-M_PI,M_PI));
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HGCGeometryTest);
