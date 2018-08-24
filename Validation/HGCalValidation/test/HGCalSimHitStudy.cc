// system include files
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
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

class HGCalSimHitStudy : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {
  
public:
  
  struct hitsinfo{
    hitsinfo() {
      phi=eta=energy=time=0.0;
      layer=0;
    }
    double phi, eta, energy, time;
    int    layer;
  };

  explicit HGCalSimHitStudy(const edm::ParameterSet&);
  ~HGCalSimHitStudy() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:

  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  
private:

  void analyzeHits (int, const std::string&, const std::vector<PCaloHit>&);
  
  // ----------member data ---------------------------
  const std::vector<std::string>        nameDetectors_, caloHitSources_;
  const double                          rmin_, rmax_, zmin_, zmax_;
  const int                             nbinR_, nbinZ_, verbosity_;
  const bool                            ifNose_;
  std::vector<const HGCalDDDConstants*> hgcons_;
  const HcalDDDRecConstants*            hcons_;
  std::vector<bool>                     heRebuild_;
  std::vector<edm::EDGetTokenT<edm::PCaloHitContainer> > tok_hits_;
  std::vector<int>                      layers_, layerFront_;
  
  //histogram related stuff
  std::vector<TH2D*>                    h_RZ_, h_EtaPhi_, h_EtFiZp_, h_EtFiZm_;
  std::vector<TH1D*>                    h_E_, h_T_, h_LayerZp_, h_LayerZm_;
};

HGCalSimHitStudy::HGCalSimHitStudy(const edm::ParameterSet& iConfig) :
  nameDetectors_(iConfig.getParameter<std::vector<std::string> >("detectorNames")),
  caloHitSources_(iConfig.getParameter<std::vector<std::string> >("caloHitSources")),
  rmin_(iConfig.getUntrackedParameter<double>("rMin",0.0)),
  rmax_(iConfig.getUntrackedParameter<double>("rMax",3000.0)),
  zmin_(iConfig.getUntrackedParameter<double>("zMin",3000.0)),
  zmax_(iConfig.getUntrackedParameter<double>("zMax",6000.0)),
  nbinR_(iConfig.getUntrackedParameter<int>("nBinR",300)),
  nbinZ_(iConfig.getUntrackedParameter<int>("nBinZ",300)),
  verbosity_(iConfig.getUntrackedParameter<int>("verbosity",0)),
  ifNose_(iConfig.getUntrackedParameter<bool>("ifNose",false)) {

  usesResource(TFileService::kSharedResource);

  for (auto const& name : nameDetectors_) {
    if (name == "HCal") heRebuild_.emplace_back(true);
    else                heRebuild_.emplace_back(false);
  }
  for (auto const& source : caloHitSources_) {
    tok_hits_.emplace_back(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits",source)));
  }
}

void HGCalSimHitStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> names   = {"HGCalEESensitive",
				      "HGCalHESiliconSensitive",
				      "Hcal"};
  std::vector<std::string> sources = {"HGCHitsEE","HGCHitsHEfront",
				      "HcalHits"};
  desc.add<std::vector<std::string> >("detectorNames", names);
  desc.add<std::vector<std::string> >("caloHitSources",sources);
  desc.addUntracked<double>("rMin",0.0);
  desc.addUntracked<double>("rMax",3000.0);
  desc.addUntracked<double>("zMin",3000.0);
  desc.addUntracked<double>("zMax",6000.0);
  desc.addUntracked<int>("nBinR",300);
  desc.addUntracked<int>("nBinZ",300);
  desc.addUntracked<int>("verbosity",0);
  desc.addUntracked<bool>("ifNose",false);
  descriptions.add("hgcalSimHitStudy",desc);
}

void HGCalSimHitStudy::analyze(const edm::Event& iEvent, 
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

void HGCalSimHitStudy::analyzeHits(int ih, std::string const& name,
				   std::vector<PCaloHit> const& hits) {
  
 
  if (verbosity_ > 0) 
    edm::LogVerbatim("HGCalValidation") << name << " with " << hits.size() 
					<< " PcaloHit elements";
  
  std::map<uint32_t,hitsinfo> map_hits;
  map_hits.clear();

  unsigned int nused(0);
  for (auto const& hit : hits) {
    double energy      = hit.energy();
    double time        = hit.time();
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
      if (ifNose_) {
	HFNoseDetId detId = HFNoseDetId(id);
	subdet   = detId.subdetId();
	cell     = detId.cellU();
	cell2    = detId.cellV();
	sector   = detId.waferU();
	sector2  = detId.waferV();
	type     = detId.type();
	layer    = detId.layer();
	zside    = detId.zside();
	xy       = hgcons_[ih]->locateCell(layer,sector,sector2,cell,cell2,
					   false,true);
      } else if ((hgcons_[ih]->geomMode() == HGCalGeometryMode::Hexagon8) ||
		 (hgcons_[ih]->geomMode() == HGCalGeometryMode::Hexagon8Full)){
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
    double tof = (gcoord.mag()*CLHEP::mm)/CLHEP::c_light; 
    if (verbosity_>1) 
      edm::LogVerbatim("HGCalValidation") << "Detector " << name
					  << " zside = " << zside
					  << " layer = " << layer
					  << " type = "  << type
					  << " wafer = " << sector << ":" << sector2
					  << " cell = "  << cell << ":" << cell2
					  << " positon = " << gcoord
					  << " energy = "  << energy
					  << " time = " << time << ":" << tof;
    time -= tof;
    if (time < 0) time = 0;
    hitsinfo   hinfo;
    if (map_hits.count(id) != 0) {
      hinfo = map_hits[id];
    } else {
      hinfo.layer  = layer+layerFront_[ih];
      hinfo.phi    = gcoord.getPhi();
      hinfo.eta    = gcoord.getEta();
      hinfo.time   = time;
    }
    hinfo.energy  += energy;
    map_hits[id]   = hinfo; 

    //Fill in histograms
    h_RZ_[0]   ->Fill(std::abs(gcoord.z()),gcoord.rho());
    h_RZ_[ih+1]->Fill(std::abs(gcoord.z()),gcoord.rho());
    h_EtaPhi_[0]   ->Fill(std::abs(hinfo.eta),hinfo.phi);
    h_EtaPhi_[ih+1]->Fill(std::abs(hinfo.eta),hinfo.phi);
  }
  if (verbosity_>0) 
    edm::LogVerbatim("HGCalValidation") << name << " with " << map_hits.size()
					<< ":" << nused << " detector elements"
					<< " being hit";

  for (auto const& hit : map_hits) {
    hitsinfo hinfo = hit.second;
    if (verbosity_>1) 
      edm::LogVerbatim("HGCalValidation") << " ----------------------   eta = "
					  << hinfo.eta << " phi = "
					  << hinfo.phi << " layer = "
					  << hinfo.layer << " E = "
					  << hinfo.energy << " T = "
					  << hinfo.time;
    h_E_[0]   ->Fill(hinfo.energy);
    h_E_[ih+1]->Fill(hinfo.energy);
    h_T_[0]    ->Fill(hinfo.time);
    h_T_[ih+1] ->Fill(hinfo.time);
    if (hinfo.eta > 0) {
      h_EtFiZp_[0]    ->Fill(std::abs(hinfo.eta),hinfo.phi,hinfo.energy);
      h_EtFiZp_[ih+1] ->Fill(std::abs(hinfo.eta),hinfo.phi,hinfo.energy);
      h_LayerZp_[0]   ->Fill(hinfo.layer,hinfo.energy);
      h_LayerZp_[ih+1]->Fill(hinfo.layer,hinfo.energy);
    } else {
      h_EtFiZm_[0]    ->Fill(std::abs(hinfo.eta),hinfo.phi,hinfo.energy);
      h_EtFiZm_[ih+1] ->Fill(std::abs(hinfo.eta),hinfo.phi,hinfo.energy);
      h_LayerZm_[0]   ->Fill(hinfo.layer,hinfo.energy);
      h_LayerZm_[ih+1]->Fill(hinfo.layer,hinfo.energy);
    }
  }
}

// ------------ method called when starting to processes a run  ------------
void HGCalSimHitStudy::beginRun(const edm::Run&, 
				const edm::EventSetup& iSetup) {
  for (unsigned int k=0; k<nameDetectors_.size(); ++k) {
    if (heRebuild_[k]) {
      edm::ESHandle<HcalDDDRecConstants> pHRNDC;
      iSetup.get<HcalRecNumberingRecord>().get( pHRNDC );
      hcons_  = &(*pHRNDC);
      layers_.emplace_back(hcons_->getMaxDepth(1));
      hgcons_.emplace_back(nullptr);
      layerFront_.emplace_back(40);
    } else {
      edm::ESHandle<HGCalDDDConstants>  pHGDC;
      iSetup.get<IdealGeometryRecord>().get(nameDetectors_[k], pHGDC);
      hgcons_.emplace_back(&(*pHGDC));
      layers_.emplace_back(hgcons_.back()->layers(false));
      if (k == 0) layerFront_.emplace_back(0);
      else        layerFront_.emplace_back(28);
    }
    if (verbosity_>0) 
      edm::LogVerbatim("HGCalValidation") << nameDetectors_[k] 
					  << " defined with "
					  << layers_.back() << " Layers with "
					  << layerFront_.back() << " in front";
  }
}

void HGCalSimHitStudy::beginJob() {

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
				      nbinZ_,zmin_,zmax_,nbinR_,rmin_,rmax_));
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
    name.str(""); title.str("");
    if (ih == 0) {
      name << "EtFiZp_AllDetectors";
      title << "#phi vs #eta (+z) for All Detectors";
    } else {
      name  << "EtFiZp_" << nameDetectors_[ih-1];
      title << "#phi vs #eta (+z) for " << nameDetectors_[ih-1];
    }
    h_EtFiZp_.emplace_back(fs->make<TH2D>(name.str().c_str(), 
					  title.str().c_str(),
					  200,1.0,3.0,200,-M_PI,M_PI));
    name.str(""); title.str("");
    if (ih == 0) {
      name << "EtFiZm_AllDetectors";
      title << "#phi vs #eta (-z) for All Detectors";
    } else {
      name  << "EtFiZm_" << nameDetectors_[ih-1];
      title << "#phi vs #eta (-z) for " << nameDetectors_[ih-1];
    }
    h_EtFiZm_.emplace_back(fs->make<TH2D>(name.str().c_str(), 
					  title.str().c_str(),
					  200,1.0,3.0,200,-M_PI,M_PI));
    name.str(""); title.str("");
    if (ih == 0) {
      name << "LayerZp_AllDetectors";
      title << "Energy vs Layer (+z) for All Detectors";
    } else {
      name  << "LayerZp_" << nameDetectors_[ih-1];
      title << "Energy vs Layer (+z) for " << nameDetectors_[ih-1];
    }
    h_LayerZp_.emplace_back(fs->make<TH1D>(name.str().c_str(), 
					   title.str().c_str(),
					   60,0.0,60.0));
    name.str(""); title.str("");
    if (ih == 0) {
      name << "LayerZm_AllDetectors";
      title << "Energy vs Layer (-z) for All Detectors";
    } else {
      name  << "LayerZm_" << nameDetectors_[ih-1];
      title << "Energy vs Layer (-z) for " << nameDetectors_[ih-1];
    }
    h_LayerZm_.emplace_back(fs->make<TH1D>(name.str().c_str(), 
					   title.str().c_str(),
					   60,0.0,60.0));

    name.str(""); title.str("");
    if (ih == 0) {
      name << "E_AllDetectors";
      title << "Energy Deposit for All Detectors";
    } else {
      name  << "E_" << nameDetectors_[ih-1];
      title << "Energy Deposit for " << nameDetectors_[ih-1];
    }
    h_E_.emplace_back(fs->make<TH1D>(name.str().c_str(), title.str().c_str(),
				     1000,0.0,1.0));

    name.str(""); title.str("");
    if (ih == 0) {
      name << "T_AllDetectors";
      title << "Time for All Detectors";
    } else {
      name  << "T_" << nameDetectors_[ih-1];
      title << "Time for " << nameDetectors_[ih-1];
    }
    h_T_.emplace_back(fs->make<TH1D>(name.str().c_str(), title.str().c_str(),
				     1000,0.0,200.0));
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HGCalSimHitStudy);
