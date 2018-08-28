// system include files
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
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
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Transform3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

class HGCalSimHitValidation : public DQMEDAnalyzer {
  
public:
  
  struct energysum{
    energysum() {etotal=0; for (int i=0; i<6; ++i) eTime[i] = 0.;}
    double eTime[6], etotal;
  };
  
  struct hitsinfo{
    hitsinfo() {
      x=y=z=phi=eta=0.0;
      cell=cell2=sector=sector2=type=layer=0;
    }
    double x, y, z, phi, eta;
    int    cell, cell2, sector, sector2, type, layer;
  };
  
  
  explicit HGCalSimHitValidation(const edm::ParameterSet&);
  ~HGCalSimHitValidation() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  
private:

  void analyzeHits (std::vector<PCaloHit>& hits);
  void fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer);
  void fillHitsInfo(std::pair<hitsinfo,energysum> hit_, unsigned int itimeslice, double esum); 
  bool defineGeometry(edm::ESTransientHandle<DDCompactView> &ddViewH);
  
  // ----------member data ---------------------------
  std::string                nameDetector_, caloHitSource_;
  const HGCalDDDConstants   *hgcons_;
  const HcalDDDRecConstants *hcons_;
  std::vector<double>        times_;
  int                        verbosity_;
  bool                       heRebuild_, testNumber_, symmDet_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hits_;
  edm::EDGetTokenT<edm::HepMCProduct>      tok_hepMC_;
  unsigned int              layers_;
  int                       firstLayer_;
  std::map<uint32_t, HepGeom::Transform3D> transMap_;
  
  std::vector<MonitorElement*> HitOccupancy_Plus_, HitOccupancy_Minus_;
  std::vector<MonitorElement*> EtaPhi_Plus_,  EtaPhi_Minus_;
  MonitorElement              *MeanHitOccupancy_Plus_, *MeanHitOccupancy_Minus_;
  static const unsigned int    maxTime_=6;
  std::vector<MonitorElement*> energy_[maxTime_];
  unsigned int                 nTimes_;
};

HGCalSimHitValidation::HGCalSimHitValidation(const edm::ParameterSet& iConfig) :
  nameDetector_(iConfig.getParameter<std::string>("DetectorName")),
  caloHitSource_(iConfig.getParameter<std::string>("CaloHitSource")),
  times_(iConfig.getParameter<std::vector<double> >("TimeSlices")),
  verbosity_(iConfig.getUntrackedParameter<int>("Verbosity",0)),
  testNumber_(iConfig.getUntrackedParameter<bool>("TestNumber",true)),
  symmDet_(true), firstLayer_(1) {

  heRebuild_     = (nameDetector_ == "HCal") ? true : false;
  tok_hepMC_     = consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"));
  tok_hits_      = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits",caloHitSource_));
  nTimes_        = (times_.size() > maxTime_) ? maxTime_ : times_.size();
}

void HGCalSimHitValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<double> times = {25.0,1000.0};
  desc.add<std::string>("DetectorName","HGCalEESensitive");
  desc.add<std::string>("CaloHitSource","HGCHitsEE");
  desc.add<std::vector<double> >("TimeSlices",times);
  desc.addUntracked<int>("Verbosity",0);
  desc.addUntracked<bool>("TestNumber",true);
  descriptions.add("hgcalSimHitValidationEE",desc);
}

void HGCalSimHitValidation::analyze(const edm::Event& iEvent, 
				    const edm::EventSetup& iSetup) {

  //Generator input
  edm::Handle<edm::HepMCProduct> evtMC;
  iEvent.getByToken(tok_hepMC_,evtMC); 
  if (!evtMC.isValid()) {
    edm::LogVerbatim("HGCalValidation") << "no HepMCProduct found";
  } else { 
    const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
    unsigned int k(0);
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	 p != myGenEvent->particles_end(); ++p, ++k) {
      edm::LogVerbatim("HGCalValidation") << "Particle[" << k << "] with pt "
					  << (*p)->momentum().perp() << " eta "
					  << (*p)->momentum().eta() << " phi "
					  << (*p)->momentum().phi();
    }
  }

  //Now the hits
  edm::Handle<edm::PCaloHitContainer> theCaloHitContainers;
  iEvent.getByToken(tok_hits_, theCaloHitContainers);
  if (theCaloHitContainers.isValid()) {
    if (verbosity_>0) 
      edm::LogVerbatim("HGCalValidation") << " PcalohitItr = " 
					  << theCaloHitContainers->size();
    std::vector<PCaloHit>               caloHits;
    caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), 
                         	    theCaloHitContainers->end());
    if (heRebuild_ && testNumber_) {
      for (unsigned int i=0; i<caloHits.size(); ++i) {
	unsigned int id_ = caloHits[i].id();
	HcalDetId hid = HcalHitRelabeller::relabel(id_,hcons_);
	if (hid.subdet()!=int(HcalEndcap)) 
	  hid = HcalDetId(HcalEmpty,hid.ieta(),hid.iphi(),hid.depth());
	caloHits[i].setID(hid.rawId());
	if (verbosity_>0)
	  edm::LogVerbatim("HGCalValidation") << "Hit[" << i << "] " << hid;
      }
    }
    analyzeHits(caloHits);
  } else if (verbosity_>0) {
    edm::LogVerbatim("HGCalValidation") << "PCaloHitContainer does not exist!";
  }
}

void HGCalSimHitValidation::analyzeHits (std::vector<PCaloHit>& hits) {

  std::map<int, int> OccupancyMap_plus, OccupancyMap_minus;
  OccupancyMap_plus.clear();   OccupancyMap_minus.clear();
  
  std::map<uint32_t,std::pair<hitsinfo,energysum> > map_hits;
  map_hits.clear();
  
  if (verbosity_ > 0) 
    edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " 
					<< hits.size() << " PcaloHit elements";
  unsigned int nused(0);
  for (unsigned int i=0; i<hits.size(); i++) {
    double energy      = hits[i].energy();
    double time        = hits[i].time();
    uint32_t id_       = hits[i].id();
    int    cell, sector, subsector(0), layer, zside;
    int    subdet(0), cell2(0), type(0);
    if (heRebuild_) {
      HcalDetId detId  = HcalDetId(id_);
      subdet           = detId.subdet();
      if (subdet != static_cast<int>(HcalEndcap)) continue;
      cell             = detId.ietaAbs();
      sector           = detId.iphi();
      subsector        = 1;
      layer            = detId.depth();
      zside            = detId.zside();
    } else if ((hgcons_->geomMode() == HGCalGeometryMode::Hexagon8) ||
	       (hgcons_->geomMode() == HGCalGeometryMode::Hexagon8Full)) {
      HGCSiliconDetId detId = HGCSiliconDetId(id_);
      subdet           = ForwardEmpty;
      cell             = detId.cellU();
      cell2            = detId.cellV();
      sector           = detId.waferU();
      subsector        = detId.waferV();
      type             = detId.type();
      layer            = detId.layer();
      zside            = detId.zside();
    } else if (hgcons_->geomMode() == HGCalGeometryMode::Square) {
      HGCalTestNumbering::unpackSquareIndex(id_, zside, layer, sector, subsector, cell);
    } else if (hgcons_->geomMode() == HGCalGeometryMode::Trapezoid) {
      HGCScintillatorDetId detId = HGCScintillatorDetId(id_);
      subdet           = ForwardEmpty;
      cell             = detId.ietaAbs();
      sector           = detId.iphi();
      subsector        = 1;
      type             = detId.type();
      layer            = detId.layer();
      zside            = detId.zside();
    } else {
      HGCalTestNumbering::unpackHexagonIndex(id_, subdet, zside, layer, sector, type, cell);
    }
    nused++;
    if (verbosity_>1) 
      edm::LogVerbatim("HGCalValidation") << "Detector "     << nameDetector_
					  << " zside = "     << zside
					  << " sector|wafer = "   << sector
					  << ":" << subsector
					  << " type = "      << type
					  << " layer = "     << layer
					  << " cell = "      << cell 
					  << ":" << cell2
					  << " energy = "    << energy
					  << " energyem = "  << hits[i].energyEM()
					  << " energyhad = " << hits[i].energyHad()
					  << " time = "      << time;

    HepGeom::Point3D<float> gcoord;
    if (heRebuild_) {
      std::pair<double,double> etaphi = hcons_->getEtaPhi(subdet,zside*cell,
							  sector);
      double rz = hcons_->getRZ(subdet,zside*cell,layer);
      if (verbosity_>2) 
	edm::LogVerbatim("HGCalValidation") << "i/p " << subdet << ":" 
					    << zside << ":" << cell << ":" 
					    << sector << ":" << layer <<" o/p "
					    << etaphi.first << ":" 
					    << etaphi.second << ":" << rz;
      gcoord = HepGeom::Point3D<float>(rz*cos(etaphi.second)/cosh(etaphi.first),
				       rz*sin(etaphi.second)/cosh(etaphi.first),
				       rz*tanh(etaphi.first));
    } else if (hgcons_->geomMode() == HGCalGeometryMode::Square) {
      std::pair<float,float> xy = hgcons_->locateCell(cell,layer,subsector,false);
      const HepGeom::Point3D<float> lcoord(xy.first,xy.second,0);
      int subs = (symmDet_ ? 0 : subsector);
      id_      = HGCalTestNumbering::packSquareIndex(zside,layer,sector,subs,0);
      gcoord   = (transMap_[id_]*lcoord);
    } else {
      std::pair<float,float> xy;
      if ((hgcons_->geomMode() == HGCalGeometryMode::Hexagon8) ||
	  (hgcons_->geomMode() == HGCalGeometryMode::Hexagon8Full)) {
	xy = hgcons_->locateCell(layer,sector,subsector,cell,cell2,false,true);
      } else if (hgcons_->geomMode() == HGCalGeometryMode::Trapezoid) {
	xy = hgcons_->locateCellTrap(layer,sector,cell,false);
      } else {
	xy = hgcons_->locateCell(cell,layer,sector,false);
      }
      double zp = hgcons_->waferZ(layer,false);
      if (zside < 0) zp = -zp;
      float  xp = (zp < 0) ? -xy.first : xy.first;
      gcoord = HepGeom::Point3D<float>(xp,xy.second,zp);
    }
    double tof = (gcoord.mag()*CLHEP::mm)/CLHEP::c_light; 
    if (verbosity_>1) 
      edm::LogVerbatim("HGCalValidation") << std::hex << id_ << std::dec
					  << " global coordinate " << gcoord
					  << " time " << time << ":" << tof;
    time -= tof;
    
    energysum  esum;
    hitsinfo   hinfo;
    if (map_hits.count(id_) != 0) {
      hinfo = map_hits[id_].first;
      esum  = map_hits[id_].second;
    } else {
      hinfo.x      = gcoord.x();
      hinfo.y      = gcoord.y();
      hinfo.z      = gcoord.z();
      hinfo.sector = sector;
      hinfo.sector2= subsector;
      hinfo.cell   = cell;
      hinfo.cell2  = cell;
      hinfo.type   = type;
      hinfo.layer  = layer-firstLayer_;
      hinfo.phi    = gcoord.getPhi();
      hinfo.eta    = gcoord.getEta();
    }
    esum.etotal += energy;
    for (unsigned int k=0; k<nTimes_; ++k) {
      if (time > 0 && time < times_[k]) esum.eTime[k] += energy;
    }

    if (verbosity_>1) 
      edm::LogVerbatim("HGCalValidation") << " -----------------------   gx = "
					  << hinfo.x << " gy = "  << hinfo.y 
					  << " gz = " << hinfo.z << " phi = " 
					  << hinfo.phi << " eta = " 
					  << hinfo.eta;
    map_hits[id_] = std::pair<hitsinfo,energysum>(hinfo,esum);
  }
  if (verbosity_>0) 
    edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " 
					<< map_hits.size()
					<< " detector elements being hit";
  
  std::map<uint32_t,std::pair<hitsinfo,energysum> >::iterator itr;
  for (itr = map_hits.begin() ; itr != map_hits.end(); ++itr)   {
    hitsinfo   hinfo = (*itr).second.first;
    energysum  esum  = (*itr).second.second;
    int        layer = hinfo.layer;
    double     eta   = hinfo.eta;
    
    for (unsigned int itimeslice = 0; itimeslice < nTimes_; itimeslice++ ) {
      fillHitsInfo((*itr).second, itimeslice, esum.eTime[itimeslice]);
    } 
    
    if (eta > 0.0) fillOccupancyMap(OccupancyMap_plus, layer);
    else           fillOccupancyMap(OccupancyMap_minus,layer);
  }
  edm::LogVerbatim("HGCalValidation") << "With map:used:total " << hits.size()
				      << "|" << nused << "|" << map_hits.size()
				      << " hits";

  for (auto const & itr : OccupancyMap_plus) {
    int layer     = itr.first;
    int occupancy = itr.second;
    HitOccupancy_Plus_.at(layer)->Fill(occupancy);
  }
  for (auto const & itr : OccupancyMap_minus) {
    int layer     = itr.first;
    int occupancy = itr.second;
    HitOccupancy_Minus_.at(layer)->Fill(occupancy);
  }
}

void HGCalSimHitValidation::fillOccupancyMap(std::map<int, int>& OccupancyMap,
					     int layer) {
  if (OccupancyMap.find(layer) != OccupancyMap.end()) {
    ++OccupancyMap[layer];
  } else {
    OccupancyMap[layer] = 1;
  }
}

void HGCalSimHitValidation::fillHitsInfo(std::pair<hitsinfo,energysum> hits, 
					 unsigned int itimeslice, double esum){

  unsigned int ilayer = hits.first.layer;
  if (ilayer < layers_) {
    energy_[itimeslice].at(ilayer)->Fill(esum);
    if (itimeslice==0) {
      EtaPhi_Plus_.at(ilayer) ->Fill(hits.first.eta , hits.first.phi);
      EtaPhi_Minus_.at(ilayer)->Fill(hits.first.eta , hits.first.phi);
    }
  } else {
    if (verbosity_>0) 
      edm::LogVerbatim("HGCalValidation") << "Problematic Hit for " 
					  << nameDetector_ << " at sector " 
					  << hits.first.sector << ":"
					  << hits.first.sector2 << " layer " 
					  << hits.first.layer << " cell " 
					  << hits.first.cell << ":"
					  << hits.first.cell2 << " energy "
					  << hits.second.etotal;
  }
}

bool HGCalSimHitValidation::defineGeometry(edm::ESTransientHandle<DDCompactView> &ddViewH) {
  if (verbosity_>0) 
    edm::LogVerbatim("HGCalValidation") << "Initialize HGCalDDDConstants for " 
					<< nameDetector_ << " : " << hgcons_;
  
  if (hgcons_->geomMode() == HGCalGeometryMode::Square) {
    const DDCompactView & cview = *ddViewH;
    std::string attribute = "Volume"; 
    std::string value     = nameDetector_;
  
    DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0)};
    DDFilteredView fv(cview,filter);
    bool dodet = fv.firstChild();
  
    while (dodet) {
      const DDSolid & sol = fv.logicalPart().solid();
      const std::string & name = sol.name().fullname();
      int isd = (name.find(nameDetector_) == std::string::npos) ? -1 : 1;
      if (isd > 0) {
	std::vector<int> copy = fv.copyNumbers();
	int nsiz = (int)(copy.size());
	int lay  = (nsiz > 0) ? copy[nsiz-1] : -1;
	int sec  = (nsiz > 1) ? copy[nsiz-2] : -1;
	int zp   = (nsiz > 3) ? copy[nsiz-4] : -1;
	if (zp !=1 ) zp = -1;
	const DDTrap & trp = static_cast<DDTrap>(sol);
	int subs = (trp.alpha1()>0 ? 1 : 0);
	symmDet_ = (trp.alpha1()==0 ? true : false);
	uint32_t id = HGCalTestNumbering::packSquareIndex(zp,lay,sec,subs,0);
	DD3Vector x, y, z;
	fv.rotation().GetComponents( x, y, z ) ;
	const CLHEP::HepRep3x3 rotation ( x.X(), y.X(), z.X(),
					  x.Y(), y.Y(), z.Y(),
					  x.Z(), y.Z(), z.Z() );
	const CLHEP::HepRotation hr ( rotation );
	const CLHEP::Hep3Vector h3v ( fv.translation().X(),
				      fv.translation().Y(),
				      fv.translation().Z()  ) ;
	const HepGeom::Transform3D ht3d (hr, h3v);
	transMap_.insert(std::make_pair(id,ht3d));
	if (verbosity_>2) 
	  edm::LogVerbatim("HGCalValidation") << HGCalDetId(id) 
					      << " Transform using " << h3v 
					      << " and " << hr;
      }
      dodet = fv.next();
    }
    if (verbosity_>0) 
      edm::LogVerbatim("HGCalValidation") << "Finds " << transMap_.size() 
					  << " elements and SymmDet_ = " 
					  << symmDet_;
  }
  return true;
}

// ------------ method called when starting to processes a run  ------------
void HGCalSimHitValidation::dqmBeginRun(const edm::Run&, 
					const edm::EventSetup& iSetup) {
  if (heRebuild_) {
    edm::ESHandle<HcalDDDRecConstants> pHRNDC;
    iSetup.get<HcalRecNumberingRecord>().get( pHRNDC );
    hcons_  = &(*pHRNDC);
    layers_ = hcons_->getMaxDepth(1);
  } else {
    edm::ESHandle<HGCalDDDConstants>  pHGDC;
    iSetup.get<IdealGeometryRecord>().get(nameDetector_, pHGDC);
    hgcons_ = &(*pHGDC);
    layers_ = hgcons_->layers(false);
    firstLayer_ = hgcons_->firstLayer();
    edm::ESTransientHandle<DDCompactView> pDD;
    iSetup.get<IdealGeometryRecord>().get( pDD );
    defineGeometry(pDD);
  }
  if (verbosity_>0) 
    edm::LogVerbatim("HGCalValidation") << nameDetector_ << " defined with "
					<< layers_ << " Layers with first at "
					<< firstLayer_;
}

void HGCalSimHitValidation::bookHistograms(DQMStore::IBooker& iB, 
					   edm::Run const&, 
					   edm::EventSetup const&) {

  iB.setCurrentFolder("HGCAL/HGCalSimHitsV/"+nameDetector_);
    
  std::ostringstream histoname;
  for (unsigned int il=0; il < layers_; ++il) {
    int ilayer = firstLayer_ + (int)(il);
    histoname.str(""); histoname << "HitOccupancy_Plus_layer_" << ilayer;
    HitOccupancy_Plus_.push_back(iB.book1D(histoname.str().c_str(), "HitOccupancy_Plus", 501, -0.5, 500.5));
    histoname.str(""); histoname << "HitOccupancy_Minus_layer_" << ilayer;
    HitOccupancy_Minus_.push_back(iB.book1D(histoname.str().c_str(), "HitOccupancy_Minus", 501, -0.5, 500.5));
      
    histoname.str(""); histoname << "EtaPhi_Plus_" << "layer_" << ilayer;
    EtaPhi_Plus_.push_back(iB.book2D(histoname.str().c_str(), "Occupancy", 31, 1.45, 3.0, 72, -CLHEP::pi, CLHEP::pi));
    histoname.str(""); histoname << "EtaPhi_Minus_" << "layer_" << ilayer;
    EtaPhi_Minus_.push_back(iB.book2D(histoname.str().c_str(), "Occupancy", 31, -3.0, -1.45, 72, -CLHEP::pi, CLHEP::pi));
      
    for (unsigned int itimeslice = 0; itimeslice < nTimes_ ; itimeslice++ ) {
      histoname.str(""); histoname << "energy_time_"<< itimeslice << "_layer_" << ilayer;
      energy_[itimeslice].push_back(iB.book1D(histoname.str().c_str(),"energy_",100,0,0.1));
    }
  }

  MeanHitOccupancy_Plus_ = iB.book1D("MeanHitOccupancy_Plus", "MeanHitOccupancy_Plus", layers_, 0.5, layers_ + 0.5);
  MeanHitOccupancy_Minus_ = iB.book1D("MeanHitOccupancy_Minus", "MeanHitOccupancy_Minus", layers_, 0.5, layers_ + 0.5);
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HGCalSimHitValidation);
