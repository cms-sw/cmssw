// -*- C++ -*-
//
// Package:    HGCalSimHitValidation
// Class:      HGCalSimHitValidation
// 
/**\class HGCalSimHitValidation HGCalSimHitValidation.cc Validaion/HGCalValidation/plugins/HGCalSimHitValidation.cc
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
#include <cmath>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "Validation/HGCalValidation/plugins/HGCalSimHitValidation.h"

HGCalSimHitValidation::HGCalSimHitValidation(const edm::ParameterSet& iConfig) :
  symmDet_(true) {

  nameDetector_  = iConfig.getParameter<std::string>("DetectorName");
  caloHitSource_ = iConfig.getParameter<std::string>("CaloHitSource");
  times_         = iConfig.getParameter<std::vector<double> >("TimeSlices");
  verbosity_     = iConfig.getUntrackedParameter<int>("Verbosity",0);
  testNumber_    = iConfig.getUntrackedParameter<bool>("TestNumber", true);
  heRebuild_     = (nameDetector_ == "HCal") ? true : false;
  tok_hepMC_     = consumes<edm::HepMCProduct>(edm::InputTag("generator"));
  tok_hits_      = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits",caloHitSource_));
  nTimes_        = (times_.size() > 6) ? 6 : times_.size();
}

HGCalSimHitValidation::~HGCalSimHitValidation() {}

void HGCalSimHitValidation::analyze(const edm::Event& iEvent, 
				    const edm::EventSetup& iSetup) {

  //Generator input
  edm::Handle<edm::HepMCProduct> evtMC;
  iEvent.getByToken(tok_hepMC_,evtMC); 
  if (!evtMC.isValid()) {
    edm::LogWarning("HGCalValidation") << "no HepMCProduct found\n";
  } else { 
    const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
    unsigned int k(0);
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	 p != myGenEvent->particles_end(); ++p, ++k) {
      edm::LogInfo("HGCalValidation") << "Particle[" << k << "] with pt "
				      << (*p)->momentum().perp() << " eta "
				      << (*p)->momentum().eta() << " phi "
				      << (*p)->momentum().phi() << std::endl;
    }
  }

  //Now the hits
  edm::Handle<edm::PCaloHitContainer> theCaloHitContainers;
  iEvent.getByToken(tok_hits_, theCaloHitContainers);
  if (theCaloHitContainers.isValid()) {
    if (verbosity_>0) 
      edm::LogInfo("HGCalValidation") << " PcalohitItr = " 
				      << theCaloHitContainers->size() << "\n";
    std::vector<PCaloHit>               caloHits;
    caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), 
                         	    theCaloHitContainers->end());
    if (heRebuild_ && testNumber_) {
      for (unsigned int i=0; i<caloHits.size(); ++i) {
	unsigned int id_ = caloHits[i].id();
	int subdet, z, depth0, eta0, phi0, lay;
	HcalTestNumbering::unpackHcalIndex(id_, subdet, z, depth0, eta0, phi0, lay);
	int sign = (z==0) ? (-1):(1);
	if (verbosity_>0)
	  edm::LogInfo("HGCalValidation") << "Hit[" << i << "] subdet "
					  << subdet << " z " << z << " depth "
					  << depth0 << " eta " << eta0 
					  << " phi " << phi0 << " lay " << lay
					  << std::endl;
	HcalDDDRecConstants::HcalID id = hcons_->getHCID(subdet, eta0, phi0, lay, depth0);
	HcalDetId hid = ((subdet==int(HcalEndcap)) ? 
			 HcalDetId(HcalEndcap,sign*id.eta,id.phi,id.depth) :
			 HcalDetId(HcalEmpty,sign*id.eta,id.phi,id.depth));
	caloHits[i].setID(hid.rawId());
	if (verbosity_>0)
	  edm::LogInfo("HGCalValidation") << "Hit[" << i << "] " << hid <<"\n";
      }
    }
    analyzeHits(caloHits);
  } else if (verbosity_>0) {
    edm::LogInfo("HGCalValidation") << "PCaloHitContainer does not exist !!!\n";
  }
}

void HGCalSimHitValidation::analyzeHits (std::vector<PCaloHit>& hits) {

  std::map<int, int> OccupancyMap_plus, OccupancyMap_minus;
  OccupancyMap_plus.clear();   OccupancyMap_minus.clear();
  
  std::map<uint32_t,std::pair<hitsinfo,energysum> > map_hits;
  map_hits.clear();
  
  if (verbosity_ > 0) 
    edm::LogInfo("HGCalValidation") << nameDetector_ << " with " << hits.size()
				    << " PcaloHit elements\n";
  unsigned int nused(0);
  for (unsigned int i=0; i<hits.size(); i++) {
    double energy      = hits[i].energy();
    double time        = hits[i].time();
    uint32_t id_       = hits[i].id();
    int    cell, sector, subsector, layer, zside;
    int    subdet(0);
    if (heRebuild_) {
      HcalDetId detId  = HcalDetId(id_);
      subdet           = detId.subdet();
      if (subdet != static_cast<int>(HcalEndcap)) continue;
      cell             = detId.ietaAbs();
      sector           = detId.iphi();
      subsector        = 1;
      layer            = detId.depth();
      zside            = detId.zside();
    } else {
      if (hgcons_->geomMode() == HGCalGeometryMode::Square) {
	HGCalTestNumbering::unpackSquareIndex(id_, zside, layer, sector, subsector, cell);
      } else {
	HGCalTestNumbering::unpackHexagonIndex(id_, subdet, zside, layer, sector, subsector, cell);
      }
    }
    nused++;
    if (verbosity_>1) 
      edm::LogInfo("HGCalValidation") << "Detector "     << nameDetector_
				      << " zside = "     << zside
				      << " sector|wafer = "   << sector
				      << " subsector|type = " << subsector
				      << " layer = "     << layer
				      << " cell = "      << cell
				      << " energy = "    << energy
				      << " energyem = "  << hits[i].energyEM()
				      << " energyhad = " << hits[i].energyHad()
				      << " time = "      << time << "\n";

    HepGeom::Point3D<float> gcoord;
    if (heRebuild_) {
      std::pair<double,double> etaphi = hcons_->getEtaPhi(subdet,zside*cell,sector);
      double rz = hcons_->getRZ(subdet,zside*cell,layer);
//    std::cout << "i/p " << subdet << ":" << zside << ":" << cell << ":" << sector << ":" << layer << " o/p " << etaphi.first << ":" << etaphi.second << ":" << rz << std::endl;
      gcoord = HepGeom::Point3D<float>(rz*cos(etaphi.second)/cosh(etaphi.first),
				       rz*sin(etaphi.second)/cosh(etaphi.first),
				       rz*tanh(etaphi.first));
    } else {
      if (hgcons_->geomMode() == HGCalGeometryMode::Square) {
	std::pair<float,float> xy = hgcons_->locateCell(cell,layer,subsector,false);
	const HepGeom::Point3D<float> lcoord(xy.first,xy.second,0);
	int subs = (symmDet_ ? 0 : subsector);
	id_      = HGCalTestNumbering::packSquareIndex(zside,layer,sector,subs,0);
	gcoord   = (transMap_[id_]*lcoord);
      } else {
	std::pair<float,float> xy = hgcons_->locateCell(cell,layer,sector,false);
	double zp = hgcons_->waferZ(layer,false);
	if (zside < 0) zp = -zp;
	float  xp = (zp < 0) ? -xy.first : xy.first;
	gcoord = HepGeom::Point3D<float>(xp,xy.second,zp);
      }
    }
    double tof = (gcoord.mag()*CLHEP::mm)/CLHEP::c_light; 
    if (verbosity_>1) 
      edm::LogInfo("HGCalValidation") << std::hex << id_ << std::dec
				      << " global coordinate " << gcoord
				      << " time " << time << ":" << tof <<"\n";
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
      hinfo.cell   = cell;
      hinfo.layer  = layer;
      hinfo.phi    = gcoord.getPhi();
      hinfo.eta    = gcoord.getEta();
    }
    esum.etotal += energy;
    for (unsigned int k=0; k<nTimes_; ++k) {
      if (time > 0 && time < times_[k]) esum.eTime[k] += energy;
    }
    if (verbosity_>1) 
      edm::LogInfo("HGCalValidation") << " --------------------------   gx = " 
				      << hinfo.x << " gy = "  << hinfo.y 
				      << " gz = " << hinfo.z << " phi = " 
				      << hinfo.phi << " eta = " << hinfo.eta
				      << std::endl;
    map_hits[id_] = std::pair<hitsinfo,energysum>(hinfo,esum);
  }
  if (verbosity_>0) 
    edm::LogInfo("HGCalValidation") << nameDetector_ << " with " 
				    << map_hits.size()
				    << " detector elements being hit\n";
  
  std::map<uint32_t,std::pair<hitsinfo,energysum> >::iterator itr;
  for (itr = map_hits.begin() ; itr != map_hits.end(); ++itr)   {
    hitsinfo   hinfo = (*itr).second.first;
    energysum  esum  = (*itr).second.second;
    int layer        = hinfo.layer;
    
    for (unsigned int itimeslice = 0; itimeslice < nTimes_; itimeslice++ ) {
      fillHitsInfo((*itr).second, itimeslice, esum.eTime[itimeslice]);
    } 
    
    double eta = hinfo.eta;
    
    if (eta > 0.0)        fillOccupancyMap(OccupancyMap_plus, layer-1);
    else                  fillOccupancyMap(OccupancyMap_minus, layer-1);
  }
  edm::LogInfo("HGCalValidation") << "With map:used:total " << hits.size()
				  << "|" << nused << "|" << map_hits.size()
				  << " hits\n";

  for (auto itr = OccupancyMap_plus.begin() ; itr != OccupancyMap_plus.end(); ++itr) {
    int layer     = (*itr).first;
    int occupancy = (*itr).second;
    HitOccupancy_Plus_.at(layer)->Fill(occupancy);
  }
  for (auto itr = OccupancyMap_minus.begin() ; itr != OccupancyMap_minus.end(); ++itr) {
    int layer     = (*itr).first;
    int occupancy = (*itr).second;
    HitOccupancy_Minus_.at(layer)->Fill(occupancy);
  }
}

void HGCalSimHitValidation::fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer){
  if (OccupancyMap.find(layer) != OccupancyMap.end()) {
    OccupancyMap[layer] ++;
  } else {
    OccupancyMap[layer] = 1;
  }
}

void HGCalSimHitValidation::fillHitsInfo(std::pair<hitsinfo,energysum> hits, 
					 unsigned int itimeslice, double esum){

  unsigned int ilayer = hits.first.layer - 1;
  if (ilayer < layers_) {
    energy_[itimeslice].at(ilayer)->Fill(esum);
    if (itimeslice==0) {
      EtaPhi_Plus_.at(ilayer)->Fill(hits.first.eta , hits.first.phi);
      EtaPhi_Minus_.at(ilayer)->Fill(hits.first.eta , hits.first.phi);
    }
  } else {
    if (verbosity_>0) 
      edm::LogInfo("HGCalValidation") << "Problematic Hit for " 
				      << nameDetector_ << " at sector " 
				      << hits.first.sector << " layer " 
				      << hits.first.layer << " cell " 
				      << hits.first.cell << " energy "
				      << hits.second.etotal << std::endl;
  }
}

bool HGCalSimHitValidation::defineGeometry(edm::ESTransientHandle<DDCompactView> &ddViewH){
  if (verbosity_>0) 
    edm::LogInfo("HGCalValidation") << "Initialize HGCalDDDConstants for " 
				    << nameDetector_ << " : " << hgcons_ <<"\n";
  
  if (hgcons_->geomMode() == HGCalGeometryMode::Square) {
    const DDCompactView & cview = *ddViewH;
    std::string attribute = "Volume"; 
    std::string value     = nameDetector_;
    DDValue val(attribute, value, 0);
  
    DDSpecificsFilter filter;
    filter.setCriteria(val, DDCompOp::equals);
    DDFilteredView fv(cview);
    fv.addFilter(filter);
    bool dodet = fv.firstChild();
  
    while (dodet) {
      const DDSolid & sol = fv.logicalPart().solid();
      std::string name = sol.name();
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
	  edm::LogInfo("HGCalValidation") << HGCalDetId(id) 
					  << " Transform using " << h3v 
					  << " and " << hr << std::endl;
      }
      dodet = fv.next();
    }
    if (verbosity_>0) 
      edm::LogInfo("HGCalValidation") << "Finds " << transMap_.size() 
				      << " elements and SymmDet_ = " 
				      << symmDet_ << std::endl;
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
    edm::ESTransientHandle<DDCompactView> pDD;
    iSetup.get<IdealGeometryRecord>().get( pDD );
    defineGeometry(pDD);
  }
  if (verbosity_>0) 
    edm::LogInfo("HGCalValidation") << nameDetector_ << " defined with "
				    << layers_ << " Layers\n";
}

void HGCalSimHitValidation::bookHistograms(DQMStore::IBooker& iB, 
					   edm::Run const&, 
					   edm::EventSetup const&) {

  iB.setCurrentFolder("HGCalSimHitsV/"+nameDetector_);
    
  std::ostringstream histoname;
  for (unsigned int ilayer = 0; ilayer < layers_; ilayer++ ) {
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

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalSimHitValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HGCalSimHitValidation);
