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
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "Validation/HGCalValidation/plugins/HGCalSimHitValidation.h"

HGCalSimHitValidation::HGCalSimHitValidation(const edm::ParameterSet& iConfig){
  dbe_           = edm::Service<DQMStore>().operator->();
  nameDetector_  = iConfig.getParameter<std::string>("DetectorName");
  caloHitSource_ = iConfig.getParameter<std::string>("CaloHitSource");
  verbosity_     = iConfig.getUntrackedParameter<int>("Verbosity",0);
  testNumber_    = iConfig.getUntrackedParameter<bool>("TestNumber", false);
  heRebuild_     = (nameDetector_ == "HCal") ? true : false;
}


HGCalSimHitValidation::~HGCalSimHitValidation() {}

void HGCalSimHitValidation::analyze(const edm::Event& iEvent, 
				    const edm::EventSetup& iSetup) {

  //Generator input
  edm::Handle<edm::HepMCProduct> evtMC;
  iEvent.getByLabel("generator",evtMC); 
  if (!evtMC.isValid()) {
    edm::LogWarning("HGCalValidation") << "no HepMCProduct found";
  } else { 
    const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
    unsigned int k(0);
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	 p != myGenEvent->particles_end(); ++p, ++k) {
      edm::LogInfo("HGCalValidation") << "Particle[" << k << "] with pt "
				      << (*p)->momentum().perp() << " eta "
				      << (*p)->momentum().eta() << " phi "
				      << (*p)->momentum().phi();
//    std::cout << "Particle[" << k << "] with pt " << (*p)->momentum().perp() << " eta " << (*p)->momentum().eta() << " phi " << (*p)->momentum().phi() << std::endl;
    }
  }

  //Now the hits
  edm::Handle<edm::PCaloHitContainer> theCaloHitContainers;
  iEvent.getByLabel("g4SimHits", caloHitSource_, theCaloHitContainers);
  if (theCaloHitContainers.isValid()) {
    if (verbosity_>0) 
      edm::LogInfo("HGCalValidation") << " PcalohitItr = " 
				      << theCaloHitContainers->size();
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
					  << " phi " << phi0 << " lay " << lay;
	HcalDDDRecConstants::HcalID id = hcons->getHCID(subdet, eta0, phi0, lay, depth0);
	HcalDetId hid = ((subdet==int(HcalEndcap)) ? 
			 HcalDetId(HcalEndcap,sign*id.eta,id.phi,id.depth) :
			 HcalDetId(HcalEmpty,sign*id.eta,id.phi,id.depth));
	caloHits[i].setID(hid.rawId());
	if (verbosity_>0)
	  edm::LogInfo("HGCalValidation") << "Hit[" << i << "] " << hid;
      }
    }
    analyzeHits(caloHits);
  } else if (verbosity_>0) {
    edm::LogInfo("HGCalValidation") << "PCaloHitContainer does not exist !!!";
  }
}

void HGCalSimHitValidation::analyzeHits (std::vector<PCaloHit>& hits) {

  for(int i=0; i<netaBins; ++i) {
    OccupancyMap_plus[i].clear();
    OccupancyMap_minus[i].clear();
  }  
  
  std::map<uint32_t,std::pair<hitsinfo,energysum> > map_hits;
  map_hits.clear();
  
  if (verbosity_ > 0) 
    edm::LogInfo("HGCalValidation") << nameDetector_ << " with " << hits.size()
				    << " PcaloHit elements";
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
      HGCalDetId detId = HGCalDetId(id_);
      cell             = detId.cell();
      subsector        = detId.subsector();
      sector           = detId.sector();
      layer            = detId.layer();
      zside            = detId.zside();
    }
    nused++;
    if (verbosity_>1) 
      edm::LogInfo("HGCalValidation") << "Detector "     << nameDetector_
				      << " zside = "     << zside
				      << " sector = "    << sector
				      << " subsector = " << subsector
				      << " layer = "     << layer
				      << " cell = "      << cell
				      << " energy = "    << energy
				      << " energyem = "  << hits[i].energyEM()
				      << " energyhad = " << hits[i].energyHad()
				      << " time = "      << time;

    HepGeom::Point3D<float> gcoord;
    if (heRebuild_) {
      std::pair<double,double> etaphi = hcons->getEtaPhi(subdet,zside*cell,sector);
      double rz = hcons->getRZ(subdet,zside*cell,layer);
//    std::cout << "i/p " << subdet << ":" << zside << ":" << cell << ":" << sector << ":" << layer << " o/p " << etaphi.first << ":" << etaphi.second << ":" << rz << std::endl;
      gcoord = HepGeom::Point3D<float>(rz*cos(etaphi.second)/cosh(etaphi.first),
				       rz*sin(etaphi.second)/cosh(etaphi.first),
				       rz*tanh(etaphi.first));
    } else {
      std::pair<float,float> xy = hgcons_->locateCell(cell,layer,subsector,false);
      const HepGeom::Point3D<float> lcoord(xy.first,xy.second,0);
      int subs = (symmDet_ ? 0 : subsector);
      id_      = HGCalDetId(ForwardEmpty,zside,layer,sector,subs,0).rawId();
      gcoord   = (transMap_[id_]*lcoord);
    }
    double tof = (gcoord.mag()*CLHEP::mm)/CLHEP::c_light; 
    if (verbosity_>1) 
      edm::LogInfo("HGCalValidation") << std::hex << id_ << std::dec
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
      hinfo.cell   = cell;
      hinfo.layer  = layer;
      hinfo.phi    = gcoord.getPhi();
      hinfo.eta    = gcoord.getEta();
      esum.etotal += energy;
      if (time > 0 && time < 1000) {
	esum.e1000 += energy;
	if (time < 250) {
	  esum.e250 += energy;
	  if (time < 100) {
	    esum.e100 += energy;
	    if (time < 50) {
	      esum.e50 += energy;
	      if (time < 25) {
		esum.e25 += energy;
		if (time < 15) {
		  esum.e15 += energy;
		}
	      }
	    }
	  }
	}
      }
    }
    if (verbosity_>1) 
      edm::LogInfo("HGCalValidation") << " --------------------------   gx = " 
				      << hinfo.x << " gy = "  << hinfo.y 
				      << " gz = " << hinfo.z << " phi = " 
				      << hinfo.phi << " eta = " << hinfo.eta;
    std::pair<hitsinfo,energysum> pair_tmp(hinfo,esum);
    map_hits[id_] = pair_tmp;
  }
  if (verbosity_>0) 
    edm::LogInfo("HGCalValidation") << nameDetector_ << " with " 
				    << map_hits.size()
				    << " detector elements being hit";
  
  std::map<uint32_t,std::pair<hitsinfo,energysum> >::iterator itr;
  for (itr = map_hits.begin() ; itr != map_hits.end(); ++itr)   {
    hitsinfo   hinfo = (*itr).second.first;
    energysum  esum  = (*itr).second.second;
    int layer        = hinfo.layer;
    
    
    std::vector<double> esumVector;
    esumVector.push_back(esum.e15);
    esumVector.push_back(esum.e25);
    esumVector.push_back(esum.e50);
    esumVector.push_back(esum.e100);
    esumVector.push_back(esum.e250);
    esumVector.push_back(esum.e1000);
    
    for(unsigned int itimeslice = 0; itimeslice < esumVector.size(); 
	itimeslice++ ) {
      FillHitsInfo((*itr).second, itimeslice, esumVector.at(itimeslice));
    } 
    
    double eta = hinfo.eta;
    
    if (eta >= 1.75 && eta <= 2.5)        fillOccupancyMap(OccupancyMap_plus[0], layer-1);
    if (eta >= 1.75 && eta <= 2.0)        fillOccupancyMap(OccupancyMap_plus[1], layer-1);
    else if (eta >= 2.0 && eta <= 2.25)   fillOccupancyMap(OccupancyMap_plus[2], layer-1);
    else if(eta >= 2.25 && eta <= 2.5)    fillOccupancyMap(OccupancyMap_plus[3], layer-1);
    
    if (eta >= -2.5 && eta <= -1.75)      fillOccupancyMap(OccupancyMap_minus[0], layer-1);
    if (eta >= -2.0 && eta <= -1.75)      fillOccupancyMap(OccupancyMap_minus[1], layer-1);
    else if (eta >= -2.25 && eta <= -2.0) fillOccupancyMap(OccupancyMap_minus[2], layer-1);
    else if (eta >= -2.5 && eta <= -2.25) fillOccupancyMap(OccupancyMap_minus[3], layer-1);
  }
  edm::LogInfo("HGCalValidation") << "With map:used:total " << hits.size()
				  << "|" << nused << "|" << map_hits.size()
				  << " hits";
  FillHitsInfo();
}

void HGCalSimHitValidation::fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer){
  if (OccupancyMap.find(layer) != OccupancyMap.end()) {
    OccupancyMap[layer] ++;
  } else {
    OccupancyMap[layer] = 1;
  }
}

void HGCalSimHitValidation::FillHitsInfo() { 
  if (geometrydefined_ && dbe_) {
    for(int indx=0; indx<netaBins;++indx) {
      for (auto itr = OccupancyMap_plus[indx].begin() ; itr != OccupancyMap_plus[indx].end(); ++itr) {
	int layer     = (*itr).first;
	int occupancy = (*itr).second;
	HitOccupancy_Plus_[indx].at(layer)->Fill(occupancy);
      }
      for (auto itr = OccupancyMap_minus[indx].begin() ; itr != OccupancyMap_minus[indx].end(); ++itr) {
	int layer     = (*itr).first;
	int occupancy = (*itr).second;
	HitOccupancy_Minus_[indx].at(layer)->Fill(occupancy);
      }
    }
  }  
}

void HGCalSimHitValidation::FillHitsInfo(std::pair<hitsinfo,energysum> hits, 
					 unsigned int itimeslice, double esum) {
  if (geometrydefined_ && dbe_) {
    unsigned int ilayer = hits.first.layer - 1;
    if (ilayer < layers_) {
      energy_[itimeslice].at(ilayer)->Fill(esum);
      if (itimeslice==1) {
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
					<< hits.second.etotal;
    }
  }
}

bool HGCalSimHitValidation::defineGeometry(edm::ESTransientHandle<DDCompactView> &ddViewH){
  const DDCompactView & cview = *ddViewH;
  hgcons_ = new HGCalDDDConstants(cview, nameDetector_);
  if (verbosity_>0) 
    edm::LogInfo("HGCalValidation") << "Initialize HGCalDDDConstants for " 
				    << nameDetector_ << " : " << hgcons_;
  
  std::string attribute = "Volume"; 
  std::string value     = nameDetector_;
  DDValue val(attribute, value, 0);
  
  DDSpecificsFilter filter;
  filter.setCriteria(val, DDSpecificsFilter::equals);
  DDFilteredView fv(cview);
  fv.addFilter(filter);
  bool dodet = fv.firstChild();
  
  while (dodet) {
    const DDSolid & sol  = fv.logicalPart().solid();
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
      uint32_t id = HGCalDetId(ForwardEmpty,zp,lay,sec,subs,0).rawId();
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
	edm::LogInfo("HGCalValidation") << HGCalDetId(id) << " Transform using "
					<< h3v << " and " << hr;
    }
    dodet = fv.next();
  }
  if (verbosity_>0) 
    edm::LogInfo("HGCalValidation") << "Finds " << transMap_.size() 
				    << " elements and SymmDet_ = " << symmDet_;
  return true;
}


// ------------ method called once each job just before starting event loop  ------------
void HGCalSimHitValidation::beginJob() {
  geometrydefined_ = false;
  symmDet_         = true;
}

// ------------ method called once each job just after ending the event loop  ------------
void HGCalSimHitValidation::endJob() { }

// ------------ method called when starting to processes a run  ------------

void HGCalSimHitValidation::beginRun(edm::Run const& iRun, 
				     edm::EventSetup const& iSetup) {
  if (!geometrydefined_ && dbe_) {
    if (heRebuild_) {
      edm::ESHandle<HcalDDDRecConstants> pHRNDC;
      iSetup.get<HcalRecNumberingRecord>().get( pHRNDC );
      hcons   = &(*pHRNDC);
      layers_ = hcons->getMaxDepth(1);
      geometrydefined_ = true;
    } else {
      edm::ESTransientHandle<DDCompactView> pDD;
      iSetup.get<IdealGeometryRecord>().get( pDD );
      geometrydefined_ = defineGeometry(pDD);
      layers_ = hgcons_->layers(false);
    }
    if (verbosity_>0) 
      edm::LogInfo("HGCalValidation") << nameDetector_ << " defined with "
				      << layers_ << " Layers";
    dbe_->setCurrentFolder("HGCalSimHitsV/"+nameDetector_);
    
    std::ostringstream histoname;
    for (unsigned int ilayer = 0; ilayer < layers_; ilayer++ ) {
      for(int indx=0; indx<netaBins; ++indx){
	histoname.str(""); histoname << "HitOccupancy_Plus"<< indx  << "_layer_" << ilayer;
	HitOccupancy_Plus_[indx].push_back(dbe_->book1D( histoname.str().c_str(), "HitOccupancy_Plus", 501, -0.5, 500.5));
	histoname.str(""); histoname << "HitOccupancy_Minus" << indx  << "_layer_" << ilayer;
	HitOccupancy_Minus_[indx].push_back(dbe_->book1D( histoname.str().c_str(), "HitOccupancy_Minus", 501, -0.5, 500.5));
      }
      
      histoname.str(""); histoname << "EtaPhi_Plus_" << "layer_" << ilayer;
      EtaPhi_Plus_.push_back(dbe_->book2D(histoname.str().c_str(), "Occupancy", 155, 1.45, 3.0, 72, -3.14, 3.14));
      histoname.str(""); histoname << "EtaPhi_Minus_" << "layer_" << ilayer;
      EtaPhi_Minus_.push_back(dbe_->book2D(histoname.str().c_str(), "Occupancy", 155, -3.0, -1.45, 72, -3.14, 3.14));
      
      for (int itimeslice = 0; itimeslice < 6 ; itimeslice++ ) {
	histoname.str(""); histoname << "energy_time_"<< itimeslice << "_layer_" << ilayer;
	energy_[itimeslice].push_back(dbe_->book1D(histoname.str().c_str(),"energy_",500,0,0.1));
      }
      
    }
    for(int indx=0; indx<netaBins; ++indx) {
      histoname.str(""); histoname << "MeanHitOccupancy_Plus"<< indx ;
      MeanHitOccupancy_Plus_[indx] = dbe_->book1D( histoname.str().c_str(), "MeanHitOccupancy_Plus", layers_, 0.5, layers_ + 0.5);
      histoname.str(""); histoname << "MeanHitOccupancy_Minus"<< indx ;
      MeanHitOccupancy_Minus_[indx] = dbe_->book1D( histoname.str().c_str(), "MeanHitOccupancy_Minus", layers_, 0.5, layers_ + 0.5);
    }
    
  }
}




// ------------ method called when ending the processing of a run  ------------

void HGCalSimHitValidation::endRun(edm::Run const&, edm::EventSetup const&) {
  if (geometrydefined_ && dbe_) {
    for(int ilayer=0; ilayer < (int)layers_; ++ilayer) {
      for(int indx=0; indx<4; ++indx){
	double meanVal = HitOccupancy_Plus_[indx].at(ilayer)->getMean();
	MeanHitOccupancy_Plus_[indx]->setBinContent(ilayer+1, meanVal);
	meanVal = HitOccupancy_Minus_[indx].at(ilayer)->getMean();
	MeanHitOccupancy_Minus_[indx]->setBinContent(ilayer+1, meanVal);
      }
    }
    
  }
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
