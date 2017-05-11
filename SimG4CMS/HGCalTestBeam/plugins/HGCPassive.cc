///////////////////////////////////////////////////////////////////////////////
// File: HGCPassive.cc
//copied from SimG4HGCalValidation
// Description: Main analysis class for HGCal Validation of G4 Hits
///////////////////////////////////////////////////////////////////////////////

#include "HGCPassive.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"
#include "G4TouchableHistory.hh"
#include "G4TransportationManager.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <utility>  

//#define EDM_ML_DEBUG

HGCPassive::HGCPassive(const edm::ParameterSet &p) : count_(0), init_(false) {

  edm::ParameterSet m_Passive = p.getParameter<edm::ParameterSet>("HGCPassive");
  LVNames_  = m_Passive.getUntrackedParameter<std::vector<std::string> >("LVNames");
  
  for (unsigned int k=0; k<LVNames_.size(); ++k) {
    produces<edm::PassiveHitContainer>(Form("%sPassiveHits",LVNames_[k].c_str()));
#ifdef EDM_ML_DEBUG
    std::cout << "Collection name[" << k << "] " << LVNames_[k] << std::endl;
#endif
  }
} 
   
HGCPassive::~HGCPassive() {
}

void HGCPassive::produce(edm::Event& e, const edm::EventSetup&) {
  
  for (unsigned int k=0; k<LVNames_.size(); ++k) {
    std::unique_ptr<edm::PassiveHitContainer> hgcPH(new edm::PassiveHitContainer);
    endOfEvent(*hgcPH, k);
    e.put(std::move(hgcPH),Form("%sPassiveHits",LVNames_[k].c_str()));
  }
}

void HGCPassive::update(const BeginOfRun * run) {

  topPV_ = getTopPV();
  if (topPV_ == 0) {
    edm::LogWarning("HGCPassive") << "Cannot find top level volume\n";
  } else {
    init_ = true;
    const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
    std::vector<G4LogicalVolume *>::const_iterator lvcite;

    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
      findLV(*lvcite);
    } 

#ifdef EDM_ML_DEBUG
    std::cout << "HGCPassive::Finds " << mapLV_.size() << " logical volumes\n";
    std::map<G4LogicalVolume*,std::pair<unsigned int, std::string> >::iterator itr;
    unsigned int k(0);
    for (itr = mapLV_.begin(); itr != mapLV_.end(); ++itr, ++k)
      std::cout << "Entry[" << k << "] " << itr->first << ": (" 
		<< (itr->second).first << ", " << (itr->second).second << ")\n";
#endif
  }
}

//=================================================================== per EVENT
void HGCPassive::update(const BeginOfEvent * evt) {
 
  int iev = (*evt)()->GetEventID();
  edm::LogInfo("ValidHGCal") << "HGCPassive: =====> Begin event = "
			     << iev << std::endl;
  
  ++count_;
  store_.clear();
}

//=================================================================== each STEP
void HGCPassive::update(const G4Step * aStep) {

  if (aStep != NULL) {

    G4VSensitiveDetector* curSD = aStep->GetPreStepPoint()->GetSensitiveDetector();
    if (curSD==NULL) {
      
      G4TouchableHistory* touchable = (G4TouchableHistory*)aStep->GetPreStepPoint()->GetTouchable();
      G4LogicalVolume* plv = (G4LogicalVolume*)touchable->GetVolume()->GetLogicalVolume();
      std::map<G4LogicalVolume*,std::pair<unsigned int,std::string>>::iterator it = (init_) ? mapLV_.find(plv) : findLV(plv);
      if (it != mapLV_.end()) {
	unsigned int copy = (unsigned int)(touchable->GetReplicaNumber(0) + 
					   1000*touchable->GetReplicaNumber(1));
	std::pair<G4LogicalVolume*,unsigned int> key(plv,copy);
	std::map<std::pair<G4LogicalVolume*,unsigned int>,std::pair<double,double>>::iterator itr = store_.find(key);
	double time = (aStep->GetPostStepPoint()->GetGlobalTime());
	if (itr == store_.end()) {
	  store_[key] = std::pair<double,double>(time,0.0);
	  itr         = store_.find(key);
	}
	double edeposit = aStep->GetTotalEnergyDeposit();
	(itr->second).second += edeposit;
#ifdef EDM_ML_DEBUG
	std::cout << "HGCPassive: Element " << (it->second).first << ":" 
		  << (it->second).second << ":" << copy << " T " 
		  << (itr->second).first << " E " << (itr->second).second 
		  << std::endl;
#endif
      }//if( it != map.end() )
    }//if (curSD==NULL)
  }//if (aStep != NULL)

    
}//end update aStep


//================================================================ End of EVENT

void HGCPassive::endOfEvent(edm::PassiveHitContainer& hgcPH, unsigned int k) {
#ifdef EDM_ML_DEBUG
  unsigned int kount(0);
#endif
  std::map<std::pair<G4LogicalVolume*,unsigned int>,std::pair<double,double>>::iterator itr;
  for (itr = store_.begin(); itr != store_.end(); ++itr) {
    G4LogicalVolume* lv = (itr->first).first;
    std::map<G4LogicalVolume*,std::pair<unsigned int,std::string>>::iterator it = mapLV_.find(lv);
    if (it != mapLV_.end()) {
      if ((it->second).first == k) {
	PassiveHit hit((it->second).second,(itr->first).second,
		       (itr->second).second,(itr->second).first);
	hgcPH.push_back(hit);
#ifdef EDM_ML_DEBUG
	std::cout << "HGCPassive[" << k << "] Hit[" << kount << "] " << hit
		  << std::endl;
	++kount;
#endif
      }
    }
  }
}

G4VPhysicalVolume * HGCPassive::getTopPV() {
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}

std::map<G4LogicalVolume*,std::pair<unsigned int,std::string>>::iterator HGCPassive::findLV(G4LogicalVolume * plv) {
  std::map<G4LogicalVolume*,std::pair<unsigned int,std::string>>::iterator itr = mapLV_.find(plv);
  if (itr == mapLV_.end()) {
    std::string name = plv->GetName();
    for (unsigned int k=0; k<LVNames_.size(); ++k) {
      if (name.find(LVNames_[k]) != std::string::npos) {
	mapLV_[plv] = std::pair<unsigned int,std::string>(k,name);
	itr = mapLV_.find(plv);
	break;
      }
    }
  }
  return itr;
}

DEFINE_SIMWATCHER (HGCPassive);

