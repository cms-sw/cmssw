///////////////////////////////////////////////////////////////////////////////
// File: HGCPassive.cc
//copied from SimG4HGCalValidation
// Description: Main analysis class for HGCal Validation of G4 Hits
///////////////////////////////////////////////////////////////////////////////

#include "HGCPassive.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4TransportationManager.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <utility>  

//#define EDM_ML_DEBUG

HGCPassive::HGCPassive(const edm::ParameterSet &p) : topPV_(0), topLV_(0),
						     count_(0), init_(false) {

  edm::ParameterSet m_Passive = p.getParameter<edm::ParameterSet>("HGCPassive");
  LVNames_   = m_Passive.getParameter<std::vector<std::string> >("LVNames");
  motherName_= m_Passive.getParameter<std::string>("MotherName");

#ifdef EDM_ML_DEBUG
  std::cout << "Name of the mother volume " << motherName_ << std::endl;
  unsigned k(0);
#endif
  for (auto name : LVNames_) {
    produces<edm::PassiveHitContainer>(Form("%sPassiveHits",name.c_str()));
#ifdef EDM_ML_DEBUG
    std::cout << "Collection name[" << k << "] " << name << std::endl;
    ++k;
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
    for (auto lvcite : *lvs) {
      findLV(lvcite);
    } 

#ifdef EDM_ML_DEBUG
    std::cout << "HGCPassive::Finds " << mapLV_.size() << " logical volumes\n";
    unsigned int k(0);
    for (const auto& lvs : mapLV_) {
      std::cout << "Entry[" << k << "] " << lvs.first << ": (" 
		<< (lvs.second).first << ", " << (lvs.second).second << ")\n";
      ++k;
    }
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
      auto it = (init_) ? mapLV_.find(plv) : findLV(plv);
      if (((aStep->GetPostStepPoint() == 0) || 
	   (aStep->GetTrack()->GetNextVolume() == 0)) &&
	  (aStep->IsLastStepInVolume())) {
#ifdef EDM_ML_DEBUG
	std::cout << plv->GetName() << " F|L Step " 
		  << aStep->IsFirstStepInVolume() << ":" 
		  << aStep->IsLastStepInVolume() << " Position" 
		  << aStep->GetPreStepPoint()->GetPosition() << " Track " 
		  << aStep->GetTrack()->GetDefinition()->GetParticleName()
		  << " at" << aStep->GetTrack()->GetPosition() << " Volume " 
		  << aStep->GetTrack()->GetVolume() << ":" 
		  << aStep->GetTrack()->GetNextVolume() << " Status " 
		  << aStep->GetTrack()->GetTrackStatus() << " KE " 
		  << aStep->GetTrack()->GetKineticEnergy() << " Deposit " 
		  << aStep->GetTotalEnergyDeposit() << " Map "
		  << (it != mapLV_.end()) << std::endl;
#endif
	double time   = aStep->GetTrack()->GetGlobalTime();
	double energy = (aStep->GetPreStepPoint()->GetKineticEnergy() +
			 aStep->GetTotalEnergyDeposit())/CLHEP::GeV;
	if (it != mapLV_.end()) {
	  storeInfo(it, plv, 0, time, energy);
	} else if (topLV_ != 0) {
	  auto itr = (init_) ? mapLV_.find(topLV_) : findLV(topLV_);
	  if (itr != mapLV_.end()) {
	    storeInfo(itr, topLV_, 0, time, energy);
	  }
	}
      } else if (it != mapLV_.end()) {
	unsigned int copy = (unsigned int)(touchable->GetReplicaNumber(0) + 
					   1000*touchable->GetReplicaNumber(1));
	double time = (aStep->GetPostStepPoint()->GetGlobalTime());
	double edeposit = (aStep->GetTotalEnergyDeposit())/CLHEP::GeV;
	storeInfo(it, plv, copy, time, edeposit);
      } else if (topLV_ != 0) {
	auto itr = findLV(topLV_);
	if (itr != mapLV_.end()) {
	  double time = (aStep->GetPostStepPoint()->GetGlobalTime());
	  double edeposit = (aStep->GetTotalEnergyDeposit())/CLHEP::GeV;
	  storeInfo(itr, topLV_, 100, time, edeposit);
	}
      }
    }//if (curSD==NULL)
  }//if (aStep != NULL)

    
}//end update aStep

//================================================================ End of EVENT

void HGCPassive::endOfEvent(edm::PassiveHitContainer& hgcPH, unsigned int k) {
#ifdef EDM_ML_DEBUG
  unsigned int kount(0);
#endif
  for (const auto& element : store_) {
    G4LogicalVolume* lv = (element.first).first;
    auto it = mapLV_.find(lv);
    if (it != mapLV_.end()) {
      if ((it->second).first == k) {
	PassiveHit hit((it->second).second,(element.first).second,
		       (element.second).second,(element.second).first);
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

HGCPassive::volumeIterator HGCPassive::findLV(G4LogicalVolume * plv) {
  auto itr = mapLV_.find(plv);
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
  if (topLV_ == 0) {
    if (std::string(plv->GetName()) == motherName_) topLV_ = plv;
  }
  return itr;
}

void HGCPassive::storeInfo(const HGCPassive::volumeIterator it,
			   G4LogicalVolume* plv, unsigned int copy, 
			   double time, double energy) {

  std::pair<G4LogicalVolume*,unsigned int> key(plv,copy);
  auto itr = store_.find(key);
  if (itr == store_.end()) {
    store_[key] = std::pair<double,double>(time,energy);
  } else {
    (itr->second).second += energy;
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HGCPassive: Element " << (it->second).first << ":" 
	    << (it->second).second << ":" << copy << " T " 
	    << (itr->second).first << " E " << (itr->second).second 
	    << std::endl;
#endif
}

DEFINE_SIMWATCHER (HGCPassive);

