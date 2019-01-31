// to make hits in EB/EE/HC
#include "SimG4CMS/Calo/interface/CaloSteppingAction.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4LogicalVolumeStore.hh"
#include "G4NavigationHistory.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4RegionStore.hh"
#include "G4Trap.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG

CaloSteppingAction::CaloSteppingAction(const edm::ParameterSet &p) : 
  count_(0) {

  edm::ParameterSet iC = p.getParameter<edm::ParameterSet>("CaloSteppingAction");
  nameEBSD_       = iC.getParameter<std::vector<std::string> >("EBSDNames");
  nameEESD_       = iC.getParameter<std::vector<std::string> >("EESDNames");
  nameHCSD_       = iC.getParameter<std::vector<std::string> >("HCSDNames");
  nameHitC_       = iC.getParameter<std::vector<std::string> >("HitCollNames");
  slopeLY_        = iC.getParameter<double>("SlopeLightYield");
  birkC1EC_       = iC.getParameter<double>("BirkC1EC")*(g/(MeV*cm2));
  birkSlopeEC_    = iC.getParameter<double>("BirkSlopeEC");
  birkCutEC_      = iC.getParameter<double>("BirkCutEC");
  birkC1HC_       = iC.getParameter<double>("BirkC1HC")*(g/(MeV*cm2));
  birkC2HC_       = iC.getParameter<double>("BirkC2HC");
  birkC3HC_       = iC.getParameter<double>("BirkC3HC");

  edm::LogVerbatim("Step") << "CaloSteppingAction:: " << nameEBSD_.size() 
			   << " names for EB SD's";
  for (unsigned int k=0; k<nameEBSD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << nameEBSD_[k];
  edm::LogVerbatim("Step") << "CaloSteppingAction:: " << nameEESD_.size() 
			   << " names for EE SD's";
  for (unsigned int k=0; k<nameEESD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << nameEESD_[k];
  edm::LogVerbatim("Step") << "CaloSteppingAction:: " << nameHCSD_.size() 
			   << " names for HC SD's";
  for (unsigned int k=0; k<nameHCSD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << nameHCSD_[k];
  edm::LogVerbatim("Step") << "CaloSteppingAction::Constants for ECAL: slope "
			   << slopeLY_ << " Birk constants " << birkC1EC_ 
			   << ":" << birkSlopeEC_ << ":" << birkCutEC_;
  edm::LogVerbatim("Step") << "CaloSteppingAction::Constants for HCAL: Birk "
			   << "constants " << birkC1HC_ << ":" << birkC2HC_
			   << ":" << birkC3HC_;
  edm::LogVerbatim("Step") << "CaloSteppingAction:: " << nameHitC_.size() 
			   << " hit collections";
  for (unsigned int k=0; k<nameHitC_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << nameHitC_[k];

  ebNumberingScheme_ = std::make_unique<EcalBarrelNumberingScheme>();
  eeNumberingScheme_ = std::make_unique<EcalEndcapNumberingScheme>();
  hcNumbering_.reset(nullptr);
  hcNumberingScheme_ = std::make_unique<HcalNumberingScheme>();
  for (int k=0; k<CaloSteppingAction::nSD_; ++k) {
    slave_[k] = std::make_unique<CaloSlaveSD>(nameHitC_[k]);
    produces<edm::PCaloHitContainer>(nameHitC_[k]);
  }
} 
   
CaloSteppingAction::~CaloSteppingAction() {
  edm::LogVerbatim("Step") << "CaloSteppingAction: -------->  Total number of "
			   << "selected entries : " << count_;
}

void CaloSteppingAction::produce(edm::Event& e, const edm::EventSetup&) {

  for (int k=0; k<CaloSteppingAction::nSD_; ++k) {
    saveHits(k);
    auto product = std::make_unique<edm::PCaloHitContainer>();
    fillHits(*product,k);
    e.put(std::move(product),nameHitC_[k]);
  }
}

void CaloSteppingAction::fillHits(edm::PCaloHitContainer& cc, int type) {
  edm::LogVerbatim("Step") << "CaloSteppingAction::fillHits for type "
			   << type << " with "
			   << slave_[type].get()->hits().size() << " hits";
  cc = slave_[type].get()->hits();
  slave_[type].get()->Clean();
}

void CaloSteppingAction::update(const BeginOfJob * job) {
  edm::LogVerbatim("Step") << "CaloSteppingAction:: Enter BeginOfJob";

  // Numbering From DDD
  edm::ESHandle<HcalDDDSimConstants>    hdc;
  (*job)()->get<HcalSimNumberingRecord>().get(hdc);
  const HcalDDDSimConstants* hcons_ = hdc.product();
  edm::LogVerbatim("Step") << "CaloSteppingAction:: Initialise "
			   << "HcalNumberingFromDDD";
  hcNumbering_ = std::make_unique<HcalNumberingFromDDD>(hcons_);
}

//==================================================================== per RUN
void CaloSteppingAction::update(const BeginOfRun * run) {

  int irun = (*run)()->GetRunID();
  edm::LogVerbatim("Step") << "CaloSteppingAction:: Begin of Run = " << irun;

  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  if (lvs) {
    std::map<const std::string, const G4LogicalVolume *> nameMap;
    std::map<const std::string, const G4LogicalVolume *>::const_iterator itr;
    for (auto lvi = lvs->begin(), lve = lvs->end(); lvi != lve; ++lvi)
      nameMap.emplace((*lvi)->GetName(), *lvi);
    for (auto const& name : nameEBSD_) {
      for (itr = nameMap.begin(); itr != nameMap.end(); ++itr) {
	const std::string &lvname = itr->first;
	if (lvname.find(name) != std::string::npos) {
	  volEBSD_.emplace_back(itr->second);
	  int type =  (lvname.find("refl") == std::string::npos) ? -1 : 1;
	  G4Trap* solid = static_cast<G4Trap*>(itr->second->GetSolid());
	  double  dz    = 2*solid->GetZHalfLength();
	  xtalMap_.insert(std::pair<const G4LogicalVolume*,double>(itr->second,dz*type));
	}
      }
    }
    for (auto const& name : nameEESD_) {
      for (itr = nameMap.begin(); itr != nameMap.end(); ++itr) {
	const std::string &lvname = itr->first;
	if (lvname.find(name) != std::string::npos)  {
	  volEESD_.emplace_back(itr->second);
	  int type =  (lvname.find("refl") == std::string::npos) ? 1 : -1;
	  G4Trap* solid = static_cast<G4Trap*>(itr->second->GetSolid());
	  double  dz    = 2*solid->GetZHalfLength();
	  xtalMap_.insert(std::pair<const G4LogicalVolume*,double>(itr->second,dz*type));
	}
      }
    }
    for (auto const& name : nameHCSD_) {
      for (itr = nameMap.begin(); itr != nameMap.end(); ++itr) {
	const std::string &lvname = itr->first;
	if (lvname.find(name) != std::string::npos) 
	  volHCSD_.emplace_back(itr->second);
      }
    }
  }
  edm::LogVerbatim("Step") << volEBSD_.size() << " logical volumes for EB SD";
  for (unsigned int k=0; k<volEBSD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << volEBSD_[k];
  edm::LogVerbatim("Step") << volEESD_.size() << " logical volumes for EE SD";
  for (unsigned int k=0; k<volEESD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << volEESD_[k];
  edm::LogVerbatim("Step") << volHCSD_.size() << " logical volumes for HC SD";
  for (unsigned int k=0; k<volHCSD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << volHCSD_[k];

}

//=================================================================== per EVENT
void CaloSteppingAction::update(const BeginOfEvent * evt) {
 
  edm::LogVerbatim("Step") <<"CaloSteppingAction: Begin of event = " 
			   << (*evt)()->GetEventID();
  for (int k=0; k<CaloSteppingAction::nSD_; ++k) {
    hitMap_[k].erase (hitMap_[k].begin(), hitMap_[k].end());
    slave_[k].get()->Initialize();
  }
}

//=================================================================== each STEP
void CaloSteppingAction::update(const G4Step * aStep) {

  //  edm::LogVerbatim("Step") <<"CaloSteppingAction: At each Step";
  NaNTrap(aStep);
  auto lv = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
  if        (std::find(volEBSD_.begin(),volEBSD_.end(),lv) != volEBSD_.end()) {
    auto unitID   = getDetIDEB(aStep);
    double dEStep = aStep->GetTotalEnergyDeposit();
    if (unitID > 0 && dEStep > 0.0) {
      fillHit(aStep, unitID, dEStep, 0);
    }
  } else if (std::find(volEESD_.begin(),volEESD_.end(),lv) != volEESD_.end()) {
    auto unitID   = getDetIDEE(aStep);
    double dEStep = aStep->GetTotalEnergyDeposit();
    if (unitID > 0 && dEStep > 0.0) {
      fillHit(aStep, unitID, dEStep, 1);
    }
  } else if (std::find(volHCSD_.begin(),volHCSD_.end(),lv) != volHCSD_.end()) {
    auto unitID   = getDetIDHC(aStep);
    double dEStep = aStep->GetTotalEnergyDeposit();
    if (unitID > 0 && dEStep > 0.0) {
      fillHit(aStep, unitID, dEStep, 2);
    }
  }
}

//================================================================ End of EVENT
void CaloSteppingAction::update(const EndOfEvent * evt) {

  ++count_;
  // Fill event input 
  edm::LogVerbatim("Step") << "CaloSteppingAction: EndOfEvent " 
			   << (*evt)()->GetEventID();
}

void CaloSteppingAction::NaNTrap(const G4Step* aStep) const {

  auto currentPos = aStep->GetTrack()->GetPosition();
  double xyz = currentPos.x() + currentPos.y() + currentPos.z();
  auto currentMom = aStep->GetTrack()->GetMomentum();
  xyz += currentMom.x() + currentMom.y() + currentMom.z();

  if (edm::isNotFinite(xyz)) {
    auto  pCurrentVol = aStep->GetPreStepPoint()->GetPhysicalVolume();
    auto& nameOfVol = pCurrentVol->GetName();
    throw cms::Exception("Unknown", "CaloSteppingAction") 
      << " Corrupted Event - NaN detected in volume " << nameOfVol << "\n";
  }
}

uint32_t CaloSteppingAction::getDetIDEB(const G4Step * aStep) const {
  EcalBaseNumber theBaseNumber = getBaseNumber(aStep);
  return ebNumberingScheme_->getUnitID(theBaseNumber);
}

uint32_t CaloSteppingAction::getDetIDEE(const G4Step * aStep) const {
  EcalBaseNumber theBaseNumber = getBaseNumber(aStep);
  return eeNumberingScheme_->getUnitID(theBaseNumber);
}

uint32_t CaloSteppingAction::getDetIDHC(const G4Step * aStep) const {

  auto const prePoint  = aStep->GetPreStepPoint(); 
  auto const touch     = prePoint->GetTouchable();
  const G4ThreeVector& hitPoint = prePoint->GetPosition();

  int depth = (touch->GetReplicaNumber(0))%10 + 1;
  int lay   = (touch->GetReplicaNumber(0)/10)%100 + 1;
  int det   = (touch->GetReplicaNumber(1))/1000;
  HcalNumberingFromDDD::HcalID tmp =  hcNumbering_.get()->unitID(det, hitPoint,
								 depth, lay);
  uint32_t id = hcNumberingScheme_.get()->getUnitID(tmp);
  return id;
}

EcalBaseNumber CaloSteppingAction::getBaseNumber(const G4Step* aStep) const {
  EcalBaseNumber theBaseNumber;
  auto touch = aStep->GetPreStepPoint()->GetTouchable();
  int theSize = touch->GetHistoryDepth()+1;
  if (theBaseNumber.getCapacity() < theSize ) theBaseNumber.setSize(theSize);
  //Get name and copy numbers
  if (theSize > 1) {
    for (int ii = 0; ii < theSize ; ii++) {
      theBaseNumber.addLevel(touch->GetVolume(ii)->GetName(),
			     touch->GetReplicaNumber(ii));
    }
  }
  return theBaseNumber;
}

void CaloSteppingAction::fillHit(const G4Step * aStep, uint32_t id, double dE,
				 int flag) {
  uint16_t   depth    = getDepth(aStep, flag);
  auto const theTrack = aStep->GetTrack();
  double     time     = theTrack->GetGlobalTime()/nanosecond;
  int        primID   = theTrack->GetTrackID();
  CaloHitID  currentID(id, time, primID, depth);
  auto const hitPoint = aStep->GetPreStepPoint();
  auto const lv       = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
  if (flag < 2) {
    auto currentLocalPoint = setToLocal(hitPoint->GetPosition(),
					hitPoint->GetTouchable());
    dE *= (curve_LY(lv,currentLocalPoint.z())*getBirkL3(aStep));
  } else {
    dE *= getBirkHC(aStep);
  }
  double edepEM(0), edepHAD(0);
  if (G4TrackToParticleID::isGammaElectronPositron(theTrack)) {
    edepEM  = dE;
  } else {
    edepHAD = dE;
  }
  auto it = hitMap_[flag].find(currentID);
  if (it != hitMap_[flag].end()) {
    (it->second).addEnergyDeposit(edepEM,edepHAD);
  } else {
    CaloGVHit aHit;
    aHit.setID(currentID);
    aHit.addEnergyDeposit(edepEM,edepHAD);
    hitMap_[flag][currentID] = aHit;
  }
}

uint16_t CaloSteppingAction::getDepth(const G4Step * aStep, int flag) const {
  uint16_t depth(0);
  if (flag < 2) {
    const G4StepPoint* hitPoint = aStep->GetPreStepPoint();
    auto currentLocalPoint = setToLocal(hitPoint->GetPosition(),
					hitPoint->GetTouchable());
    auto lv = hitPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();
    auto ite = xtalMap_.find(lv);
    double crystalDepth = (ite == xtalMap_.end()) 
      ? 0.0 : (std::abs(0.5*(ite->second)+currentLocalPoint.z()));
    uint16_t depth1 = (ite == xtalMap_.end()) ? 0 : (((ite->second) >= 0) ? 0 :
						     PCaloHit::kEcalDepthRefz);
    double radl = hitPoint->GetMaterial()->GetRadlen();
    uint16_t depth2 = (uint16_t)floor(crystalDepth/radl);
    depth          |= (((depth2&PCaloHit::kEcalDepthMask) << PCaloHit::kEcalDepthOffset) | depth1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Step") << "CaloSteppingAction::getDepth radl " << radl
			     << ":" << crystalDepth << " depth " << depth;
#endif
  } else {
  }
  return depth;
}

G4ThreeVector CaloSteppingAction::setToLocal(const G4ThreeVector& global, 
					     const G4VTouchable* touch) const {
  return touch->GetHistory()->GetTopTransform().TransformPoint(global);
}

double CaloSteppingAction::curve_LY(const G4LogicalVolume* lv, double z) {

  double weight = 1.;
  auto ite = xtalMap_.find(lv);
  double crystalLength = ((ite == xtalMap_.end()) ? 230.0 : 
			  std::abs(ite->second));
  double crystalDepth = ((ite == xtalMap_.end()) ? 0.0 :
			 (std::abs(0.5*(ite->second)+z)));
  double dapd = crystalLength - crystalDepth;
  if (dapd >= -0.1 || dapd <= crystalLength+0.1) {
    if (dapd <= 100.)
      weight = 1.0 + slopeLY_ - dapd * 0.01 * slopeLY_;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Step") << "CaloSteppingAction::curve_LY " << crystalDepth
			     << ":" << crystalLength << ":" << dapd << ":" 
			     << weight;
#endif
  } else {
    edm::LogWarning("Step") << "CaloSteppingAction: light coll curve : wrong "
			    << "distance to APD " << dapd << " crlength = " 
			    << crystalLength <<" crystal name = " 
			    << lv->GetName() << " z of localPoint = " << z
			    << " take weight = " << weight;
  }
  return weight;
}

double CaloSteppingAction::getBirkL3(const G4Step* aStep) {

  double weight = 1.;
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();

  if (preStepPoint->GetCharge() != 0. && aStep->GetStepLength() > 0.) {
    const G4Material* mat = preStepPoint->GetMaterial();
    double density = mat->GetDensity();
    double dedx    = aStep->GetTotalEnergyDeposit()/aStep->GetStepLength();
    double rkb     = birkC1EC_/density;
    if (dedx > 0) {
      weight         = 1. - birkSlopeEC_*log(rkb*dedx);
      if (weight < birkCutEC_) weight = birkCutEC_;
      else if (weight > 1.)    weight = 1.;
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Step") << "CaloSteppingAction::getBirkL3 in " 
			     << mat->GetName() << " Charge " 
			     << preStepPoint->GetCharge() << " dE/dx " << dedx
			     << " Birk Const " << rkb << " Weight = " << weight
			     << " dE " << aStep->GetTotalEnergyDeposit();
#endif
  }
  return weight;
}

double CaloSteppingAction::getBirkHC(const G4Step* aStep) {

  double weight = 1.;
  double charge = aStep->GetPreStepPoint()->GetCharge();
  double length = aStep->GetStepLength();

  if (charge != 0. && length > 0.) {
    double density = aStep->GetPreStepPoint()->GetMaterial()->GetDensity();
    double dedx    = aStep->GetTotalEnergyDeposit()/length;
    double rkb     = birkC1HC_/density;
    double c       = birkC2HC_*rkb*rkb;
    if (std::abs(charge) >= 2.) rkb /= birkC3HC_;
    weight = 1./(1.+rkb*dedx+c*dedx*dedx);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Step") << "CaloSteppingAction::getBirkHC in " 
			     << aStep->GetPreStepPoint()->GetMaterial()->GetName() 
			     << " Charge " << charge << " dE/dx " << dedx 
			     << " Birk Const " << rkb << ", " << c 
			     << " Weight = " << weight << " dE "
			     << aStep->GetTotalEnergyDeposit();
#endif
  }
  return weight;
}

void CaloSteppingAction::saveHits(int type) {

  edm::LogVerbatim("Step") << "CaloSteppingAction:: saveHits for type " 
			   << type << " with " << hitMap_[type].size()
			   << " hits";
  slave_[type].get()->ReserveMemory(hitMap_[type].size());
  for (auto const& hit : hitMap_[type]) {
    slave_[type].get()->processHits(hit.second.getUnitID(),
				    hit.second.getEM()/GeV, 
				    hit.second.getHadr()/GeV,
				    hit.second.getTimeSlice(),
				    hit.second.getTrackID(),
				    hit.second.getDepth());
  }
}
