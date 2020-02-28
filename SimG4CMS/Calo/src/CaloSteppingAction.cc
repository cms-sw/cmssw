// to make hits in EB/EE/HC
#include "SimG4CMS/Calo/interface/CaloSteppingAction.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#ifdef HcalNumberingTest
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#endif

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

CaloSteppingAction::CaloSteppingAction(const edm::ParameterSet& p) : count_(0) {
  edm::ParameterSet iC = p.getParameter<edm::ParameterSet>("CaloSteppingAction");
  nameEBSD_ = iC.getParameter<std::vector<std::string> >("EBSDNames");
  nameEESD_ = iC.getParameter<std::vector<std::string> >("EESDNames");
  nameHCSD_ = iC.getParameter<std::vector<std::string> >("HCSDNames");
  nameHitC_ = iC.getParameter<std::vector<std::string> >("HitCollNames");
  allSteps_ = iC.getParameter<int>("AllSteps");
  slopeLY_ = iC.getParameter<double>("SlopeLightYield");
  birkC1EC_ = iC.getParameter<double>("BirkC1EC");
  birkSlopeEC_ = iC.getParameter<double>("BirkSlopeEC");
  birkCutEC_ = iC.getParameter<double>("BirkCutEC");
  birkC1HC_ = iC.getParameter<double>("BirkC1HC");
  birkC2HC_ = iC.getParameter<double>("BirkC2HC");
  birkC3HC_ = iC.getParameter<double>("BirkC3HC");
  timeSliceUnit_ = iC.getUntrackedParameter<double>("TimeSliceUnit", 1.0);

  edm::LogVerbatim("Step") << "CaloSteppingAction:: " << nameEBSD_.size() << " names for EB SD's";
  for (unsigned int k = 0; k < nameEBSD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << nameEBSD_[k];
  edm::LogVerbatim("Step") << "CaloSteppingAction:: " << nameEESD_.size() << " names for EE SD's";
  for (unsigned int k = 0; k < nameEESD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << nameEESD_[k];
  edm::LogVerbatim("Step") << "CaloSteppingAction:: " << nameHCSD_.size() << " names for HC SD's";
  for (unsigned int k = 0; k < nameHCSD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << nameHCSD_[k];
  edm::LogVerbatim("Step") << "CaloSteppingAction::Constants for ECAL: slope " << slopeLY_ << " Birk constants "
                           << birkC1EC_ << ":" << birkSlopeEC_ << ":" << birkCutEC_;
  edm::LogVerbatim("Step") << "CaloSteppingAction::Constants for HCAL: Birk "
                           << "constants " << birkC1HC_ << ":" << birkC2HC_ << ":" << birkC3HC_;
  edm::LogVerbatim("Step") << "CaloSteppingAction::Constant for time slice " << timeSliceUnit_;
  edm::LogVerbatim("Step") << "CaloSteppingAction:: " << nameHitC_.size() << " hit collections";
  for (unsigned int k = 0; k < nameHitC_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << nameHitC_[k];

  ebNumberingScheme_ = std::make_unique<EcalBarrelNumberingScheme>();
  eeNumberingScheme_ = std::make_unique<EcalEndcapNumberingScheme>();
  hcNumberingPS_ = std::make_unique<HcalNumberingFromPS>(iC);
  hcNumberingScheme_ = std::make_unique<HcalNumberingScheme>();
#ifdef HcalNumberingTest
  hcNumbering_.reset(nullptr);
#endif
  for (int k = 0; k < CaloSteppingAction::nSD_; ++k) {
    slave_[k] = std::make_unique<CaloSlaveSD>(nameHitC_[k]);
    produces<edm::PCaloHitContainer>(nameHitC_[k]);
  }
  if (allSteps_ != 0)
    produces<edm::PassiveHitContainer>("AllPassiveHits");
  edm::LogVerbatim("Step") << "CaloSteppingAction:: All Steps Flag " << allSteps_ << " for passive hits";
}

CaloSteppingAction::~CaloSteppingAction() {
  edm::LogVerbatim("Step") << "CaloSteppingAction: -------->  Total number of "
                           << "selected entries : " << count_;
}

void CaloSteppingAction::produce(edm::Event& e, const edm::EventSetup&) {
  for (int k = 0; k < CaloSteppingAction::nSD_; ++k) {
    saveHits(k);
    auto product = std::make_unique<edm::PCaloHitContainer>();
    fillHits(*product, k);
    e.put(std::move(product), nameHitC_[k]);
  }
  if (allSteps_ != 0) {
    std::unique_ptr<edm::PassiveHitContainer> hgcPH(new edm::PassiveHitContainer);
    fillPassiveHits(*hgcPH);
    e.put(std::move(hgcPH), "AllPassiveHits");
  }
}

void CaloSteppingAction::fillHits(edm::PCaloHitContainer& cc, int type) {
  edm::LogVerbatim("Step") << "CaloSteppingAction::fillHits for type " << type << " with "
                           << slave_[type].get()->hits().size() << " hits";
  cc = slave_[type].get()->hits();
  slave_[type].get()->Clean();
}

void CaloSteppingAction::fillPassiveHits(edm::PassiveHitContainer& cc) {
  edm::LogVerbatim("Step") << "CaloSteppingAction::fillPassiveHits with " << store_.size() << " hits";
  for (const auto& element : store_) {
    auto lv = std::get<0>(element);
    auto it = mapLV_.find(lv);
    if (it != mapLV_.end()) {
      PassiveHit hit(it->second,
                     std::get<1>(element),
                     std::get<5>(element),
                     std::get<6>(element),
                     std::get<4>(element),
                     std::get<2>(element),
                     std::get<3>(element),
                     std::get<7>(element),
                     std::get<8>(element),
                     std::get<9>(element),
                     std::get<10>(element));
      cc.emplace_back(hit);
    }
  }
}

void CaloSteppingAction::update(const BeginOfJob* job) {
  edm::LogVerbatim("Step") << "CaloSteppingAction:: Enter BeginOfJob";

#ifdef HcalNumberingTest
  // Numbering From DDD
  edm::ESHandle<HcalDDDSimConstants> hdc;
  (*job)()->get<HcalSimNumberingRecord>().get(hdc);
  const HcalDDDSimConstants* hcons_ = hdc.product();
  edm::LogVerbatim("Step") << "CaloSteppingAction:: Initialise "
                           << "HcalNumberingFromDDD";
  hcNumbering_ = std::make_unique<HcalNumberingFromDDD>(hcons_);
#endif
}

//==================================================================== per RUN
void CaloSteppingAction::update(const BeginOfRun* run) {
  int irun = (*run)()->GetRunID();
  edm::LogVerbatim("Step") << "CaloSteppingAction:: Begin of Run = " << irun;

  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  if (lvs) {
    std::map<const std::string, const G4LogicalVolume*> nameMap;
    std::map<const std::string, const G4LogicalVolume*>::const_iterator itr;
    for (auto lvi = lvs->begin(), lve = lvs->end(); lvi != lve; ++lvi) {
      nameMap.emplace((*lvi)->GetName(), *lvi);
      if (allSteps_ < 0)
        mapLV_[*lvi] = (*lvi)->GetName();
    }

    for (auto const& name : nameEBSD_) {
      for (itr = nameMap.begin(); itr != nameMap.end(); ++itr) {
        const std::string& lvname = itr->first;
        if (lvname.find(name) != std::string::npos) {
          volEBSD_.emplace_back(itr->second);
          int type = (lvname.find("refl") == std::string::npos) ? -1 : 1;
          G4Trap* solid = static_cast<G4Trap*>(itr->second->GetSolid());
          double dz = 2 * solid->GetZHalfLength() / CLHEP::mm;
          xtalMap_.insert(std::pair<const G4LogicalVolume*, double>(itr->second, dz * type));
          if ((allSteps_ > 0) && ((allSteps_ % 10) > 0))
            mapLV_[itr->second] = itr->first;
        }
      }
    }
    for (auto const& name : nameEESD_) {
      for (itr = nameMap.begin(); itr != nameMap.end(); ++itr) {
        const std::string& lvname = itr->first;
        if (lvname.find(name) != std::string::npos) {
          volEESD_.emplace_back(itr->second);
          int type = (lvname.find("refl") == std::string::npos) ? 1 : -1;
          G4Trap* solid = static_cast<G4Trap*>(itr->second->GetSolid());
          double dz = 2 * solid->GetZHalfLength() / CLHEP::mm;
          xtalMap_.insert(std::pair<const G4LogicalVolume*, double>(itr->second, dz * type));
          if ((allSteps_ > 0) && (((allSteps_ / 10) % 10) > 0))
            mapLV_[itr->second] = itr->first;
        }
      }
    }

    for (auto const& name : nameHCSD_) {
      for (itr = nameMap.begin(); itr != nameMap.end(); ++itr) {
        const std::string& lvname = itr->first;
        if (lvname.find(name) != std::string::npos) {
          volHCSD_.emplace_back(itr->second);
          if ((allSteps_ > 0) && (((allSteps_ / 100) % 10) > 0))
            mapLV_[itr->second] = itr->first;
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Step") << volEBSD_.size() << " logical volumes for EB SD";
  for (unsigned int k = 0; k < volEBSD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << volEBSD_[k];
  edm::LogVerbatim("Step") << volEESD_.size() << " logical volumes for EE SD";
  for (unsigned int k = 0; k < volEESD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << volEESD_[k];
  edm::LogVerbatim("Step") << volHCSD_.size() << " logical volumes for HC SD";
  for (unsigned int k = 0; k < volHCSD_.size(); ++k)
    edm::LogVerbatim("Step") << "[" << k << "] " << volHCSD_[k];
  edm::LogVerbatim("Step") << mapLV_.size() << " logical volumes for Passive hits";
  unsigned int k(0);
  for (auto itr = mapLV_.begin(); itr != mapLV_.end(); ++itr) {
    edm::LogVerbatim("Step") << "[" << k << "] " << itr->second << ":" << itr->first;
    ++k;
  }
#endif
}

//=================================================================== per EVENT
void CaloSteppingAction::update(const BeginOfEvent* evt) {
  eventID_ = (*evt)()->GetEventID();
  edm::LogVerbatim("Step") << "CaloSteppingAction: Begin of event = " << eventID_;
  for (int k = 0; k < CaloSteppingAction::nSD_; ++k) {
    hitMap_[k].erase(hitMap_[k].begin(), hitMap_[k].end());
    slave_[k].get()->Initialize();
  }
  if (allSteps_ != 0)
    store_.clear();
}

//=================================================================== each STEP
void CaloSteppingAction::update(const G4Step* aStep) {
  //  edm::LogVerbatim("Step") <<"CaloSteppingAction: At each Step";
  NaNTrap(aStep);
  auto lv = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
  bool hc = (std::find(volHCSD_.begin(), volHCSD_.end(), lv) != volHCSD_.end());
  bool eb = (std::find(volEBSD_.begin(), volEBSD_.end(), lv) != volEBSD_.end());
  bool ee = (std::find(volEESD_.begin(), volEESD_.end(), lv) != volEESD_.end());
  uint32_t unitID(0);
  if (hc || eb || ee) {
    double dEStep = aStep->GetTotalEnergyDeposit() / CLHEP::MeV;
    auto const theTrack = aStep->GetTrack();
    double time = theTrack->GetGlobalTime() / CLHEP::nanosecond;
    int primID = theTrack->GetTrackID();
    bool em = G4TrackToParticleID::isGammaElectronPositron(theTrack);
    auto const touch = aStep->GetPreStepPoint()->GetTouchable();
    auto const& hitPoint = aStep->GetPreStepPoint()->GetPosition();
    if (hc) {
      int depth = (touch->GetReplicaNumber(0)) % 10 + 1;
      int lay = (touch->GetReplicaNumber(0) / 10) % 100 + 1;
      int det = (touch->GetReplicaNumber(1)) / 1000;
      unitID = getDetIDHC(det, lay, depth, math::XYZVectorD(hitPoint.x(), hitPoint.y(), hitPoint.z()));
      if (unitID > 0 && dEStep > 0.0) {
        dEStep *= getBirkHC(dEStep,
                            (aStep->GetStepLength() / CLHEP::cm),
                            aStep->GetPreStepPoint()->GetCharge(),
                            (aStep->GetPreStepPoint()->GetMaterial()->GetDensity() / (CLHEP::g / CLHEP::cm3)));
        fillHit(unitID, dEStep, time, primID, 0, em, 2);
      }
    } else {
      EcalBaseNumber theBaseNumber;
      int size = touch->GetHistoryDepth() + 1;
      if (theBaseNumber.getCapacity() < size)
        theBaseNumber.setSize(size);
      //Get name and copy numbers
      if (size > 1) {
        for (int ii = 0; ii < size; ii++) {
          theBaseNumber.addLevel(touch->GetVolume(ii)->GetName(), touch->GetReplicaNumber(ii));
        }
      }
      unitID = (eb ? (ebNumberingScheme_->getUnitID(theBaseNumber)) : (eeNumberingScheme_->getUnitID(theBaseNumber)));
      if (unitID > 0 && dEStep > 0.0) {
        auto local = touch->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
        auto ite = xtalMap_.find(lv);
        double crystalLength = ((ite == xtalMap_.end()) ? 230.0 : std::abs(ite->second));
        double crystalDepth =
            ((ite == xtalMap_.end()) ? 0.0 : (std::abs(0.5 * (ite->second) + (local.z() / CLHEP::mm))));
        double radl = aStep->GetPreStepPoint()->GetMaterial()->GetRadlen() / CLHEP::mm;
        bool flag = ((ite == xtalMap_.end()) ? true : (((ite->second) >= 0) ? true : false));
        auto depth = getDepth(flag, crystalDepth, radl);
        dEStep *= (getBirkL3(dEStep,
                             (aStep->GetStepLength() / CLHEP::cm),
                             aStep->GetPreStepPoint()->GetCharge(),
                             (aStep->GetPreStepPoint()->GetMaterial()->GetDensity() / (CLHEP::g / CLHEP::cm3))) *
                   curve_LY(crystalLength, crystalDepth));
        fillHit(unitID, dEStep, time, primID, depth, em, (eb ? 0 : 1));
      }
    }
  }

  if (allSteps_ != 0) {
    auto it = mapLV_.find(lv);
    if (it != mapLV_.end()) {
      double energy = aStep->GetTotalEnergyDeposit() / CLHEP::MeV;
      auto const touch = aStep->GetPreStepPoint()->GetTouchable();
      double time = aStep->GetTrack()->GetGlobalTime() / CLHEP::nanosecond;
      int trackId = aStep->GetTrack()->GetTrackID();
      int pdg = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
      double stepl = (aStep->GetStepLength() / CLHEP::cm);
      double xp = aStep->GetPreStepPoint()->GetPosition().x() / CLHEP::cm;
      double yp = aStep->GetPreStepPoint()->GetPosition().y() / CLHEP::cm;
      double zp = aStep->GetPreStepPoint()->GetPosition().z() / CLHEP::cm;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("Step") << "CaloSteppingAction: Volume " << lv->GetName() << " History "
                               << touch->GetHistoryDepth() << " Pointers " << aStep->GetPostStepPoint() << ":"
                               << aStep->GetTrack()->GetNextVolume() << ":" << aStep->IsLastStepInVolume() << " E "
                               << energy << " T " << time << " PDG " << pdg << " step " << stepl << " Position (" << xp
                               << ", " << yp << ", " << zp << ")";
#endif
      uint32_t copy = (allSteps_ < 0) ? 0 : unitID;
      if (((aStep->GetPostStepPoint() == nullptr) || (aStep->GetTrack()->GetNextVolume() == nullptr)) &&
          (aStep->IsLastStepInVolume())) {
        energy += (aStep->GetPreStepPoint()->GetKineticEnergy() / CLHEP::MeV);
      } else {
        time = aStep->GetPostStepPoint()->GetGlobalTime() / CLHEP::nanosecond;
        if (copy == 0)
          copy = (touch->GetHistoryDepth() < 1)
                     ? static_cast<uint32_t>(touch->GetReplicaNumber(0))
                     : static_cast<uint32_t>(touch->GetReplicaNumber(0) + 1000 * touch->GetReplicaNumber(1));
      }
      PassiveData key(std::make_tuple(lv, copy, trackId, pdg, time, energy, energy, stepl, xp, yp, zp));
      store_.push_back(key);
    }
  }
}

//================================================================ End of EVENT
void CaloSteppingAction::update(const EndOfEvent* evt) {
  ++count_;
  // Fill event input
  edm::LogVerbatim("Step") << "CaloSteppingAction: EndOfEvent " << (*evt)()->GetEventID();
}

void CaloSteppingAction::NaNTrap(const G4Step* aStep) const {
  auto currentPos = aStep->GetTrack()->GetPosition();
  double xyz = currentPos.x() + currentPos.y() + currentPos.z();
  auto currentMom = aStep->GetTrack()->GetMomentum();
  xyz += currentMom.x() + currentMom.y() + currentMom.z();

  if (edm::isNotFinite(xyz)) {
    auto pCurrentVol = aStep->GetPreStepPoint()->GetPhysicalVolume();
    auto& nameOfVol = pCurrentVol->GetName();
    throw cms::Exception("Unknown", "CaloSteppingAction")
        << " Corrupted Event - NaN detected in volume " << nameOfVol << "\n";
  }
}

uint32_t CaloSteppingAction::getDetIDHC(int det, int lay, int depth, const math::XYZVectorD& pos) const {
  HcalNumberingFromDDD::HcalID tmp = hcNumberingPS_.get()->unitID(det, lay, depth, pos);
#ifdef HcalNumberingTest
  auto id0 = hcNumberingScheme_.get()->getUnitID(tmp);
  HcalNumberingFromDDD::HcalID tmpO = hcNumbering_.get()->unitID(det, pos, depth, lay);
  auto idO = hcNumberingScheme_.get()->getUnitID(tmpO);
  std::string error = (id0 == idO) ? " ** OK **" : " ** ERROR **";
  edm::LogVerbatim("Step") << "Det ID " << HcalDetId(id0) << " Original " << HcalDetId(idO) << error;
#endif
  return (hcNumberingScheme_.get()->getUnitID(tmp));
}

void CaloSteppingAction::fillHit(uint32_t id, double dE, double time, int primID, uint16_t depth, double em, int flag) {
  CaloHitID currentID(id, time, primID, depth, timeSliceUnit_);
  double edepEM = (em ? dE : 0);
  double edepHAD = (em ? 0 : dE);
  std::pair<int, CaloHitID> evID = std::make_pair(eventID_, currentID);
  auto it = hitMap_[flag].find(evID);
  if (it != hitMap_[flag].end()) {
    (it->second).addEnergyDeposit(edepEM, edepHAD);
  } else {
    CaloGVHit aHit;
    aHit.setEventID(eventID_);
    aHit.setID(currentID);
    aHit.addEnergyDeposit(edepEM, edepHAD);
    hitMap_[flag][evID] = aHit;
  }
}

uint16_t CaloSteppingAction::getDepth(bool flag, double crystalDepth, double radl) const {
  uint16_t depth1 = (flag ? 0 : PCaloHit::kEcalDepthRefz);
  uint16_t depth2 = (uint16_t)floor(crystalDepth / radl);
  uint16_t depth = (((depth2 & PCaloHit::kEcalDepthMask) << PCaloHit::kEcalDepthOffset) | depth1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Step") << "CaloSteppingAction::getDepth radl " << radl << ":" << crystalDepth << " depth " << depth;
#endif
  return depth;
}

double CaloSteppingAction::curve_LY(double crystalLength, double crystalDepth) const {
  double weight = 1.;
  double dapd = crystalLength - crystalDepth;
  if (dapd >= -0.1 || dapd <= crystalLength + 0.1) {
    if (dapd <= 100.)
      weight = 1.0 + slopeLY_ - dapd * 0.01 * slopeLY_;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Step") << "CaloSteppingAction::curve_LY " << crystalDepth << ":" << crystalLength << ":" << dapd
                             << ":" << weight;
#endif
  } else {
    edm::LogWarning("Step") << "CaloSteppingAction: light coll curve : wrong "
                            << "distance to APD " << dapd << " crlength = " << crystalLength
                            << " crystal Depth = " << crystalDepth << " weight = " << weight;
  }
  return weight;
}

double CaloSteppingAction::getBirkL3(double dEStep, double step, double charge, double density) const {
  double weight = 1.;
  if (charge != 0. && step > 0.) {
    double dedx = dEStep / step;
    double rkb = birkC1EC_ / density;
    if (dedx > 0) {
      weight = 1. - birkSlopeEC_ * log(rkb * dedx);
      if (weight < birkCutEC_)
        weight = birkCutEC_;
      else if (weight > 1.)
        weight = 1.;
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Step") << "CaloSteppingAction::getBirkL3 Charge " << charge << " dE/dx " << dedx << " Birk Const "
                             << rkb << " Weight = " << weight << " dE " << dEStep << " step " << step;
#endif
  }
  return weight;
}

double CaloSteppingAction::getBirkHC(double dEStep, double step, double charge, double density) const {
  double weight = 1.;
  if (charge != 0. && step > 0.) {
    double dedx = dEStep / step;
    double rkb = birkC1HC_ / density;
    double c = birkC2HC_ * rkb * rkb;
    if (std::abs(charge) >= 2.)
      rkb /= birkC3HC_;
    weight = 1. / (1. + rkb * dedx + c * dedx * dedx);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Step") << "CaloSteppingAction::getBirkHC Charge " << charge << " dE/dx " << dedx << " Birk Const "
                             << rkb << ", " << c << " Weight = " << weight << " dE " << dEStep;
#endif
  }
  return weight;
}

void CaloSteppingAction::saveHits(int type) {
  edm::LogVerbatim("Step") << "CaloSteppingAction:: saveHits for type " << type << " with " << hitMap_[type].size()
                           << " hits";
  slave_[type].get()->ReserveMemory(hitMap_[type].size());
  for (auto const& hit : hitMap_[type]) {
    slave_[type].get()->processHits(hit.second.getUnitID(),
                                    0.001 * hit.second.getEM(),
                                    0.001 * hit.second.getHadr(),
                                    hit.second.getTimeSlice(),
                                    hit.second.getTrackID(),
                                    hit.second.getDepth());
  }
}
