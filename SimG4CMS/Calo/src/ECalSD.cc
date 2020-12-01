///////////////////////////////////////////////////////////////////////////////
// File: ECalSD.cc
// Description: Sensitive Detector class for electromagnetic calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/ECalSD.h"
#include "SimG4CMS/Calo/interface/EcalDumpGeometry.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalPreshowerNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/ESTBNumberingScheme.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DD4hep/Filter.h"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"

#include <algorithm>

//#define EDM_ML_DEBUG

using namespace geant_units::operators;

template <class T>
bool any(const std::vector<T>& v, const T& what) {
  return std::find(v.begin(), v.end(), what) != v.end();
}

ECalSD::ECalSD(const std::string& name,
               const edm::EventSetup& es,
               const SensitiveDetectorCatalog& clg,
               edm::ParameterSet const& p,
               const SimTrackManager* manager)
    : CaloSD(name,
             es,
             clg,
             p,
             manager,
             (float)(p.getParameter<edm::ParameterSet>("ECalSD").getParameter<double>("TimeSliceUnit")),
             p.getParameter<edm::ParameterSet>("ECalSD").getParameter<bool>("IgnoreTrackID")),
      ecalSimParameters_(nullptr),
      numberingScheme_(nullptr) {
  //   static SimpleConfigurable<bool>   on1(false,  "ECalSD:UseBirkLaw");
  //   static SimpleConfigurable<double> bk1(0.00463,"ECalSD:BirkC1");
  //   static SimpleConfigurable<double> bk2(-0.03,  "ECalSD:BirkC2");
  //   static SimpleConfigurable<double> bk3(1.0,    "ECalSD:BirkC3");
  // Values from NIM A484 (2002) 239-244: as implemented in Geant3
  //   useBirk          = on1.value();
  //   birk1            = bk1.value()*(g/(MeV*cm2));
  //   birk2            = bk2.value()*(g/(MeV*cm2))*(g/(MeV*cm2));
  edm::ParameterSet m_EC = p.getParameter<edm::ParameterSet>("ECalSD");
  useBirk = m_EC.getParameter<bool>("UseBirkLaw");
  useBirkL3 = m_EC.getParameter<bool>("BirkL3Parametrization");
  double bunit = (CLHEP::g / (CLHEP::MeV * CLHEP::cm2));
  birk1 = m_EC.getParameter<double>("BirkC1") * bunit;
  birk2 = m_EC.getParameter<double>("BirkC2");
  birk3 = m_EC.getParameter<double>("BirkC3");
  birkSlope = m_EC.getParameter<double>("BirkSlope");
  birkCut = m_EC.getParameter<double>("BirkCut");
  slopeLY = m_EC.getParameter<double>("SlopeLightYield");
  storeTrack = m_EC.getParameter<bool>("StoreSecondary");
  crystalMat = m_EC.getUntrackedParameter<std::string>("XtalMat", "E_PbWO4");
  bool isItTB = m_EC.getUntrackedParameter<bool>("TestBeam", false);
  bool nullNS = m_EC.getUntrackedParameter<bool>("NullNumbering", false);
  storeRL = m_EC.getUntrackedParameter<bool>("StoreRadLength", false);
  scaleRL = m_EC.getUntrackedParameter<double>("ScaleRadLength", 1.0);
  int dumpGeom = m_EC.getUntrackedParameter<int>("DumpGeometry", 0);

  //Changes for improved timing simulation
  storeLayerTimeSim = m_EC.getUntrackedParameter<bool>("StoreLayerTimeSim", false);

  ageingWithSlopeLY = m_EC.getUntrackedParameter<bool>("AgeingWithSlopeLY", false);
  if (ageingWithSlopeLY)
    ageing.setLumies(p.getParameter<edm::ParameterSet>("ECalSD").getParameter<double>("DelivLuminosity"),
                     p.getParameter<edm::ParameterSet>("ECalSD").getParameter<double>("InstLuminosity"));

  edm::ESHandle<EcalSimulationParameters> esp;
  es.get<IdealGeometryRecord>().get(name, esp);
  if (esp.isValid()) {
    ecalSimParameters_ = esp.product();
  } else {
    edm::LogError("EcalSim") << "ECalSD : Cannot find EcalSimulationParameters for " << name;
    throw cms::Exception("Unknown", "ECalSD") << "Cannot find EcalSimulationParameters for " << name << "\n";
  }

  // Use of Weight
  useWeight = ecalSimParameters_->useWeight_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "ECalSD:: useWeight " << useWeight;
#endif
  depth1Name = ecalSimParameters_->depth1Name_;
  depth2Name = ecalSimParameters_->depth2Name_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "Names (Depth 1):" << depth1Name << " (Depth 2):" << depth2Name << std::endl;
#endif
  int type(-1);
  bool dump(false);
  EcalNumberingScheme* scheme = nullptr;
  if (nullNS) {
    scheme = nullptr;
  } else if (name == "EcalHitsEB") {
    scheme = dynamic_cast<EcalNumberingScheme*>(new EcalBarrelNumberingScheme());
    type = 0;
    dump = ((dumpGeom % 10) > 0);
  } else if (name == "EcalHitsEE") {
    scheme = dynamic_cast<EcalNumberingScheme*>(new EcalEndcapNumberingScheme());
    type = 1;
    dump = (((dumpGeom / 10) % 10) > 0);
  } else if (name == "EcalHitsES") {
    if (isItTB)
      scheme = dynamic_cast<EcalNumberingScheme*>(new ESTBNumberingScheme());
    else
      scheme = dynamic_cast<EcalNumberingScheme*>(new EcalPreshowerNumberingScheme());
    useWeight = false;
    type = 2;
    dump = (((dumpGeom / 100) % 10) > 0);
  } else {
    edm::LogWarning("EcalSim") << "ECalSD: ReadoutName not supported";
  }
  int type0 = dumpGeom / 1000;
  type += (10 * type0);

  if (scheme)
    setNumberingScheme(scheme);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "Constructing a ECalSD  with name " << GetName();
#endif
  if (useWeight) {
    edm::LogVerbatim("EcalSim") << "ECalSD:: Use of Birks law is set to " << useBirk
                                << " with three constants kB = " << birk1 / bunit << ", C1 = " << birk2
                                << ", C2 = " << birk3 << "\n         Use of L3 parametrization " << useBirkL3
                                << " with slope " << birkSlope << " and cut off " << birkCut << "\n"
                                << "         Slope for Light yield is set to " << slopeLY;
  } else {
    edm::LogVerbatim("EcalSim") << "ECalSD:: energy deposit is not corrected "
                                << " by Birk or light yield curve";
  }

  edm::LogVerbatim("EcalSim") << "ECalSD:: Suppression Flag " << suppressHeavy << "\tprotons below "
                              << kmaxProton / CLHEP::MeV << " MeV,\tneutrons below " << kmaxNeutron / CLHEP::MeV
                              << " MeV,\tions below " << kmaxIon / CLHEP::MeV << " MeV \n\tDepth1 Name = " << depth1Name
                              << "\tDepth2 Name = " << depth2Name << "\n\tstoreRL " << storeRL << ":" << scaleRL
                              << "\tstoreLayerTimeSim " << storeLayerTimeSim << "\n\ttime Granularity "
                              << p.getParameter<edm::ParameterSet>("ECalSD").getParameter<double>("TimeSliceUnit")
                              << " ns";
  if (useWeight)
    initMap();
#ifdef plotDebug
  edm::Service<TFileService> tfile;
  if (tfile.isAvailable()) {
    TFileDirectory ecDir = tfile->mkdir("ProfileFromECalSD");
    static const std::string ctype[4] = {"EB", "EBref", "EE", "EERef"};
    for (int k = 0; k < 4; ++k) {
      std::string name = "ECLL_" + ctype[k];
      std::string title = "Local vs Global for " + ctype[k];
      double xmin = (k > 1) ? 3000.0 : 1000.0;
      g2L_[k] = ecDir.make<TH2F>(name.c_str(), title.c_str(), 100, xmin, xmin + 1000., 100, 0.0, 3000.);
    }
  } else {
    for (int k = 0; k < 4; ++k)
      g2L_[k] = 0;
  }
#endif
  if (dump) {
    const auto& lvNames = clg.logicalNames(name);
    EcalDumpGeometry geom(lvNames, depth1Name, depth2Name, type);
    geom.update();
  }
}

ECalSD::~ECalSD() { delete numberingScheme_; }

double ECalSD::getEnergyDeposit(const G4Step* aStep) {
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4Track* theTrack = aStep->GetTrack();
  double edep = aStep->GetTotalEnergyDeposit();

  // take into account light collection curve for crystals
  double weight = 1.;
  if (suppressHeavy) {
    TrackInformation* trkInfo = (TrackInformation*)(theTrack->GetUserInformation());
    if (trkInfo) {
      int pdg = theTrack->GetDefinition()->GetPDGEncoding();
      if (!(trkInfo->isPrimary())) {  // Only secondary particles
        double ke = theTrack->GetKineticEnergy();
        if (((pdg / 1000000000 == 1 && ((pdg / 10000) % 100) > 0 && ((pdg / 10) % 100) > 0)) && (ke < kmaxIon))
          weight = 0;
        if ((pdg == 2212) && (ke < kmaxProton))
          weight = 0;
        if ((pdg == 2112) && (ke < kmaxNeutron))
          weight = 0;
      }
    }
  }
  const G4LogicalVolume* lv = preStepPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();
  double wt1 = 1.0;
  if (useWeight && !any(noWeight, lv)) {
    weight *= curve_LY(lv);
    if (useBirk) {
      if (useBirkL3)
        weight *= getBirkL3(aStep);
      else
        weight *= getAttenuation(aStep, birk1, birk2, birk3);
    }
    wt1 = getResponseWt(theTrack);
  }
  edep *= weight * wt1;
  // Russian Roulette
  double wt2 = theTrack->GetWeight();
  if (wt2 > 0.0) {
    edep *= wt2;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << lv->GetName() << " " << dd4hep::dd::noNamespace(lv->GetName())
                              << " Light Collection Efficiency " << weight << ":" << wt1 << " wt2= " << wt2
                              << " Weighted Energy Deposit " << edep / CLHEP::MeV << " MeV at "
                              << preStepPoint->GetPosition();
#endif
  return edep;
}

double ECalSD::EnergyCorrected(const G4Step& step, const G4Track* track) {
  double edep = step.GetTotalEnergyDeposit();
  const G4StepPoint* hitPoint = step.GetPreStepPoint();
  const G4LogicalVolume* lv = hitPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();

  if (useWeight && !any(noWeight, lv)) {
    currentLocalPoint = setToLocal(hitPoint->GetPosition(), hitPoint->GetTouchable());
    auto ite = xtalLMap.find(lv);
    crystalLength = (ite == xtalLMap.end()) ? 230._mm : std::abs(ite->second);
    crystalDepth = (ite == xtalLMap.end()) ? 0.0 : (std::abs(0.5 * (ite->second) + currentLocalPoint.z()));
    edep *= curve_LY(lv) * getResponseWt(track);
  }
  return edep;
}

int ECalSD::getTrackID(const G4Track* aTrack) {
  int primaryID(0);
  if (storeTrack && depth > 0) {
    forceSave = true;
    primaryID = aTrack->GetTrackID();
  } else {
    primaryID = CaloSD::getTrackID(aTrack);
  }
  return primaryID;
}

uint16_t ECalSD::getDepth(const G4Step* aStep) {
  // this method should be called first at a step
  const G4StepPoint* hitPoint = aStep->GetPreStepPoint();
  currentLocalPoint = setToLocal(hitPoint->GetPosition(), hitPoint->GetTouchable());
  const G4LogicalVolume* lv = hitPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();

  auto ite = xtalLMap.find(lv);
  crystalLength = (ite == xtalLMap.end()) ? 230._mm : std::abs(ite->second);
  crystalDepth = (ite == xtalLMap.end()) ? 0.0 : (std::abs(0.5 * (ite->second) + currentLocalPoint.z()));
  depth = any(useDepth1, lv) ? 1 : (any(useDepth2, lv) ? 2 : 0);
  uint16_t depth1(0), depth2(0);
  if (storeRL) {
    depth1 = (ite == xtalLMap.end()) ? 0 : (((ite->second) >= 0) ? 0 : PCaloHit::kEcalDepthRefz);
    depth2 = getRadiationLength(hitPoint, lv);
    depth |= (((depth2 & PCaloHit::kEcalDepthMask) << PCaloHit::kEcalDepthOffset) | depth1);
  } else if (storeLayerTimeSim) {
    depth2 = getLayerIDForTimeSim();
    depth |= ((depth2 & PCaloHit::kEcalDepthMask) << PCaloHit::kEcalDepthOffset);
  }
#ifdef EDM_ML_DEBUG
  if (isXtal(lv))
    edm::LogVerbatim("EcalSimX") << "ECalSD::Volume " << lv->GetName() << " DetId " << std::hex << setDetUnitId(aStep)
                                 << std::dec << " Global " << (hitPoint->GetPosition()).rho() << ":"
                                 << (hitPoint->GetPosition()).z() << " Local Z " << currentLocalPoint.z() << " Depth "
                                 << crystalDepth;
  edm::LogVerbatim("EcalSim") << "ECalSD::Depth " << std::hex << depth1 << ":" << depth2 << ":" << depth << std::dec
                              << " L " << (ite == xtalLMap.end()) << ":" << ite->second << " local "
                              << currentLocalPoint << " Crystal length " << crystalLength << ":" << crystalDepth;
#endif
  return depth;
}

uint16_t ECalSD::getRadiationLength(const G4StepPoint* hitPoint, const G4LogicalVolume* lv) {
  uint16_t thisX0 = 0;
  if (useWeight) {
    double radl = hitPoint->GetMaterial()->GetRadlen();
    thisX0 = (uint16_t)floor(scaleRL * crystalDepth / radl);
#ifdef plotDebug
    const std::string& lvname = dd4hep::dd::noNamespace(lv->GetName());
    int k1 = (lvname.find("EFRY") != std::string::npos) ? 2 : 0;
    int k2 = (lvname.find("refl") != std::string::npos) ? 1 : 0;
    int kk = k1 + k2;
    double rz = (k1 == 0) ? (hitPoint->GetPosition()).rho() : std::abs((hitPoint->GetPosition()).z());
    edm::LogVerbatim("EcalSim") << lvname << " # " << k1 << ":" << k2 << ":" << kk << " rz " << rz << " D " << thisX0;
    g2L_[kk]->Fill(rz, thisX0);
#endif
#ifdef EDM_ML_DEBUG
    G4ThreeVector localPoint = setToLocal(hitPoint->GetPosition(), hitPoint->GetTouchable());
    edm::LogVerbatim("EcalSim") << lv->GetName() << " " << dd4hep::dd::noNamespace(lv->GetName()) << " Global "
                                << hitPoint->GetPosition() << ":" << (hitPoint->GetPosition()).rho() << " Local "
                                << localPoint << " Crystal Length " << crystalLength << " Radl " << radl
                                << " crystalDepth " << crystalDepth << " Index " << thisX0 << " : "
                                << getLayerIDForTimeSim();
#endif
  }
  return thisX0;
}

uint16_t ECalSD::getLayerIDForTimeSim() {
  const double invLayerSize = 0.1;  //layer size in 1/mm
  return (int)crystalDepth * invLayerSize;
}

uint32_t ECalSD::setDetUnitId(const G4Step* aStep) {
  if (numberingScheme_ == nullptr) {
    return EBDetId(1, 1)();
  } else {
    getBaseNumber(aStep);
    return numberingScheme_->getUnitID(theBaseNumber);
  }
}

void ECalSD::setNumberingScheme(EcalNumberingScheme* scheme) {
  if (scheme != nullptr) {
    edm::LogVerbatim("EcalSim") << "EcalSD: updates numbering scheme for " << GetName();
    if (numberingScheme_)
      delete numberingScheme_;
    numberingScheme_ = scheme;
  }
}

void ECalSD::initMap() {
  std::vector<const G4LogicalVolume*> lvused;
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  std::map<const std::string, const G4LogicalVolume*> nameMap;
  for (auto lvi = lvs->begin(), lve = lvs->end(); lvi != lve; ++lvi)
    nameMap.emplace(dd4hep::dd::noNamespace((*lvi)->GetName()), *lvi);

  for (unsigned int it = 0; it < ecalSimParameters_->lvNames_.size(); ++it) {
    const std::string& matname = ecalSimParameters_->matNames_[it];
    const std::string& lvname = ecalSimParameters_->lvNames_[it];
    const G4LogicalVolume* lv = nameMap[lvname];
    int ibec = (lvname.find("EFRY") == std::string::npos) ? 0 : 1;
    int iref = (lvname.find("refl") == std::string::npos) ? 0 : 1;
    int type = (ibec + iref == 1) ? 1 : -1;
    if (depth1Name != " ") {
      if (strncmp(lvname.c_str(), depth1Name.c_str(), 4) == 0) {
        if (!any(useDepth1, lv)) {
          useDepth1.push_back(lv);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EcalSim") << "ECalSD::initMap Logical Volume " << lvname << " in Depth 1 volume list";
#endif
        }
        const G4LogicalVolume* lvr = nameMap[lvname + "_refl"];
        if (lvr != nullptr && !any(useDepth1, lvr)) {
          useDepth1.push_back(lvr);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EcalSim") << "ECalSD::initMap Logical Volume " << lvname << "_refl"
                                      << " in Depth 1 volume list";
#endif
        }
      }
    }
    if (depth2Name != " ") {
      if (strncmp(lvname.c_str(), depth2Name.c_str(), 4) == 0) {
        if (!any(useDepth2, lv)) {
          useDepth2.push_back(lv);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EcalSim") << "ECalSD::initMap Logical Volume " << lvname << " in Depth 2 volume list";
#endif
        }
        const G4LogicalVolume* lvr = nameMap[lvname + "_refl"];
        if (lvr != nullptr && !any(useDepth2, lvr)) {
          useDepth2.push_back(lvr);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EcalSim") << "ECalSD::initMap Logical Volume " << lvname << "_refl"
                                      << " in Depth 2 volume list";
#endif
        }
      }
    }
    if (lv != nullptr) {
      if (crystalMat.size() == matname.size() && !strcmp(crystalMat.c_str(), matname.c_str())) {
        if (!any(lvused, lv)) {
          lvused.push_back(lv);
          double dz = ecalSimParameters_->dzs_[it];
          xtalLMap.insert(std::pair<const G4LogicalVolume*, double>(lv, dz * type));
          lv = nameMap[lvname + "_refl"];
          if (lv != nullptr) {
            xtalLMap.insert(std::pair<const G4LogicalVolume*, double>(lv, -dz * type));
          }
        }
      } else {
        if (!any(noWeight, lv)) {
          noWeight.push_back(lv);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EcalSim") << "ECalSD::initMap Logical Volume " << lvname << " Material " << matname
                                      << " in noWeight list";
#endif
        }
        lv = nameMap[lvname];
        if (lv != nullptr && !any(noWeight, lv)) {
          noWeight.push_back(lv);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EcalSim") << "ECalSD::initMap Logical Volume " << lvname << " Material " << matname
                                      << " in noWeight list";
#endif
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "ECalSD: Length Table:";
  int i = 0;
  for (auto ite : xtalLMap) {
    std::string name("Unknown");
    if (ite.first != nullptr)
      name = dd4hep::dd::noNamespace((ite.first)->GetName());
    edm::LogVerbatim("EcalSim") << " " << i << " " << ite.first << " " << name << " L = " << ite.second;
    ++i;
  }
#endif
}

double ECalSD::curve_LY(const G4LogicalVolume* lv) {
  double weight = 1.;
  if (ageingWithSlopeLY) {
    //position along the crystal in mm from 0 to 230 (in EB)
    if (crystalDepth >= -0.1 || crystalDepth <= crystalLength + 0.1)
      weight = ageing.calcLightCollectionEfficiencyWeighted(currentID.unitID(), crystalDepth / crystalLength);
  } else {
    double dapd = crystalLength - crystalDepth;
    if (dapd >= -0.1 || dapd <= crystalLength + 0.1) {
      if (dapd <= 100.)
        weight = 1.0 + slopeLY - dapd * 0.01 * slopeLY;
    } else {
      edm::LogWarning("EcalSim") << "ECalSD: light coll curve : wrong distance "
                                 << "to APD " << dapd << " crlength = " << crystalLength << ":" << crystalDepth
                                 << " crystal name = " << lv->GetName() << " " << dd4hep::dd::noNamespace(lv->GetName())
                                 << " z of localPoint = " << currentLocalPoint.z() << " take weight = " << weight;
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "ECalSD: light coll curve : crlength = " << crystalLength << " Depth " << crystalDepth
                              << " crystal name = " << lv->GetName() << " " << dd4hep::dd::noNamespace(lv->GetName())
                              << " z of localPoint = " << currentLocalPoint.z() << " take weight = " << weight;
#endif
  return weight;
}

void ECalSD::getBaseNumber(const G4Step* aStep) {
  theBaseNumber.reset();
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int theSize = touch->GetHistoryDepth() + 1;
  if (theBaseNumber.getCapacity() < theSize)
    theBaseNumber.setSize(theSize);
  //Get name and copy numbers
  if (theSize > 1) {
    for (int ii = 0; ii < theSize; ii++) {
      std::string_view name = dd4hep::dd::noNamespace(touch->GetVolume(ii)->GetName());
      theBaseNumber.addLevel(std::string(name), touch->GetReplicaNumber(ii));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EcalSim") << "ECalSD::getBaseNumber(): Adding level " << ii << ": " << name << "["
                                  << touch->GetReplicaNumber(ii) << "]";
#endif
    }
  }
}

double ECalSD::getBirkL3(const G4Step* aStep) {
  double weight = 1.;
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  double charge = preStepPoint->GetCharge();

  if (charge != 0. && aStep->GetStepLength() > 0.) {
    const G4Material* mat = preStepPoint->GetMaterial();
    double density = mat->GetDensity();
    double dedx = aStep->GetTotalEnergyDeposit() / aStep->GetStepLength();
    double rkb = birk1 / density;
    if (dedx > 0) {
      weight = 1. - birkSlope * log(rkb * dedx);
      if (weight < birkCut)
        weight = birkCut;
      else if (weight > 1.)
        weight = 1.;
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalSim") << "ECalSD::getBirkL3 in " << dd4hep::dd::noNamespace(mat->GetName()) << " Charge "
                                << charge << " dE/dx " << dedx << " Birk Const " << rkb << " Weight = " << weight
                                << " dE " << aStep->GetTotalEnergyDeposit();
#endif
  }
  return weight;
}

bool ECalSD::isXtal(const G4LogicalVolume* lv) { return (xtalLMap.find(lv) != xtalLMap.end()); }
