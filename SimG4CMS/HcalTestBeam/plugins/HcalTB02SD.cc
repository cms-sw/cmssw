// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02SD
//
// Implementation:
//     Sensitive Detector class for Hcal Test Beam 2002 detectors
//
// Original Author:
//         Created:  Sun 21 10:14:34 CEST 2006
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "HcalTB02SD.h"
#include "HcalTB02HcalNumberingScheme.h"
#include "HcalTB02XtalNumberingScheme.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include "G4SystemOfUnits.hh"

//#define EDM_ML_DEBUG

//
// constructors and destructor
//

HcalTB02SD::HcalTB02SD(const std::string& name,
                       const edm::EventSetup& es,
                       const SensitiveDetectorCatalog& clg,
                       edm::ParameterSet const& p,
                       const SimTrackManager* manager)
    : CaloSD(name, clg, p, manager) {
  numberingScheme_.reset(nullptr);
  edm::ParameterSet m_SD = p.getParameter<edm::ParameterSet>("HcalTB02SD");
  useBirk_ = m_SD.getUntrackedParameter<bool>("UseBirkLaw", false);
  birk1_ = m_SD.getUntrackedParameter<double>("BirkC1", 0.013) * (CLHEP::g / (CLHEP::MeV * CLHEP::cm2));
  birk2_ = m_SD.getUntrackedParameter<double>("BirkC2", 0.0568);
  birk3_ = m_SD.getUntrackedParameter<double>("BirkC3", 1.75);
  useWeight_ = true;

  HcalTB02NumberingScheme* scheme = nullptr;
  if (name == "EcalHitsEB") {
    scheme = dynamic_cast<HcalTB02NumberingScheme*>(new HcalTB02XtalNumberingScheme());
    useBirk_ = false;
  } else if (name == "HcalHits") {
    scheme = dynamic_cast<HcalTB02NumberingScheme*>(new HcalTB02HcalNumberingScheme());
    useWeight_ = false;
  } else {
    edm::LogWarning("HcalTBSim") << "HcalTB02SD: ReadoutName " << name << " not supported\n";
  }

  if (scheme)
    setNumberingScheme(scheme);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "***************************************************\n"
                                << "*                                                 *\n"
                                << "* Constructing a HcalTB02SD  with name " << GetName() << "\n"
                                << "*                                                 *\n"
                                << "***************************************************";
  edm::LogVerbatim("HcalTBSim") << "HcalTB02SD:: Use of Birks law is set to      " << useBirk_
                                << "        with three constants kB = " << birk1_ << ", C1 = " << birk2_
                                << ", C2 = " << birk3_;
#endif
  // Get pointers to HcalTB02Parameters
  edm::ESHandle<HcalTB02Parameters> hdc;
  es.get<IdealGeometryRecord>().get(name, hdc);
  if (hdc.isValid()) {
    hcalTB02Parameters_ = hdc.product();
  } else {
    throw cms::Exception("Unknown", "HcalTB02SD") << "Cannot find HcalTB02Parameters for " << name << "\n";
  }
}

HcalTB02SD::~HcalTB02SD() {}

//
// member functions
//

double HcalTB02SD::getEnergyDeposit(const G4Step* aStep) {
  auto const preStepPoint = aStep->GetPreStepPoint();
  std::string nameVolume = static_cast<std::string>(preStepPoint->GetPhysicalVolume()->GetName());

  // take into account light collection curve for crystals
  double weight = 1.;
  if (useWeight_)
    weight *= curve_LY(nameVolume, preStepPoint);
  if (useBirk_)
    weight *= getAttenuation(aStep, birk1_, birk2_, birk3_);
  double edep = aStep->GetTotalEnergyDeposit() * weight;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB02SD:: " << nameVolume << " Light Collection Efficiency " << weight
                                << " Weighted Energy Deposit " << edep / CLHEP::MeV << " MeV";
#endif
  return edep;
}

uint32_t HcalTB02SD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme_ == nullptr ? 0 : (uint32_t)(numberingScheme_->getUnitID(aStep)));
}

void HcalTB02SD::setNumberingScheme(HcalTB02NumberingScheme* scheme) {
  if (scheme != nullptr) {
    edm::LogVerbatim("HcalTBSim") << "HcalTB02SD: updates numbering scheme for " << GetName();
    numberingScheme_.reset(scheme);
  }
}

double HcalTB02SD::curve_LY(const std::string& nameVolume, const G4StepPoint* stepPoint) {
  double weight = 1.;
  G4ThreeVector localPoint = setToLocal(stepPoint->GetPosition(), stepPoint->GetTouchable());
  double crlength = crystalLength(nameVolume);
  double dapd = 0.5 * crlength - localPoint.z();
  if (dapd >= -0.1 || dapd <= crlength + 0.1) {
    if (dapd <= 100.)
      weight = 1.05 - dapd * 0.0005;
  } else {
    edm::LogWarning("HcalTBSim") << "HcalTB02SD: light coll curve : wrong "
                                 << "distance to APD " << dapd << " crlength = " << crlength
                                 << " crystal name = " << nameVolume << " z of localPoint = " << localPoint.z()
                                 << " take weight = " << weight;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB02SD, light coll curve : " << dapd << " crlength = " << crlength
                                << " crystal name = " << nameVolume << " z of localPoint = " << localPoint.z()
                                << " take weight = " << weight;
#endif
  return weight;
}

double HcalTB02SD::crystalLength(const std::string& name) {
  double length = 230.;
  std::map<std::string, double>::const_iterator it = hcalTB02Parameters_->lengthMap_.find(name);
  if (it != hcalTB02Parameters_->lengthMap_.end())
    length = it->second;
  return length;
}
