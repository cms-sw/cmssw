///////////////////////////////////////////////////////////////////////////////
// File: HcalTB06BeamSD.cc
// Description: Sensitive Detector class for beam counters in TB06 setup
///////////////////////////////////////////////////////////////////////////////

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimDataFormats/HcalTestBeam/interface/HcalTestBeamNumbering.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HcalTB06BeamSD.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Material.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

HcalTB06BeamSD::HcalTB06BeamSD(const std::string& name,
                               const edm::EventSetup& es,
                               const SensitiveDetectorCatalog& clg,
                               edm::ParameterSet const& p,
                               const SimTrackManager* manager)
    : CaloSD(name, clg, p, manager) {
  // Values from NIM 80 (1970) 239-244: as implemented in Geant3
  edm::ParameterSet m_HC = p.getParameter<edm::ParameterSet>("HcalTB06BeamSD");
  useBirk_ = m_HC.getParameter<bool>("UseBirkLaw");
  birk1_ = m_HC.getParameter<double>("BirkC1") * (CLHEP::g / (CLHEP::MeV * CLHEP::cm2));
  birk2_ = m_HC.getParameter<double>("BirkC2");
  birk3_ = m_HC.getParameter<double>("BirkC3");

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB06BeamSD:: Use of Birks law is set to " << useBirk_
                                << "  with three constants kB = " << birk1_ << ", C1 = " << birk2_
                                << ", C2 = " << birk3_;
#endif

  // Get pointers to HcalTB06BeamParameters
  edm::ESHandle<HcalTB06BeamParameters> hdc;
  es.get<IdealGeometryRecord>().get(hdc);
  if (hdc.isValid()) {
    hcalBeamPar_ = hdc.product();
  } else {
    throw cms::Exception("Unknown", "HCalTB06BeamSD") << "Cannot find HcalTB06BeamParameters";
  }
}

HcalTB06BeamSD::~HcalTB06BeamSD() {}

double HcalTB06BeamSD::getEnergyDeposit(const G4Step* aStep) {
  double destep = aStep->GetTotalEnergyDeposit();
  double weight = 1;
  if (useBirk_ && aStep->GetPreStepPoint()->GetMaterial()->GetName() == static_cast<G4String>(hcalBeamPar_->material_))
    weight *= getAttenuation(aStep, birk1_, birk2_, birk3_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB06BeamSD: Detector "
                                << aStep->GetPreStepPoint()->GetTouchable()->GetVolume()->GetName() << " weight "
                                << weight;
#endif
  return weight * destep;
}

uint32_t HcalTB06BeamSD::setDetUnitId(const G4Step* aStep) {
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  std::string name = static_cast<std::string>(preStepPoint->GetPhysicalVolume()->GetName());

  int det = 1;
  int lay = 0, x = 0, y = 0;
  if (!isItWireChamber(name)) {
    lay = (touch->GetReplicaNumber(0));
  } else {
    det = 2;
    lay = (touch->GetReplicaNumber(1));
    G4ThreeVector localPoint = setToLocal(preStepPoint->GetPosition(), touch);
    x = (int)(localPoint.x() / (0.2 * mm));
    y = (int)(localPoint.y() / (0.2 * mm));
  }

  return HcalTestBeamNumbering::packIndex(det, lay, x, y);
}

bool HcalTB06BeamSD::isItWireChamber(const std::string& name) {
  std::vector<std::string>::const_iterator it = hcalBeamPar_->wchambers_.begin();
  for (; it != hcalBeamPar_->wchambers_.end(); it++)
    if (name == *it)
      return true;
  return false;
}
