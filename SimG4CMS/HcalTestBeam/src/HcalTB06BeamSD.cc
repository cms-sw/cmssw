///////////////////////////////////////////////////////////////////////////////
// File: HcalTB06BeamSD.cc
// Description: Sensitive Detector class for beam counters in TB06 setup
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/HcalTestBeam/interface/HcalTB06BeamSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimDataFormats/HcalTestBeam/interface/HcalTestBeamNumbering.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Material.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

HcalTB06BeamSD::HcalTB06BeamSD(const G4String& name, const DDCompactView & cpv,
			       const SensitiveDetectorCatalog & clg,
			       edm::ParameterSet const & p, 
			       const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager) {

  // Values from NIM 80 (1970) 239-244: as implemented in Geant3
  edm::ParameterSet m_HC = p.getParameter<edm::ParameterSet>("HcalTB06BeamSD");
  useBirk    = m_HC.getParameter<bool>("UseBirkLaw");
  birk1      = m_HC.getParameter<double>("BirkC1")*(g/(MeV*cm2));
  birk2      = m_HC.getParameter<double>("BirkC2");
  birk3      = m_HC.getParameter<double>("BirkC3");

  edm::LogInfo("HcalTB06BeamSD") << "HcalTB06BeamSD:: Use of Birks law is set to " 
			    << useBirk << "  with three constants kB = "
			    << birk1 << ", C1 = " <<birk2 << ", C2 = " <<birk3;

  std::string attribute, value;

  // Wire Chamber volume names
  attribute = "Volume";
  value     = "WireChamber";
  DDSpecificsMatchesValueFilter filter1{DDValue(attribute,value,0)};
  DDFilteredView fv1(cpv,filter1);
  wcNames = getNames(fv1);
  edm::LogInfo("HcalTB06BeamSD") 
    << "HcalTB06BeamSD:: Names to be tested for " 
    << attribute << " = " << value << ": " << wcNames.size() << " paths";
  for (unsigned int i=0; i<wcNames.size(); i++)
    edm::LogInfo("HcalTB06BeamSD") << "HcalTB06BeamSD:: (" << i << ") " 
				   << wcNames[i];

  //Material list for scintillator detector
  attribute = "ReadOutName";
  DDSpecificsMatchesValueFilter filter2{DDValue(attribute,name,0)};
  DDFilteredView fv2(cpv,filter2);
  bool dodet = fv2.firstChild();

  std::vector<G4String> matNames;
  std::vector<int>      nocc;
  while (dodet) {
    const DDLogicalPart & log = fv2.logicalPart();
    matName = log.material().name().name();
    bool notIn = true;
    for (unsigned int i=0; i<matNames.size(); i++) {
      if (matName == matNames[i]) {notIn = false; nocc[i]++;}
    }
    if (notIn) {
      matNames.push_back(matName);
      nocc.push_back(0);
    }
    dodet = fv2.next();
  }
  if (!matNames.empty()) {
    matName = matNames[0];
    int occ = nocc[0];
    for (unsigned int i = 0; i < matNames.size(); i++) {
      if (nocc[i] > occ) {
	occ     = nocc[i];
	matName = matNames[i];
      }
    }
  } else {
    matName = "Not Found";
  }

  edm::LogInfo("HcalTB06BeamSD") 
    << "HcalTB06BeamSD: Material name for " 
    << attribute << " = " << name << ":" << matName;
}

HcalTB06BeamSD::~HcalTB06BeamSD() {}

double HcalTB06BeamSD::getEnergyDeposit(G4Step* aStep) {

  double destep = aStep->GetTotalEnergyDeposit();
  double weight = 1;
  if (useBirk && matName == aStep->GetPreStepPoint()->GetMaterial()->GetName()) {
    weight *= getAttenuation(aStep, birk1, birk2, birk3);
  }
  LogDebug("HcalTB06BeamSD") 
    << "HcalTB06BeamSD: Detector " 
    << aStep->GetPreStepPoint()->GetTouchable()->GetVolume()->GetName()
    << " weight " << weight;
  return weight*destep;
}

uint32_t HcalTB06BeamSD::setDetUnitId(G4Step * aStep) { 

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  G4String name             = preStepPoint->GetPhysicalVolume()->GetName();

  int det = 1;
  int lay = 0, x = 0, y = 0;
  if (!isItWireChamber(name)) {
    lay     = (touch->GetReplicaNumber(0));
  } else {
    det = 2;
    lay = (touch->GetReplicaNumber(1));
    G4ThreeVector localPoint  = setToLocal(preStepPoint->GetPosition(), touch);
    x   = (int)(localPoint.x()/(0.2*mm));
    y   = (int)(localPoint.y()/(0.2*mm));
  }

  return HcalTestBeamNumbering::packIndex (det, lay, x, y);
}

std::vector<G4String> HcalTB06BeamSD::getNames(DDFilteredView& fv) {
 
  std::vector<G4String> tmp;
  bool dodet = fv.firstChild();
  while (dodet) {
    const DDLogicalPart & log = fv.logicalPart();
    G4String name = log.name().name();
    bool ok = true;
    for (unsigned int i=0; i<tmp.size(); i++)
      if (name == tmp[i]) ok = false;
    if (ok) tmp.push_back(name);
    dodet = fv.next();
  }
  return tmp;
}
 
bool HcalTB06BeamSD::isItWireChamber (G4String name) {
 
  std::vector<G4String>::const_iterator it = wcNames.begin();
  for (; it != wcNames.end(); it++)
    if (name == *it) return true;
  return false;
}
