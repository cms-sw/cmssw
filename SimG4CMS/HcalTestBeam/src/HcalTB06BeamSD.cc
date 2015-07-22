///////////////////////////////////////////////////////////////////////////////
// File: HcalTB06BeamSD.cc
// Description: Sensitive Detector class for beam counters in TB06 setup
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/HcalTestBeam/interface/HcalTB06BeamSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
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
#include "CLHEP/Units/GlobalSystemOfUnits.h"

HcalTB06BeamSD::HcalTB06BeamSD(G4String name, const DDCompactView & cpv,
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

  LogDebug("HcalTBSim") <<"***************************************************"
			<< "\n"
			<<"*                                                 *"
			<< "\n"
			<< "* Constructing a HcalTB06BeamSD  with name " 
			<< name << "\n"
			<<"*                                                 *"
			<< "\n"
			<<"***************************************************";

  edm::LogInfo("HcalTBSim") << "HcalTB06BeamSD:: Use of Birks law is set to " 
			    << useBirk << "  with three constants kB = "
			    << birk1 << ", C1 = " <<birk2 << ", C2 = " <<birk3;

  std::string attribute, value;

  // Wire Chamber volume names
  attribute = "Volume";
  value     = "WireChamber";
  DDSpecificsFilter filter1;
  DDValue           ddv1(attribute,value,0);
  filter1.setCriteria(ddv1,DDCompOp::equals);
  DDFilteredView fv1(cpv);
  fv1.addFilter(filter1);
  wcNames = getNames(fv1);
  edm::LogInfo("HcalTBSim") << "HcalTB06BeamSD:: Names to be tested for " 
			    << attribute << " = " << value << ": " 
			    << wcNames.size() << " paths";
  for (unsigned int i=0; i<wcNames.size(); i++)
    edm::LogInfo("HcalTBSim") << "HcalTB06BeamSD:: (" << i << ") " 
			      << wcNames[i];

  //Material list for scintillator detector
  attribute = "ReadOutName";
  DDSpecificsFilter filter2;
  DDValue           ddv2(attribute,name,0);
  filter2.setCriteria(ddv2,DDCompOp::equals);
  DDFilteredView fv2(cpv);
  fv2.addFilter(filter2);
  bool dodet = fv2.firstChild();

  std::vector<G4String> matNames;
  std::vector<int>      nocc;
  while (dodet) {
    const DDLogicalPart & log = fv2.logicalPart();
    matName = log.material().name().name();
    bool notIn = true;
    for (unsigned int i=0; i<matNames.size(); i++) 
      if (matName == matNames[i]) {notIn = false; nocc[i]++;}
    if (notIn) {
      matNames.push_back(matName);
      nocc.push_back(0);
    }
    dodet = fv2.next();
  }
  if (matNames.size() > 0) {
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

  edm::LogInfo("HcalTBSim") << "HcalTB06BeamSD: Material name for " 
			    << attribute << " = " << name << ":" << matName;
}

HcalTB06BeamSD::~HcalTB06BeamSD() {}

double HcalTB06BeamSD::getEnergyDeposit(G4Step* aStep) {

  double destep = aStep->GetTotalEnergyDeposit();
  double weight = 1;
  if (useBirk) {
    G4Material* mat = aStep->GetPreStepPoint()->GetMaterial();
    if (mat->GetName() == matName)
      weight *= getAttenuation(aStep, birk1, birk2, birk3);
  }
  LogDebug("HcalTBSim") << "HcalTB06BeamSD: Detector " 
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
    G4ThreeVector hitPoint    = preStepPoint->GetPosition();
    G4ThreeVector localPoint  = setToLocal(hitPoint, touch);
    x   = (int)(localPoint.x()/(0.2*mm));
    y   = (int)(localPoint.y()/(0.2*mm));
  }

  return packIndex (det, lay, x, y);
}

uint32_t HcalTB06BeamSD::packIndex(int det, int lay, int x, int y) {

  int ix = 0, ixx = x;
  if (x < 0) { ix = 1; ixx =-x;}
  int iy = 0, iyy = y;
  if (y < 0) { iy = 1; iyy =-y;}
  uint32_t idx = (det&15)<<28;      //bits 28-31
  idx         += (lay&127)<<21;     //bits 21-27
  idx         += (iy&1)<<19;        //bit  19
  idx         += (iyy&511)<<10;     //bits 10-18
  idx         += (ix&1)<<9;         //bit   9
  idx         += (ixx&511);         //bits  0-8

  LogDebug("HcalTBSim") << "HcalTB06BeamSD: Detector " << det << " Layer "
			<< lay << " x " << x << " " << ix << " " << ixx 
			<< " y " << y << " " << iy << " " << iyy << " ID " 
			<< std::hex << idx << std::dec;
  return idx;
}

void HcalTB06BeamSD::unpackIndex(const uint32_t & idx, int& det, int& lay,
				 int& x, int& y) {

  det  = (idx>>28)&15;
  lay  = (idx>>21)&127;
  y    = (idx>>10)&511; if (((idx>>19)&1) == 1) y = -y;
  x    = (idx)&511;     if (((idx>>9)&1)  == 1) x = -x;

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
