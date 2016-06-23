///////////////////////////////////////////////////////////////////////////////
// File: HGCalTB16SD01.cc
// Description: Sensitive Detector class for beam counters in TB06 setup
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/HGCalTestBeam/interface/HGCalTB16SD01.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Material.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog

HGCalTB16SD01::HGCalTB16SD01(G4String name, const DDCompactView & cpv,
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

#ifdef DebugLog
  std::cout << "HGCalTB16SD01:: Use of Birks law is set to "  << useBirk 
	    << "  with three constants kB = " << birk1 << ", C1 = " << birk2 
	    << ", C2 = " << birk3 << std::endl;
#endif

  //Material list for scintillator detector
  std::string attribute = "ReadOutName";
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,name,0);
  filter.setCriteria(ddv,DDCompOp::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  bool dodet = fv.firstChild();

  std::map<G4String,int> matNames;
  while (dodet) {
    const DDLogicalPart & log = fv.logicalPart();
    matName = log.material().name().name();
    std::map<G4String,int>::iterator itr = matNames.find(matName);
    if (itr != matNames.end()) {
      ++itr;
    } else {
      matNames[matName] = 1;
    }
    dodet = fv.next();
  }
  if (matNames.size() > 0) {
    int occ(0);
    for (std::map<G4String,int>::iterator itr=matNames.begin(); 
	 itr != matNames.end(); ++itr) {
      if (itr->second > occ) {
	occ     = itr->second;
	matName = itr->first;
      }
    }
  } else {
    matName = "Not Found";
  }

#ifdef DebugLog
  std::cout << "HGCalTB16SD01: Material name for "  << attribute << " = " 
	    << name << ":" << matName << std::endl;
#endif
  matScin = nullptr;
  const G4MaterialTable * matTab = G4Material::GetMaterialTable();
  std::vector<G4Material*>::const_iterator matite;
  unsigned int kount(0);
  for (matite = matTab->begin(); matite != matTab->end(); ++matite, ++kount)
    std::cout << "Material[" << kount << "] " << (*matite)->GetName() << std::endl;
  matScin = G4Material::GetMaterial(matName);
}

HGCalTB16SD01::~HGCalTB16SD01() {}

void HGCalTB16SD01::initRun() {

  const G4MaterialTable * matTab = G4Material::GetMaterialTable();
  std::vector<G4Material*>::const_iterator matite;
  std::cout << "Material table pointers " << (matTab->begin() == matTab->end()) << std::endl;
  unsigned int kount(0);
  for (matite = matTab->begin(); matite != matTab->end(); ++matite, ++kount)
    std::cout << "Material[" << kount << "] " << (*matite)->GetName() << std::endl;
}

double HGCalTB16SD01::getEnergyDeposit(G4Step* aStep) {

  double destep = aStep->GetTotalEnergyDeposit();
  double weight = 1;
  if (useBirk && matScin == aStep->GetPreStepPoint()->GetMaterial()) {
    weight *= getAttenuation(aStep, birk1, birk2, birk3);
  }
#ifdef DebugLog
  std::cout << "HGCalTB16SD01: Detector " 
	    << aStep->GetPreStepPoint()->GetTouchable()->GetVolume()->GetName()
	    << " weight " << weight << std::endl;
#endif
  return weight*destep;
}

uint32_t HGCalTB16SD01::setDetUnitId(G4Step * aStep) { 

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  G4String name             = preStepPoint->GetPhysicalVolume()->GetName();

  int det(1), x(0), y(0);
  int lay = (touch->GetReplicaNumber(0));

  return packIndex (det, lay, x, y);
}

uint32_t HGCalTB16SD01::packIndex(int det, int lay, int x, int y) {

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

#ifdef DebugLog
  std::cout << "HGCalTB16SD01: Detector " << det << " Layer "  << lay << " x "
	    << x << " " << ix << " " << ixx << " y " << y << " " << iy << " " 
	    << iyy << " ID " << std::hex << idx << std::dec << std::endl;
#endif
  return idx;
}

void HGCalTB16SD01::unpackIndex(const uint32_t & idx, int& det, int& lay,
				 int& x, int& y) {

  det  = (idx>>28)&15;
  lay  = (idx>>21)&127;
  y    = (idx>>10)&511; if (((idx>>19)&1) == 1) y = -y;
  x    = (idx)&511;     if (((idx>>9)&1)  == 1) x = -x;

}

std::vector<G4String> HGCalTB16SD01::getNames(DDFilteredView& fv) {
 
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
