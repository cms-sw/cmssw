#include "SimG4CMS/CherenkovAnalysis/interface/DreamSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

DreamSD::DreamSD(G4String name, const DDCompactView & cpv,
	       SensitiveDetectorCatalog & clg, 
	       edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager) {

  edm::ParameterSet m_EC = p.getParameter<edm::ParameterSet>("ECalSD");
  useBirk= m_EC.getParameter<bool>("UseBirkLaw");
  birk1  = m_EC.getParameter<double>("BirkC1")*(g/(MeV*cm2));
  birk2  = m_EC.getParameter<double>("BirkC2")*(g/(MeV*cm2))*(g/(MeV*cm2));
  slopeLY= m_EC.getParameter<double>("SlopeLightYield");
  
  edm::LogInfo("EcalSim")  << "Constructing a DreamSD  with name " 
			   << GetName() << "\n"
			   << "DreamSD:: Use of Birks law is set to      " 
			   << useBirk << "        with the two constants C1 = "
			   << birk1 << ", C2 = " << birk2 << "\n"
			   << "         Slope for Light yield is set to "
			   << slopeLY;

  initMap(name,cpv);

}

DreamSD::~DreamSD() {}

double DreamSD::getEnergyDeposit(G4Step * aStep) {
  
  if (aStep == NULL) {
    return 0;
  } else {
    preStepPoint        = aStep->GetPreStepPoint();
    G4String nameVolume = preStepPoint->GetPhysicalVolume()->GetName();

    // take into account light collection curve for crystals
    double weight = 1.;
    weight *= curve_LY(aStep);
    if (useBirk)   weight *= getAttenuation(aStep, birk1, birk2);
    double edep   = aStep->GetTotalEnergyDeposit() * weight;
    LogDebug("EcalSim") << "DreamSD:: " << nameVolume
			<<" Light Collection Efficiency " << weight 
			<< " Weighted Energy Deposit " << edep/MeV << " MeV";
    return edep;
  } 
}

uint32_t DreamSD::setDetUnitId(G4Step * aStep) { 

  const G4VTouchable* touch = aStep->GetPostStepPoint()->GetTouchable();
  return touch->GetReplicaNumber(0);
}

void DreamSD::initMap(G4String sd, const DDCompactView & cpv) {

  G4String attribute = "ReadOutName";
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,sd,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  fv.firstChild();

  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  bool dodet=true;
  while (dodet) {
    const DDSolid & sol  = fv.logicalPart().solid();
    const std::vector<double> & paras = sol.parameters();
    G4String name = DDSplit(sol.name()).first;
    G4LogicalVolume* lv=0;
    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) 
      if ((*lvcite)->GetName() == name) {
	lv = (*lvcite);
	break;
      }
    LogDebug("EcalSim") << "DreamSD::initMap (for " << sd << "): Solid " 
			<< name	<< " Shape " << sol.shape() <<" Parameter 0 = "
			<< paras[0] << " Logical Volume " << lv;
    double dz = 0;
    if (sol.shape() == ddbox) {
      dz = 2*paras[2];
    } else if (sol.shape() == ddtrap) {
      dz = 2*paras[0];
    }
    xtalLMap.insert(std::pair<G4LogicalVolume*,double>(lv,dz));
    dodet = fv.next();
  }
  LogDebug("EcalSim") << "DreamSD: Length Table for " << attribute << " = " 
		      << sd << ":";   
  std::map<G4LogicalVolume*,double>::const_iterator ite = xtalLMap.begin();
  int i=0;
  for (; ite != xtalLMap.end(); ite++, i++) {
    G4String name = "Unknown";
    if (ite->first != 0) name = (ite->first)->GetName();
    LogDebug("EcalSim") << " " << i << " " << ite->first << " " << name 
			<< " L = " << ite->second;
  }
}

double DreamSD::curve_LY(G4Step* aStep) {

  G4StepPoint*     stepPoint = aStep->GetPreStepPoint();
  G4LogicalVolume* lv        = stepPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();
  G4String         nameVolume= lv->GetName();

  double weight = 1.;
  G4ThreeVector  localPoint = setToLocal(stepPoint->GetPosition(),
					 stepPoint->GetTouchable());
  double crlength = crystalLength(lv);
  double dapd = 0.5 * crlength - localPoint.z();
  if (dapd >= -0.1 || dapd <= crlength+0.1) {
    if (dapd <= 100.)
      weight = 1.0 + slopeLY - dapd * 0.01 * slopeLY;
  } else {
    edm::LogWarning("EcalSim") << "DreamSD: light coll curve : wrong distance "
			       << "to APD " << dapd << " crlength = " 
			       << crlength << " crystal name = " << nameVolume 
			       << " z of localPoint = " << localPoint.z() 
			       << " take weight = " << weight;
  }
  LogDebug("EcalSim") << "DreamSD, light coll curve : " << dapd 
		      << " crlength = " << crlength
		      << " crystal name = " << nameVolume 
		      << " z of localPoint = " << localPoint.z() 
		      << " take weight = " << weight;
  return weight;
}

double DreamSD::crystalLength(G4LogicalVolume* lv) {

  double length= 230.;
  std::map<G4LogicalVolume*,double>::const_iterator ite = xtalLMap.find(lv);
  if (ite != xtalLMap.end()) length = ite->second;
  return length;
}
