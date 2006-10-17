///////////////////////////////////////////////////////////////////////////////
// File: ECalSD.cc
// Description: Sensitive Detector class for electromagnetic calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/ECalSD.h"
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalPreshowerNumberingScheme.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

ECalSD::ECalSD(G4String name, const DDCompactView & cpv,
	       edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, p, manager), numberingScheme(0) {
  
  //   static SimpleConfigurable<bool>   on1(false, "ECalSD:UseBirkLaw");
  //   static SimpleConfigurable<double> bk1(0.013, "ECalSD:BirkC1");
  //   static SimpleConfigurable<double> bk2(9.6e-6,"ECalSD:BirkC2");
  // Values from NIM 80 (1970) 239-244: as implemented in Geant3
  //   useBirk          = on1.value();
  //   birk1            = bk1.value()*(g/(MeV*cm2));
  //   birk2            = bk2.value()*(g/(MeV*cm2))*(g/(MeV*cm2));

  edm::ParameterSet m_ECalSD = p.getParameter<edm::ParameterSet>("ECalSD");
  useBirk= m_ECalSD.getParameter<bool>("UseBirkLaw");
  birk1  = m_ECalSD.getParameter<double>("BirkC1")*(g/(MeV*cm2));
  birk2  = m_ECalSD.getParameter<double>("BirkC2")*(g/(MeV*cm2))*(g/(MeV*cm2));
  slopeLY= m_ECalSD.getUntrackedParameter<double>("SlopeLightYield", 0.02);
  useWeight= true;

  EcalNumberingScheme* scheme=0;
  if      (name == "EcalHitsEB") scheme = dynamic_cast<EcalNumberingScheme*>(new EcalBarrelNumberingScheme());
  else if (name == "EcalHitsEE") scheme = dynamic_cast<EcalNumberingScheme*>(new EcalEndcapNumberingScheme());
  else if (name == "EcalHitsES") {
    scheme = dynamic_cast<EcalNumberingScheme*>(new EcalPreshowerNumberingScheme());
    useWeight = false;
  } else {edm::LogWarning("EcalSim") << "ECalSD: ReadoutName not supported\n";}

  if (scheme)  setNumberingScheme(scheme);
  LogDebug("EcalSim") 
    << "***************************************************" 
    << "\n"
    << "*                                                 *" 
    << "\n"
    << "* Constructing a ECalSD  with name " << GetName()
    << "\n"
    << "*                                                 *"
    << "\n"
    << "***************************************************" ;
  edm::LogInfo("EcalSim")  << "ECalSD:: Use of Birks law is set to      " 
			   << useBirk << "        with the two constants C1 = "
			   << birk1 << ", C2 = " << birk2 << "\n"
			   << "         Slope for Light yield is set to "
			   << slopeLY;

  if (useWeight) initMap(name,cpv);

}

ECalSD::~ECalSD() {
  if (numberingScheme) delete numberingScheme;
}

double ECalSD::getEnergyDeposit(G4Step * aStep) {
  
  if (aStep == NULL) {
    return 0;
  } else {
    preStepPoint        = aStep->GetPreStepPoint();
    G4String nameVolume = preStepPoint->GetPhysicalVolume()->GetName();

    // take into account light collection curve for crystals
    double weight = 1.;
    if (useWeight) {
      weight *= curve_LY(nameVolume, preStepPoint);
      if (useBirk)   weight *= getAttenuation(aStep, birk1, birk2);
    }
    double edep   = aStep->GetTotalEnergyDeposit() * weight;
    LogDebug("EcalSim") << "ECalSD:: " << nameVolume
			<<" Light Collection Efficiency " << weight 
			<< " Weighted Energy Deposit " << edep/MeV << " MeV";
    return edep;
  } 
}

uint32_t ECalSD::setDetUnitId(G4Step * aStep) { 
  return (numberingScheme == 0 ? 0 : numberingScheme->getUnitID(getBaseNumber(aStep)));
}

void ECalSD::setNumberingScheme(EcalNumberingScheme* scheme) {
  if (scheme != 0) {
    edm::LogInfo("EcalSim") << "EcalSD: updates numbering scheme for " 
			    << GetName() << "\n";
    if (numberingScheme) delete numberingScheme;
    numberingScheme = scheme;
  }
}

void ECalSD::initMap(G4String sd, const DDCompactView & cpv) {

  G4String attribute = "ReadOutName";
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,sd,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  fv.firstChild();

  bool dodet=true;
  while (dodet) {
    const DDSolid & sol  = fv.logicalPart().solid();
    const std::vector<double> & paras = sol.parameters();
    G4String name = DDSplit(sol.name()).first;
    LogDebug("EcalSim") << "ECalSD::initMap (for " << sd << "): Solid " << name
			<< " Shape " << sol.shape() << " Parameter 0 = " 
			<< paras[0];
    if (sol.shape() == ddtrap) {
      double dz = 2*paras[0];
      lengthMap.insert(pair<G4String,double>(name,dz));
    }
    dodet = fv.next();
  }
  LogDebug("EcalSim") << "ECalSD: Length Table for " << attribute << " = " 
		      << sd << ":";   
  map<G4String,double>::const_iterator it = lengthMap.begin();
  int i=0;
  for (; it != lengthMap.end(); it++, i++) {
    LogDebug("EcalSim") << " " << i << " " << it->first << " L = " 
			<< it->second;
  }
}

double ECalSD::curve_LY(G4String& nameVolume, G4StepPoint* stepPoint) {

  double weight = 1.;
  G4ThreeVector  localPoint = setToLocal(stepPoint->GetPosition(),
					 stepPoint->GetTouchable());
  double crlength = crystalLength(nameVolume);
  double dapd = 0.5 * crlength - localPoint.z();
  if (dapd >= -0.1 || dapd <= crlength+0.1) {
    if (dapd <= 100.)
      weight = 1.0 + slopeLY - dapd * 0.01 * slopeLY;
  } else {
    edm::LogWarning("EcalSim") << "ECalSD: light coll curve : wrong distance "
			       << "to APD " << dapd << " crlength = " 
			       << crlength << " crystal name = " << nameVolume 
			       << " z of localPoint = " << localPoint.z() 
			       << " take weight = " << weight;
  }
  LogDebug("EcalSim") << "ECalSD, light coll curve : " << dapd 
		      << " crlength = " << crlength
		      << " crystal name = " << nameVolume 
		      << " z of localPoint = " << localPoint.z() 
		      << " take weight = " << weight;
  return weight;
}

double ECalSD::crystalLength(G4String name) {

  double length = 230.;
  map<G4String,double>::const_iterator it = lengthMap.find(name);
  if (it != lengthMap.end()) length = it->second;
  return length;
}

EcalBaseNumber ECalSD::getBaseNumber(const G4Step* aStep) const {

  EcalBaseNumber aBaseNumber;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  //Get name and copy numbers
  if (touch->GetHistoryDepth() > 0) {
    for (int ii = 0; ii <= touch->GetHistoryDepth() ; ii++) {
      aBaseNumber.addLevel(touch->GetVolume(ii)->GetName(),touch->GetReplicaNumber(ii));
      LogDebug("EcalSim") << "ECalSD::getBaseNumber(): Adding level " << ii 
			  << ": " << touch->GetVolume(ii)->GetName() << "[" 
			  << touch->GetReplicaNumber(ii) << "]";
    }
  }
  return aBaseNumber;
}
