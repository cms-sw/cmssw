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
// $Id: HcalTB02SD.cc,v 1.5 2009/09/09 10:31:49 fabiocos Exp $
//
  
// system include files
  
// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02SD.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02HcalNumberingScheme.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02XtalNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

//
// constructors and destructor
//

HcalTB02SD::HcalTB02SD(G4String name, const DDCompactView & cpv,
		       SensitiveDetectorCatalog & clg, 
		       edm::ParameterSet const & p, 
		       const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager), numberingScheme(0) {
  
  edm::ParameterSet m_SD = p.getParameter<edm::ParameterSet>("HcalTB02SD");
  useBirk= m_SD.getUntrackedParameter<bool>("UseBirkLaw",false);
  birk1  = m_SD.getUntrackedParameter<double>("BirkC1",0.013)*(g/(MeV*cm2));
  birk2  = m_SD.getUntrackedParameter<double>("BirkC2",0.0568);
  birk3  = m_SD.getUntrackedParameter<double>("BirkC3",1.75);
  useWeight= true;

  HcalTB02NumberingScheme* scheme=0;
  if      (name == "EcalHitsEB") {
    scheme = dynamic_cast<HcalTB02NumberingScheme*>(new HcalTB02XtalNumberingScheme());
    useBirk = false;
  } else if (name == "HcalHits") {
    scheme = dynamic_cast<HcalTB02NumberingScheme*>(new HcalTB02HcalNumberingScheme());
      useWeight= false;
  } else {edm::LogWarning("HcalTBSim") << "HcalTB02SD: ReadoutName " << name
				       << " not supported\n";}

  if (scheme)  setNumberingScheme(scheme);
  LogDebug("HcalTBSim") 
    << "***************************************************" 
    << "\n"
    << "*                                                 *" 
    << "\n"
    << "* Constructing a HcalTB02SD  with name " << GetName()
    << "\n"
    << "*                                                 *"
    << "\n"
    << "***************************************************" ;
  edm::LogInfo("HcalTBSim")  << "HcalTB02SD:: Use of Birks law is set to      "
			     << useBirk << "        with three constants kB = "
			     << birk1 << ", C1 = " << birk2 << ", C2 = "
			     << birk3;

  if (useWeight) initMap(name,cpv);

}

HcalTB02SD::~HcalTB02SD() {
  if (numberingScheme) delete numberingScheme;
}

//
// member functions
//
 
double HcalTB02SD::getEnergyDeposit(G4Step * aStep) {
  
  if (aStep == NULL) {
    return 0;
  } else {
    preStepPoint        = aStep->GetPreStepPoint();
    G4String nameVolume = preStepPoint->GetPhysicalVolume()->GetName();

    // take into account light collection curve for crystals
    double weight = 1.;
    if (useWeight) weight *= curve_LY(nameVolume, preStepPoint);
    if (useBirk)   weight *= getAttenuation(aStep, birk1, birk2, birk3);
    double edep   = aStep->GetTotalEnergyDeposit() * weight;
    LogDebug("HcalTBSim") << "HcalTB02SD:: " << nameVolume
			  <<" Light Collection Efficiency " << weight 
			  << " Weighted Energy Deposit " << edep/MeV << " MeV";
    return edep;
  } 
}

uint32_t HcalTB02SD::setDetUnitId(G4Step * aStep) { 
  return (numberingScheme == 0 ? 0 : (uint32_t)(numberingScheme->getUnitID(aStep)));
}

void HcalTB02SD::setNumberingScheme(HcalTB02NumberingScheme* scheme) {
  if (scheme != 0) {
    edm::LogInfo("HcalTBSim") << "HcalTB02SD: updates numbering scheme for " 
			      << GetName();
    if (numberingScheme) delete numberingScheme;
    numberingScheme = scheme;
  }
}

void HcalTB02SD::initMap(G4String sd, const DDCompactView & cpv) {

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
    G4String name = sol.name().name();
    LogDebug("HcalTBSim") << "HcalTB02SD::initMap (for " << sd << "): Solid " 
			  << name << " Shape " << sol.shape() 
			  << " Parameter 0 = " << paras[0];
    if (sol.shape() == ddtrap) {
      double dz = 2*paras[0];
      lengthMap.insert(std::pair<G4String,double>(name,dz));
    }
    dodet = fv.next();
  }
  LogDebug("HcalTBSim") << "HcalTB02SD: Length Table for " << attribute 
			<< " = " << sd << ":";   
  std::map<G4String,double>::const_iterator it = lengthMap.begin();
  int i=0;
  for (; it != lengthMap.end(); it++, i++) {
    LogDebug("HcalTBSim") << " " << i << " " << it->first << " L = " 
			  << it->second;
  }
}

double HcalTB02SD::curve_LY(G4String& nameVolume, G4StepPoint* stepPoint) {

  double weight = 1.;
  G4ThreeVector  localPoint = setToLocal(stepPoint->GetPosition(),
					 stepPoint->GetTouchable());
  double crlength = crystalLength(nameVolume);
  double dapd = 0.5 * crlength - localPoint.z();
  if (dapd >= -0.1 || dapd <= crlength+0.1) {
    if (dapd <= 100.)
      weight = 1.05 - dapd * 0.0005;
  } else {
    edm::LogWarning("HcalTBSim") << "HcalTB02SD: light coll curve : wrong "
				 << "distance to APD " << dapd <<" crlength = "
				 << crlength << " crystal name = " <<nameVolume
				 << " z of localPoint = " << localPoint.z() 
				 << " take weight = " << weight;
  }
  LogDebug("HcalTBSim") << "HcalTB02SD, light coll curve : " << dapd 
			<< " crlength = " << crlength
			<< " crystal name = " << nameVolume 
			<< " z of localPoint = " << localPoint.z() 
			<< " take weight = " << weight;
  return weight;
}

double HcalTB02SD::crystalLength(G4String name) {

  double length = 230.;
  std::map<G4String,double>::const_iterator it = lengthMap.find(name);
  if (it != lengthMap.end()) length = it->second;
  return length;
}
