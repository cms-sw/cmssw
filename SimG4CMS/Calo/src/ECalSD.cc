///////////////////////////////////////////////////////////////////////////////
// File: ECalSD.cc
// Description: Sensitive Detector class for electromagnetic calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/ECalSD.h"
#include "SimG4CMS/Calo/interface/EcalBarrelNumberingScheme.h"
#include "SimG4CMS/Calo/interface/EcalEndcapNumberingScheme.h"
#include "SimG4CMS/Calo/interface/ShowerForwardNumberingScheme.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#define debug


ECalSD::ECalSD(G4String name, const DDCompactView & cpv,
	       edm::ParameterSet const & p) : CaloSD(name, cpv, p), 
					      numberingScheme(0) {
  
  //   static SimpleConfigurable<bool>   on1(false, "ECalSD:UseBirkLaw");
  //   static SimpleConfigurable<double> bk1(0.013, "ECalSD:BirkC1");
  //   static SimpleConfigurable<double> bk2(9.6e-6,"ECalSD:BirkC2");
  // Values from NIM 80 (1970) 239-244: as implemented in Geant3
  //   useBirk          = on1.value();
  //   birk1            = bk1.value()*(g/(MeV*cm2));
  //   birk2            = bk2.value()*(g/(MeV*cm2))*(g/(MeV*cm2));

  edm::ParameterSet m_ECalSD = p.getParameter<edm::ParameterSet>("ECalSD");
  verbosity= m_ECalSD.getParameter<int>("Verbosity");
  useBirk= m_ECalSD.getParameter<bool>("UseBirkLaw");
  birk1  = m_ECalSD.getParameter<double>("BirkC1")*(g/(MeV*cm2));
  birk2  = m_ECalSD.getParameter<double>("BirkC2")*(g/(MeV*cm2))*(g/(MeV*cm2));
  useWeight= true;

  int verbn = verbosity/10;
  EcalNumberingScheme* scheme=0;
  if      (name == "EcalHitsEB") scheme = dynamic_cast<EcalNumberingScheme*>(new EcalBarrelNumberingScheme(verbn));
  else if (name == "EcalHitsEE") scheme = dynamic_cast<EcalNumberingScheme*>(new EcalEndcapNumberingScheme(verbn));
  else if (name == "EcalHitsES") {
    scheme = dynamic_cast<EcalNumberingScheme*>(new ShowerForwardNumberingScheme(verbn));
    useWeight = false;
  } else {std::cout << "ECalSD: ReadoutName not supported" << std::endl;}

  if (scheme)  setNumberingScheme(scheme);
  verbosity %= 10;
#ifdef debug 
  if (verbosity>1) 
    std::cout << "***************************************************" 
	      << std::endl
	      << "*                                                 *" 
	      << std::endl
	      << "* Constructing a ECalSD  with name " << GetName()
	      << std::endl
	      << "*                                                 *"
	      << std::endl
	      << "***************************************************" 
	      << std::endl;
#endif
  if (verbosity>0) 
    std::cout << "ECalSD:: Use of Birks law is set to      " 
	      << useBirk << "         with the two constants C1 =     "
	      << birk1 << ", C2 = " << birk2 << std::endl;

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
#ifdef debug
    if (verbosity>1)
      std::cout << "ECalSD:: " << nameVolume <<" Light Collection Efficiency "
		<< weight << " Weighted Energy Deposit " << edep/MeV << " MeV"
		<< std::endl;
#endif
    return edep;
  } 
}

uint32_t ECalSD::setDetUnitId(G4Step * aStep) { 
  return (numberingScheme == 0 ? 0 : numberingScheme->getUnitID(aStep));
}

void ECalSD::setNumberingScheme(EcalNumberingScheme* scheme) {
  if (scheme != 0) {
    std::cout << "EcalSD: updates numbering scheme for " << GetName() 
	      << std::endl;
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
#ifdef debug
    if (verbosity>2)
      std::cout << "ECalSD::initMap (for " << sd << "): Solid " << name
		<< " Shape " << sol.shape() << " Parameter 0 = " 
		<< paras[0] << std::endl;
#endif
    if (sol.shape() == ddtrap) {
      double dz = 2*paras[0];
      lengthMap.insert(pair<G4String,double>(name,dz));
    }
    dodet = fv.next();
   }
  if (verbosity>0) {
    std::cout << "ECalSD: Length Table for " << attribute << " = " 
	      << sd << ":" << std::endl << "       ";
    map<G4String,double>::const_iterator it = lengthMap.begin();
    int i=0;
    for (; it != lengthMap.end(); it++, i++) {
      std::cout << " " << i << " " << it->first << " L = " << it->second;
      if (i%5 == 4) std::cout << std::endl << "       ";
    }
    std::cout << std::endl;
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
      weight = 1.05 - dapd * 0.0005;
  } else {
    if (verbosity>0) 
      std::cout << "ECalSD: light coll curve : wrong distance to APD " << dapd
		<< " crlength = " << crlength
		<< " crystal name = " << nameVolume 
		<< " z of localPoint = " << localPoint.z() 
		<< " take weight = " << weight << std::endl;
  }
#ifdef debug
  if (verbosity>2) 
    std::cout << "ECalSD, light coll curve : " << dapd 
	      << " crlength = " << crlength
	      << " crystal name = " << nameVolume 
	      << " z of localPoint = " << localPoint.z() 
	      << " take weight = " << weight << std::endl;
#endif
  return weight;
}

double ECalSD::crystalLength(G4String name) {

  double length = 230.;
  map<G4String,double>::const_iterator it = lengthMap.find(name);
  if (it != lengthMap.end()) length = it->second;
  return length;
}
