///////////////////////////////////////////////////////////////////////////////
// File: ECalSD.cc
// Description: Sensitive Detector class for electromagnetic calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/ECalSD.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#define debug


ECalSD::ECalSD(G4String name) : 
  CaloSD(name) {
  
  //   static SimpleConfigurable<bool>   on1(false, "ECalSD:UseBirkLaw");
  //   static SimpleConfigurable<double> bk1(0.013, "ECalSD:BirkC1");
  //   static SimpleConfigurable<double> bk2(9.6e-6,"ECalSD:BirkC2");
  // Values from NIM 80 (1970) 239-244: as implemented in Geant3
  //   useBirk          = on1.value();
  //   birk1            = bk1.value()*(g/(MeV*cm2));
  //   birk2            = bk2.value()*(g/(MeV*cm2))*(g/(MeV*cm2));


  useBirk          = false;
  birk1            = 0.013*(g/(MeV*cm2));
  birk2            = 9.6e-6*(g/(MeV*cm2))*(g/(MeV*cm2));
#ifdef debug  
  std::cout << "***************************************************" <<std::endl
	       << "*                                                 *" <<std::endl
	       << "* Constructing a ECalSD  with name " << name         <<std::endl
	       << "*                                                 *" <<std::endl
	       << "***************************************************" <<std::endl;
#endif
  std::cout << "ECalSD:: Use of Birks law is set to      " 
	       << useBirk << "         with the two constants C1 =     "
	       << birk1 << ", C2 = " << birk2 << std::endl;

  initMap(name);

}

ECalSD::~ECalSD() {
  //  delete scheme;
}

double ECalSD::getEnergyDeposit(G4Step * aStep ) {
  
  if (aStep == NULL) {
    return 0;
  } else {
    preStepPoint        = aStep->GetPreStepPoint();
    G4String nameVolume = preStepPoint->GetPhysicalVolume()->GetName();

    // take into account light collection curve for crystals
    double weight = curve_LY(nameVolume, preStepPoint);
    double edep   = aStep->GetTotalEnergyDeposit() * weight;
    if (useBirk) edep *= getAttenuation(aStep, birk1, birk2);
#ifdef debug_verbose
    std::cout << "ECalSD:: " << nameVolume <<" Light Collection Efficiency "
		 << weight << " Weighted Energy Deposit " << edep/MeV << " MeV"
		 << std::endl;
#endif
    return edep;
  } 
}


void ECalSD::initMap(G4String sd) {

  G4String attribute = "ReadOutName";
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,sd,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDCompactView cpv;
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  fv.firstChild();

  bool dodet=true;
  while (dodet) {
    const DDSolid & sol  = fv.logicalPart().solid();
    const std::vector<double> & paras = sol.parameters();
    G4String name = DDSplit(sol.name()).first;
#ifdef debug_verbose
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
#ifdef debug_verbose
  std::cout << "ECalSD: Length Table for " << attribute << " = " 
	       << sd << ":" << std::endl << "       ";
  map<G4String,double>::const_iterator it = lengthMap.begin();
  int i=0;
  for (; it != lengthMap.end(); it++, i++) {
    std::cout << " " << i << " " << it->first << " L = " << it->second;
    if (i%5 == 4) std::cout << std::endl << "       ";
  }
  std::cout << std::endl;
#endif
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
    std::cout << "ECalSD: light coll curve : wrong distance to APD " << dapd
		 << " crlength = " << crlength
		 << " crystal name = " << nameVolume 
		 << " z of localPoint = " << localPoint.z() 
		 << " take weight = " << weight << std::endl;
  }
#ifdef debug_verbose
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
