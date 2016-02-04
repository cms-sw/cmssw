// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02HcalNumberingScheme
//
// Implementation:
//     Numbering scheme for hadron calorimeter in 2002 test beam
//
// Original Author:
//         Created:  Sun 21 10:14:34 CEST 2006
// $Id: HcalTB02HcalNumberingScheme.cc,v 1.2 2006/11/13 10:32:15 sunanda Exp $
//
  
// system include files
  
// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02HcalNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

//
// constructors and destructor
//

HcalTB02HcalNumberingScheme::HcalTB02HcalNumberingScheme() : 
  HcalTB02NumberingScheme(), phiScale(1000000), etaScale(10000) {
  edm::LogInfo("HcalTBSim") << "Creating HcalTB02HcalNumberingScheme";
}

HcalTB02HcalNumberingScheme::~HcalTB02HcalNumberingScheme() {
  edm::LogInfo("HcalTBSim") << "Deleting HcalTB02HcalNumberingScheme";
}

//
// member functions
//
 
int HcalTB02HcalNumberingScheme::getUnitID(const G4Step* aStep) const {

  int scintID = 0;

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint(); 
  G4ThreeVector    hitPoint = preStepPoint->GetPosition();
  float hx = hitPoint.x();
  float hy = hitPoint.y();
  float hz = hitPoint.z();
  float hr = std::sqrt( pow(hx,2)+pow(hy,2) );

  // Check if hit happened in first HO layer or second.

  if ( (hr > 3.*m) && (hr < 3.830*m) ) return scintID=17;
  if (hr > 3.830*m)                    return scintID=18;

  // Compute the scintID in the HB.

  float hR = hitPoint.mag();//sqrt( pow(hx,2)+pow(hy,2)+pow(hz,2) );
  float htheta =  (hR == 0. ? 0. : acos(max(min(hz/hR,float(1.)),float(-1.))));
  float hsintheta = sin(htheta);
  float hphi = (hR*hsintheta == 0. ? 0. :acos( max(min(hx/(hR*hsintheta),float(1.)),float(-1.)) ) );
  float heta = ( fabs(hsintheta) == 1.? 0. : -log(fabs(tan(htheta/2.))) );
  int eta = int(heta/0.087);
  int phi = int(hphi/(5.*degree));

  G4VPhysicalVolume*  thePV = preStepPoint->GetPhysicalVolume();
  int ilayer = ((thePV->GetCopyNo())/10)%100;
  LogDebug("HcalTBSim") << "HcalTB02HcalNumberingScheme:: Layer " 
			<< thePV->GetName() << " found at phi = " << phi
			<< " eta = " << eta << " lay = " << thePV->GetCopyNo()
			<< " " << ilayer;

  scintID = phiScale*phi + etaScale*eta + ilayer;
  if (hy<0.) scintID = -scintID;

  return scintID;
}

int HcalTB02HcalNumberingScheme::getlayerID(int sID) const {

  sID = abs(sID);
  int layerID = sID;
  if ( (layerID != 17) && (layerID != 18) )
    layerID = sID - int(float(sID)/float(etaScale))*etaScale;

  LogDebug("HcalTBSim") << "HcalTB02HcalNumberingScheme:: scintID " << sID 
			<< " layer = " << layerID;
  return layerID;
}      

int HcalTB02HcalNumberingScheme::getphiID(int sID) const {

  float IDsign = 1.;
  if (sID<0) IDsign = -1;
  sID = abs(sID);
  int phiID = int(float(sID)/float(phiScale));
  LogDebug("HcalTBSim") << "HcalTB02HcalNumberingScheme:: scintID " << sID 
			<< " phi = " << phiID;
  if (IDsign>0) {
    phiID += 4;
  } else {
    phiID = abs(phiID-3);
  }
  return phiID;
}      

int HcalTB02HcalNumberingScheme::getetaID(int sID) const {

  sID = abs(sID);
  int aux = sID - int(float(sID)/float(phiScale))*phiScale;
  int etaID = int(float(aux)/float(etaScale));

  LogDebug("HcalTBSim") << "HcalTB02HcalNumberingScheme:: scintID " << sID
			<< " eta = " << etaID;
  return etaID;

}    
