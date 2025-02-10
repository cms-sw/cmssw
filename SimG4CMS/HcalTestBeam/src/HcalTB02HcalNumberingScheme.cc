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
//

// system include files

// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02HcalNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <CLHEP/Units/SystemOfUnits.h>
using CLHEP::degree;
using CLHEP::m;

//#define EDM_ML_DEBUG
//
// constructors and destructor
//

HcalTB02HcalNumberingScheme::HcalTB02HcalNumberingScheme()
    : HcalTB02NumberingScheme(), phiScale(1000000), etaScale(10000) {
  edm::LogVerbatim("HcalTBSim") << "Creating HcalTB02HcalNumberingScheme";
}

HcalTB02HcalNumberingScheme::~HcalTB02HcalNumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "Deleting HcalTB02HcalNumberingScheme";
#endif
}

//
// member functions
//

int HcalTB02HcalNumberingScheme::getUnitID(const G4Step* aStep) const {
  G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4ThreeVector& hitPoint = preStepPoint->GetPosition();
  float hx = hitPoint.x();
  float hy = hitPoint.y();
  float hr = std::sqrt(pow(hx, 2) + pow(hy, 2));

  // Check if hit happened in first HO layer or second.

  if ((hr > 3. * m) && (hr < 3.830 * m))
    return 17;
  if (hr > 3.830 * m)
    return 18;

  // Compute the scintID in the HB.
  int scintID = 0;
  float hz = hitPoint.z();
  float hR = hitPoint.mag();  //sqrt( pow(hx,2)+pow(hy,2)+pow(hz,2) );
  float htheta = (hR == 0. ? 0. : acos(std::max(std::min(hz / hR, float(1.)), float(-1.))));
  float hsintheta = sin(htheta);
  float hphi = (hR * hsintheta == 0. ? 0. : acos(std::max(std::min(hx / (hR * hsintheta), float(1.)), float(-1.))));
  float heta = (std::fabs(hsintheta) == 1. ? 0. : -std::log(std::fabs(tan(htheta / 2.))));
  int eta = int(heta / 0.087);
  int phi = int(hphi / (5. * degree));

  G4VPhysicalVolume* thePV = preStepPoint->GetPhysicalVolume();
  int ilayer = ((thePV->GetCopyNo()) / 10) % 100;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB02HcalNumberingScheme:: Layer " << thePV->GetName()
                                << " found at phi = " << phi << " eta = " << eta << " lay = " << thePV->GetCopyNo()
                                << " " << ilayer;
#endif
  scintID = phiScale * phi + etaScale * eta + ilayer;
  if (hy < 0.)
    scintID = -scintID;

  return scintID;
}

int HcalTB02HcalNumberingScheme::getlayerID(int sID) const {
  sID = std::abs(sID);
  int layerID = sID;
  if ((layerID != 17) && (layerID != 18))
    layerID = sID - int(float(sID) / float(etaScale)) * etaScale;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB02HcalNumberingScheme:: scintID " << sID << " layer = " << layerID;
#endif
  return layerID;
}

int HcalTB02HcalNumberingScheme::getphiID(int sID) const {
  float IDsign = 1.;
  if (sID < 0)
    IDsign = -1;
  sID = std::abs(sID);
  int phiID = int(float(sID) / float(phiScale));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB02HcalNumberingScheme:: scintID " << sID << " phi = " << phiID;
#endif
  if (IDsign > 0) {
    phiID += 4;
  } else {
    phiID = std::abs(phiID - 3);
  }
  return phiID;
}

int HcalTB02HcalNumberingScheme::getetaID(int sID) const {
  sID = std::abs(sID);
  int aux = sID - int(float(sID) / float(phiScale)) * phiScale;
  int etaID = int(float(aux) / float(etaScale));

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB02HcalNumberingScheme:: scintID " << sID << " eta = " << etaID;
#endif
  return etaID;
}
