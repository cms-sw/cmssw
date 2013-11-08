///////////////////////////////////////////////////////////////////////////////
// File: EcalTBH4BeamSD.cc
// Description: Sensitive Detector class for electromagnetic calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/EcalTestBeam/interface/EcalTBH4BeamSD.h"
#include "Geometry/EcalTestBeam/interface/EcalHodoscopeNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include "G4SystemOfUnits.hh"

EcalTBH4BeamSD::EcalTBH4BeamSD(G4String name, const DDCompactView & cpv,
			       SensitiveDetectorCatalog & clg, 
			       edm::ParameterSet const & p, 
			       const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager), numberingScheme(0) {
  
  edm::ParameterSet m_EcalTBH4BeamSD = p.getParameter<edm::ParameterSet>("EcalTBH4BeamSD");
  useBirk= m_EcalTBH4BeamSD.getParameter<bool>("UseBirkLaw");
  birk1  = m_EcalTBH4BeamSD.getParameter<double>("BirkC1")*(g/(MeV*cm2));
  birk2  = m_EcalTBH4BeamSD.getParameter<double>("BirkC2");
  birk3  = m_EcalTBH4BeamSD.getParameter<double>("BirkC3");

  EcalNumberingScheme* scheme=0;
  if     (name == "EcalTBH4BeamHits") { 
    scheme = dynamic_cast<EcalNumberingScheme*>(new EcalHodoscopeNumberingScheme());
  } 
  else {edm::LogWarning("EcalTBSim") << "EcalTBH4BeamSD: ReadoutName not supported\n";}

  if (scheme)  setNumberingScheme(scheme);
  edm::LogInfo("EcalTBSim") << "Constructing a EcalTBH4BeamSD  with name " 
			    << GetName();
  edm::LogInfo("EcalTBSim") << "EcalTBH4BeamSD:: Use of Birks law is set to  " 
			    << useBirk << "        with three constants kB = "
			    << birk1 << ", C1 = " << birk2 << ", C2 = " 
			    << birk3;
}

EcalTBH4BeamSD::~EcalTBH4BeamSD() {
  if (numberingScheme) delete numberingScheme;
}

double EcalTBH4BeamSD::getEnergyDeposit(G4Step * aStep) {
  
  if (aStep == NULL) {
    return 0;
  } else {
    preStepPoint        = aStep->GetPreStepPoint();
    G4String nameVolume = preStepPoint->GetPhysicalVolume()->GetName();

    // take into account light collection curve for crystals
    double weight = 1.;
    if (useBirk)   weight *= getAttenuation(aStep, birk1, birk2, birk3);
    double edep   = aStep->GetTotalEnergyDeposit() * weight;
    LogDebug("EcalTBSim") << "EcalTBH4BeamSD:: " << nameVolume
			<<" Light Collection Efficiency " << weight 
			<< " Weighted Energy Deposit " << edep/MeV << " MeV";
    return edep;
  } 
}

uint32_t EcalTBH4BeamSD::setDetUnitId(G4Step * aStep) { 
  getBaseNumber(aStep);
  return (numberingScheme == 0 ? 0 : numberingScheme->getUnitID(theBaseNumber));
}

void EcalTBH4BeamSD::setNumberingScheme(EcalNumberingScheme* scheme) {
  if (scheme != 0) {
    edm::LogInfo("EcalTBSim") << "EcalTBH4BeamSD: updates numbering scheme for " 
			    << GetName() << "\n";
    if (numberingScheme) delete numberingScheme;
    numberingScheme = scheme;
  }
}


void EcalTBH4BeamSD::getBaseNumber(const G4Step* aStep) {

  theBaseNumber.reset();
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int theSize = touch->GetHistoryDepth()+1;
  if ( theBaseNumber.getCapacity() < theSize ) theBaseNumber.setSize(theSize);
  //Get name and copy numbers
  if ( theSize > 1 ) {
    for (int ii = 0; ii < theSize ; ii++) {
      theBaseNumber.addLevel(touch->GetVolume(ii)->GetName(),touch->GetReplicaNumber(ii));
      LogDebug("EcalTBSim") << "EcalTBH4BeamSD::getBaseNumber(): Adding level " << ii 
                            << ": " << touch->GetVolume(ii)->GetName() << "["
                            << touch->GetReplicaNumber(ii) << "]";
    }
  }
}
