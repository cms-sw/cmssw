#include "SimG4CMS/Forward/interface/MtdSD.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"

#include <iostream>

//#define EDM_ML_DEBUG
//-------------------------------------------------------------------
MtdSD::MtdSD(const std::string& name,
             const SensitiveDetectorCatalog& clg,
             edm::ParameterSet const& p,
             const SimTrackManager* manager)
    : TimingSD(name, clg, manager), numberingScheme(nullptr) {
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("MtdSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");

  SetVerboseLevel(verbn);

  MTDNumberingScheme* scheme = nullptr;
  if (name == "FastTimerHitsBarrel") {
    scheme = dynamic_cast<MTDNumberingScheme*>(new BTLNumberingScheme());
    isBTL = true;
  } else if (name == "FastTimerHitsEndcap") {
    scheme = dynamic_cast<MTDNumberingScheme*>(new ETLNumberingScheme());
    isETL = true;
  } else {
    scheme = nullptr;
    edm::LogWarning("MtdSim") << "MtdSD: ReadoutName not supported";
  }
  if (scheme)
    setNumberingScheme(scheme);

  energyCut =
        m_p.getParameter<double>("EnergyThresholdForPersistencyInGeV") * CLHEP::GeV;  //default must be 0.5
  energyHistoryCut =
        m_p.getParameter<double>("EnergyThresholdForHistoryInGeV") * CLHEP::GeV;  //default must be 0.05

  setCuts(energyCut, energyHistoryCut);

  double newTimeFactor = 1. / m_p.getParameter<double>("TimeSliceUnit");
  edm::LogVerbatim("MtdSim") << "New time factor = " << newTimeFactor;
  setTimeFactor(newTimeFactor);

  edm::LogVerbatim("MtdSim") << "MtdSD: Instantiation completed for " << name;
}

MtdSD::~MtdSD() {}

uint32_t MtdSD::setDetUnitId(const G4Step* aStep) {
  if (numberingScheme == nullptr) {
    return MTDDetId();
  } else {
    getBaseNumber(aStep);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MtdSim") << "DetId = " << numberingScheme->getUnitID(theBaseNumber);
#endif
    return numberingScheme->getUnitID(theBaseNumber);
  }
}

void MtdSD::setNumberingScheme(MTDNumberingScheme* scheme) {
  if (scheme != nullptr) {
    edm::LogVerbatim("MtdSim") << "MtdSD: updates numbering scheme for " << GetName();
    if (numberingScheme)
      delete numberingScheme;
    numberingScheme = scheme;
  }
}

void MtdSD::getBaseNumber(const G4Step* aStep) {
  theBaseNumber.reset();
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int theSize = touch->GetHistoryDepth() + 1;
  if (theBaseNumber.getCapacity() < theSize)
    theBaseNumber.setSize(theSize);
  //Get name and copy numbers
  if (theSize > 1) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MtdSim") << "Building MTD basenumber:";
#endif
    for (int ii = 0; ii < theSize; ii++) {
      theBaseNumber.addLevel(touch->GetVolume(ii)->GetName(), touch->GetReplicaNumber(ii));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MtdSim") << "MtdSD::getBaseNumber(): Adding level " << ii << ": "
                                 << touch->GetVolume(ii)->GetName() << "[" << touch->GetReplicaNumber(ii) << "]";
#endif
    }
  }
}
