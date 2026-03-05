///////////////////////////////////////////////////////////////////////////////
// File: FscSD.cc
// Date: 02.2026
// Description: Sensitive Detector class for Fsc
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Forward/interface/FscSD.h"
#include "SimG4CMS/Forward/interface/FscNumberingScheme.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"
#include <iostream>

#define EDM_ML_DEBUG

//-------------------------------------------------------------------
FscSD::FscSD(const std::string& name,
             const SensitiveDetectorCatalog& clg,
             edm::ParameterSet const& p,
             const SimTrackManager* manager)
    : TimingSD(name, clg, manager) {
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("FscSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");

  SetVerboseLevel(verbn);

  if (name == "FSCHits") {
    if (verbn > 0)
      edm::LogVerbatim("FscSim") << "name = FSCHits and  new FscNumberingSchem";
  } else {
    edm::LogWarning("FscSim") << "FscSD: ReadoutName " << name << " not supported";
  }
}

FscSD::~FscSD() {}

uint32_t FscSD::setDetUnitId(const G4Step* aStep) { return FscNumberingScheme::getUnitID(aStep); }
