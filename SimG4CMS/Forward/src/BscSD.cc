///////////////////////////////////////////////////////////////////////////////
// File: BscSD.cc
// Date: 02.2006
// Description: Sensitive Detector class for Bsc
// Modifications:
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Forward/interface/BscSD.h"
#include "SimG4CMS/Forward/interface/BscG4Hit.h"
#include "SimG4CMS/Forward/interface/BscG4HitCollection.h"
#include "SimG4CMS/Forward/interface/BscNumberingScheme.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"

#include <iostream>

#define debug
//-------------------------------------------------------------------
BscSD::BscSD(const std::string& name,
             const edm::EventSetup& es,
             const SensitiveDetectorCatalog& clg,
             edm::ParameterSet const& p,
             const SimTrackManager* manager)
    : TimingSD(name, clg, manager), numberingScheme(nullptr) {
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("BscSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");

  SetVerboseLevel(verbn);

  if (name == "BSCHits") {
    if (verbn > 0) {
      edm::LogInfo("BscSim") << "name = BSCHits and  new BscNumberingSchem";
    }
    numberingScheme = new BscNumberingScheme();
  } else {
    edm::LogWarning("BscSim") << "BscSD: ReadoutName " << name << " not supported";
  }
}

BscSD::~BscSD() { delete numberingScheme; }

uint32_t BscSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme == nullptr ? 0 : numberingScheme->getUnitID(aStep));
}
