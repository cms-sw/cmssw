#include "SimG4CMS/Forward/interface/BHMSD.h"
#include "SimG4CMS/Forward/interface/BHMNumberingScheme.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"

#include <iostream>

//-------------------------------------------------------------------
BHMSD::BHMSD(const std::string& name,
             const SensitiveDetectorCatalog& clg,
             edm::ParameterSet const& p,
             const SimTrackManager* manager)
    : TimingSD(name, clg, manager) {
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("BHMSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");

  SetVerboseLevel(verbn);

  if (verbn > 0) {
    edm::LogVerbatim("BHMSim") << "name = " << name << " and new BHMNumberingScheme";
  }
}

BHMSD::~BHMSD() {}

uint32_t BHMSD::setDetUnitId(const G4Step* aStep) { return BHMNumberingScheme::getUnitID(aStep); }
