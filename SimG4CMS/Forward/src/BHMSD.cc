#include "SimG4CMS/Forward/interface/BHMSD.h"
#include "SimG4CMS/Forward/interface/BHMNumberingScheme.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"

#include <iostream>

//#define debug
//-------------------------------------------------------------------
BHMSD::BHMSD(const std::string& name,
             const edm::EventSetup& es,
             const SensitiveDetectorCatalog& clg,
             edm::ParameterSet const& p,
             const SimTrackManager* manager)
    : TimingSD(name, clg, manager), numberingScheme(nullptr) {
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("BHMSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");

  SetVerboseLevel(verbn);

  if (verbn > 0) {
    edm::LogInfo("BHMSim") << "name = " << name << " and new BHMNumberingScheme";
  }
  numberingScheme = new BHMNumberingScheme();
}

BHMSD::~BHMSD() { delete numberingScheme; }

uint32_t BHMSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme == nullptr ? 0 : numberingScheme->getUnitID(aStep));
}
