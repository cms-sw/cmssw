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
    : CaloSD(name,
             clg,
             p,
             manager,
             (float)(p.getParameter<edm::ParameterSet>("FscSD").getParameter<double>("TimeSliceUnit")),
             p.getParameter<edm::ParameterSet>("FscSD").getParameter<bool>("IgnoreTrackID")) {
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("FscSD");
  verbn_ = m_p.getUntrackedParameter<int>("Verbosity");
  useBirk_ = m_p.getParameter<bool>("UseBirkLaw");
  birk1_ = m_p.getParameter<double>("BirkC1") * (CLHEP::g / (CLHEP::MeV * CLHEP::cm2));
  birk2_ = m_p.getParameter<double>("BirkC2");
  birk3_ = m_p.getParameter<double>("BirkC3");

  if (verbn_ > 0)
    edm::LogVerbatim("FscSim") << "Use " << name << " and FscNumberingSchem \nUse of Birks law is set to      " << useBirk_ << "  with three constants kB = " << birk1_ << ", C1 = " << birk2_ << ", C2 = " << birk3_;
}

uint32_t FscSD::setDetUnitId(const G4Step* aStep) { return FscNumberingScheme::getUnitID(aStep); }

double FscSD::getEnergyDeposit(const G4Step* aStep) {
  double destep = aStep->GetTotalEnergyDeposit();
  double weight = ((useBirk_) ? getAttenuation(aStep, birk1_, birk2_, birk3_) : 1.0);
  double edep = weight * destep;
  if (verbn_ > 0)
    edm::LogVerbatim("FscSim") << "TotemT2ScintSD: edep= " << destep << ":" << weight << ":" << edep;
  return edep;
}
