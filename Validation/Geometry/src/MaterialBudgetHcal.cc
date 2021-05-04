#include "Validation/Geometry/interface/MaterialBudgetHcal.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

#include "G4Step.hh"
#include "G4Track.hh"

#include <iostream>
#include <memory>

MaterialBudgetHcal::MaterialBudgetHcal(const edm::ParameterSet& p) {
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("MaterialBudgetHcal");
  rMax_ = m_p.getUntrackedParameter<double>("RMax", 4.5) * CLHEP::m;
  zMax_ = m_p.getUntrackedParameter<double>("ZMax", 13.0) * CLHEP::m;
  fromdd4hep_ = m_p.getUntrackedParameter<bool>("Fromdd4hep", false);
  bool doHcal = m_p.getUntrackedParameter<bool>("DoHCAL", true);
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcal initialized with rMax " << rMax_ << " mm and zMax " << zMax_
                                     << " mm doHcal is set to " << doHcal << " and Fromdd4hep to " << fromdd4hep_;
  if (doHcal) {
    theHistoHcal_ = std::make_unique<MaterialBudgetHcalHistos>(m_p);
    theHistoCastor_.reset(nullptr);
  } else {
    theHistoHcal_.reset(nullptr);
    theHistoCastor_ = std::make_unique<MaterialBudgetCastorHistos>(m_p);
  }
}

void MaterialBudgetHcal::update(const BeginOfJob* job) {
  //----- Check that selected volumes are indeed part of the geometry
  // Numbering From DDD
  if (fromdd4hep_) {
    edm::ESTransientHandle<cms::DDCompactView> pDD;
    (*job)()->get<IdealGeometryRecord>().get(pDD);
    if (theHistoHcal_)
      theHistoHcal_->fillBeginJob((*pDD));
  } else {
    edm::ESTransientHandle<DDCompactView> pDD;
    (*job)()->get<IdealGeometryRecord>().get(pDD);
    if (theHistoHcal_)
      theHistoHcal_->fillBeginJob((*pDD));
  }
}

void MaterialBudgetHcal::update(const BeginOfTrack* trk) {
  const G4Track* aTrack = (*trk)();  // recover G4 pointer if wanted
  if (theHistoHcal_)
    theHistoHcal_->fillStartTrack(aTrack);
  if (theHistoCastor_)
    theHistoCastor_->fillStartTrack(aTrack);
}

void MaterialBudgetHcal::update(const G4Step* aStep) {
  //---------- each step
  if (theHistoHcal_)
    theHistoHcal_->fillPerStep(aStep);
  if (theHistoCastor_)
    theHistoCastor_->fillPerStep(aStep);

  //----- Stop tracking after selected position
  if (stopAfter(aStep)) {
    G4Track* track = aStep->GetTrack();
    track->SetTrackStatus(fStopAndKill);
  }
}

void MaterialBudgetHcal::update(const EndOfTrack* trk) {
  if (theHistoHcal_)
    theHistoHcal_->fillEndTrack();
  if (theHistoCastor_)
    theHistoCastor_->fillEndTrack();
}

bool MaterialBudgetHcal::stopAfter(const G4Step* aStep) {
  G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
  double rr = hitPoint.perp();
  double zz = std::abs(hitPoint.z());

  if (rr > rMax_ || zz > zMax_) {
    edm::LogVerbatim("MaterialBudget") << " MaterialBudgetHcal::StopAfter R = " << rr << " and Z = " << zz;
    return true;
  } else {
    return false;
  }
}
