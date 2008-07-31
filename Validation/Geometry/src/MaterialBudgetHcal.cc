#include "Validation/Geometry/interface/MaterialBudgetHcal.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "G4Step.hh"
#include "G4Track.hh"

#include <iostream>

MaterialBudgetHcal::MaterialBudgetHcal(const edm::ParameterSet& p) {
  
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("MaterialBudgetHcal");
  rMax = m_p.getUntrackedParameter<double>("RMax", 4.5)*m;
  zMax = m_p.getUntrackedParameter<double>("ZMax", 13.0)*m;
  edm::LogInfo("MaterialBudget") << "MaterialBudgetHcal initialized with rMax "
				 << rMax << " mm and zMax " << zMax << " mm";
  theHistos   = new MaterialBudgetHcalHistos(m_p);

}

MaterialBudgetHcal::~MaterialBudgetHcal() {
  delete theHistos;
}

void MaterialBudgetHcal::update(const BeginOfJob* job)
{
  //----- Check that selected volumes are indeed part of the geometry
  // Numbering From DDD
  edm::ESHandle<DDCompactView> pDD;
  (*job)()->get<IdealGeometryRecord>().get(pDD);
  theHistos->fillBeginJob((*pDD));

}

void MaterialBudgetHcal::update(const BeginOfTrack* trk) {

  const G4Track * aTrack = (*trk)(); // recover G4 pointer if wanted
  theHistos->fillStartTrack(aTrack);
}
 
void MaterialBudgetHcal::update(const G4Step* aStep) {

  //---------- each step
  theHistos->fillPerStep(aStep);

  //----- Stop tracking after selected position
  if (stopAfter(aStep)) {
    G4Track* track = aStep->GetTrack();
    track->SetTrackStatus( fStopAndKill );
  }
}


void MaterialBudgetHcal::update(const EndOfTrack* trk) {

  theHistos->fillEndTrack();
}

bool MaterialBudgetHcal::stopAfter(const G4Step* aStep) {

  G4ThreeVector hitPoint    = aStep->GetPreStepPoint()->GetPosition();
  double        rr = hitPoint.perp();
  double        zz = std::abs(hitPoint.z());

  if (rr > rMax || zz > zMax) {
    LogDebug("MaterialBudget") << " MaterialBudgetHcal::StopAfter R = " << rr
			       << " and Z = " << zz;
    return true;
  } else {
    return false;
  }
}
