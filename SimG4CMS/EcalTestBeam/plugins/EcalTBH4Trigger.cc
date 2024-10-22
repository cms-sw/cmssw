// -*- C++ -*-
//
// Package:     HelpfulWatchers
// Class  :     EcalTBH4Trigger
//
/**\class EcalTBH4Trigger EcalTBH4Trigger.h
 SimG4Core/HelpfulWatchers/interface/EcalTBH4Trigger.h

 Description: Simulates ECALTBH4 trigger an throw exception in case of non
 triggering event

 Usage:
    <usage>

*/
//

// system include files
#include <sstream>

// user include files
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Step.hh"
#include "G4VProcess.hh"
#include "G4VTouchable.hh"

#include <CLHEP/Units/SystemOfUnits.h>

class EcalTBH4Trigger : public SimWatcher,
                        public Observer<const BeginOfEvent *>,
                        public Observer<const G4Step *>,
                        public Observer<const EndOfEvent *> {
public:
  EcalTBH4Trigger(const edm::ParameterSet &pSet)
      : m_verbose(pSet.getUntrackedParameter<bool>("verbose", false)),
        nTriggeredEvents_(0),
        trigEvents_(pSet.getUntrackedParameter<int>("trigEvents", -1)) {}
  // virtual ~EcalTBH4Trigger();

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void update(const BeginOfEvent *anEvent) override;
  void update(const G4Step *iStep) override;
  void update(const EndOfEvent *anEvent) override;

private:
  // ---------- member data --------------------------------
  bool m_verbose;
  bool m_passedTrg1;
  bool m_passedTrg3;
  bool m_passedTrg4;
  bool m_passedTrg5;
  bool m_passedTrg6;
  int nTriggeredEvents_;
  int trigEvents_;
};

void EcalTBH4Trigger::update(const BeginOfEvent *anEvent) {
  m_passedTrg1 = false;
  m_passedTrg3 = false;
  m_passedTrg4 = false;
  m_passedTrg5 = false;
  m_passedTrg6 = false;
}

void EcalTBH4Trigger::update(const G4Step *iStep) {
  if (trigEvents_ >= 0 && nTriggeredEvents_ >= trigEvents_)
    throw SimG4Exception("Number of needed trigger events reached in ECALTBH4");

  const G4StepPoint *pre = iStep->GetPreStepPoint();
  const G4StepPoint *post = iStep->GetPostStepPoint();
  if (m_verbose) {
    std::ostringstream st1;
    st1 << "++ signal G4Step";
    const G4VTouchable *touch = iStep->GetPreStepPoint()->GetTouchable();
    // Get name and copy numbers
    if (touch->GetHistoryDepth() > 0) {
      for (int ii = 0; ii <= touch->GetHistoryDepth(); ii++) {
        st1 << "EcalTBH4::Level " << ii << ": " << touch->GetVolume(ii)->GetName() << "[" << touch->GetReplicaNumber(ii)
            << "]";
      }
    }
    edm::LogVerbatim("EcalTBInfo") << st1.str();

    std::ostringstream st2;
    const G4Track *theTrack = iStep->GetTrack();
    const G4ThreeVector &pos = post->GetPosition();
    st2 << "( " << pos.x() << "," << pos.y() << "," << pos.z() << ") ";
    st2 << " released energy (MeV) " << iStep->GetTotalEnergyDeposit() / CLHEP::MeV;
    if (theTrack) {
      const G4ThreeVector mom = theTrack->GetMomentum();
      st2 << " track length (cm) " << theTrack->GetTrackLength() / CLHEP::cm << " particle type "
          << theTrack->GetDefinition()->GetParticleName() << " momentum "
          << "( " << mom.x() << "," << mom.y() << "," << mom.z() << ") ";
      if (theTrack->GetCreatorProcess()) {
        st2 << " created by " << theTrack->GetCreatorProcess()->GetProcessName();
      }
    }
    if (post->GetPhysicalVolume()) {
      st2 << " " << pre->GetPhysicalVolume()->GetName() << "->" << post->GetPhysicalVolume()->GetName();
    }
    edm::LogVerbatim("EcalTBInfo") << st2.str();
  }

  if (post && post->GetPhysicalVolume()) {
    if (!m_passedTrg1 && post->GetPhysicalVolume()->GetName() == "TRG1")
      m_passedTrg1 = true;
    if (!m_passedTrg3 && post->GetPhysicalVolume()->GetName() == "TRG3")
      m_passedTrg3 = true;
    if (!m_passedTrg4 && post->GetPhysicalVolume()->GetName() == "TRG4")
      m_passedTrg4 = true;
    if (!m_passedTrg5 && post->GetPhysicalVolume()->GetName() == "TRG5")
      m_passedTrg5 = true;
    if (!m_passedTrg6 && post->GetPhysicalVolume()->GetName() == "TRG6")
      m_passedTrg6 = true;
    if (post->GetPhysicalVolume()->GetName() == "CMSSE")  // Exiting TBH4BeamLine
      if (!(m_passedTrg1 && m_passedTrg6))                // Trigger defined as Trg4 && Trg6
        throw SimG4Exception("Event is not triggered by ECALTBH4");
  }

  /*     if (!m_enteringTBH4BeamLine && ( post->GetPhysicalVolume()->GetName()
   * ==  */
}

void EcalTBH4Trigger::update(const EndOfEvent *anEvent) {
  edm::LogVerbatim("EcalTBInfo") << "++ signal BeginOfEvent ";
  nTriggeredEvents_++;
}

DEFINE_SIMWATCHER(EcalTBH4Trigger);
