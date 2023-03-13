
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4Track.hh"
#include "G4TrackStatus.hh"
#include "G4Step.hh"
#include "G4SteppingManager.hh"
#include "G4SteppingVerbose.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"

CMSSteppingVerbose::CMSSteppingVerbose(
    G4int verb, G4double ekin, std::vector<G4int>& evtNum, std::vector<G4int>& primV, std::vector<G4int>& trNum)
    : m_PrintEvent(false),
      m_PrintTrack(false),
      m_smInitialized(false),
      m_verbose(verb),
      m_EventNumbers(evtNum),
      m_PrimaryVertex(primV),
      m_TrackNumbers(trNum),
      m_EkinThreshold(ekin) {
  m_nEvents = m_EventNumbers.size();
  m_nVertex = m_PrimaryVertex.size();
  m_nTracks = m_TrackNumbers.size();
  m_g4SteppingVerbose = new G4SteppingVerbose();
  G4VSteppingVerbose::SetInstance(m_g4SteppingVerbose);
  m_g4SteppingVerbose->SetSilent(1);
}

void CMSSteppingVerbose::beginOfEvent(const G4Event* evt) {
  m_PrintEvent = false;
  if (0 >= m_verbose) {
    return;
  }
  if (m_nEvents == 0) {
    m_PrintEvent = true;
  } else {
    for (G4int i = 0; i < m_nEvents; ++i) {
      // check event number
      if (evt->GetEventID() == m_EventNumbers[i]) {
        // check number of vertex
        if (m_nVertex == m_nEvents && evt->GetNumberOfPrimaryVertex() != m_PrimaryVertex[i]) {
          continue;
        }
        m_PrintEvent = true;
        break;
      }
    }
  }
  if (!m_PrintEvent) {
    return;
  }
  G4cout << "========== Event #" << evt->GetEventID() << "   " << evt->GetNumberOfPrimaryVertex()
         << " primary vertexes ======" << G4endl;
  G4cout << G4endl;
}

void CMSSteppingVerbose::stopEventPrint() {
  m_PrintEvent = false;
  m_PrintTrack = false;
  m_verbose = 0;
}

void CMSSteppingVerbose::trackStarted(const G4Track* track, bool isKilled) {
  m_PrintTrack = false;
  if (!m_PrintEvent) {
    return;
  }

  if (!m_smInitialized) {
    G4SteppingManager* stepman = G4EventManager::GetEventManager()->GetTrackingManager()->GetSteppingManager();
    m_g4SteppingVerbose->SetManager(stepman);
    stepman->SetVerboseLevel(m_verbose);
    m_smInitialized = true;
  }

  if (m_nTracks == 0) {
    if (track->GetKineticEnergy() >= m_EkinThreshold) {
      m_PrintTrack = true;
    }

  } else {
    for (G4int i = 0; i < m_nTracks; ++i) {
      if (track->GetTrackID() == m_TrackNumbers[i]) {
        m_PrintTrack = true;
        break;
      }
    }
  }
  if (!m_PrintTrack) {
    return;
  }

  G4cout << "*********************************************************************************************************"
         << G4endl;
  const G4ParticleDefinition* pd = track->GetDefinition();
  G4cout << "* G4Track Information:   Particle = ";
  if (pd) {
    G4cout << pd->GetParticleName();
  }
  G4cout << ",   Track ID = " << track->GetTrackID() << ",   Parent ID = " << track->GetParentID() << G4endl;
  G4cout << "*********************************************************************************************************"
         << G4endl;

  G4cout << std::setw(5) << "Step#"
         << " " << std::setw(8) << "X(cm)"
         << " " << std::setw(8) << "Y(cm)"
         << " " << std::setw(8) << "Z(cm)"
         << " " << std::setw(9) << "KinE(GeV)"
         << " " << std::setw(8) << "dE(MeV)"
         << " " << std::setw(8) << "Step(mm)"
         << " " << std::setw(9) << "TrackL(cm)"
         << " " << std::setw(30) << "PhysVolume"
         << " " << std::setw(8) << "ProcName" << G4endl;

  G4int prec = G4cout.precision(4);

  G4cout << std::setw(5) << track->GetCurrentStepNumber() << " " << std::setw(8) << track->GetPosition().x() / CLHEP::cm
         << " " << std::setw(8) << track->GetPosition().y() / CLHEP::cm << " " << std::setw(8)
         << track->GetPosition().z() / CLHEP::cm << " " << std::setw(9) << track->GetKineticEnergy() / CLHEP::GeV << " "
         << std::setw(8) << "         " << std::setw(8) << "         " << std::setw(9) << "          ";
  if (track->GetVolume() != nullptr) {
    G4cout << std::setw(30) << track->GetVolume()->GetName() << " ";
  }
  if (isKilled) {
    G4cout << "isKilled";
  }
  G4cout << G4endl;
  G4cout.precision(prec);
}

void CMSSteppingVerbose::stackFilled(const G4Track* track, bool isKilled) const {
  if (2 >= m_verbose || !m_PrintTrack || track->GetKineticEnergy() < m_EkinThreshold) {
    return;
  }
  G4int prec = G4cout.precision(4);

  G4cout << std::setw(10) << track->GetTrackID() << " " << std::setw(8) << track->GetPosition().x() / CLHEP::cm << " "
         << std::setw(8) << track->GetPosition().y() / CLHEP::cm << " " << std::setw(8)
         << track->GetPosition().z() / CLHEP::cm << " " << std::setw(9) << track->GetKineticEnergy() / CLHEP::GeV
         << " ";
  if (track->GetVolume() != nullptr) {
    G4cout << std::setw(24) << track->GetVolume()->GetName() << " ";
  }
  if (isKilled) {
    G4cout << "isKilled";
  }
  G4cout << G4endl;
  G4cout.precision(prec);
}

void CMSSteppingVerbose::nextStep(const G4Step* step, const G4SteppingManager* sManager, bool isKilled) const {
  if (!m_PrintTrack) {
    return;
  }

  G4int prec;
  const G4Track* track = step->GetTrack();
  const G4StepPoint* preStep = step->GetPreStepPoint();
  const G4StepPoint* postStep = step->GetPostStepPoint();

  if (3 <= m_verbose) {
    m_g4SteppingVerbose->SetSilent(0);
    m_g4SteppingVerbose->DPSLStarted();
    m_g4SteppingVerbose->DPSLAlongStep();
    m_g4SteppingVerbose->DPSLPostStep();
    if (4 <= m_verbose) {
      m_g4SteppingVerbose->AlongStepDoItAllDone();
      m_g4SteppingVerbose->PostStepDoItAllDone();
    }
    m_g4SteppingVerbose->SetSilent(1);

    prec = G4cout.precision(16);

    G4cout << G4endl;
    G4cout << "    ++G4Step Information " << G4endl;
    G4cout << "      Address of G4Track    : " << track << G4endl;
    G4cout << "      Step Length (mm)      : " << track->GetStepLength() << G4endl;
    G4cout << "      Energy Deposit (MeV)  : " << step->GetTotalEnergyDeposit() << G4endl;

    G4cout << "   -------------------------------------------------------"
           << "-------------------------------" << G4endl;
    G4cout << "  StepPoint Information  " << std::setw(30) << "PreStep" << std::setw(30) << "PostStep" << G4endl;
    G4cout << "   -------------------------------------------------------"
           << "-------------------------------" << G4endl;
    G4cout << "      Position - x (cm)   : " << std::setw(30) << preStep->GetPosition().x() / CLHEP::cm << std::setw(30)
           << postStep->GetPosition().x() / CLHEP::cm << G4endl;
    G4cout << "      Position - y (cm)   : " << std::setw(30) << preStep->GetPosition().y() / CLHEP::cm << std::setw(30)
           << postStep->GetPosition().y() / CLHEP::cm << G4endl;
    G4cout << "      Position - z (cm)   : " << std::setw(30) << preStep->GetPosition().z() / CLHEP::cm << std::setw(30)
           << postStep->GetPosition().z() / CLHEP::cm << G4endl;
    G4cout << "      Global Time (ns)    : " << std::setw(30) << preStep->GetGlobalTime() / CLHEP::ns << std::setw(30)
           << postStep->GetGlobalTime() / CLHEP::ns << G4endl;
    G4cout << "      Local Time (ns)     : " << std::setw(30) << preStep->GetLocalTime() / CLHEP::ns << std::setw(30)
           << postStep->GetLocalTime() / CLHEP::ns << G4endl;
    G4cout << "      Proper Time (ns)    : " << std::setw(30) << preStep->GetProperTime() / CLHEP::ns << std::setw(30)
           << postStep->GetProperTime() / CLHEP::ns << G4endl;
    G4cout << "      Momentum Direct - x : " << std::setw(30) << preStep->GetMomentumDirection().x() << std::setw(30)
           << postStep->GetMomentumDirection().x() << G4endl;
    G4cout << "      Momentum Direct - y : " << std::setw(30) << preStep->GetMomentumDirection().y() << std::setw(30)
           << postStep->GetMomentumDirection().y() << G4endl;
    G4cout << "      Momentum Direct - z : " << std::setw(30) << preStep->GetMomentumDirection().z() << std::setw(30)
           << postStep->GetMomentumDirection().z() << G4endl;
    G4cout << "      Momentum - x (GeV/c): " << std::setw(30) << preStep->GetMomentum().x() / CLHEP::GeV
           << std::setw(30) << postStep->GetMomentum().x() / CLHEP::GeV << G4endl;
    G4cout << "      Momentum - y (GeV/c): " << std::setw(30) << preStep->GetMomentum().y() / CLHEP::GeV
           << std::setw(30) << postStep->GetMomentum().y() / CLHEP::GeV << G4endl;
    G4cout << "      Momentum - z (GeV/c): " << std::setw(30) << preStep->GetMomentum().z() / CLHEP::GeV
           << std::setw(30) << postStep->GetMomentum().z() / CLHEP::GeV << G4endl;
    G4cout << "      Total Energy (GeV)  : " << std::setw(30) << preStep->GetTotalEnergy() / CLHEP::GeV << std::setw(30)
           << postStep->GetTotalEnergy() / CLHEP::GeV << G4endl;
    G4cout << "      Kinetic Energy (GeV): " << std::setw(30) << preStep->GetKineticEnergy() / CLHEP::GeV
           << std::setw(30) << postStep->GetKineticEnergy() / CLHEP::GeV << G4endl;
    G4cout << "      Velocity (mm/ns)    : " << std::setw(30) << preStep->GetVelocity() << std::setw(30)
           << postStep->GetVelocity() << G4endl;
    G4cout << "      Volume Name         : " << std::setw(30) << preStep->GetPhysicalVolume()->GetName();
    G4String volName = (postStep->GetPhysicalVolume()) ? postStep->GetPhysicalVolume()->GetName() : "OutOfWorld";

    G4cout << std::setw(30) << volName << G4endl;
    G4cout << "      Safety (mm)         : " << std::setw(30) << preStep->GetSafety() << std::setw(30)
           << postStep->GetSafety() << G4endl;
    G4cout << "      Polarization - x    : " << std::setw(30) << preStep->GetPolarization().x() << std::setw(30)
           << postStep->GetPolarization().x() << G4endl;
    G4cout << "      Polarization - y    : " << std::setw(30) << preStep->GetPolarization().y() << std::setw(30)
           << postStep->GetPolarization().y() << G4endl;
    G4cout << "      Polarization - Z    : " << std::setw(30) << preStep->GetPolarization().z() << std::setw(30)
           << postStep->GetPolarization().z() << G4endl;
    G4cout << "      Weight              : " << std::setw(30) << preStep->GetWeight() << std::setw(30)
           << postStep->GetWeight() << G4endl;
    G4cout << "      Step Status         : ";
    G4StepStatus tStepStatus = preStep->GetStepStatus();
    if (tStepStatus == fGeomBoundary) {
      G4cout << std::setw(30) << "Geom Limit";
    } else if (tStepStatus == fAlongStepDoItProc) {
      G4cout << std::setw(30) << "AlongStep Proc.";
    } else if (tStepStatus == fPostStepDoItProc) {
      G4cout << std::setw(30) << "PostStep Proc";
    } else if (tStepStatus == fAtRestDoItProc) {
      G4cout << std::setw(30) << "AtRest Proc";
    } else if (tStepStatus == fUndefined) {
      G4cout << std::setw(30) << "Undefined";
    }

    tStepStatus = postStep->GetStepStatus();
    if (tStepStatus == fGeomBoundary) {
      G4cout << std::setw(30) << "Geom Limit";
    } else if (tStepStatus == fAlongStepDoItProc) {
      G4cout << std::setw(30) << "AlongStep Proc.";
    } else if (tStepStatus == fPostStepDoItProc) {
      G4cout << std::setw(30) << "PostStep Proc";
    } else if (tStepStatus == fAtRestDoItProc) {
      G4cout << std::setw(30) << "AtRest Proc";
    } else if (tStepStatus == fUndefined) {
      G4cout << std::setw(30) << "Undefined";
    }

    G4cout << G4endl;
    G4cout << "      Process defined Step: ";
    if (preStep->GetProcessDefinedStep() == nullptr) {
      G4cout << std::setw(30) << "Undefined";
    } else {
      G4cout << std::setw(30) << preStep->GetProcessDefinedStep()->GetProcessName();
    }
    if (postStep->GetProcessDefinedStep() == nullptr) {
      G4cout << std::setw(30) << "Undefined";
    } else {
      G4cout << std::setw(30) << postStep->GetProcessDefinedStep()->GetProcessName();
    }
    G4cout.precision(prec);

    G4cout << G4endl;
    G4cout << "   -------------------------------------------------------"
           << "-------------------------------" << G4endl;
  }

  // geometry information
  if (4 <= m_verbose) {
    const G4VTouchable* tch1 = preStep->GetTouchable();
    const G4VTouchable* tch2 = postStep->GetTouchable();

    G4double x = postStep->GetPosition().x();
    G4double y = postStep->GetPosition().y();
    G4cout << "Touchable depth pre= " << tch1->GetHistoryDepth() << " post= " << tch2->GetHistoryDepth()
           << " trans1= " << tch1->GetTranslation(tch1->GetHistoryDepth())
           << " trans2= " << tch2->GetTranslation(tch2->GetHistoryDepth()) << " r= " << std::sqrt(x * x + y * y)
           << G4endl;
    const G4VPhysicalVolume* pv1 = preStep->GetPhysicalVolume();
    const G4VPhysicalVolume* pv2 = postStep->GetPhysicalVolume();
    const G4RotationMatrix* rotm = pv1->GetFrameRotation();
    G4cout << "PreStepVolume: " << pv1->GetName() << G4endl;
    G4cout << "       Translation: " << pv1->GetObjectTranslation() << G4endl;
    if (nullptr != rotm)
      G4cout << "       Rotation:    " << *rotm << G4endl;
    const G4VSolid* sv1 = pv1->GetLogicalVolume()->GetSolid();
    sv1->StreamInfo(G4cout);
    G4cout << G4endl;
    if (pv2 && pv2 != pv1) {
      G4cout << "PostStepVolume: " << pv2->GetName() << G4endl;
      G4cout << "       Translation: " << pv2->GetObjectTranslation() << G4endl;
      rotm = pv2->GetFrameRotation();
      if (nullptr != rotm)
        G4cout << "       Rotation:    " << *rotm << G4endl;
      const G4VSolid* sv2 = pv2->GetLogicalVolume()->GetSolid();
      sv2->StreamInfo(G4cout);
    }
    G4cout << G4endl;

    if (5 <= m_verbose) {
      G4cout << "##### Geometry Depth Analysis" << G4endl;
      for (G4int k = 1; k < tch1->GetHistoryDepth(); ++k) {
        const G4VPhysicalVolume* pv = tch1->GetVolume(k);
        if (pv) {
          const G4VSolid* sol = pv->GetLogicalVolume()->GetSolid();
          G4cout << " Depth # " << k << "  PhysVolume " << pv->GetName() << G4endl;
          G4cout << "       Translation: " << pv->GetObjectTranslation() << G4endl;
          const G4RotationMatrix* rotm = pv->GetFrameRotation();
          if (nullptr != rotm)
            G4cout << "       Rotation:    " << *rotm << G4endl;
          sol->StreamInfo(G4cout);
        }
      }
    }
    G4cout << G4endl;
  }

  // verbose == 1
  prec = G4cout.precision(4);
  G4cout << std::setw(5) << track->GetCurrentStepNumber() << " " << std::setw(8) << track->GetPosition().x() / CLHEP::cm
         << " " << std::setw(8) << track->GetPosition().y() / CLHEP::cm << " " << std::setw(8)
         << track->GetPosition().z() / CLHEP::cm << " " << std::setw(9) << track->GetKineticEnergy() / CLHEP::GeV
         << " ";
  G4cout.precision(prec);
  prec = G4cout.precision(3);
  G4cout << std::setw(8) << step->GetTotalEnergyDeposit() / CLHEP::MeV << " " << std::setw(8)
         << step->GetStepLength() / CLHEP::mm << " " << std::setw(9) << track->GetTrackLength() / CLHEP::cm << " ";

  G4bool endTracking = false;
  if (track->GetNextVolume() != nullptr) {
    G4cout << std::setw(30) << track->GetVolume()->GetName() << " ";
  } else {
    G4cout << std::setw(30) << "OutOfWorld"
           << " ";
    endTracking = true;
  }
  if (isKilled) {
    G4cout << "isKilled";
    endTracking = true;
  } else if (postStep->GetProcessDefinedStep() != nullptr) {
    G4cout << postStep->GetProcessDefinedStep()->GetProcessName();
  }
  G4cout << G4endl;
  G4cout.precision(prec);

  if (1 >= m_verbose) {
    return;
  }
  // verbose > 1
  if (!endTracking && fStopAndKill != track->GetTrackStatus()) {
    return;
  }

  prec = G4cout.precision(4);
  const G4TrackVector* tv = step->GetSecondary();
  G4int nt = tv->size();
  if (nt > 0) {
    G4cout << "    ++List of " << nt << " secondaries generated; "
           << " Ekin > " << m_EkinThreshold << " MeV are shown:" << G4endl;
  }
  for (G4int i = 0; i < nt; ++i) {
    if ((*tv)[i]->GetKineticEnergy() < m_EkinThreshold) {
      continue;
    }
    G4cout << "      (" << std::setw(9) << (*tv)[i]->GetPosition().x() / CLHEP::cm << " " << std::setw(9)
           << (*tv)[i]->GetPosition().y() / CLHEP::cm << " " << std::setw(9) << (*tv)[i]->GetPosition().z() / CLHEP::cm
           << ") cm, " << std::setw(9) << (*tv)[i]->GetKineticEnergy() / CLHEP::GeV << " GeV, " << std::setw(9)
           << (*tv)[i]->GetGlobalTime() / CLHEP::ns << " ns, " << std::setw(18)
           << (*tv)[i]->GetDefinition()->GetParticleName() << G4endl;
  }
  G4cout.precision(prec);
}
