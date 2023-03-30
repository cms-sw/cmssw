#include "G4Version.hh"
#if G4VERSION_NUMBER >= 1100

#include "SimG4Core/PhysicsLists/interface/CMSHepEmTrackingManager.h"

#include "G4EventManager.hh"
#include "G4ProcessManager.hh"
#include "G4StackManager.hh"
#include "G4TrackingManager.hh"

CMSHepEmTrackingManager::CMSHepEmTrackingManager(G4double highEnergyLimit) : fHighEnergyLimit(highEnergyLimit) {}

CMSHepEmTrackingManager::~CMSHepEmTrackingManager() = default;

void CMSHepEmTrackingManager::BuildPhysicsTable(const G4ParticleDefinition& part) {
  G4HepEmTrackingManager::BuildPhysicsTable(part);

  G4ProcessManager* pManager = part.GetProcessManager();
  G4ProcessManager* pManagerShadow = part.GetMasterProcessManager();

  G4ProcessVector* pVector = pManager->GetProcessList();
  for (std::size_t j = 0; j < pVector->size(); ++j) {
    if (pManagerShadow == pManager) {
      (*pVector)[j]->BuildPhysicsTable(part);
    } else {
      (*pVector)[j]->BuildWorkerPhysicsTable(part);
    }
  }
}

void CMSHepEmTrackingManager::PreparePhysicsTable(const G4ParticleDefinition& part) {
  G4HepEmTrackingManager::PreparePhysicsTable(part);

  G4ProcessManager* pManager = part.GetProcessManager();
  G4ProcessManager* pManagerShadow = part.GetMasterProcessManager();

  G4ProcessVector* pVector = pManager->GetProcessList();
  for (std::size_t j = 0; j < pVector->size(); ++j) {
    if (pManagerShadow == pManager) {
      (*pVector)[j]->PreparePhysicsTable(part);
    } else {
      (*pVector)[j]->PrepareWorkerPhysicsTable(part);
    }
  }
}

void CMSHepEmTrackingManager::HandOverOneTrack(G4Track* aTrack) {
  if (aTrack->GetKineticEnergy() < fHighEnergyLimit) {
    // Fully track with G4HepEm.
    G4HepEmTrackingManager::HandOverOneTrack(aTrack);
  } else {
    // Track with the Geant4 kernel and all registered processes.
    G4EventManager* eventManager = G4EventManager::GetEventManager();
    G4TrackingManager* trackManager = eventManager->GetTrackingManager();

    trackManager->ProcessOneTrack(aTrack);
    if (aTrack->GetTrackStatus() != fStopAndKill) {
      G4Exception("CMSHepEmTrackingManager::HandOverOneTrack", "NotStopped", FatalException, "track was not stopped");
    }

    G4TrackVector* secondaries = trackManager->GimmeSecondaries();
    eventManager->StackTracks(secondaries);
    delete aTrack;
  }
}

#endif
