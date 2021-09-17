//
// S.Y. Jun, August 2007
//
#include "SimG4Core/GFlash/interface/GflashHadronWrapperProcess.h"

#include "G4GPILSelection.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4Track.hh"
#include "G4VParticleChange.hh"
#include "G4VProcess.hh"

using namespace CLHEP;

GflashHadronWrapperProcess::GflashHadronWrapperProcess(G4String processName)
    : particleChange(nullptr), pmanager(nullptr), fProcessVector(nullptr), fProcess(nullptr) {
  theProcessName = processName;
}

GflashHadronWrapperProcess::~GflashHadronWrapperProcess() {}

G4VParticleChange *GflashHadronWrapperProcess::PostStepDoIt(const G4Track &track, const G4Step &step) {
  // process PostStepDoIt for the original process

  particleChange = pRegProcess->PostStepDoIt(track, step);

  // specific actions of the wrapper process

  // update step/track information after PostStep of the original process is
  // done these steps will be repeated again without additional conflicts even
  // if the parameterized physics process doesn't take over the original process

  particleChange->UpdateStepForPostStep(const_cast<G4Step *>(&step));

  // we may not want to update G4Track during this wrapper process if
  // G4Track accessors are not used in the G4VFastSimulationModel model
  //  (const_cast<G4Step *> (&step))->UpdateTrack();

  // update safety after each invocation of PostStepDoIts
  // G4double safety = std::max(endpointSafety - (endpointSafOrigin -
  // fPostStepPoint->GetPosition()).mag(),0.);
  // step.GetPostStepPoint()->SetSafety(safety);

  // the secondaries from the original process are also created at the wrapper
  // process, but they must be deleted whether the parameterized physics is
  // an 'ExclusivelyForced' process or not.  Normally the secondaries will be
  // created after this wrapper process in G4SteppingManager::InvokePSDIP

  // store the secondaries from ParticleChange to SecondaryList

  G4TrackVector *fSecondary = (const_cast<G4Step *>(&step))->GetfSecondary();
  G4int nSecondarySave = fSecondary->size();

  G4int num2ndaries = particleChange->GetNumberOfSecondaries();
  G4Track *tempSecondaryTrack;

  for (G4int DSecLoop = 0; DSecLoop < num2ndaries; DSecLoop++) {
    tempSecondaryTrack = particleChange->GetSecondary(DSecLoop);

    // Set the process pointer which created this track
    tempSecondaryTrack->SetCreatorProcess(pRegProcess);

    // add secondaries from this wrapper process to the existing list
    fSecondary->push_back(tempSecondaryTrack);

  }  // end of loop on secondary

  // Now we can still impose conditions on the secondaries from ModelTrigger,
  // such as the number of secondaries produced by the original process as well
  // as those by other continuous processes - for the hadronic inelastic
  // interaction, it is always true that
  // particleChange->GetNumberOfSecondaries() > 0

  // at this stage, updated step information after all processes involved
  // should be available

  //  Print(step);

  // call ModelTrigger for the second time - the primary loop of PostStepGPIL
  // is inside G4SteppingManager::DefinePhysicalStepLength().

  pmanager = track.GetDefinition()->GetProcessManager();

  G4double testGPIL = DBL_MAX;
  G4double fStepLength = 0.0;
  G4ForceCondition fForceCondition = InActivated;

  fStepLength = step.GetStepLength();

  fProcessVector = pmanager->GetPostStepProcessVector(typeDoIt);

  // keep the current status of track, use fPostponeToNextEvent word for
  // this particular PostStep GPIL and then restore G4TrackStatus if the
  // paramterized physics doesn't meet trigger conditions in ModelTrigger

  const G4TrackStatus keepStatus = track.GetTrackStatus();

  (const_cast<G4Track *>(&track))->SetTrackStatus(fPostponeToNextEvent);
  int fpv_entries = fProcessVector->entries();
  for (G4int ipm = 0; ipm < fpv_entries; ipm++) {
    fProcess = (*fProcessVector)(ipm);

    if (fProcess->GetProcessType() == fParameterisation) {
      // test ModelTrigger via PostStepGPIL

      testGPIL = fProcess->PostStepGPIL(track, fStepLength, &fForceCondition);

      // if G4FastSimulationModel:: ModelTrigger is true, then the parameterized
      // physics process takes over the current process

      if (fForceCondition == ExclusivelyForced) {
        // clean up memory for changing the process - counter clean up for
        // the secondaries created by new G4Track in
        // G4HadronicProcess::FillTotalResult
        G4int nsec = particleChange->GetNumberOfSecondaries();
        for (G4int DSecLoop = 0; DSecLoop < nsec; DSecLoop++) {
          G4Track *tempSecondaryTrack = particleChange->GetSecondary(DSecLoop);
          delete tempSecondaryTrack;
        }
        particleChange->Clear();

        // updating G4Step between PostStepGPIL and PostStepDoIt for the
        // parameterized process may not be necessary, but do it anyway

        (const_cast<G4Step *>(&step))->SetStepLength(testGPIL);
        (const_cast<G4Track *>(&track))->SetStepLength(testGPIL);

        step.GetPostStepPoint()->SetStepStatus(fExclusivelyForcedProc);
        ;
        step.GetPostStepPoint()->SetProcessDefinedStep(fProcess);
        step.GetPostStepPoint()->SetSafety(0.0);

        // invoke PostStepDoIt: equivalent steps for
        // G4SteppingManager::InvokePSDIP
        particleChange = fProcess->PostStepDoIt(track, step);

        // update PostStepPoint of Step according to ParticleChange
        particleChange->UpdateStepForPostStep(const_cast<G4Step *>(&step));

        // update G4Track according to ParticleChange after each PostStepDoIt
        (const_cast<G4Step *>(&step))->UpdateTrack();

        // update safety after each invocation of PostStepDoIts - acutally this
        // is not necessary for the parameterized physics process, but do it
        // anyway
        step.GetPostStepPoint()->SetSafety(0.0);

        // additional nullification
        (const_cast<G4Track *>(&track))->SetTrackStatus(particleChange->GetTrackStatus());
      } else {
        // restore TrackStatus if fForceCondition !=  ExclusivelyForced
        (const_cast<G4Track *>(&track))->SetTrackStatus(keepStatus);
      }
      // assume that there is one and only one parameterized physics
      break;
    }
  }

  // remove secondaries of this wrapper process that were added to the secondary
  // list since they will be added in the normal stepping procedure after
  // this->PostStepDoIt in G4SteppingManager::InvokePSDIP

  // move the iterator to the (nSecondarySave+1)th element in the secondary list
  G4TrackVector::iterator itv = fSecondary->begin();
  itv += nSecondarySave;

  // delete next num2ndaries tracks from the secondary list
  fSecondary->erase(itv, itv + num2ndaries);

  // end of specific actions of this wrapper process

  return particleChange;
}

void GflashHadronWrapperProcess::Print(const G4Step &step) {
  std::cout << " GflashHadronWrapperProcess ProcessName, PreStepPosition, "
               "preStepPoint KE, PostStepPoint KE, DeltaEnergy Nsec \n "
            << step.GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() << " "
            << step.GetPostStepPoint()->GetPosition() << " " << step.GetPreStepPoint()->GetKineticEnergy() / GeV << " "
            << step.GetPostStepPoint()->GetKineticEnergy() / GeV << " " << step.GetDeltaEnergy() / GeV << " "
            << particleChange->GetNumberOfSecondaries() << std::endl;
}
