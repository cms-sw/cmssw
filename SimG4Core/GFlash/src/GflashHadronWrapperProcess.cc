//
// S.Y. Jun, August 2007
//
#include "SimG4Core/GFlash/interface/GflashHadronWrapperProcess.h"

#include "G4Track.hh"
#include "G4VParticleChange.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4VProcess.hh"
#include "G4GPILSelection.hh"

GflashHadronWrapperProcess::GflashHadronWrapperProcess(G4String processName) :
  particleChange(0), 
  pmanager(0), 
  fProcessVector(0),
  fProcess(0) 
{
  theProcessName = processName;
}

GflashHadronWrapperProcess::~GflashHadronWrapperProcess() {
}

G4VParticleChange* GflashHadronWrapperProcess::PostStepDoIt(const G4Track& track, const G4Step& step)
{
  // process PostStepDoIt for the original process

  particleChange = pRegProcess->PostStepDoIt(track, step);

  // specific actions of the wrapper process 

  // update step/track information after PostStep of the original process is done
  // these steps will be repeated again without additional conflicts even if the 
  // parameterized physics process doesn't take over the original process

  particleChange->UpdateStepForPostStep(const_cast<G4Step *> (&step));

  // we may not want to update G4Track according to particleChange
  //  (const_cast<G4Step *> (&step))->UpdateTrack();

  // update safety after each invocation of PostStepDoIts
  // G4double safety = std::max(endpointSafety - (endpointSafOrigin - fPostStepPoint->GetPosition()).mag(),0.);
  // step.GetPostStepPoint()->SetSafety(safety);

  // the secondaries from the original process will not be created at the wrapper
  // process, so they will not be deleted them if the parameterized physics is not
  // an 'ExclusivelyForced' process.  Normally the secondaries should be created after
  // this wrapper process in G4SteppingManager::InvokePSDIP

  // We can still impose conditions on the secondaries,
  // such as the number of secondaries produced by the original process - for the 
  // hadronic inelastic interaction, it is always true that a few of secondaries are 
  // created, i.e., particleChange->GetNumberOfSecondaries() > 0

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

  (const_cast<G4Track *> (&track))->SetTrackStatus(fPostponeToNextEvent);
  
  for(G4int ipm = 0 ; ipm < fProcessVector->entries() ; ipm++) {
    fProcess = (*fProcessVector)(ipm);  

    if ( fProcess->GetProcessType() == fParameterisation ) {
      // test ModelTrigger via PostStepGPIL

      testGPIL = fProcess->PostStepGPIL(track,fStepLength,&fForceCondition );

      //restore TrackStatus if fForceCondition !=  ExclusivelyForced
      (const_cast<G4Track *> (&track))->SetTrackStatus(keepStatus);

      // if G4FastSimulationModel:: ModelTrigger is true, then the parameterized 
      // physics process takes over the current process 
      
      if( fForceCondition == ExclusivelyForced) {     

	// updating G4Step between PostStepGPIL and PostStepDoIt for the parameterized 
        // process may not be necessary, but do it anyway

	(const_cast<G4Step *> (&step))->SetStepLength(testGPIL);
        (const_cast<G4Track *> (&track))->SetStepLength(testGPIL);

	step.GetPostStepPoint()->SetStepStatus(fExclusivelyForcedProc);;
	step.GetPostStepPoint()->SetProcessDefinedStep(fProcess);
        step.GetPostStepPoint()->SetSafety(0.0);

	// invoke PostStepDoIt: equivalent steps for G4SteppingManager::InvokePSDIP 
        particleChange = fProcess->PostStepDoIt(track,step);

	// update PostStepPoint of Step according to ParticleChange
	particleChange->UpdateStepForPostStep(const_cast<G4Step *> (&step));

	// update G4Track according to ParticleChange after each PostStepDoIt
        (const_cast<G4Step *> (&step))->UpdateTrack();

	// update safety after each invocation of PostStepDoIts - acutally this
	// is not necessary for the parameterized physics process, but do it anyway
        step.GetPostStepPoint()->SetSafety(0.0);

	// additional nullification 

	(const_cast<G4Track *> (&track))->SetTrackStatus( particleChange->GetTrackStatus() );

	//	(const_cast<G4Step *> (&step))->DeleteSecondaryVector();

      }
      // assume that there is one and only one parameterized physics
      break;
    }
  }

  return particleChange;
}

void GflashHadronWrapperProcess::Print(const G4Step& step) {

  std::cout << " GflashHadronWrapperProcess ProcessName, PreStepPosition, preStepPoint KE, PostStepPoint KE, DeltaEnergy Nsec \n " 
         << step.GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() << " " 
         << step.GetPostStepPoint()->GetPosition() << " "  
         << step.GetPreStepPoint()->GetKineticEnergy()/GeV  << " "  
         << step.GetPostStepPoint()->GetKineticEnergy()/GeV  << " " 
         << step.GetDeltaEnergy()/GeV << " "
         << particleChange->GetNumberOfSecondaries() << std::endl;
}
