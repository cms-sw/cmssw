//
// S.Y. Jun, August 2007
//
#ifndef GflashHadronWrapperProcess_HH
#define GflashHadronWrapperProcess_HH 1

#include "G4WrapperProcess.hh"

class G4VParticleChange;
class G4ProcessManager;
class G4ProcessVector;
class G4VProcess;

class GflashHadronWrapperProcess : public G4WrapperProcess {
public:
  GflashHadronWrapperProcess(G4String processName);
  //  GflashHadronWrapperProcess();

  ~GflashHadronWrapperProcess() override;

  // Override PostStepDoIt  method
  G4VParticleChange *PostStepDoIt(const G4Track &track, const G4Step &step) override;

  G4String GetName() { return theProcessName; };

  void Print(const G4Step &astep);

private:
  G4String theProcessName;

  G4VParticleChange *particleChange;
  G4ProcessManager *pmanager;
  G4ProcessVector *fProcessVector;
  G4VProcess *fProcess;
};

#endif
