#ifndef ToyModelHadronicProcess_h
#define ToyModelHadronicProcess_h 1

#include "G4VDiscreteProcess.hh"

class HadronicProcessHelper;
class HistoHelper;
class TProfile;
class TH1D;
class TH2D;


class ToyModelHadronicProcess : public G4VDiscreteProcess
{
public:
  
  ToyModelHadronicProcess(HadronicProcessHelper * aHelper, const G4String& processName = "ToyModelHadronicProcess");
    
  virtual ~ToyModelHadronicProcess(){};
  
  G4bool IsApplicable(const G4ParticleDefinition& aP);
  
  G4VParticleChange *PostStepDoIt(const G4Track &aTrack, const G4Step &aStep);
 
  void setVerbosity(int level) { m_verboseLevel = level; }
private:    
  
  virtual G4double GetMicroscopicCrossSection( const G4DynamicParticle *aParticle, 
					       const G4Element *anElement, 
					       G4double aTemp );
  
  G4double GetMeanFreePath(const G4Track& aTrack, G4double, G4ForceCondition*);
  

  const G4DynamicParticle* FindRhadron(G4ParticleChange*);

  int m_verboseLevel;
  HadronicProcessHelper* m_helper;
  G4ParticleChange m_particleChange;
  bool m_detachCloud;
  


};

#endif
 
