#ifndef SimG4Core_GenericHadronicProcess_H
#define SimG4Core_GenericHadronicProcess_H

#include "G4HadronicProcess.hh"

class HadronicProcessHelper;

class GenericHadronicProcess : public G4HadronicProcess
{
public:
    GenericHadronicProcess(const std::string & processName = "LElastic");   
    virtual ~GenericHadronicProcess() {}
    bool IsApplicable(const G4ParticleDefinition & aP);
    G4VParticleChange * PostStepDoIt(const G4Track & aTrack, const G4Step & aStep); 
    void setVerbosity(int level) { m_verboseLevel = level; }
private:    
    virtual double GetMicroscopicCrossSection(const G4DynamicParticle * aParticle, 
					      const G4Element * anElement, 
					      double aTemp);  
    HadronicProcessHelper * m_helper;
    G4ParticleChange m_particleChange;
    int m_verboseLevel;
};

#endif
 
