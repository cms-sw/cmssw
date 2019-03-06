
#ifndef G4AntiSIMP_h
#define G4AntiSIMP_h 1

#include "globals.hh"
#include "G4ios.hh"
#include "G4ParticleDefinition.hh"


class G4AntiSIMP : public G4ParticleDefinition {

  private:
    static G4AntiSIMP* theInstance;
    G4AntiSIMP(){}
    ~G4AntiSIMP(){}

  public:
    static G4AntiSIMP* Definition(double mass);
    static G4AntiSIMP* AntiSIMPDefinition(double mass);
    static G4AntiSIMP* AntiSIMP();

};

#endif
