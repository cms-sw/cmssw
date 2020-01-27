
#ifndef G4SIMP_h
#define G4SIMP_h 1

#include "globals.hh"
#include "G4ios.hh"
#include "G4ParticleDefinition.hh"


class G4SIMP : public G4ParticleDefinition {

  private:
    static G4SIMP* theInstance;
    G4SIMP(){}
    ~G4SIMP(){}

  public:
    static G4SIMP* Definition(double mass);
    static G4SIMP* SIMPDefinition(double mass);
    static G4SIMP* SIMP();

};

#endif
