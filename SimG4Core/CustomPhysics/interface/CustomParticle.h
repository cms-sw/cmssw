#ifndef CustomParticle_h
#define CustomParticle_h 1

#include "G4ParticleDefinition.hh"
#include "globals.hh"

// ######################################################################
// ###                          CustomParticle                                ###
// ######################################################################
class CustomParticleFactory;

class CustomParticle : public G4ParticleDefinition
{
 friend class CustomParticleFactory;
 private:
   CustomParticle(
       const G4String&     aName,        G4double            mass,
       G4double            width,        G4double            charge,   
       G4int               iSpin,        G4int               iParity,    
       G4int               iConjugation, G4int               iIsospin,   
       G4int               iIsospin3,    G4int               gParity,
       const G4String&     pType,        G4int               lepton,      
       G4int               baryon,       G4int               encoding,
       G4bool              stable,       G4double            lifetime,
       G4DecayTable        *decaytable
   );
   G4ParticleDefinition* m_cloud;
   G4ParticleDefinition* m_spec;
 public:
   void SetCloud(G4ParticleDefinition* theCloud);
   void SetSpectator(G4ParticleDefinition* theSpectator);
   G4ParticleDefinition* GetCloud();
   G4ParticleDefinition* GetSpectator();
   virtual ~CustomParticle() {}
};

inline void CustomParticle::SetCloud(G4ParticleDefinition* theCloud){ m_cloud = theCloud; }
inline G4ParticleDefinition* CustomParticle::GetCloud(){ return m_cloud; }
inline void CustomParticle::SetSpectator(G4ParticleDefinition* theSpectator){ m_spec = theSpectator; }
inline G4ParticleDefinition* CustomParticle::GetSpectator(){ return m_spec; }

#endif
