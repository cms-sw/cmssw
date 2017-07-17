#ifndef FullModelHadronicProcess_h
#define FullModelHadronicProcess_h 1

#include "globals.hh"
#include "G4VDiscreteProcess.hh"
#include "G4EnergyRangeManager.hh"
#include "G4Nucleus.hh"
#include "G4ReactionProduct.hh"
#include <vector>
#include "G4HadronicException.hh"
#include "FullModelReactionDynamics.hh"

class G4ProcessHelper;
//class HistoHelper;

class FullModelHadronicProcess : public G4VDiscreteProcess
{
public:
  
  FullModelHadronicProcess(G4ProcessHelper * aHelper, 
			   const G4String& processName = "FullModelHadronicProcess");
    
  virtual ~FullModelHadronicProcess();

  
  G4bool IsApplicable(const G4ParticleDefinition& aP);
  
  G4VParticleChange *PostStepDoIt(const G4Track &aTrack, const G4Step &aStep);

protected:
  
  const G4ParticleDefinition *theParticle;
  G4ParticleDefinition *newParticle;
  
  G4ParticleChange theParticleChange;
  
private:    
  
  virtual G4double GetMicroscopicCrossSection( const G4DynamicParticle *aParticle, 
					       const G4Element *anElement, 
					       G4double aTemp );

  G4double GetMeanFreePath(const G4Track& aTrack, G4double, G4ForceCondition*);
  
  void CalculateMomenta( G4FastVector<G4ReactionProduct,MYGHADLISTSIZE> &vec,
			 G4int &vecLen,
			 const G4HadProjectile *originalIncident,
			 const G4DynamicParticle *originalTarget,
			 G4ReactionProduct &modifiedOriginal,
			 G4Nucleus &targetNucleus,
			 G4ReactionProduct &currentParticle,
			 G4ReactionProduct &targetParticle,
			 G4bool &incidentHasChanged,
			 G4bool &targetHasChanged,
			 G4bool quasiElastic );

  G4bool MarkLeadingStrangeParticle(
				    const G4ReactionProduct &currentParticle,
				    const G4ReactionProduct &targetParticle,
				    G4ReactionProduct &leadParticle );
    
  void Rotate(G4FastVector<G4ReactionProduct,MYGHADLISTSIZE> &vec, G4int &vecLen);

  const G4DynamicParticle* FindRhadron(G4ParticleChange*);

  G4ProcessHelper* theHelper;
  G4bool toyModel;
  G4double cache;
  G4ThreeVector what;
};

#endif
 
