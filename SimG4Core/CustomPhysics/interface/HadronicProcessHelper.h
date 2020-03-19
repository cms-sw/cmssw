#ifndef SimG4Core_CustomPhysics_HadronicProcessHelper_H
#define SimG4Core_CustomPhysics_HadronicProcessHelper_H

#include "globals.hh"
#include "G4ParticleDefinition.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4Track.hh"

#include <vector>
#include <map>
#include <string>

class G4ParticleTable;

class HadronicProcessHelper {
public:
  //Typedefs just made to make life easier :-)
  typedef std::vector<G4int> ReactionProduct;
  typedef std::vector<ReactionProduct> ReactionProductList;
  typedef std::map<G4int, ReactionProductList> ReactionMap;

  HadronicProcessHelper(const std::string& fileName);

  HadronicProcessHelper(const HadronicProcessHelper&);

  HadronicProcessHelper& operator=(const HadronicProcessHelper&);

  HadronicProcessHelper* instance() { return this; };

  G4bool applicabilityTester(const G4ParticleDefinition& particle);

  G4double inclusiveCrossSection(const G4DynamicParticle* particle, const G4Element* element);

  //Make sure the element is known (for n/p-decision)
  ReactionProduct finalState(const G4Track& track, G4ParticleDefinition*& target) {
    return finalState(track.GetDynamicParticle(), track.GetMaterial(), target);
  }
  ReactionProduct finalState(const G4DynamicParticle* particle,
                             const G4Material* material,
                             G4ParticleDefinition*& target);

private:
  G4ParticleDefinition* m_proton;
  G4ParticleDefinition* m_neutron;

  //  ReactionMap* m_reactionMap;

  G4double m_phaseSpace(const ReactionProduct& aReaction,
                        const G4DynamicParticle* aDynamicParticle,
                        G4ParticleDefinition* target);

  G4double m_reactionProductMass(const ReactionProduct& aReaction,
                                 const G4DynamicParticle* aDynamicParticle,
                                 G4ParticleDefinition* target);

  G4bool m_reactionIsPossible(const ReactionProduct& aReaction,
                              const G4DynamicParticle* aDynamicParticle,
                              G4ParticleDefinition* target);

  void m_readAndParse(const G4String& str, std::vector<G4String>& tokens, const G4String& delimiters = " ");

  //Map of applicable particles
  std::map<const G4ParticleDefinition*, G4bool> m_knownParticles;

  //Proton-scattering processes
  ReactionMap m_protonReactionMap;

  //Neutron-scattering processes
  ReactionMap m_neutronReactionMap;

  G4ParticleTable* m_particleTable;

  ////Debug stuff
  G4double m_checkFraction;
  G4int m_n22;
  G4int m_n23;
};

#endif
