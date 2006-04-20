#ifndef SimG4Core_HadronicProcessHelper_H
#define SimG4Core_HadronicProcessHelper_H

#include "G4ParticleDefinition.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4Track.hh"

#include <vector>
#include <map>

class G4ParticleTable;

class HadronicProcessHelper 
{
public:
    // Typedefs just made to make life easier :-) 
    typedef std::vector<int> ReactionProduct;
    typedef std::vector<ReactionProduct > ReactionProductList;
    typedef std::map<int , ReactionProductList> ReactionMap;

    static HadronicProcessHelper * instance();

    bool applicabilityTester(const G4ParticleDefinition & particle);
    double inclusiveCrossSection(const G4DynamicParticle * particle,
				 const G4Element *	element);
    // Make sure the element is known (for n/p-decision)
    ReactionProduct finalState(const G4Track & track,G4ParticleDefinition*& target);
protected:
    HadronicProcessHelper();
    HadronicProcessHelper(const HadronicProcessHelper&);
    HadronicProcessHelper& operator= (const HadronicProcessHelper&);
private:
    G4ParticleDefinition * m_proton;
    G4ParticleDefinition* m_neutron;

    static HadronicProcessHelper * s_instance;

    double m_phaseSpace(const ReactionProduct & aReaction,
			const G4DynamicParticle * aDynamicParticle,
		        G4ParticleDefinition * target);
    double m_reactionProductMass(const ReactionProduct & aReaction,
				 const G4DynamicParticle * aDynamicParticle,
				 G4ParticleDefinition * target);
    bool m_reactionIsPossible(const ReactionProduct & aReaction, 
			      const G4DynamicParticle * aDynamicParticle,
			      G4ParticleDefinition * target);

    void m_readAndParse(const std::string& str,
			std::vector<std::string>& tokens,
			const std::string& delimiters = " ");

    // Map of applicable particles
    std::map<const G4ParticleDefinition *,bool> m_knownParticles;
    // Proton-scattering processes
    ReactionMap m_protonReactionMap;
    // Neutron-scattering processes
    ReactionMap m_neutronReactionMap;
    G4ParticleTable * m_particleTable;
    // Debug stuff
    double m_checkFraction;
    int m_n22;
    int m_n23;
};

#endif
