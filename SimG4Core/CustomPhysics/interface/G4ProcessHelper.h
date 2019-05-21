#ifndef SimG4Core_CustomPhysics_G4ProcessHelper_H
#define SimG4Core_CustomPhysics_G4ProcessHelper_H

#include "globals.hh"
#include "G4ParticleDefinition.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4Track.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <map>

//Typedefs just made to make life easier :-)
typedef std::vector<G4int> ReactionProduct;
typedef std::vector<ReactionProduct> ReactionProductList;
typedef std::map<G4int, ReactionProductList> ReactionMap;

class G4ParticleTable;
class CustomParticleFactory;
class HistoHelper;
class TProfile;
class TH1D;

class G4ProcessHelper {
public:
  G4ProcessHelper(const edm::ParameterSet& p, CustomParticleFactory* ptr);

  ~G4ProcessHelper();

  G4bool ApplicabilityTester(const G4ParticleDefinition& aPart);

  G4double GetInclusiveCrossSection(const G4DynamicParticle* aParticle, const G4Element* anElement);

  //Make sure the element is known (for n/p-decision)
  ReactionProduct GetFinalState(const G4Track& aTrack, G4ParticleDefinition*& aTarget);

private:
  G4ProcessHelper(const G4ProcessHelper&) = delete;
  G4ProcessHelper& operator=(const G4ProcessHelper&) = delete;

  G4double Regge(const double boost);
  G4double Pom(const double boost);

  G4double checkfraction;
  G4int n_22;
  G4int n_23;

  G4ParticleDefinition* theTarget;
  G4ParticleDefinition* theProton;
  G4ParticleDefinition* theNeutron;
  G4ParticleDefinition* theRmesoncloud;
  G4ParticleDefinition* theRbaryoncloud;

  ReactionMap* theReactionMap;

  G4double PhaseSpace(const ReactionProduct& aReaction, const G4DynamicParticle* aDynamicParticle);

  G4double ReactionProductMass(const ReactionProduct& aReaction, const G4DynamicParticle* aDynamicParticle);

  G4bool ReactionGivesBaryon(const ReactionProduct& aReaction);

  G4bool ReactionIsPossible(const ReactionProduct& aReaction, const G4DynamicParticle* aDynamicParticle);

  void ReadAndParse(const G4String& str, std::vector<G4String>& tokens, const G4String& delimiters = " ");

  //Map of applicable particles
  std::map<const G4ParticleDefinition*, G4bool> known_particles;

  //Map for physics parameters, name to value
  std::map<G4String, G4double> parameters;

  //The parameters themselves
  bool resonant;
  double ek_0;
  double gamma;
  double amplitude;
  double suppressionfactor;
  bool reggemodel;
  double hadronlifetime;
  double mixing;

  //Proton-scattering processes
  ReactionMap pReactionMap;

  //Neutron-scattering processes
  ReactionMap nReactionMap;

  CustomParticleFactory* fParticleFactory;
  G4ParticleTable* particleTable;
  HistoHelper* theHistoHelper;
  TProfile* h_xsec_lab;
  TProfile* h_xsec_cms;
  TH1D* h_sqrts;
  TProfile* h_q_p;
  TProfile* h_q_gamma;
};
#endif
