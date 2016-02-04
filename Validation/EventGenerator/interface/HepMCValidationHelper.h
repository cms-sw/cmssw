#ifndef Validation_EventGenerator_HepMCValidationHelper 
#define Validation_EventGenerator_HepMCValidationHelper 

#include <HepMC/GenEvent.h>
#include <vector>
#include "TLorentzVector.h"

namespace HepMCValidationHelper {
  template <class T> inline bool GreaterByE (const T& a1, const T& a2) {return a1.E() > a2.E();}

  //sort by descending pt 
  inline bool sortByPt(const HepMC::GenParticle* a , const HepMC::GenParticle* b) {return a->momentum().perp() > b->momentum().perp(); }
  
  //sort by energy
  inline bool sortByE(const HepMC::GenParticle* a , const HepMC::GenParticle* b) {return a->momentum().e() > b->momentum().e(); }

  //sort by rapidity
  inline bool sortByRapidity(const HepMC::GenParticle* a , const HepMC::GenParticle* b) {
    const HepMC::FourVector& amom = a->momentum(); 
    const HepMC::FourVector& bmom = b->momentum(); 
    double rapa = 0.5 * std::log( (amom.e() + amom.z()) / (amom.e() - amom.z()) ); 
    double rapb = 0.5 * std::log( (bmom.e() + bmom.z()) / (bmom.e() - bmom.z()) ); 
    return rapa > rapb;
  }

  //sort by pseudorapidity
  inline bool sortByPseudoRapidity(const HepMC::GenParticle* a , const HepMC::GenParticle* b) {return a->momentum().eta() > b->momentum().eta(); }

  void findDescendents(const HepMC::GenParticle* a, std::vector<const HepMC::GenParticle*>& descendents);

  //get all status 1 particles
  void allStatus1(const HepMC::GenEvent* all, std::vector<const HepMC::GenParticle*>& status1);

  //get all status 2 particles
  void allStatus2(const HepMC::GenEvent* all, std::vector<const HepMC::GenParticle*>& status2);

  //get all status 3 particles
  void allStatus3(const HepMC::GenEvent* all, std::vector<const HepMC::GenParticle*>& status3);

  //given a collection of leptons and the collection of all particles in the event, 
  //get the collection of photons closer than deltaR from any of the leptons
  void findFSRPhotons(const std::vector<const HepMC::GenParticle*>& leptons,
                      const std::vector<const HepMC::GenParticle*>& all,
                      double deltaR,
                      std::vector<const HepMC::GenParticle*>& photons);
 
  
  //given a collection of leptons and the collection of all particles in the event, 
  //get the collection of photons closer than deltaR from any of the leptons
  void findFSRPhotons(const std::vector<const HepMC::GenParticle*>& leptons, 
                      const HepMC::GenEvent* all,
                      double deltaR,
                      std::vector<const HepMC::GenParticle*>& photons);

  //returns true if a status 3 particle is a tau or if a status 1 particle is either an electron or a neutrino
  bool isChargedLepton(const HepMC::GenParticle* part);

  //returns true if a status 1 particle is a neutrino
  bool isNeutrino(const HepMC::GenParticle* part );

  //returns true is status 3 particle is tau
  bool isTau(const HepMC::GenParticle* part); 
  
  //removes isolated leptons (their decay products in the case of taus) and possible fsr photons from the list of particles.
  //this should result in a list of particles to be used for hadronic activity studies, such as construction of jets
  //note: deltaR is both the cone in wich we compute "isolation" and the radius in which we look for fsr photons 
  void removeIsolatedLeptons(const HepMC::GenEvent* all, double deltaR, double sumPt, std::vector<const HepMC::GenParticle*>& pruned);                     
  
  //get all status 1 particles
  void allStatus1(const HepMC::GenEvent* all, std::vector<const HepMC::GenParticle*>& status1);

  //get all visible status1 particles
  void allVisibleParticles(const HepMC::GenEvent* all, std::vector<const HepMC::GenParticle*>& visible);

  //compute generated met
  TLorentzVector genMet(const HepMC::GenEvent* all, double etamin = -9999., double etamax = 9999.);			

}

#endif
