#ifndef SimG4Core_CustomPhysics_RHadronPythiaDecayer_H
#define SimG4Core_CustomPhysics_RHadronPythiaDecayer_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "G4Decay.hh"
#include "G4VExtDecayer.hh"
#include "G4ThreeVector.hh"
#include <string>
#include <vector>
#include <utility>
#include <memory>

namespace gen {
  class P8RndmEngine;
  typedef std::shared_ptr<P8RndmEngine> P8RndmEnginePtr;
}  // namespace gen

namespace Pythia8 {
  class Pythia;
  class Event;
  class Rndm;
}  // namespace Pythia8

class G4DynamicParticle;
class G4DecayProducts;
class G4VParticleChange;
class G4Step;
class G4Track;
class RHadronPythiaDecayer : public G4Decay, public G4VExtDecayer {
public:
  RHadronPythiaDecayer(edm::ParameterSet const& p);
  ~RHadronPythiaDecayer() override;

  //What Geant calls to decay the Rhadron
  G4VParticleChange* DecayIt(const G4Track& aTrack, const G4Step& aStep) override;

  //Tell pythia to decay the Rhadron and return the products in Geant format
  G4DecayProducts* ImportDecayProducts(const G4Track&) override;

private:
  // Strip the RHadron down to its constituents in preperation for decaying the gluino or squark
  void RHadronToConstituents(Pythia8::Event& event);

  std::pair<int, int> fromIdWithSquark(int idRHad) const;
  std::pair<int, int> fromIdWithGluino(int idRHad, Pythia8::Rndm* rndmPtr) const;

  bool isGluinoRHadron(int pdgId) const;

  //Fill a Pythia8 event with the information from a G4Track
  void fillParticle(const G4Track&, Pythia8::Event& event) const;

  //Function to decay the RHadron and return products in G4 format
  void pythiaDecay(const G4Track&, std::vector<G4DynamicParticle*>&);

  std::unique_ptr<Pythia8::Pythia> pythia_;
  std::vector<G4ThreeVector> secondaryDisplacements_;
};

#endif
