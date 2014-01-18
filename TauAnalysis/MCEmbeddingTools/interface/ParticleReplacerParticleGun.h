// -*- C++ -*-
#ifndef TauAnalysis_MCEmbeddingTools_ParticleReplacerParticleGun_h
#define TauAnalysis_MCEmbeddingTools_ParticleReplacerParticleGun_h

//
// Package:    MCEmbeddingtools
// Class:      ParticleReplacerBase
//
/**\class ParticleReplacerParticleGun ParticleReplacerParticleGun.cc TauAnalysis/MCEmbeddingTools/src/ParticleReplacerParticleGun.cc

 Description: Particle gun replacer algorithm

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sami Lehti
//
//

#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerBase.h"

#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaInterfaceBase.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaFactory.h"

#include<string>
#include<vector>

class ParticleReplacerParticleGun: public ParticleReplacerBase {
public:
  explicit ParticleReplacerParticleGun(const edm::ParameterSet&, bool);
  virtual ~ParticleReplacerParticleGun();

  virtual void beginJob();
  virtual void endJob();

  std::auto_ptr<HepMC::GenEvent> produce(const reco::MuonCollection& muons, const reco::Vertex *pvtx=0, const HepMC::GenEvent *genEvt=0);

protected:

private:
  void correctTauMass(const reco::MuonCollection& muons, std::vector<HepMC::FourVector>& corrected);
  void forceTauolaTauDecays();
  void tauola_forParticleGun(int tau_idx, int pdg_id, const HepMC::FourVector& particle_momentum);
  float tauHelicity(int pdg_id);
  float randomPolarization();

  gen::TauolaInterfaceBase* tauola_;
  gen::Pythia6Service pythia_;

  std::string particleOrigin_;
  std::string forceTauPolarization_;
  std::string forceTauDecay_;
  std::string generatorMode_;
  int gunParticle_;
  int forceTauPlusHelicity_;
  int forceTauMinusHelicity_;

  float pol1_[4];
  float pol2_[4];

  bool printout_;

};

#endif
