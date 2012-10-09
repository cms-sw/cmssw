#ifndef TauAnalysis_MCEmbeddingTools_ParticleReplacerParticleGun_h
#define TauAnalysis_MCEmbeddingTools_ParticleReplacerParticleGun_h

/** \class ParticleReplacerParticleGun
 *
 * Particle gun replacer algorithm
 *
 * \author Sami Lehti
 *
 * \version $Revision: 1.13 $
 *
 * $Id: ParticleReplacerParticleGun.h,v 1.13 2012/10/07 13:09:35 veelken Exp $
 *
 */

#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerBase.h"

#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"
#include "GeneratorInterface/ExternalDecays/interface/TauolaInterface.h"

#include<string>
#include<vector>

class ParticleReplacerParticleGun: public ParticleReplacerBase 
{
 public:
  explicit ParticleReplacerParticleGun(const edm::ParameterSet&);
  virtual ~ParticleReplacerParticleGun() {}

  virtual void beginJob();
  virtual void endJob();

  std::auto_ptr<HepMC::GenEvent> produce(const std::vector<reco::Particle>&, const reco::Vertex* = 0, const HepMC::GenEvent* = 0);

 private:
  void correctTauMass(const std::vector<reco::Particle>&, std::vector<HepMC::FourVector>&);
  void forceTauolaTauDecays();
  void tauola_forParticleGun(int, int, const HepMC::FourVector&);
  float tauHelicity(int);
  float randomPolarization();

  //gen::TauolaInterface* tauola_;
  gen::TauolaInterface tauola_;
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
};

#endif
