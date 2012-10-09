#ifndef TauAnalysis_MCEmbeddingTools_ParticleReplacerClass_h
#define TauAnalysis_MCEmbeddingTools_ParticleReplacerClass_h

/** \class ParticleReplacerClass
 *
 * Auxiliary class to replace muons reconstructed in selected Z --> mu+ mu- events 
 * by generator level particles, which will be passed to detector simulation & reconstruction modules
 * to create "hybrid" events ("embedded" leptons from Monte Carlo simulation, rest of the event taken from data)
 *
 * Per default, the reconstructed muons are replaced by generator level tau leptons,
 * which are passed to TAUOLA in order to produce generator level tau decay products.
 *
 * For systematic/background studies, it is possible also to:
 *  - replace generator level muons
 *  - "embed" electrons or muons 
 * 
 * \author Manuel Zeise 
 *
 * \version $Revision: 1.13 $
 *
 * $Id: ParticleReplacerClass.cc,v 1.13 2012/10/07 13:09:35 veelken Exp $
 *
 */

#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerBase.h"

#include "DataFormats/Candidate/interface/Particle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h" // needed by TauolaInterface
#include "GeneratorInterface/ExternalDecays/interface/TauolaInterface.h"

#include <TTree.h>

#include<string>

class ParticleReplacerClass : public ParticleReplacerBase
{
 public:
  explicit ParticleReplacerClass(const edm::ParameterSet&);
  ~ParticleReplacerClass() {}

  virtual std::auto_ptr<HepMC::GenEvent> produce(const std::vector<reco::Particle>&, const reco::Vertex* = 0, const HepMC::GenEvent* = 0);
  virtual void beginRun(edm::Run&, const edm::EventSetup&);
  virtual void endJob();

 private:
  void transformMuMu2LepLep(reco::Particle*, reco::Particle*);
  void transformMuMu2TauNu(reco::Particle*, reco::Particle*);

  HepMC::GenEvent* processEventWithTauola(HepMC::GenEvent*);
  HepMC::GenEvent* processEventWithPythia(HepMC::GenEvent*);
	
  bool testEvent(HepMC::GenEvent*);	

  void cleanEvent(HepMC::GenEvent*, HepMC::GenVertex*);
  void repairBarcodes(HepMC::GenEvent*);

  std::string generatorMode_;
  double sqrtS_; // center-of-mass energy in GeV

  // this variable defines the type of decay to simulate
  //  0 - mumu -> mumu 
  //  1 - mumu -> tautau (default value)
  //  2 - mumu -> ee
  //  3 - mumu -> taunu
  //  4 - munu -> taunu
  unsigned int transformationMode_;

  // keep track if this instance of ParticleReplacerClass is the first one.
  // Needed to avoid multiple initializations of TAUOLA interface,
  // which makes TAUOLA crash.
  static int numInstances_;
  bool isFirstInstance_;

  int motherParticleID_;
  bool useExternalGenerators_;
  bool useTauola_;
  bool useTauolaPolarization_;
	
  gen::TauolaInterface tauola_;

  bool printEvent_;

  struct MinVisPtCut 
  { 
    enum { kELEC, kMU, kHAD, kTAU };
    int type_; 
    unsigned index_; 
    double threshold_; 
  };
  typedef std::vector<MinVisPtCut> MinVisPtCutCollection;
  std::vector<MinVisPtCutCollection> minVisPtCuts_;
	
  double targetParticle1Mass_;
  int targetParticle1AbsPdgID_;
  double targetParticle2Mass_;
  int targetParticle2AbsPdgID_;
	
  TTree* outTree_;
  int attempts_;
  int maxNumberOfAttempts_;
};

#endif


