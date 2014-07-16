#ifndef TauAnalysis_MCEmbeddingTools_ParticleReplacerZtautau_h
#define TauAnalysis_MCEmbeddingTools_ParticleReplacerZtautau_h

/** \class ParticleReplacerZtautau
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
 * \version $Revision: 1.6 $
 *
 * $Id: ParticleReplacerZtautau.h,v 1.6 2013/01/31 09:07:18 veelken Exp $
 *
 */

#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerBase.h"

#include "DataFormats/Candidate/interface/Particle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaInterfaceBase.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaFactory.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "TauAnalysis/MCEmbeddingTools/interface/GenMuonRadiationAlgorithm.h"

#include <TTree.h>

#include<string>

class ParticleReplacerZtautau : public ParticleReplacerBase
{
 public:
  explicit ParticleReplacerZtautau(const edm::ParameterSet&);
  ~ParticleReplacerZtautau();

  virtual void declareExtraProducts(MCParticleReplacer*);

  virtual std::auto_ptr<HepMC::GenEvent> produce(const std::vector<reco::Particle>&, const reco::Vertex* = 0, const HepMC::GenEvent* = 0, MCParticleReplacer* = 0);
  virtual void beginRun(edm::Run&, const edm::EventSetup&);
  virtual void beginJob();
  virtual void endJob();

 private:
  void transformMuMu2LepLep(CLHEP::HepRandomEngine& randomEngine, reco::Particle*, reco::Particle*);
  void transformMuMu2TauNu(reco::Particle*, reco::Particle*);

  HepMC::GenEvent* processEventWithTauola(HepMC::GenEvent*);
  HepMC::GenEvent* processEventWithPythia(HepMC::GenEvent*);
	
  bool testEvent(HepMC::GenEvent*);	

  void cleanEvent(HepMC::GenEvent*, HepMC::GenVertex*);

  std::string generatorMode_;
  double beamEnergy_; // proton beam energy in GeV

  // this variable defines the type of decay to simulate
  //  0 - mumu -> mumu 
  //  1 - mumu -> tautau (default value)
  //  2 - mumu -> ee
  //  3 - mumu -> taunu
  //  4 - munu -> taunu
  unsigned int transformationMode_;

  int motherParticleID_;
  bool useExternalGenerators_;
  bool useTauola_;
  bool useTauolaPolarization_;
  double rfRotationAngle_; // angle of rotation around Z-direction of embedded leptons wrt. reconstructed muons
                           // (used to "place" simulated leptons in a detector region different from reconstructed muons,
                           // Note this does not preserve the Z polarization!!!
  bool rfMirror_; // mirror the muon momentum vectors at the plane defined by the Z axis and the proton axis
                  // This preserves the Z polarization from what we have seen so far.

  gen::TauolaInterfaceBase* tauola_;
  // keep track if TAUOLA interface has already been initialized.
  // Needed to avoid multiple initializations of TAUOLA interface,
  // which makes TAUOLA crash.
  static bool tauola_isInitialized_;

  bool applyMuonRadiationCorrection_;
  GenMuonRadiationAlgorithm* muonRadiationAlgo_;

  gen::Pythia6Service pythia_;

  bool printEvent_;

  struct MinVisPtCut 
  { 
    enum { kELEC, kMU, kHAD, kTAU };
    int type_; 
    unsigned index_; 
    double threshold_; 
    void print(std::ostream& stream) const
    {
      std::string type_string = "";
      if      ( type_ == kELEC ) type_string = "elec";
      else if ( type_ == kMU   ) type_string = "mu";
      else if ( type_ == kHAD  ) type_string = "had";
      else if ( type_ == kTAU  ) type_string = "tau";
      stream << type_string << " #" << index_ << ": threshold = " << threshold_ << std::endl;
    }
  };
  struct MinVisPtCutCombination
  {
    std::string cut_string_;
    std::vector<MinVisPtCut> cuts_;
    void print(std::ostream& stream) const
    {
      stream << "<MinVisPtCutCombination::print>:" << std::endl;
      stream << " cut = " << cut_string_ << std::endl;
      stream << "elements:" << std::endl;
      for ( std::vector<MinVisPtCut>::const_iterator cut = cuts_.begin();
	    cut != cuts_.end(); ++cut ) {
	cut->print(stream);
      }
    }
  };
  std::vector<MinVisPtCutCombination> minVisPtCuts_;
	
  double targetParticle1Mass_;
  int targetParticle1AbsPdgID_;
  double targetParticle2Mass_;
  int targetParticle2AbsPdgID_;
	
  int maxNumberOfAttempts_;

};

#endif


