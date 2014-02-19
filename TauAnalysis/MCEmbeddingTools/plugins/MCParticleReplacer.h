#ifndef TauAnalysis_MCEmbeddingTools_MCParticleReplacer_h
#define TauAnalysis_MCEmbeddingTools_MCParticleReplacer_h

/** \class MCParticleReplacer
 *
 * Replace muons reconstructed in selected Z --> mu+ mu- events 
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
 * \version $Revision: 1.3 $
 *
 * $Id: MCParticleReplacer.h,v 1.3 2013/01/31 09:07:18 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerBase.h"

#include<string>

#include<boost/shared_ptr.hpp>

class MCParticleReplacer : public edm::EDProducer
{
 public:
  explicit MCParticleReplacer(const edm::ParameterSet&);
  ~MCParticleReplacer();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void beginRun(edm::Run&,const edm::EventSetup&);
  virtual void endRun();
  virtual void beginJob();
  virtual void endJob();
	
  template <typename T>
  void call_produces(const std::string& instanceName)
  {
    produces<T>(instanceName);
  }

  template <typename T>
  void call_put(T& product, const std::string& instanceName)
  {
    evt_->put(product, instanceName);
  }

  edm::StreamID getStreamID() const { assert(evt_ != nullptr); return evt_->streamID(); }

 private:
  enum HepMcMode { kInvalid = 0, kNew, kReplace };
  static HepMcMode stringToHepMcMode(const std::string& name);

  edm::InputTag src_;
  edm::InputTag srcHepMC_;
  HepMcMode hepMcMode_;
  ParticleReplacerBase* replacer_;
  edm::Event* evt_;

  int verbosity_;
};

#endif
