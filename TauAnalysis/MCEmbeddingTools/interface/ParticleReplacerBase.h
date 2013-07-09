// -*- C++ -*-
#ifndef TauAnalysis_MCEmbeddingTools_ParticleReplacerBase_h
#define TauAnalysis_MCEmbeddingTools_ParticleReplacerBase_h

//
// Package:    MCEmbeddingtools
// Class:      ParticleReplacerBase
//
/**\class ParticleReplacerBase ParticleReplacerBase.cc TauAnalysis/MCEmbeddingTools/src/ParticleReplacerBase.cc

 Description: Base class for particle replacer algorithms

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Matti Kortelainen
//
//
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "HepMC/GenEvent.h"

#include<memory>

class ParticleReplacerBase {
public:
  explicit ParticleReplacerBase(const edm::ParameterSet& iConfig);
  virtual ~ParticleReplacerBase();

  virtual void beginJob();
  virtual void beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);
  virtual void endRun();
  virtual void endJob();

  virtual std::auto_ptr<HepMC::GenEvent> produce(const reco::MuonCollection&, const reco::Vertex *pvtx=0, const HepMC::GenEvent *genEvt=0) = 0;

  unsigned int tried;
  unsigned int passed;

protected:

  const double tauMass;
private:
};


#endif
