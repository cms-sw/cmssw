// -*- C++ -*-
#ifndef TauAnalysis_MCEmbeddingTools_ParticleReplacerFactory_h
#define TauAnalysis_MCEmbeddingTools_ParticleReplacerFactory_h

//
// Package:    MCEmbeddingtools
// Class:      ParticleReplacerFactory
//
/**\class ParticleReplacerFactory ParticleReplacerFactory.cc TauAnalysis/MCEmbeddingTools/src/ParticleReplacerFactory.cc

 Description: Factory class for creating particle replacer algorithm objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Matti Kortelainen
//
//

#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<string>

#include<boost/shared_ptr.hpp>

class ParticleReplacerFactory {
public:
  static boost::shared_ptr<ParticleReplacerBase> create(const std::string& algo, const edm::ParameterSet& iConfig);
};

#endif
