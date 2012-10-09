#ifndef TauAnalysis_MCEmbeddingTools_ParticleReplacerFactory_h
#define TauAnalysis_MCEmbeddingTools_ParticleReplacerFactory_h

/** \class ParticleReplacerFactory
 *
 * Create plugins producing HepMC::Event object of embedded event
 * 
 * \author Matti Kortelainen
 *
 * \version $Revision: 1.13 $
 *
 * $Id: ParticleReplacerFactory.cc,v 1.13 2012/10/07 13:09:35 veelken Exp $
 *
 */

#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<string>

#include<boost/shared_ptr.hpp>

class ParticleReplacerFactory 
{
 public:
  static boost::shared_ptr<ParticleReplacerBase> create(const std::string&, const edm::ParameterSet&);
};

#endif
