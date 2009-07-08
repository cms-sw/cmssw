// -*- C++ -*-
//
// Package:    Replacer
// Class:      Replacer
//
/**\class Replacer Replacer.cc TauAnalysis/MCEmbeddingTools/src/Replacer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  M. Zeise
//         Created:  Tue Oct  14 13:04:54 CEST 2008
//
//


// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerClass.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

using namespace std;
using namespace edm;

class MCParticleReplacer : public edm::EDProducer
{
public:
	explicit MCParticleReplacer(const edm::ParameterSet&);
	~MCParticleReplacer();

	virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
	virtual void beginJob(const edm::EventSetup& );
	virtual void endJob();
	
private:
	int desiredReplacerClass;
	ParticleReplacerClass * replacer1;
};


