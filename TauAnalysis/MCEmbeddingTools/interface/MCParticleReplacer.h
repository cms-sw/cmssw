// -*- C++ -*-
#ifndef TauAnalysis_MCEmbeddingTools_MCParticleReplacer_h
#define TauAnalysis_MCEmbeddingTools_MCParticleReplacer_h
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
#include "FWCore/Framework/interface/EDProducer.h"

#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerBase.h"

#include<string>

#include<boost/shared_ptr.hpp>

class MCParticleReplacer : public edm::EDProducer
{
public:
	explicit MCParticleReplacer(const edm::ParameterSet&);
	~MCParticleReplacer();

	virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
	virtual void beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup) override;
	virtual void endRun(const edm::Run& iRun,const edm::EventSetup& iSetup) override;
	virtual void beginJob();
	virtual void endJob();
	
private:
        enum HepMcMode { kInvalid=0, kNew, kReplace };
        static HepMcMode stringToHepMcMode(const std::string& name);

        edm::InputTag src_;
        edm::InputTag srcHepMC_;
        HepMcMode hepMcMode_;
        boost::shared_ptr<ParticleReplacerBase> replacer_;
};


#endif
