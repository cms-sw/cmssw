#include "TauAnalysis/MCEmbeddingTools/interface/MCParticleReplacer.h"

MCParticleReplacer::MCParticleReplacer(const edm::ParameterSet& pset)
{
	desiredReplacerClass = pset.getUntrackedParameter<int>("desiredReplacerClass",1);
	
	switch (desiredReplacerClass)
	{
		case 1:
			replacer1 = new ParticleReplacerClass(pset);
			break;
		case 2:
			//replacer2 = new (pset);
			//break;
		default:
			throw cms::Exception("MCParticleReplacer") << "desired replacer class not present" << std::endl;
	}
}

MCParticleReplacer::~MCParticleReplacer()
{
}

// ------------ method called to produce the data  ------------
void
MCParticleReplacer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	using namespace edm;
	switch (desiredReplacerClass)
	{
		case 1:
			replacer1->produce(iEvent, iSetup);
			break;
		case 2:
			//replacer2->produce(iEvent, iSetup);
			//break;
		default:
			throw cms::Exception("MCParticleReplacer") << "desired replacer class not present" << std::endl;
	}
}

// ------------ method called once each job just before starting event loop  ------------
void 
MCParticleReplacer::beginJob(const edm::EventSetup& iSetup)
{
	using namespace edm;
	switch (desiredReplacerClass)
	{
		case 1:
			replacer1->beginJob(iSetup);
			break;
		case 2:
			//replacer2->beginJob(iSetup);
			//break;
		default:
			throw cms::Exception("MCParticleReplacer") << "desired replacer class not present" << std::endl;
	}
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MCParticleReplacer::endJob()
{
	switch (desiredReplacerClass)
	{
		case 1:
			replacer1->endJob();
			break;
		case 2:
			//replacer2->endJob();
			//break;
		default:
			throw cms::Exception("MCParticleReplacer") << "desired replacer class not present" << std::endl;
	}
}

//define this as a plug-in
DEFINE_FWK_MODULE(MCParticleReplacer);
