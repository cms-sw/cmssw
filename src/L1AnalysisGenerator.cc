#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisGenerator.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace reco;

L1Analysis::L1AnalysisGenerator::L1AnalysisGenerator() 
{
}

L1Analysis::L1AnalysisGenerator::~L1AnalysisGenerator()
{
}

void L1Analysis::L1AnalysisGenerator::Set(const edm::Event& e)
{
  edm::Handle<reco::GenParticleCollection> genParticles;
   e.getByLabel("genParticles", genParticles);
   for(size_t i = 0; i < genParticles->size(); ++ i) {
     const GenParticle & p = (*genParticles)[i];
     int id = p.pdgId();
     //int st = p.status();  
		if (abs(id) == 13) {
 			unsigned int nMo=p.numberOfMothers();
//			std::cout << "id " << id << "; st " << st 
//							<< "; nMo " << nMo << std::endl;
			for(unsigned int i=0;i<nMo;++i){
//				int thisParentID = dynamic_cast	<const reco::GenParticle*>(p.mother(i))->pdgId();
//				std::cout << "   mother ID " << thisParentID << std::endl;
			}
		}
			
//
// See if the parent was interesting
		int parentID = -10000;
 		unsigned int nMo=p.numberOfMothers();
		for(unsigned int i=0;i<nMo;++i){
			int thisParentID = dynamic_cast
					<const reco::GenParticle*>(p.mother(i))->pdgId();
//
// Is this a bottom hadron?
			int hundredsIndex = abs(thisParentID)/100;
			int thousandsIndex = abs(thisParentID)/1000;
			if ( ((abs(thisParentID) >= 23) && 
						(abs(thisParentID) <= 25)) ||
						(abs(thisParentID) == 6) ||
						(hundredsIndex == 5) ||
						(hundredsIndex == 4) ||
						(thousandsIndex == 5) ||
						(thousandsIndex == 4) 
					)
				parentID = thisParentID;
		}
		if ((parentID == -10000) && (nMo > 0)) 
			parentID = dynamic_cast
					<const reco::GenParticle*>(p.mother(0))->pdgId();
//
// If the parent of this particle is interesting, store all of the info
		if ((parentID != p.pdgId()) &&
			((parentID > -9999) 
			   || (abs(id) == 11)
			   || (abs(id) == 13)
			   || (abs(id) == 23)
			   || (abs(id) == 24)
			   || (abs(id) == 25)
			   || (abs(id) == 4)
			   || (abs(id) == 5)
			   || (abs(id) == 6))
			)
		{
         generator_.id.push_back(p.pdgId());
			generator_.status.push_back(p.status());
			generator_.px.push_back(p.px());
			generator_.py.push_back(p.py());
			generator_.pz.push_back(p.pz());
			generator_.e.push_back(p.energy());
			generator_.parent_id.push_back(parentID);
		}
   }

}
