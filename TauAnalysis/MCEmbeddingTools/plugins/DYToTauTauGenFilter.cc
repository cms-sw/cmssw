#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class DYToTauTauGenFilter: public edm::stream::EDFilter<> {
   public:
      explicit DYToTauTauGenFilter(const edm::ParameterSet&);
      ~DYToTauTauGenFilter() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      void beginStream(edm::StreamID) override;
      bool filter(edm::Event&, const edm::EventSetup&) override;
      void endStream() override;
      
      edm::InputTag inputTag_;
      edm::EDGetTokenT<reco::GenParticleCollection> genParticleCollection_;

      edm::Handle<reco::GenParticleCollection> gen_handle;
      
      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
};




DYToTauTauGenFilter::DYToTauTauGenFilter(const edm::ParameterSet& iConfig)
{
  inputTag_= iConfig.getParameter<edm::InputTag>("inputTag");
  genParticleCollection_ = consumes<reco::GenParticleCollection>(inputTag_);
}


DYToTauTauGenFilter::~DYToTauTauGenFilter() {
}

bool photoncheck(const reco::Candidate *d, int depth)
{
	// Debug Output
	// std::cout << std::string(4*depth, '-') << "-----------------" << std::endl;
	// std::cout << std::string(4*depth, '-') << "|  Depth       " << depth << std::endl;
	// std::cout << std::string(4*depth, '-') << "|  ID:         " << d->pdgId() << std::endl;
	// std::cout << std::string(4*depth, '-') << "|  Status:     " << d->status() << std::endl;
	// std::cout << std::string(4*depth, '-') << "|  NDaughters: " << d->numberOfDaughters()<< std::endl;
	bool check = false;
	if (d->status()!= 1)
	{
		if(d->numberOfDaughters() == 3)
		{
			if(std::abs(d->daughter(0)->pdgId()) == 14 
			|| std::abs(d->daughter(1)->pdgId()) == 14 
			|| std::abs(d->daughter(2)->pdgId()) == 14
			|| std::abs(d->daughter(0)->pdgId()) == 12 
			|| std::abs(d->daughter(1)->pdgId()) == 12 
			|| std::abs(d->daughter(2)->pdgId()) == 12)
			{	
				check = true;
			}
		}
		else if (d->numberOfDaughters() > 3) return false;
		if (d->numberOfDaughters() < 4)
		{
			for (unsigned int k = 0; k < d->numberOfDaughters(); k++)
			{
				// std::cout << std::string(4*depth, '-') << "| Daughter Number " << k << std::endl;
				int new_depth = depth + 1;
				if(photoncheck(d->daughter(k), new_depth) == true)
					check = true;
			}
		}
			
	}
	return check;
	
}

bool DYToTauTauGenFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {


    iEvent.getByToken(genParticleCollection_, gen_handle);
    
    for(unsigned int i = 0; i < gen_handle->size(); i++)
    {
      const reco::GenParticle gen_particle = (*gen_handle)[i];
		// Check if Z Boson decayed into two leptons
		if (gen_particle.pdgId() == 23 && gen_particle.numberOfDaughters() == 2)
		{
			//Debug output
			// std::cout << "pdgId" << gen_particle.pdgId() << std::endl;
			// std::cout << "nDau" << gen_particle.numberOfDaughters() << std::endl;
			// std::cout << "Dau1 " << gen_particle.daughter(0)->pdgId() << std::endl;
			// std::cout << "Dau2 " << gen_particle.daughter(1)->pdgId() << std::endl;
			// std::cout << gen_particle.daughter(1)->pdgId()+gen_particle.daughter(0)->pdgId() << std::endl;
			
			// Check if daugther particles are taus
		  if (std::abs(gen_particle.daughter(0)->pdgId()) == 15 
		      && std::abs(gen_particle.daughter(0)->eta())<2.3  
		      && std::abs(gen_particle.daughter(1)->eta())<2.3
		      && ((gen_particle.daughter(0)->pt()>30
		      && gen_particle.daughter(1)->pt()>35)||(gen_particle.daughter(0)->pt()>35
		      && gen_particle.daughter(1)->pt()>30)))
			{
				if (!photoncheck(gen_particle.daughter(0),1) && !photoncheck(gen_particle.daughter(1),1))
				{
					//std::cout << "Hadronic Decay Check was positive" << std::endl;
					return true;
				}
				//std::cout << "Hadronic Decay Check was negative" << std::endl;	
			}
		return false;
		}
    }
    return false;
}
// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
DYToTauTauGenFilter::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
DYToTauTauGenFilter::endStream() {
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DYToTauTauGenFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}



DEFINE_FWK_MODULE(DYToTauTauGenFilter);
