/*
// system include files

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <stack>
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

using namespace std;
using namespace edm;

class CheckHepMCEvent : public edm::EDAnalyzer {
   public:
	explicit CheckHepMCEvent(const edm::ParameterSet&);
	~CheckHepMCEvent();

	virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
	virtual void beginJob(const edm::EventSetup& );
	virtual void endJob();
  private:
	std::string HepMCSource_;
	void checkParticle(HepMC::GenParticle * part);
};
*/
