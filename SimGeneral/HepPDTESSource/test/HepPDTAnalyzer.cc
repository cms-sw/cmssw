#include "FWCore/Framework/interface/EventSetup.h"
#include "SimGeneral/HepPDTESSource/test/HepPDTAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

using namespace edm;
using namespace std;

HepPDTAnalyzer::HepPDTAnalyzer(const edm::ParameterSet& iConfig) :
  particleName_( iConfig.getParameter<std::string>( "particleName" ) ) {
}

HepPDTAnalyzer::~HepPDTAnalyzer() {
}

void
HepPDTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  
  ESHandle <ParticleDataTable> pdt;
  iSetup.getData( pdt );
  
  const ParticleData * part = pdt->particle( particleName_ );
  cout << " Particle properties of the " <<  part->name() << " are:" << endl;  
  cout << " Particle ID = " <<  part->pid() <<  endl;  
  cout << " Charge = " <<  part->charge() << endl;  
  cout << " Mass = " <<  part->mass() << endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HepPDTAnalyzer);
