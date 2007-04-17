#include "FWCore/Framework/interface/EventSetup.h"
#include "SimGeneral/HepPDTESSource/test/HepPDTAnalyzer.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

using namespace edm;
using namespace std;

HepPDTAnalyzer::HepPDTAnalyzer(const edm::ParameterSet& iConfig) :
pName( iConfig.getParameter<std::string>( "particleName" ) )
{
   //now do what ever initialization is needed

}


HepPDTAnalyzer::~HepPDTAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HepPDTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  
//   Handle<ExampleData> pIn;
//   iEvent.getByLabel("example",pIn);
   
//   ESHandle<SetupData> pSetup;
//   iSetup.get<SetupRecord>().get(pSetup);

// getting the table from the EventSource es and filling it into pdt
 

 ESHandle < ParticleDataTable > pdt;
 iSetup.getData( pdt );
 
// for( vstring::const_iterator e = pname.begin(); 
//       e != pname.end(); ++ e ) {
  const ParticleData * part = pdt->particle( pName );     // get the particle data  
  cout << " Particle properties of the " <<  part->name() << " are:" << endl;  
  cout << " Particle ID = " <<  part->pid() <<  endl;  
  cout << " Charge = " <<  part->charge() << endl;  
  cout << " Mass = " <<  part->mass() << endl;
 //cout << " Spin = " <<  part->spin() << endl;
 //}


}

//define this as a plug-in
DEFINE_FWK_MODULE(HepPDTAnalyzer);
