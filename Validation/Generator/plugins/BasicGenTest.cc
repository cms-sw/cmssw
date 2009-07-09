/* Please Note ****  Kenneth Smith had NOTHING to do with this code!  Thank you */


#include "BasicGenTest.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace edm;

BasicGenTest::BasicGenTest(const edm::ParameterSet& iPSet)
{

  bnum = 0;
  topnumber = 0;
  wnum = 0;
  dusnum = 0;
  cnum = 0;
  Znum = 0;
  partc = 0;
  charstanum = 0;
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();

}

BasicGenTest::~BasicGenTest() {}

void BasicGenTest::beginJob(const edm::EventSetup& iSetup)
{
   


  if(dbe){
    meTestInt = 0;
    
    dbe->setCurrentFolder("ConverterTest/Int");
    meTestInt = dbe->bookInt("TestInt");
    dbe->setCurrentFolder("ConverterTest/Validation");
    bNumber = dbe->book1D("bNumber", "Number of b's per event", 20, -.5, 19.5);
    particle_number = dbe->book1D("partnum", "Number of particles per event", 1000, 399.5, 1399.5);
    WNumber = dbe->book1D("WNumber", "Number of W's per event", 20, -.5, 19.5);
    dusNumber = dbe->book1D("dusNumber", "Number of d,u, and s's per event", 100, -.5, 99.5);
    cNumber = dbe->book1D("cNumber", "Number of c's per event", 20, -.5, 19.5);
    tNumber = dbe->book1D("tNumber", "Number of t's per event", 20, -.5, 19.5);
    ZNumber = dbe->book1D("ZNumber", "Number of Z's per event", 20, -.5, 19.5);
    ChargStableNumber = dbe->book1D("ChargStableNumber", "Number of Charged Stable particles per event", 100, -.5, 999.5);
    PartonNumber = dbe->book1D("PartonNumber", "Number of each parton (organized by PDG ID) for the whole input file", 100, -.5, 99.5);

    dbe->tag(meTestInt->getFullname(),1);
    dbe->tag(bNumber->getFullname(),2);
    dbe->tag(particle_number->getFullname(),3);
    dbe->tag(WNumber->getFullname(),4);
    dbe->tag(dusNumber->getFullname(),5);
    dbe->tag(cNumber->getFullname(),6);
    dbe->tag(tNumber->getFullname(),7);
    dbe->tag(ZNumber->getFullname(),8);
    dbe->tag(ChargStableNumber->getFullname(),9);
    dbe->tag(PartonNumber->getFullname(),10);
  }

  return;
}

void BasicGenTest::endJob()
{
  return;
}

void BasicGenTest::beginRun(const edm::Run& iRun, 
				const edm::EventSetup& iSetup)
{


  return;
}

void BasicGenTest::endRun(const edm::Run& iRun, 
			      const edm::EventSetup& iSetup)
{
  meTestInt->Fill(100);
  
  return;
}

void BasicGenTest::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup)
{
 
  int counter2 = 0;
  bnum = 0;
  topnumber = 0;
  wnum = 0;
  dusnum = 0;
  cnum = 0;
  Znum = 0;
  charstanum = 0;



  edm::Handle<HepMCProduct> evt;
  
  iEvent.getByLabel("generator", evt);
  
  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
  HepMC::GenEvent::particle_const_iterator begin = myGenEvent->particles_begin();
  HepMC::GenEvent::particle_const_iterator end = myGenEvent->particles_end();

  for(int i =0; i < 100; ++i) part_counter[i] = 0;
 
  for(HepMC::GenEvent::particle_const_iterator it = begin;it!=end;++it)
     {

       HepMC::GenParticle* particle = *it;
       int Id = particle->pdg_id();
       int status = particle->status();       
       //int charge = particle->flow(1);
       //double pT = particle->momentum().perp();

       if (abs(Id) == 6){
          ++topnumber;
         }
       
       if (abs(Id) == 24 && status == 2){
          ++wnum;
         }
       
       if(abs(Id) == 5 && status == 2){
          ++bnum;
         }
	
       if((abs(Id) == 1 || abs(Id) == 2 || abs(Id) == 3) && status == 2){
	 ++dusnum;	 
         }

       if(abs(Id) == 4 && status == 2){
	 ++cnum;
         }
 
       if(abs(Id) == 23 && status == 2){
	 ++Znum;
         }
       
       //  if(abs(Id) == 21 && status == 2)
     
       // if(charge != 0 && status == 1 && pT < .5){
       // ++charstanum;
       // std::cout << charge << std::endl;
	 // std::cout <<"hi!" << std::endl;
       // }

       if( 0 < Id && 100 > Id){
	 ++part_counter[Id];
	 // std::cout  << Id << std::endl;
       }
       
       ++counter2;

	}//for(HepMC::
      if(topnumber != wnum){
      
	std::cout << "WOOT WOOT! We have a problem! Tops= " << topnumber << "!= W's = " << wnum << std::endl;
	myGenEvent->print(); 
      
      }//if(top
      
      if(bnum == 1 || bnum == 3 || bnum == 5){
	std::cout << "We have odd number of b's! b = " << bnum << std::endl;
	myGenEvent->print();
      }

   std:: cout << "\n"  <<"We are inside the analyze loop: BTagPerformanceAnalyzerMC: iEvent = " << counter2 << " Event Number= " << iEvent.id() << "!!!" << "\n"<< std::endl;
   
   
   particle_number->Fill(counter2);
   bNumber->Fill(bnum);
   WNumber->Fill(wnum);
   dusNumber->Fill(dusnum);
   cNumber->Fill(cnum);
   tNumber->Fill(topnumber);
   ZNumber->Fill(Znum);
  
   for(int i = 0; i < 100; ++i){
     PartonNumber->Fill(float(i), float(part_counter[i]));
   }
   //     ChargStableNumber->Fill(charstanum);
 }

DEFINE_FWK_MODULE(BasicGenTest);
