/*class BasicGenTest
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  BasicGenTest:
 *  $Date: 2009/07/09 04:23:19 $
 *  $Revision: 1.1 $
 *  \author Joseph Zennamo SUNY-Buffalo; Based on: ConverterTester*/
 
#include "BasicGenTest.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace edm;

BasicGenTest::BasicGenTest(const edm::ParameterSet& iPSet)
{
  glunum = 0;
  dusnum = 0;
  cnum = 0;
  bnum = 0;
  topnumber = 0;
  Wnum = 0;
  Znum = 0;

  //charstanum = 0;
 
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
}

BasicGenTest::~BasicGenTest() {}

void BasicGenTest::beginJob(const edm::EventSetup& iSetup)
{
  if(dbe){
       
    dbe->setCurrentFolder("Generator/Particles");

    gluonNumber = dbe->book1D("gluonNumber", "No. gluons", 100, -.5, 9999.5);
    dusNumber = dbe->book1D("dusNumber", "No. uds", 20, -.5, 59.5);
    cNumber = dbe->book1D("cNumber", "No. c", 15, -.5, 14.5);
    bNumber = dbe->book1D("bNumber", "No. b", 15, -.5, 14.5);
    tNumber = dbe->book1D("tNumber", "No. t", 6, -.5, 5.5);
    WNumber = dbe->book1D("WNumber", "No. W", 15, -.5, 14.5);
    ZNumber = dbe->book1D("ZNumber", "No. Z", 6, -.5, 5.5);

    particle_number_nogammagluon = dbe->book1D("particle_number_nogammagluon", "No. ptcls & partons by PDGId w/o gamma & gluons", 100, -.5, 99.5);
    particle_number = dbe->book1D("partnum", "No. ptcls", 100, 99.5, 1199.5);
    PartonNumber = dbe->book1D("PartonNumber", "No. parton by PDGId ", 100, -.5, 99.5);
    // ChargStableNumber = dbe->book1D("ChargStableNumber", "Number of Charged Stable particles per event", 100, -.5, 999.5);

    dbe->tag(gluonNumber->getFullname(),1);
    dbe->tag(dusNumber->getFullname(),2);
    dbe->tag(cNumber->getFullname(),3);
    dbe->tag(bNumber->getFullname(),4);
    dbe->tag(tNumber->getFullname(),5);
    dbe->tag(WNumber->getFullname(),6);
    dbe->tag(ZNumber->getFullname(),7);
    dbe->tag(particle_number_nogammagluon->getFullname(),8);
    dbe->tag(particle_number->getFullname(),9);
    dbe->tag(PartonNumber->getFullname(),10);
    // dbe->tag(ChargStableNumber->getFullname(),11);
 }

  return;
}

void BasicGenTest::endJob(){return;}
void BasicGenTest::beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup){return;}
void BasicGenTest::endRun(const edm::Run& iRun,const edm::EventSetup& iSetup){return;}
void BasicGenTest::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
 
  int counterstable = 0; // To count "stable", status == 1, particles
  bnum = 0;
  topnumber = 0;
  Wnum = 0;
  dusnum = 0;
  cnum = 0;
  Znum = 0;
  // charstanum = 0;
  for(int i=0; i < 100; ++i) {
    part_counter[i] = 0;
    part_noglu_counter[i] = 0;
  }

  edm::Handle<HepMCProduct> evt;
  
  iEvent.getByLabel("generator", evt);
  
  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
  HepMC::GenEvent::particle_const_iterator begin = myGenEvent->particles_begin();
  HepMC::GenEvent::particle_const_iterator end = myGenEvent->particles_end();

  for(HepMC::GenEvent::particle_const_iterator it = begin;it!=end;++it)
     {
       HepMC::GenParticle* particle = *it;
       int Id = particle->pdg_id();
       int status = particle->status();       
       //int charge = particle->flow(1);
       //double pT = particle->momentum().perp();

       if (abs(Id) == 6) ++topnumber;
       if (abs(Id) == 24) ++Wnum;
       if(abs(Id) == 5) ++bnum;
       if(abs(Id) == 1 || abs(Id) == 2 || abs(Id) == 3) ++dusnum;	 
       if(abs(Id) == 4) ++cnum;
       if(abs(Id) == 23) ++Znum;
       if(abs(Id) == 21) ++glunum;
       if( 0 < Id && 100 > Id) ++part_counter[Id];
       if( 0 < Id && 100 > Id && Id != 21 && Id != 22) ++ part_noglu_counter[Id];
       if(status == 1) ++counterstable;

       // if(charge != 0 && status == 1 && pT < .5){
       // ++charstanum;
       // }

	}//for(HepMC::

  if((2*topnumber) != Wnum){ ///Since top decays from status 3, but W does not
      
	std::cout << "WOOT WOOT! We have a problem! Tops= " << topnumber << "!= W's = " << Wnum << std::endl;
	myGenEvent->print(); 
      
      }//if(top
      
 if(bnum == 1 || bnum == 3 || bnum == 5){
	std::cout << "We have odd number of b's! b = " << bnum << std::endl;
	myGenEvent->print();
      }

 // std:: cout << "\n"  <<"We are inside the analyze loop: BTagPerformanceAnalyzerMC: iEvent = " << counter2 << " Event Number= " << iEvent.id() << "!!!" << "\n"<< std::endl;

  gluonNumber->Fill(glunum);  
  dusNumber->Fill(dusnum);
  cNumber->Fill(cnum);
  bNumber->Fill(bnum);
  tNumber->Fill(topnumber);
  WNumber->Fill(Wnum);
  ZNumber->Fill(Znum);
  particle_number->Fill(counterstable);
  
  for(int i = 0; i < 100; ++i){
     PartonNumber->Fill(float(i), float(part_counter[i]));
     particle_number_nogammagluon->Fill(float(i), float(part_noglu_counter[i]));  
   }
   //     ChargStableNumber->Fill(charstanum);
 }


DEFINE_FWK_MODULE(BasicGenTest);
