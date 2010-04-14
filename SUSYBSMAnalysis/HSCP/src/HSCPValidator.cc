// -*- C++ -*-
//
// Package:    HSCP
// Class:      HSCPValidator
// 
/**\class HSCPValidator HSCPValidator.cc HSCPValidation/HSCPValidator/src/HSCPValidator.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Seth Cooper,27 1-024,+41227672342,
//         Created:  Wed Apr 14 14:27:52 CEST 2010
// $Id$
//
//


// system include files
#include <memory>
#include <vector>
#include <string>
#include <map>

// user include files
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1.h"
#include "TGraph.h"
#include "TCanvas.h"

#include "SUSYBSMAnalysis/HSCP/interface/HSCPValidator.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
edm::Service<TFileService> fileService;

//
// constructors and destructor
//
HSCPValidator::HSCPValidator(const edm::ParameterSet& iConfig) :
  label_ (iConfig.getParameter<edm::InputTag>("generatorLabel")),
  particleIds_ (iConfig.getParameter< std::vector<int> >("particleIds")),
  particleStatus_ (iConfig.getUntrackedParameter<int>("particleStatus",3))
{
  //now do what ever initialization is needed
  particleEtaHist_ = fileService->make<TH1F>("particleEta","Eta of gen particle",100,-5,5);
  particlePhiHist_ = fileService->make<TH1F>("particlePhi","Phi of gen particle",180,-3.15,3.15);
  particlePHist_ = fileService->make<TH1F>("particleP","Momentum of gen particle",500,0,2000);
  particlePtHist_ = fileService->make<TH1F>("particlePt","P_{T} of gen particle",500,0,2000);
  particleMassHist_ = fileService->make<TH1F>("particleMass","Mass of gen particle",1000,0,2000);
  particleStatusHist_ = fileService->make<TH1F>("particleStatus","Status of gen particle",10,0,10);
  particleBetaHist_ = fileService->make<TH1F>("particleBeta","Beta of gen particle",100,0,1);
  particleBetaInverseHist_ = fileService->make<TH1F>("particleBetaInverse","1/#beta of gen particle",100,0,5);
}


HSCPValidator::~HSCPValidator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HSCPValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<HepMCProduct> evt;
   iEvent.getByLabel(label_, evt);

   HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));


   for(HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
       p != myGenEvent->particles_end(); ++p )
   {
     // Check if the particleId is in our list
     std::vector<int>::const_iterator partIdItr = find(particleIds_.begin(),particleIds_.end(),(*p)->pdg_id());
     if(partIdItr==particleIds_.end())
       continue;
     
     if((*p)->status() != particleStatus_)
       continue;

     std::pair<std::map<int,int>::iterator,bool> pair = particleIdsFoundMap_.insert(std::make_pair<int,int>((*p)->pdg_id(),1));
     if(!pair.second)
     {
       ++(pair.first->second);
     }

     particleEtaHist_->Fill((*p)->momentum().eta());
     particlePhiHist_->Fill((*p)->momentum().phi());
     particlePHist_->Fill((*p)->momentum().mag());
     particlePtHist_->Fill((*p)->momentum().perp());
     particleMassHist_->Fill((*p)->generated_mass());
     particleStatusHist_->Fill((*p)->status());
     float particleP = (*p)->momentum().mag();
     float particleM = (*p)->generated_mass();
     particleBetaHist_->Fill(particleP/sqrt(particleP*particleP+particleM*particleM));
     particleBetaInverseHist_->Fill(sqrt(particleP*particleP+particleM*particleM)/particleP);

         //std::cout << "FOUND PARTICLE WITH PDGid: " << (*p)->pdg_id() << std::endl;
         //std::cout << "FOUND PARTICLE in param array where its id is " << particleID[i] << std::endl;
         //std::cout << "\tParticle -- eta: " << (*p)->momentum().eta() << std::endl;

         //if((*p)->momentum().perp() > ptMin[i] && (*p)->momentum().eta() > etaMin[i] 
         //    && (*p)->momentum().eta() < etaMax[i] && ((*p)->status() == status[i] || status[i] == 0))
         //{
         //  std::cout << "!!!!PARTICLE ACCEPTED" << std::endl;
         //}  


   }

    delete myGenEvent; 
}


// ------------ method called once each job just before starting event loop  ------------
void 
HSCPValidator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCPValidator::endJob()
{
  // make pngs -- GEN
  std::string command = "mkdir GenPlots";
  system(command.c_str());
  TCanvas t;
  t.cd();
  particleEtaHist_->Draw();
  t.Print("GenPlots/eta.png");
  particlePhiHist_->Draw();
  t.Print("GenPlots/phi.png");
  particlePtHist_->Draw();
  t.Print("GenPlots/pt.png");
  particlePHist_->Draw();
  t.Print("GenPlots/p.png");
  particleBetaHist_->Draw();
  t.Print("GenPlots/beta.png");
  particleBetaInverseHist_->Draw();
  t.Print("GenPlots/1beta.png");
  std::string frequencies = "";
  for(std::map<int,int>::const_iterator itr = particleIdsFoundMap_.begin();
      itr != particleIdsFoundMap_.end(); ++itr)
  {
      frequencies+="PDG ID: ";
      frequencies+=intToString(itr->first);
      frequencies+=" Frequency: ";
      frequencies+=intToString(itr->second);
      frequencies+="\n";
  }
  std::cout << "Found PDGIds: " << "\n\n" << frequencies << std::endl;


}

// ------------- Convert int to string for printing -------------------------------------
std::string HSCPValidator::intToString(int num)
{
  using namespace std;
  ostringstream myStream;
  myStream << num << flush;
  return(myStream.str()); //returns the string form of the stringstream object
}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPValidator);
