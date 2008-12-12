#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/Examples/plugins/TopGenEventAnalyzer.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"
 
TopGenEventAnalyzer::TopGenEventAnalyzer(const edm::ParameterSet& cfg):
  inputGenEvent_(cfg.getParameter<edm::InputTag>("genEvent"))





{ 
  edm::Service<TFileService> fs;
  //TH1F *semilep = new TH1F("semilep","semilep",2,0,1);

  //Num_Leptons = fs->make<TH1F> ("Number_of_Leptons", "Number_of_Leptons",3,0,2);
  semilep = fs->make<TH1F>("semilep", "semilep", 3,-1,3);
  fulllep = fs->make<TH1F>("fulllep", "fulllep", 3,-1,3);
  fullhad = fs->make<TH1F>("fullhad", "fullhad", 3,-1,3);
  Summe = fs->make<TH1F>("Summe", "Summe", 3,-0.5,2.5);
  number_of_Daughters = fs->make<TH1F>("number_of_Daughters", "number_of_Daughters", 3,-1,3);
  Daughters = fs->make<TH1F>("Daughters", "Daughters", 3,0.5,3.5);
  pdg = fs->make<TH1F>("pdg", "pdg", 40,-20,20);
  leptype = fs->make<TH1F>("leptype", "leptype", 3,0.5,3.5);
  Mothers = fs->make<TH1F>("Mothers", "Mothers", 3,0.5,3.5);
  Daughters_of_Tau = fs->make<TH1F>("Daughters_of_Tau"," Daughters_of_Tau",3,0.5,2.5);
  Mothers_of_Mu = fs->make<TH1F>("Mothers_of_Mu","Mothers_of_Mu",3,0.5,2.5);


 

 }

TopGenEventAnalyzer::~TopGenEventAnalyzer()
{
}

void
TopGenEventAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<TtGenEvent> genEvent;
  evt.getByLabel(inputGenEvent_, genEvent);






  // std::cout << "Hallo world" << std::endl;
 
  //if(genEvent->isTtBar()){std::cout << "This is really a ttbar event [" << genEvent->isTtBar() << "]" << std::endl;
 
  //std::cout << "But is it really semi-leptonic? [" << genEvent->isSemiLeptonic() << "]" << std::endl;
 
  //std::cout << "But is it really Full-leptonic? [" << genEvent->isFullLeptonic() << "]" << std::endl;

 //std::cout << "But is it really Full-hadronic? [" << genEvent->isFullHadronic() << "]" << std::endl;


  // if(genEvent->isSemiLeptonic()){std::cout << "This is really a semi-leptonic  event [" << genEvent->isSemiLeptonic() << "]" << std::endl;

  if(genEvent->isSemiLeptonic())
    
    {
  //std::cout<< genEvent->semiLeptonicChannel() << std::endl;

  // const reco::GenParticle* hadW = genEvent->hadronicDecayW();
  //std::cout << "hadW: numberOfDaughters = " << hadW->numberOfDaughters() << std::endl;
 
  //number_of_Daughters->Fill(hadW->numberOfDaughters());
  //Daughters->Fill(genEvent->semiLeptonicChannel());
  //number_of_Mothers->Fill(hadW->numberOfMothers());
  //Mothers->Fill(genEvent->hadronicDecayW()->numberOfMothers());
  // std::cout << genEvent->leptonicDecayW()->mother()->pdgId() << std::endl;

  // const reco::GenParticle* hadT = genEvent->hadronicDecayTop();
  //std::cout << genEvent->hadronicDecayTop()->mother()->pdgId() << std::endl;         //   

      //if (genEvent->lepton() && genEvent->lepton()->pdgId() == 15)
      //{
      //  std::cout << "no of tau daughters: " << genEvent->lepton()->numberOfDaughters() << std::endl;
      //  for(unsigned int d = 0; d < genEvent->lepton()->numberOfDaughters(); ++d )
      //    {
      //      //std::cout << genEvent->lepton()->daughter(d)->pdgId() << std::endl;
      //    }
      //}

      // if( genEvent->lepton() ){
      //std::cout << "Wert ist:" <<  genEvent->lepton()->mother()->pdgId() << std::endl;
      //if(genEvent->lepton()->mother()->pdgId()    ==  15){
      //  std::cout << "Juhu!!!!!!!!!\n" << std::endl << std::endl;
      //}
      //}
      // else{
      //std::cout << "Wert ist:" <<  genEvent->leptonBar()->mother()->pdgId() << std::endl;
      //if(genEvent->leptonBar()->mother()->pdgId() == -15){ 
      //  std::cout << "Juhu!!!!!!!!!\n" << std::endl << std::endl;
      //}
      
      // }

      int t=0;
      int u=0;

      if (genEvent->lepton() && genEvent->lepton()->pdgId() == 15)
	{
	std::cout << "\nJuhu!!!!!!!!!\n" << std::endl << std::endl;
     
	std::cout << "Mother is :" <<  genEvent->lepton()->mother()->pdgId() << std::endl << std::endl;

	std::cout << "Number of Daughters is :" <<  genEvent->lepton()->numberOfDaughters() << std::endl << std::endl;
	
	for(unsigned int d = 0; d < genEvent->lepton()->mother()->numberOfDaughters(); ++d )
          {
            std::cout << "My Sister is: " << genEvent->lepton()->mother()->daughter(d)->pdgId() << std::endl << std::endl;
          }
	

 	if ( genEvent->lepton()->numberOfDaughters() != 0)
 	  {
 	    t = 1;
	    Daughters_of_Tau->Fill(t);
	  }
	}
      
      if (genEvent->leptonBar() && genEvent->leptonBar()->pdgId() == -15)
	{
	  std::cout << "\nJuhu!!!!!!!!!\n" << std::endl << std::endl;
	  
	  std::cout << "Mother is :" <<  genEvent->leptonBar()->mother()->pdgId() << std::endl << std::endl;
	  
	  std::cout << "Number of Daughters is :" <<  genEvent->leptonBar()->numberOfDaughters() << std::endl << std::endl;

	  for(unsigned int d = 0; d < genEvent->leptonBar()->mother()->numberOfDaughters(); ++d )
	    {
	      std::cout << "My Sister is: " << genEvent->leptonBar()->mother()->daughter(d)->pdgId() << std::endl << std::endl;
	    }

	
	  if ( genEvent->leptonBar()->numberOfDaughters() != 0)
	    {
	      t = 2;
	      Daughters_of_Tau->Fill(t);
	    }
	}
     


      if (genEvent->lepton() && genEvent->lepton()->pdgId() == 13)
	{
	  std::cout << "\nJuhu!!!!!!!!!\n" << std::endl << std::endl;
	  
	  std::cout << "Mother is :" <<  genEvent->lepton()->mother()->pdgId() << std::endl << std::endl;
	  
	  // std::cout << "Number of Daughters is :" <<  genEvent->lepton()->numberOfDaughters() << std::endl << std::endl;
	  if ( genEvent->lepton()->mother()->pdgId() == 15 )
	    {
	      u = 1;
	      Mothers_of_Mu->Fill(u);
	    }
	}
      
      if (genEvent->leptonBar() && genEvent->leptonBar()->pdgId() == -13)
	{
	  std::cout << "\nJuhu!!!!!!!!!\n" << std::endl << std::endl;
	  
	  std::cout << "Mother is :" <<  genEvent->leptonBar()->mother()->pdgId() << std::endl << std::endl;
	  
	  //std::cout << "Number of Daughters is :" <<  genEvent->leptonBar()->numberOfDaughters() << std::endl << std::endl;
	  if ( genEvent->leptonBar()->mother()->pdgId() == -15)
	    {
	      u = 2;
	      Mothers_of_Mu->Fill(u);
	    }
	}
    


      // else{
      //std::cout << "Wert ist:" <<  genEvent->leptonBar()->mother()->pdgId() << std::endl;
	



  //std::cout <<  genEvent->lepton()->mother->pdgId() << std::endl;                    //   
 
  Summe->Fill(genEvent->isSemiLeptonic()-1);
    

    };

  int a=0;
  int b=0;


  if(genEvent->isFullLeptonic())
 
 {
     Summe->Fill(genEvent->isFullLeptonic());
    

     //if ( genEvent->leptonicDecayW()->pdgId() == 24)
     //   {
     // std::cout << genEvent->leptonicDecayW()->pdgId() << std::endl;
     //   };


     //const reco::GenParticle* l1 = genEvent->lepton();
    //  if(genEvent->lepton())
//        std::cout <<  genEvent->lepton()->mother()->pdgId() << std::endl;
//      else if(genEvent->leptonBar())
//        std::cout <<  genEvent->leptonBar()->mother()->pdgId() << std::endl;
//      else{
//        std::cout << "weder lepton jnoch lepton bar existiert" << std::endl;
//      }
     //int a = genEvent->lepton()->pdgId();

if( genEvent->lepton()->pdgId() == 11) {
       a  = 1;
       
     };

if( genEvent->lepton()->pdgId() == 13) {
       a  = 2;
 };


if( genEvent->lepton()->pdgId() == 15) {
       a  = 3;
       // std::cout << "15!!!" << std::endl;
 };


if( genEvent->leptonBar()->pdgId() == -11) {
       b  = 1;
       
     };

if( genEvent->leptonBar()->pdgId() == -13) {
       b  = 2;
 };


if( genEvent->leptonBar()->pdgId() == -15) {
       b  = 3;
       // std::cout << "15!!!" << std::endl;

 };

 leptype->Fill(a);
 leptype->Fill(b);

     //pdg->Fill(genEvent->lepton()->pdgId());
     //pdg->Fill(genEvent->leptonBar()->pdgId());


     // Daughters->Fill(genEvent->()+4);
 };


   if(genEvent->isFullHadronic())
  
 {  
   //
   //if (genEvent->hadronicDecayW()->pdgId() == 24)
   // {
   //    std::cout <<  genEvent->hadronicDecayW()->pdgId() << std::endl;
     
   // std::cout << genEvent->hadronicDecayW()->mother()->pdgId() << std::endl;
       // };


 Summe->Fill(genEvent->isFullHadronic()+1);

 

//std::cout <<  genEvent->leptonicDecayW()->mother()->pdgId() << std::endl;
 //std::cout <<  genEvent->leptonBar()->pdgId() << std::endl;


 };
   



   

}

void TopGenEventAnalyzer::beginJob(const edm::EventSetup&)
{

} 

void TopGenEventAnalyzer::endJob()
{
}
