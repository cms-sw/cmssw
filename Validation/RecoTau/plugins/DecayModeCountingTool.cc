#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/RecoTau/interface/DecayModeCountingTool.h"

using namespace edm;
using namespace std;
using namespace reco;

DecayModeCountingTool::DecayModeCountingTool(const edm::ParameterSet& mc)
{
 
  //One Parameter Set per Collection

  MC_      = mc.getUntrackedParameter<edm::InputTag>("GenParticles");
  ptMinMCTau_ = mc.getUntrackedParameter<double>("ptMinTau",5.);
  ptMinMCMuon_ = mc.getUntrackedParameter<double>("ptMinMuon",2.);
  ptMinMCElectron_ = mc.getUntrackedParameter<double>("ptMinElectron",5.);
  m_PDG_   = mc.getUntrackedParameter<std::vector<int> >("BosonID");
  etaMax = mc.getUntrackedParameter<double>("EtaMax",2.5);

 // The output histograms can be stored or not
  saveoutputhistograms_ = mc.getParameter<bool>("SaveOutputHistograms");

  cerr << " Here " << endl;

  tversion = edm::getReleaseVersion();

  cerr << " Here 2 " << endl;

  if (!saveoutputhistograms_) {
    LogInfo("OutputInfo") << " Tau Decay Mode Histogram will not be saved ";
  } else {
    outPutFile_.append("DecayMode_");    
    tversion.erase(0,1);
    tversion.erase(tversion.size()-1,1);
    outPutFile_.append(tversion);
    outPutFile_.append("_"+ MC_.label());
    outPutFile_.append(".root");
  }
  cerr << " Here 3 " << MC_ <<" "<< ptMinMCTau_ <<" "<< saveoutputhistograms_ <<" "<< tversion <<" "<<  outPutFile_ << endl;
 
  dbeDecay = edm::Service<DQMStore>().operator->();


  cerr << " Here 4 " << endl;
}



void DecayModeCountingTool::beginJob(const edm::EventSetup& mc)
{ 
  cerr << " Here 5 " << endl;

  if(dbeDecay) {
     
    // What kind of Taus do we originally have!
    dbeDecay->setCurrentFolder("RecoTauV/TausAtGenLevel");  
    hGenTauDecay_DecayModes_ = dbeDecay->book1D("genDecayModeChosen", "DecayModeChosen", kOther + 1, -0.5, kOther + 0.5);

  }
  

} 

void DecayModeCountingTool::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<GenParticleCollection> genParticles;
  iEvent.getByLabel(MC_, genParticles);
  
  GenParticleCollection::const_iterator p = genParticles->begin();
  
  for (;p != genParticles->end(); ++p ) {
    //Check the PDG ID
    bool pdg_ok = true;
    for(size_t pi =0;pi<m_PDG_.size();++pi)
      {
	if(abs((*p).pdgId())== m_PDG_[pi] && abs((*p).status()) == 3 ){
   	  pdg_ok = true;
	  //	  cout<<" Bsoson particles: "<< (*p).pdgId()<< " " <<(*p).status() << " "<< pdg_ok<<endl;
   	}
      }
    
    // Check if the boson is one of interest and if there is a valid vertex
    if(  pdg_ok )
      {
	
	std::vector<GenParticle*> decayProducts;
	
	//	TLorentzVector Boson((*p).px(),(*p).py(),(*p).pz(),(*p).energy());	
	
	for (GenParticle::const_iterator BosonIt=(*p).begin(); BosonIt != (*p).end(); BosonIt++){

	  decayProducts.clear();
	  // cout<<" Dparticles: "<< (*BosonIt).pdgId() << " "<< (*BosonIt).status()<<endl;
	  
	  if (abs((*BosonIt).pdgId()) == 15 && ((*BosonIt).status()==3)) //if it is a Tau and decayed
	    {
	      
	      //	      cout<<" Boson daugther particles: "<< (*BosonIt).pdgId() << " "<< (*BosonIt).status()<< endl;	      
	      for (GenParticle::const_iterator TauIt = (*BosonIt).begin(); TauIt != (*BosonIt).end(); TauIt++) {
		//	cout<<" Tau daughter particles: "<< (*TauIt).pdgId() << " "<< (*TauIt).status()<<endl;
		
		if (abs((*TauIt).pdgId()) == 15 && ((*TauIt).status()==2)) //if it is a Tau and decayed
		  {		    
		    decayProducts = getGenStableDecayProducts((reco::GenParticle*) & (*TauIt));	 
		    //  for (GenParticle::const_iterator TauIt2 = (*TauIt).begin(); TauIt2 != (*TauIt).end(); TauIt2++) {
		    //		      cout<<" Real Tau particles: "<< (*TauIt2).pdgId() << " "<< (*TauIt2).status()<< " mother: "<< (*TauIt2).mother()->pdgId() << endl;
		    // }
		  }
	      }
	      
	      
	      if ( !decayProducts.empty() )
		{
		  
		  LorentzVector Visible_Taus(0.,0.,0.,0.);
		  LorentzVector TauDecayProduct(0.,0.,0.,0.);
		  LorentzVector Neutrino(0.,0.,0.,0.);
		  
		  int numElectrons      = 0;
		  int numMuons          = 0;
		  int numChargedPions   = 0;
		  int numNeutralPions   = 0;
		  int numNeutrinos      = 0;
		  int numOtherParticles = 0;
		  
		  
		  for (std::vector<GenParticle*>::iterator pit = decayProducts.begin(); pit != decayProducts.end(); pit++)
		    {
		      int pdg_id = abs((*pit)->pdgId());
		      if (pdg_id == 11) numElectrons++;
		      else if (pdg_id == 13) numMuons++;
		      else if (pdg_id == 211) numChargedPions++;
		      else if (pdg_id == 111) numNeutralPions++;
		      else if (pdg_id == 12 || 
			       pdg_id == 14 || 
			       pdg_id == 16) {
			numNeutrinos++;
			if (pdg_id == 16) {
			  Neutrino.SetPxPyPzE((*pit)->px(),(*pit)->py(),(*pit)->pz(),(*pit)->energy());
			}
		      }
		      else if (pdg_id != 22) {
			numOtherParticles++;
		      }
		      
		      if (pdg_id != 12 &&
			  pdg_id != 14 && 
			  pdg_id != 16){
			TauDecayProduct.SetPxPyPzE((*pit)->px(),(*pit)->py(),(*pit)->pz(),(*pit)->energy());
			Visible_Taus+=TauDecayProduct;
		      }	
		      //		  cout<< "This has to be the same: " << (*pit)->pdgId() << " "<< (*pit)->status()<< " mother: "<< (*pit)->mother()->pdgId() << endl;
		    }
		  
		  int tauDecayMode = kOther;
		  
		  if ( numOtherParticles == 0 ){
		    if ( numElectrons == 1 ){
		      //--- tau decays into electrons
		      tauDecayMode = kElectron;
		    } else if ( numMuons == 1 ){
		      //--- tau decays into muons
		      tauDecayMode = kMuon;
		    } else {
		      //--- hadronic tau decays
		      switch ( numChargedPions ){
		      case 1 : 
			switch ( numNeutralPions ){
			case 0 : 
			  tauDecayMode = kOneProng0pi0;
			  break;
			case 1 : 
			  tauDecayMode = kOneProng1pi0;
			  break;
			case 2 : 
			  tauDecayMode = kOneProng2pi0;
			  break;
			}
			break;
		      case 3 : 
			switch ( numNeutralPions ){
			case 0 : 
			  tauDecayMode = kThreeProng0pi0;
			  break;
			case 1 : 
			  tauDecayMode = kThreeProng1pi0;
			  break;
			}
			break;
		      }		    
		    }
		  }		  

		  hGenTauDecay_DecayModes_->Fill(tauDecayMode);
		  
		}
	    }
	}
      }
  } 
}


void DecayModeCountingTool::endJob()
{  
  if (!outPutFile_.empty() && &*edm::Service<DQMStore>() && saveoutputhistograms_) dbeDecay->save (outPutFile_);
}


// Helper Function

std::vector<reco::GenParticle*> DecayModeCountingTool::getGenStableDecayProducts(const reco::GenParticle* particle)
{
  std::vector<GenParticle*> decayProducts;
  decayProducts.clear();

  //  std::cout << " Are we ever here?: "<< (*particle).numberOfDaughters() << std::endl;
  for ( GenParticle::const_iterator daughter_particle = (*particle).begin();daughter_particle != (*particle).end(); ++daughter_particle ){   

    int pdg_id = abs((*daughter_particle).pdgId());

//    // check if particle is stable
    if ( pdg_id == 11 || pdg_id == 12 || pdg_id == 13 || pdg_id == 14 || pdg_id == 16 ||  pdg_id == 111 || pdg_id == 211 ){
      // stable particle, identified by pdg code
      decayProducts.push_back((reco::GenParticle*) &(* daughter_particle));
    } 
    else {
//      // unstable particle, identified by non-zero decay vertex
      std::vector<GenParticle*> addDecayProducts = getGenStableDecayProducts((reco::GenParticle*) &(*daughter_particle));
      for ( std::vector<GenParticle*>::const_iterator adddaughter_particle = addDecayProducts.begin(); adddaughter_particle != addDecayProducts.end(); ++adddaughter_particle ){
	decayProducts.push_back((*adddaughter_particle));
      }
    }
  }
  return decayProducts;
}
