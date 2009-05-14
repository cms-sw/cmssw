// -*- C++ -*-
//
// Package:    CaloTauTagVal
// Class:      CaloTauTagVal
// 
/**\class CaloTauTagVal CaloTauTagVal.cc Validation/RecoTau/src/CaloTauTagVal.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ricardo Vasquez
//         Created:  Thu Feb 28 10:35:28 CST 2008
// $Id: CaloTauTagVal.cc,v 1.3 2008/03/30 12:47:26 vasquez Exp $
//
//

// user include files
#include "Validation/RecoTau/interface/CaloTauTagVal.h"

using namespace edm;
using namespace reco;
using namespace std;

CaloTauTagVal::CaloTauTagVal(const edm::ParameterSet& iConfig)

{
  outputhistograms_ = iConfig.getParameter<string>("OutPutHistograms");
  //  jetEMTagSrc_ = iConfig.getParameter<InputTag>("JetEMTagProd");
  genJetSrc_ = iConfig.getParameter<edm::InputTag>("GenJetProd");  
  ExtensionName_ = iConfig.getParameter<edm::InputTag>("ExtensionName");
  outPutFile_ = iConfig.getParameter<std::string>("OutPutFile"); 
  dataType_ = iConfig.getParameter<std::string>("DataType");
  CaloTauProducer_ = iConfig.getParameter<std::string>("CaloTauProducer");
  CaloTauDiscriminatorByIsolationProducer_ = iConfig.getParameter<std::string>("CaloTauDiscriminatorByIsolationProducer");
  //  CaloTauDiscriminatorAgainstElectronProducer_ = iConfig.getParameter<std::string>("CaloTauDiscriminatorAgainstElectronProducer");

  DQMStore* dbe = &*edm::Service<DQMStore>();

  if(dbe) {

    // What kind of Taus do we originally have!
    dbe->setCurrentFolder("RecoTauV/TausAtGenLevel_" + ExtensionName_.label());    
	
    ptTauMC_    = dbe->book1D("pt_Tau_GenLevel", "pt_Tau_GenLevel", 75, 0., 150.);
    etaTauMC_   = dbe->book1D("eta_Tau_GenLevel", "eta_Tau_GenLevel", 60, -3.0, 3.0 );
    phiTauMC_   = dbe->book1D("phi_Tau_GenLevel", "phi_Tau_GenLevel", 36, -180., 180.);
    energyTauMC_= dbe->book1D("Energy_Tau_GenLevel", "Energy_Tau_GenLevel", 45, 0., 450.0);
    hGenTauDecay_DecayModes_ = dbe->book1D("genDecayMode", "DecayMode", kOther + 1, -0.5, kOther + 0.5);
    hGenTauDecay_DecayModesChosen_ = dbe->book1D("genDecayModeChosen", "DecayModeChosen", kOther + 1, -0.5, kOther + 0.5);
     
    nMCTaus_ptTauJet_ =     dbe->book1D("nMC_Taus_vs_ptTauJet", "nMC_Taus_vs_ptTauJet", 75, 0., 150.); 
    nMCTaus_etaTauJet_ =    dbe->book1D("nMC_Taus_vs_etaTauJet", "nMC_Taus_vs_etaTauJet", 60, -3.0, 3.0 );
    nMCTaus_phiTauJet_ =    dbe->book1D("nMC_Taus_vs_phiTauJet", "nMC_Taus_vs_phiTauJet", 36, -180., 180.);
    nMCTaus_energyTauJet_ = dbe->book1D("nMC_Taus_vs_energyTauJet", "nMC_Taus_vs_energyTauJet", 45, 0., 450.0);

    // Number of CaloJets matched to MC Taus
    dbe->setCurrentFolder("RecoTauV/CaloJetMatched_"+ExtensionName_.label());

    nCaloJet_ptTauJet_     = dbe->book1D("CaloJet_vs_ptTauJet", "CaloJet_vs_ptTauJet",  75, 0., 150.); 	     
    nCaloJet_etaTauJet_	   = dbe->book1D("CaloJet_vs_etaTauJet", "CaloJet_vs_etaTauJet", 60, -3.0, 3.0 );
    nCaloJet_phiTauJet_    = dbe->book1D("CaloJet_vs_phiTauJet", "CaloJet_vs_phiTauJet", 36, -180.,180.);	     
    nCaloJet_energyTauJet_ = dbe->book1D("CaloJet_vs_energyTauJet", "CaloJet_vs_energyTauJet", 45, 0., 450.0); 	     
    
    nCaloJet_Tracks_       = dbe->book1D("CaloJet_Number_Tracks", "CaloJet_Number_Tracks", 15, -0.5, 14.5);
    nCaloJet_isolationECALhitsEtSum_ =  dbe->book1D("CaloJet_isolationECALhitsEtSum", "CaloJet_isolationECALhitsEtSum", 75, 0., 75.); 

    // Number of CaloJets with a Leading Track (within a cone of 0.1 around the jet axis and a minimum pt of 5. GeV)
    dbe->setCurrentFolder("RecoTauV/LeadingTrackFinding_"+ExtensionName_.label());
    nCaloJet_LeadingTrack_ptTauJet_    	 =dbe->book1D("LeadingTrack_vs_ptTauJet", "LeadingTrack_vs_ptTauJet", 75, 0., 150.);   
    nCaloJet_LeadingTrack_etaTauJet_  	 =dbe->book1D("LeadingTrack_vs_etaTauJet", "LeadingTrack_vs_etaTauJet", 60, -3.0, 3.0 );          
    nCaloJet_LeadingTrack_phiTauJet_     =dbe->book1D("LeadingTrack_vs_phiTauJet", "LeadingTrack_vs_phiTauJet", 36, -180., 180);        
    nCaloJet_LeadingTrack_energyTauJet_  =dbe->book1D("LeadingTrack_vs_energyTauJet", "LeadingTrack_vs_energyTauJet", 45, 0., 450.0);     	    
    					              
    nCaloJet_LeadingTrack_signalTracksInvariantMass_ =dbe->book1D("LeadingTrack_signalTracksInvariantMass","LeadingTrack_signalTracksInvariantMass",  75, 0., 150.); 
    nCaloJet_LeadingTrack_signalTracks_              =dbe->book1D("LeadingTrack_signalTracks", "LeadingTrack_NumberSignalTracks" , 15, -0.5, 14.5);         
    nCaloJet_LeadingTrack_isolationTracks_           =dbe->book1D("LeadingTrack_isolationTracks", "LeadingTrack_isolationTracks",  15, -0.5, 14.5);      
    nCaloJet_LeadingTrack_isolationECALhitsEtSum_     =dbe->book1D("LeadingTrack_isolationECALhitsEtSum", "LeadingTrack_isolationECALhitsEtSum", 75, 0., 75.); 
    						                 
    // Track Isolated CaloTau with a Leading Track		  
    dbe->setCurrentFolder("RecoTauV/TrackIsolated_"+ExtensionName_.label());

    nTrackIsolated_ptTauJet_       =dbe->book1D("TrackIsolated_vs_ptTauJet","TrackIsolated_vs_ptTauJet", 75, 0., 150.);   		      
    nTrackIsolated_etaTauJet_      =dbe->book1D("TrackIsolated_vs_etaTauJet","TrackIsolated_vs_etaTauJet", 60, -3.0, 3.0 );    	  
    nTrackIsolated_phiTauJet_      =dbe->book1D("TrackIsolated_vs_phiTauJet","TrackIsolated_vs_phiTauJet", 36, -180., 180);	  	  
    nTrackIsolated_energyTauJet_   =dbe->book1D("TrackIsolated_vs_energyTauJet","TrackIsolated_vs_energyTauJet", 45, 0., 450.0);    
     
    nTrackIsolated_isolationECALhitsEtSum_    =dbe->book1D("TrackIsolated_isolationECALhitsEtSum", "TrackIsolated_isolationECALhitsEtSum", 75, 0., 75.);
    nTrackIsolated_signalTracksInvariantMass_ =dbe->book1D("TrackIsolated_signalTracksInvariantMass","TrackIsolated_signalTracksInvariantMass", 75, 0., 150.); 
    nTrackIsolated_signalTracks_              =dbe->book1D("TrackIsolated_signalTracks","TrackIsolated_signalTracks", 15, -0.5, 14.5);  

    // EM Isolated CaloTau with a Leading with no tracks in the Isolation Annulus
    dbe->setCurrentFolder("RecoTauV/EMIsolated_"+ExtensionName_.label());

    nEMIsolated_ptTauJet_       =dbe->book1D("EMIsolated_vs_ptTauJet","EMIsolated_vs_ptTauJet", 75, 0., 150.);          		   
    nEMIsolated_etaTauJet_      =dbe->book1D("EMIsolated_vs_etaTauJet","EMIsolated_vs_etaTauJet", 60, -3.0, 3.0 );               
    nEMIsolated_phiTauJet_      =dbe->book1D("EMIsolated_vs_phiTauJet","EMIsolated_vs_phiTauJet", 36, -180., 180);               
    nEMIsolated_energyTauJet_	  =dbe->book1D("EMIsolated_vs_energyTauJet","EMIsolated_vs_energyTauJet", 45, 0., 450.0);          
    
    nEMIsolated_signalTracksInvariantMass_  =dbe->book1D("EMIsolated_signalTracksInvariantMass","EMIsolated_signalTracksInvariantMass", 75, 0., 150.); 
    nEMIsolated_signalTracks_          =dbe->book1D("EMIsolated_signalTracks","EMIsolated_signalTracks", 15, -0.5, 14.5);    
    
    tversion = edm::getReleaseVersion();
    std::cout<<std::endl<<"-----------------------*******************************Version: " << tversion<<std::endl;  
  }

  if (outPutFile_.empty ()) {
    edm::LogInfo("OutputInfo") << " TauJet histograms will NOT be saved";
  } else {
    int sizeofstring = outPutFile_.size();
    if (sizeofstring > 5);
    outPutFile_.erase(sizeofstring-5);
    outPutFile_.append("_");    
    tversion.erase(0,1);
    tversion.erase(tversion.size()-1,1);
    outPutFile_.append(tversion);
    outPutFile_.append("_"+ outputhistograms_ + "_");
    outPutFile_.append(dataType_+".root");

    std::cout<< std::endl<< outPutFile_<<std::endl;
    edm::LogInfo("OutputInfo") << " TauJethistograms will be saved to file:" << outPutFile_;
  }

  //---- book-keeping information ---
  numEvents_ = 0 ;
}

// ------------ method called once each job just before starting event loop  ------------
void CaloTauTagVal::beginJob()
{
}

//
// member functions
//

// ------------ method called to for each event  ------------
void
CaloTauTagVal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //  using namespace edm;
  //  using namespace reco;
  // using namespace std;
  numEvents_++;
  double matching_criteria=0.15;


  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByLabel("source", evt);

  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));

  // ------------------------------ CaloTauCollection---------------------------------------------------------
  edm::Handle< GenJetCollection > genJets ;
  iEvent.getByLabel( genJetSrc_, genJets ) ;

  // Get a TLorentzVector with the Visible Taus at Generator level (the momentum of the neutrino is substracted
  std::vector<TLorentzVector> TauJetsMC;
  if(dataType_ == "CALOTAU"){
    TauJetsMC=getVectorOfVisibleTauJets(myGenEvent);
    matching_criteria=0.15;
  }

  if(dataType_ == "QCD")  {
    TauJetsMC=getVectorOfGenJets(genJets);
    matching_criteria=0.30;
  }
  
  edm::Handle<CaloTauCollection> theCaloTauHandle;
  iEvent.getByLabel(CaloTauProducer_,theCaloTauHandle);

  edm::Handle<CaloTauDiscriminatorByIsolation> theCaloTauDiscriminatorByIsolation;
  iEvent.getByLabel(CaloTauDiscriminatorByIsolationProducer_,theCaloTauDiscriminatorByIsolation);

  //  Handle<CaloTauDiscriminatorAgainstElectron> theCaloTauDiscriminatorAgainstElectron;
  // iEvent.getByLabel(CaloTauDiscriminatorAgainstElectronProducer_,theCaloTauDiscriminatorAgainstElectron);

  std::cout<<"***"<<std::endl;
  std::cout<<"Found "<<theCaloTauHandle->size()<<" had. tau-jet candidates"<<std::endl;
  int i_CaloTau=0;
  int numTrueCaloTausCand=0;


  for (CaloTauCollection::size_type iCaloTau=0;iCaloTau<theCaloTauHandle->size();iCaloTau++) {
    CaloTauRef theCaloTau(theCaloTauHandle,iCaloTau);

    TLorentzVector CaloTauDirection((*theCaloTau).px(), (*theCaloTau).py(), (*theCaloTau).pz());
    bool trueCaloTau=false;
   
    std::vector<TLorentzVector>::iterator MCjet;
    for (MCjet = TauJetsMC.begin(); MCjet != TauJetsMC.end(); MCjet++){ 
     
      if ( MCjet->DeltaR(CaloTauDirection) <  matching_criteria ) {
	trueCaloTau=true;
	numTrueCaloTausCand++;
        break;
      }
    }
    // If the CaloTau is matched to a MC Tau then continue

    if (trueCaloTau) {
      
      nCaloJet_ptTauJet_->Fill(MCjet->Perp());
      nCaloJet_etaTauJet_->Fill(MCjet->Eta());
      nCaloJet_phiTauJet_->Fill(MCjet->Phi()*180.0/TMath::Pi());
      nCaloJet_energyTauJet_->Fill(MCjet->E());

      nCaloJet_Tracks_->Fill((*theCaloTau).caloTauTagInfoRef()->Tracks().size());	     
      nCaloJet_isolationECALhitsEtSum_->Fill((*theCaloTau).isolationECALhitsEtSum());
      
      // Leading Track finding
      TrackRef theLeadTk=(*theCaloTau).leadTrack();
      if(!theLeadTk){
	//cout<<"No Lead Tk "<<endl;
      }else{

	nCaloJet_LeadingTrack_ptTauJet_->Fill(MCjet->Perp()); 
	nCaloJet_LeadingTrack_etaTauJet_->Fill(MCjet->Eta());  
	nCaloJet_LeadingTrack_phiTauJet_->Fill(MCjet->Phi()*180.0/TMath::Pi()); 
	nCaloJet_LeadingTrack_energyTauJet_->Fill(MCjet->E());                 

	nCaloJet_LeadingTrack_signalTracksInvariantMass_->Fill((*theCaloTau).signalTracksInvariantMass());
	nCaloJet_LeadingTrack_signalTracks_->Fill((*theCaloTau).signalTracks().size()); 
	nCaloJet_LeadingTrack_isolationTracks_->Fill((*theCaloTau).isolationTracks().size());
	nCaloJet_LeadingTrack_isolationECALhitsEtSum_->Fill((*theCaloTau).isolationECALhitsEtSum());

	// Track isolation 
	if ((*theCaloTau).isolationTracks().size()==0){
	  nTrackIsolated_ptTauJet_->Fill(MCjet->Perp());      	    
	  nTrackIsolated_etaTauJet_->Fill(MCjet->Eta());    	    
	  nTrackIsolated_phiTauJet_->Fill(MCjet->Phi()*180.0/TMath::Pi());     	    
          nTrackIsolated_energyTauJet_->Fill(MCjet->E());		            
	                                            
	  nTrackIsolated_isolationECALhitsEtSum_->Fill((*theCaloTau).isolationECALhitsEtSum());   
	  nTrackIsolated_signalTracksInvariantMass_->Fill((*theCaloTau).signalTracksInvariantMass());
	  nTrackIsolated_signalTracks_->Fill((*theCaloTau).signalTracks().size());             
	  //EM isolation
	  if (!((*theCaloTau).isolationECALhitsEtSum()>5.0)){
	    nEMIsolated_ptTauJet_->Fill(MCjet->Perp());      	    	    	 
	    nEMIsolated_etaTauJet_->Fill(MCjet->Eta());    	    	    		 
	    nEMIsolated_phiTauJet_->Fill(MCjet->Phi()*180.0/TMath::Pi());  		 
	    nEMIsolated_energyTauJet_->Fill(MCjet->E());		            	 
	                                           
	    nEMIsolated_signalTracksInvariantMass_->Fill((*theCaloTau).signalTracksInvariantMass());
	    nEMIsolated_signalTracks_->Fill((*theCaloTau).signalTracks().size());             
	  }
	}
      }
    }
    i_CaloTau++;
  }

  delete myGenEvent;
}

// ------------ method called once each job just after ending the event loop  ------------
void CaloTauTagVal::endJob() {

  if (!outPutFile_.empty() && &*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (outPutFile_);

}

// Helper function  

// Get all the daughter stable particles of a particle

std::vector<TLorentzVector> CaloTauTagVal::getVectorOfVisibleTauJets(HepMC::GenEvent *theEvent)
{
  std::vector<TLorentzVector> tempvec;   
  HepMC::GenVertex* TauDecVtx = 0 ;
  int numMCTaus = 0;
  TLorentzVector TauJetMC(0.0,0.0,0.0,0.0);
  for ( HepMC::GenEvent::particle_iterator p = theEvent->particles_begin(); // Loop over all particles
	p != theEvent->particles_end(); ++p ) 
    {
      
      if(abs((*p)->pdg_id())==15 && ((*p)->status()==2))   // Is it a Tau and Decayed?
	{
	  bool FinalTau=true;                           // This looks like a final decayed Tau 
	  TLorentzVector TauDecayProduct(0.0,0.0,0.0,0.);   // Neutrino momentum from the Tau decay at GenLevel
	  TLorentzVector TauJetMC(0.0,0.0,0.0,0.); 

	  std::vector<HepMC::GenParticle*> TauDaught;
	  //  TauDaught=Daughters((*p));
	  TauDaught=getGenStableDecayProducts(*p);
          TauDecVtx = (*p)->end_vertex();
	  
	  if ( TauDecVtx != 0 )
	    {	    
	      int numElectrons      = 0;
	      int numMuons          = 0;
	      int numChargedPions   = 0;
	      int numNeutralPions   = 0;
	      int numNeutrinos      = 0;
	      int numOtherParticles = 0;
	      TString output7="";
	      // Loop over Tau Daughter particles and store the Tau neutrino momentum for future use
	      for(std::vector<HepMC::GenParticle*>::const_iterator pit=TauDaught.begin();pit!=TauDaught.end();++pit) 
		{
		  int pdg_id = abs((*pit)->pdg_id());
		  output7+=" PDG_ID = ";
		  std::stringstream out;
		  out<<pdg_id;
		  output7+=out.str();
		  if (pdg_id == 11) numElectrons++;
		  else if (pdg_id == 13) numMuons++;
		  else if (pdg_id == 211) numChargedPions++;
		  else if (pdg_id == 111) numNeutralPions++;
		  else if (pdg_id == 12 || 
			   pdg_id == 14 || 
			   pdg_id == 16)  numNeutrinos++;
		  else if (pdg_id != 22) {
		    numOtherParticles++;
		    //    cout<< " PDG_ID " << pdg_id << endl;
		  }

		  if (pdg_id != 12 &&
		      pdg_id != 14 && 
		      pdg_id != 16){
		    TauDecayProduct=TLorentzVector((*pit)->momentum().px(),(*pit)->momentum().py(),(*pit)->momentum().pz(),(*pit)->momentum().e());
		    TauJetMC+=TauDecayProduct;
		  }
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
	      //	      cout<<" tauDecayMode: "<<tauDecayMode<<endl;
	      if (tauDecayMode == kOther){
//		cout<<output7<<endl;
//                   std::cout << "HepMCProduct INFO" << std::endl;
//		   theEvent->print();
//		   std::cout << std::endl;
	      }

	      hGenTauDecay_DecayModes_->Fill(tauDecayMode);

	      FinalTau=false;

	      if ( tauDecayMode == kOneProng0pi0   ||
		   tauDecayMode == kOneProng1pi0   ||
		   tauDecayMode == kOneProng2pi0   ||
		   tauDecayMode == kThreeProng0pi0 ||
		   tauDecayMode == kThreeProng1pi0 ){
		   //  ||		   tauDecayMode == kOther) { 
		if (outputhistograms_ == "OneProngAndThreeProng")
                  FinalTau=true;
                else if ( outputhistograms_ == "OneProng" &&
                          (tauDecayMode == kOneProng0pi0 ||
                           tauDecayMode == kOneProng1pi0   ||
                           tauDecayMode == kOneProng2pi0 ))
                  FinalTau=true;
                else if ( outputhistograms_ == "ThreeProng" &&
			  (tauDecayMode == kThreeProng0pi0 ||
			   tauDecayMode == kThreeProng1pi0 ))
                  FinalTau=true;
	      }
	      else {
		FinalTau=false;
		//		if ( numOtherParticles != 0 )
		//  cout <<" This decay mode has no cathegory. "<< endl;
		
	      }
	      
	      if(FinalTau) {  // Meaning: did it find a Neutrino in the list of Daughter particles? Then fill histograms of the original Tau info
	  	  hGenTauDecay_DecayModesChosen_->Fill(tauDecayMode);		
		  ptTauMC_->Fill((*p)->momentum().perp());
		  etaTauMC_->Fill((*p)->momentum().eta()); 
                  phiTauMC_->Fill((*p)->momentum().phi()*(180./TMath::Pi()));
		  energyTauMC_->Fill((*p)->momentum().e());		   
                  
		  // Get the Tau Lorentz Vector
		  //		  TLorentzVector theTau((*p)->momentum().x(),(*p)->momentum().y(),(*p)->momentum().z(),(*p)->momentum().e());
		  //	  TauJetMC=theTau-TauNeutrino;      // Substract the Neutrino Lorentz Vector from the Tau
		  if (abs(TauJetMC.Eta())<2.5 && TauJetMC.Perp()>5.0) {
		    nMCTaus_ptTauJet_->Fill(TauJetMC.Perp());  // Fill the histogram with the Pt, Eta, Energy of the Tau Jet at Generator level
		    nMCTaus_etaTauJet_->Fill(TauJetMC.Eta()); 
                    nMCTaus_phiTauJet_->Fill(TauJetMC.Phi()*180./TMath::Pi());
		    nMCTaus_energyTauJet_->Fill(TauJetMC.E());
		    tempvec.push_back(TauJetMC);
		    ++numMCTaus;
		  }
	    }
	    }
	}
    } // closing the loop over the Particles at Generator level

  return tempvec;
  //  cout<<" Number of Taus at Generator Level: "<< numMCTaus << endl; 
}

std::vector<HepMC::GenParticle*> CaloTauTagVal::getGenStableDecayProducts(const HepMC::GenParticle* particle)
{
  HepMC::GenVertex* vertex = particle->end_vertex();

  std::vector<HepMC::GenParticle*> decayProducts;
  for ( HepMC::GenVertex::particles_out_const_iterator daughter_particle = vertex->particles_out_const_begin(); 
	daughter_particle != vertex->particles_out_const_end(); ++daughter_particle ){
    int pdg_id = abs((*daughter_particle)->pdg_id());

    // check if particle is stable
    if ( pdg_id == 11 || pdg_id == 12 || pdg_id == 13 || pdg_id == 14 || pdg_id == 16 ||  pdg_id == 111 || pdg_id == 211 ){
      // stable particle, identified by pdg code
      decayProducts.push_back((*daughter_particle));
    } else if ( (*daughter_particle)->end_vertex() != NULL ){
      // unstable particle, identified by non-zero decay vertex

      std::vector<HepMC::GenParticle*> addDecayProducts = getGenStableDecayProducts(*daughter_particle);

      for ( std::vector<HepMC::GenParticle*>::const_iterator adddaughter_particle = addDecayProducts.begin(); adddaughter_particle != addDecayProducts.end(); ++adddaughter_particle ){
	decayProducts.push_back((*adddaughter_particle));
      }
    } else {
      // stable particle, not identified by pdg code
      decayProducts.push_back((*daughter_particle));
    }
  }
   
  return decayProducts;
}


//Get a list of Gen Jets
std::vector<TLorentzVector> CaloTauTagVal::getVectorOfGenJets(edm::Handle< GenJetCollection >& genJets ) {
  int jjj=0;
  std::vector<TLorentzVector> GenJets;
  GenJets.clear();
  GenJetCollection::const_iterator jetItr = genJets->begin();
  if(jetItr != genJets->end() )
    {
      math::XYZTLorentzVector p4 = jetItr->p4() ;
      TLorentzVector genJetMC(p4.x(),p4.y(),p4.z(),p4.e());
      if (abs(genJetMC.Eta())<2.5 && genJetMC.Perp()>5.0) {
	if(jjj<2.) {
	  
	  nMCTaus_ptTauJet_->Fill(genJetMC.Perp());  // Fill the histogram with the Pt, Eta, Energy of the Tau Jet at Generator level
	  nMCTaus_etaTauJet_->Fill(genJetMC.Eta()); 
	  nMCTaus_phiTauJet_->Fill(genJetMC.Phi()*180./TMath::Pi());
	  nMCTaus_energyTauJet_->Fill(genJetMC.E());
	  GenJets.push_back(genJetMC);
	  jjj++;
	  
	}
      }
    }
  return GenJets;
}

