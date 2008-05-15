// -*- C++ -*-
//
// Package:    PFTauTagVal
// Class:      PFTauTagVal
// 
/**\class PFTauTagVal PFTauTagVal.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ricardo Vasquez Sierra
//         Created:  October 8, 2007 
// $Id: PFTauTagVal.cc,v 1.7 2008/03/01 17:28:22 gennai Exp $
//
//
// user include files

#include "Validation/RecoTau/interface/PFTauTagVal.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace edm;
using namespace std;
using namespace reco;

PFTauTagVal::PFTauTagVal(const edm::ParameterSet& iConfig)
{
  dataType_ = iConfig.getParameter<string>("DataType");
  outputhistograms_ = iConfig.getParameter<string>("OutPutHistograms");
  //jetTagSrc_ = iConfig.getParameter<InputTag>("JetTagProd");
  //jetEMTagSrc_ = iConfig.getParameter<InputTag>("JetEMTagProd");
  genJetSrc_ = iConfig.getParameter<InputTag>("GenJetProd");
  
  ExtensionName_ = iConfig.getParameter<InputTag>("ExtensionName");
  outPutFile_ = iConfig.getParameter<string>("OutPutFile"); 
  dataType_ = iConfig.getParameter<string>("DataType");
  PFTauProducer_ = iConfig.getParameter<string>("PFTauProducer");
  PFTauDiscriminatorByIsolationProducer_ = iConfig.getParameter<string>("PFTauDiscriminatorByIsolationProducer");
  

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

    // Number of PFTau Candidates matched to MC Taus
    dbe->setCurrentFolder("RecoTauV/PFTauCandidatesMatched_"+ExtensionName_.label());

    nPFTauCand_ptTauJet_ =     dbe->book1D("n_PFTauCand_vs_ptTauJet", "n_PFTauCand_vs_ptTauJet", 75, 0., 150.);
    nPFTauCand_etaTauJet_ =    dbe->book1D("n_PFTauCand_vs_etaTauJet", "n_PFTauCand_vs_etaTauJet",60, -3.0, 3.0 );
    nPFTauCand_phiTauJet_ =    dbe->book1D("n_PFTauCand_vs_phiTauJet", "n_PFTauCand_vs_phiTauJet",36, -180.,180.);
    nPFTauCand_energyTauJet_ = dbe->book1D("n_PFTauCand_vs_energyTauJet", "n_PFTauCand_vs_energyTauJet", 45, 0., 450.0);  
    nPFTauCand_ChargedHadronsSignal_     =dbe->book1D("nPFTauCand_ChargedHadronsSignal","nPFTauCand_ChargedHadronsSignal", 21, -0.5, 20.5);
    nPFTauCand_ChargedHadronsIsolAnnulus_=dbe->book1D("nPFTauCand_ChargedHadronsIsolAnnulus","nPFTauCand_ChargedHadronsIsolAnnulus", 21, -0.5, 20.5);
    nPFTauCand_GammasSignal_             =dbe->book1D("nPFTauCand_GammasSignal","nPFTauCand_GammasSignal",21, -0.5, 20.5);
    nPFTauCand_GammasIsolAnnulus_        =dbe->book1D("nPFTauCand_GammasIsolAnnulus","nPFTauCand_GammasIsolAnnulus",21, -0.5, 20.5);  
    nPFTauCand_NeutralHadronsSignal_     =dbe->book1D("nPFTauCand_NeutralHadronsSignal","nPFTauCand_NeutralHadronsSignal",21, -0.5, 20.5);
    nPFTauCand_NeutralHadronsIsolAnnulus_=dbe->book1D("nPFTauCand_NeutralHadronsIsolAnnulus","nPFTauCand_NeutralHadronsIsolAnnulus",21, -0.5, 20.5);   

    // Number of PFTau Candidates with a Leading charged hadron in it (within a cone of 0.1 avound the jet axis and a minimum pt of 6 GeV)
    dbe->setCurrentFolder("RecoTauV/PFTauPlusLeadingChargedHadron_"+ExtensionName_.label());

    nPFTau_LeadingChargedHadron_ptTauJet_ =     dbe->book1D("n_PFTau_LeadingChargedHadron_vs_ptTauJet", "n_PFTau_LeadingChargedHadron_vs_ptTauJet", 75, 0., 150.);
    nPFTau_LeadingChargedHadron_etaTauJet_ =    dbe->book1D("n_PFTau_LeadingChargedHadron_vs_etaTauJet", "n_PFTau_LeadingChargedHadron_vs_etaTauJet",60, -3.0, 3.0 );
    nPFTau_LeadingChargedHadron_phiTauJet_ =    dbe->book1D("n_PFTau_LeadingChargedHadron_vs_phiTauJet", "n_PFTau_LeadingChargedHadron_vs_phiTauJet",36,-180.,180);
    nPFTau_LeadingChargedHadron_energyTauJet_ = dbe->book1D("n_PFTau_LeadingChargedHadron_vs_energyTauJet", "n_PFTau_LeadingChargedHadron_vs_energyTauJet", 45, 0., 450.0); 
    nPFTau_LeadingChargedHadron_ChargedHadronsSignal_     =dbe->book1D("nPFTau_LeadingChargedHadron_ChargedHadronsSignal","nPFTau_LeadingChargedHadron_ChargedHadronsSignal",21, -0.5, 20.5);
    nPFTau_LeadingChargedHadron_ChargedHadronsIsolAnnulus_=dbe->book1D("nPFTau_LeadingChargedHadron_ChargedHadronsIsolAnnulus", "nPFTau_LeadingChargedHadron_ChargedHadronsIsolAnnulus",21, -0.5, 20.5);
    nPFTau_LeadingChargedHadron_GammasSignal_             =dbe->book1D("nPFTau_LeadingChargedHadron_GammasSignal","nPFTau_LeadingChargedHadron_GammasSignal",21, -0.5, 20.5);
    nPFTau_LeadingChargedHadron_GammasIsolAnnulus_        =dbe->book1D("nPFTau_LeadingChargedHadron_GammasIsolAnnulus","nPFTau_LeadingChargedHadron_GammasIsolAnnulus",21, -0.5, 20.5);          
    nPFTau_LeadingChargedHadron_NeutralHadronsSignal_     =dbe->book1D("nPFTau_LeadingChargedHadron_NeutralHadronsSignal","nPFTau_LeadingChargedHadron_NeutralHadronsSignal",21, -0.5, 20.5);
    nPFTau_LeadingChargedHadron_NeutralHadronsIsolAnnulus_=dbe->book1D("nPFTau_LeadingChargedHadron_NeutralHadronsIsolAnnulus","nPFTau_LeadingChargedHadron_NeutralHadronsIsolAnnulus",21, -0.5, 20.5);
    
    // Isolated PFTau with a Leading charged hadron with no Charged Hadrons inside the isolation annulus
    dbe->setCurrentFolder("RecoTauV/Isolated_NoChargedHadrons_"+ExtensionName_.label());

    nIsolated_NoChargedHadrons_ptTauJet_ =       dbe->book1D("n_Isolated_NoChargedHadrons_vs_ptTauJet","n_Isolated_NoChargedHadrons_vs_ptTauJet", 75, 0., 150.);
    nIsolated_NoChargedHadrons_etaTauJet_ =      dbe->book1D("n_Isolated_NoChargedHadrons_vs_etaTauJet","n_Isolated_NoChargedHadrons_vs_etaTauJet", 60, -3.0, 3.0 );
    nIsolated_NoChargedHadrons_phiTauJet_ =      dbe->book1D("n_Isolated_NoChargedHadrons_vs_phiTauJet","n_Isolated_NoChargedHadrons_vs_phiTauJets", 36, -180., 180);
    nIsolated_NoChargedHadrons_energyTauJet_ =   dbe->book1D("n_Isolated_NoChargedHadrons_vs_energyTauJet", "n_Isolated_NoChargedHadrons_vs_energyTauJet", 45, 0., 450.0);    
    nIsolated_NoChargedHadrons_ChargedHadronsSignal_     =dbe->book1D("n_Isolated_NoChargedHadrons_ChargedHadronsSignal","nIsolated_NoChargedHadrons_ChargedHadronsSignal",21, -0.5, 20.5);	  
    //  nIsolated_NoChargedHadrons_ChargedHadronsIsolAnnulus_=dbe->book1D("nIsolated_NoChargedHadrons_ChargedHadronsIsolAnnulus","nIsolated_NoChargedHadrons_ChargedHadronsIsolAnnulus",21, -0.5, 20.5); 
    nIsolated_NoChargedHadrons_GammasSignal_             =dbe->book1D("nIsolated_NoChargedHadrons_GammasSignal","nIsolated_NoChargedHadrons_GammasSignal",21, -0.5, 20.5);		  
    nIsolated_NoChargedHadrons_GammasIsolAnnulus_        =dbe->book1D("nIsolated_NoChargedHadrons_GammasIsolAnnulus","nIsolated_NoChargedHadrons_GammasIsolAnnulus",21, -0.5, 20.5);         
    nIsolated_NoChargedHadrons_NeutralHadronsSignal_     =dbe->book1D("nIsolated_NoChargedHadrons_NeutralHadronsSignal","nIsolated_NoChargedHadrons_NeutralHadronsSignal",21, -0.5, 20.5);
    nIsolated_NoChargedHadrons_NeutralHadronsIsolAnnulus_=dbe->book1D("nIsolated_NoChargedHadrons_NeutralHadronsIsolAnnulus","nIsolated_NoChargedHadrons_NeutralHadronsIsolAnnulus",21, -0.5, 20.5);
    
    // Isolated PFTau with a Leading charge hadron with no Charged Hadron inside the isolation annulus with no Ecal/Gamma candidates in the isolation annulus
    dbe->setCurrentFolder("RecoTauV/Isolated_NoChargedNoGammas_"+ExtensionName_.label());

    nIsolated_NoChargedNoGammas_ptTauJet_ =       dbe->book1D("n_Isolated_NoChargedNoGammas_vs_ptTauJet","n_Isolated_NoChargedNoGammas_vs_ptTauJet", 75, 0., 150.);
    nIsolated_NoChargedNoGammas_etaTauJet_ =      dbe->book1D("n_Isolated_NoChargedNoGammas_vs_etaTauJet","n_Isolated_NoChargedNoGammas_vs_etaTauJet", 60, -3.0, 3.0 );
    nIsolated_NoChargedNoGammas_phiTauJet_ =      dbe->book1D("n_Isolated_NoChargedNoGammas_vs_phiTauJet","n_Isolated_NoChargedNoGammas_vs_phiTauJets", 36, -180., 180);
    nIsolated_NoChargedNoGammas_energyTauJet_ =   dbe->book1D("n_Isolated_NoChargedNoGammas_vs_energyTauJet", "n_Isolated_NoChargedNoGammas_vs_energyTauJet", 45, 0., 450.0); 
    nIsolated_NoChargedNoGammas_ChargedHadronsSignal_     =dbe->book1D("nIsolated_NoChargedNoGammas_ChargedHadronsSignal","nIsolated_NoChargedNoGammas_ChargedHadronsSignal",21, -0.5, 20.5);	  
    //  nIsolated_NoChargedNoGammas_ChargedHadronsIsolAnnulus_=dbe->book1D("nIsolated_NoChargedNoGammas_ChargedHadronsIsolAnnulus","nIsolated_NoChargedNoGammas_ChargedHadronsIsolAnnulus",21, -0.5, 20.5);    
    nIsolated_NoChargedNoGammas_GammasSignal_             =dbe->book1D("nIsolated_NoChargedNoGammas_GammasSignal","nIsolated_NoChargedNoGammas_GammasSignal",21, -0.5, 20.5);	  
    // nIsolated_NoChargedNoGammas_GammasIsolAnnulus_        =dbe->book1D("nIsolated_NoChargedNoGammas_GammasIsolAnnulus","nIsolated_NoChargedNoGammas_GammasIsolAnnulus",21, -0.5, 20.5);         
    nIsolated_NoChargedNoGammas_NeutralHadronsSignal_     =dbe->book1D("nIsolated_NoChargedNoGammas_NeutralHadronsSignal","nIsolated_NoChargedNoGammas_NeutralHadronsSignal",21, -0.5, 20.5);
    nIsolated_NoChargedNoGammas_NeutralHadronsIsolAnnulus_=dbe->book1D("nIsolated_NoChargedNoGammas_NeutralHadronsIsolAnnulus","nIsolated_NoChargedNoGammas_NeutralHadronsIsolAnnulus",21, -0.5, 20.5);

    dbe->setCurrentFolder("RecoTauV/PFCandidates_in_SignalOrIsolationAnnulus_"+ExtensionName_.label());

    nChargedHadronsSignalCone_isolated_= dbe->book1D("nChargedHadronsSignalCone_isolated","nChargedHadronsSignalCone_isolated", 21,-0.5,20.5);
    nGammasSignalCone_isolated_        = dbe->book1D("nGammasSignalCone_isolated","nGammasSignalCone_isolated",21,-0.5,20.5);
    nNeutralHadronsSignalCone_isolated_= dbe->book1D("nNeutralHadronsSignalCone_isolated","nNeutralHadronsSignalCone_isolated",21,-0.5,20.5);
    
    N_1_ChargedHadronsSignal_          = dbe->book1D("N_1_ChargedHadronsSignal","N_1_ChargedHadronsSignal",21,-0.5,20.5);	  
    N_1_ChargedHadronsIsolAnnulus_     = dbe->book1D("N_1_ChargedHadronsIsolAnnulus","N_1_ChargedHadronsIsolAnnulus",21,-0.5,20.5); 
    N_1_GammasSignal_                  = dbe->book1D("N_1_GammasSignal","N_1_GammasSignal",21,-0.5,20.5);		  
    N_1_GammasIsolAnnulus_             = dbe->book1D("N_1_GammasIsolAnnulus","N_1_GammasIsolAnnulus",21,-0.5,20.5);
    N_1_NeutralHadronsSignal_          = dbe->book1D("N_1_NeutralHadronsSignal","N_1_NeutralHadronsSignal",21,-0.5,20.5);	      
    N_1_NeutralHadronsIsolAnnulus_     = dbe->book1D("N_1_NeutralHadronsIsolAnnulus","N_1_NeutralHadronsIsolAnnulus",21,-0.5,20.5);
    
    tversion = edm::getReleaseVersion();
    cout<<endl<<"-----------------------*******************************Version: " << tversion<<endl;
 
  }
    
  if (outPutFile_.empty ()) {
    LogInfo("OutputInfo") << " TauJet histograms will NOT be saved";
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

    cout<<endl<< outPutFile_<<endl;
    LogInfo("OutputInfo") << " TauJethistograms will be saved to file:" << outPutFile_;
  }

  //---- book-keeping information ---
  numEvents_ = 0 ;
}

void PFTauTagVal::beginJob()
{ 


} 

// -- method called to produce fill all the histograms --------------------
void PFTauTagVal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  using namespace std;
  numEvents_++;
  double matching_criteria=0.15;

  //  std::cout << "--------------------------------------------------------------"<<endl;
  //std::cout << " RunNumber: " << iEvent.id().run() << ", EventNumber: " << iEvent.id().event() << std:: endl;
  //std::cout << "Event number: " << ++numEvents_ << endl;
  //std::cout << "--------------------------------------------------------------"<<endl;

  // ------------------------ MC product stuff -------------------------------------------------------------------------
  Handle<HepMCProduct> evt;
  iEvent.getByLabel("source", evt);

  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));

  edm::Handle< GenJetCollection > genJets ;
  iEvent.getByLabel( genJetSrc_, genJets ) ;

  // Get a TLorentzVector with the Visible Taus at Generator level (the momentum of the neutrino is substracted
  vector<TLorentzVector> TauJetsMC;
  if(dataType_ == "PFTAU"){
    TauJetsMC=getVectorOfVisibleTauJets(myGenEvent);
    matching_criteria=0.15;
  }

  if(dataType_ == "QCD")  {
    TauJetsMC=getVectorOfGenJets(genJets);
    matching_criteria=0.30;
  }
  
  
  //  myGenEvent->print();
    
  // ------------------------------ PFTauCollection---------------------------------------------------------
  Handle<PFTauCollection> thePFTauHandle;
  iEvent.getByLabel(PFTauProducer_,thePFTauHandle);
  
  cout<<"***"<<endl;
  cout<<"Found "<<thePFTauHandle->size()<<" had. tau-jet candidates"<<endl;
  int i_PFTau=0;
  int numTruePFTausCand=0;

  for (PFTauCollection::size_type iPFTau=0;iPFTau<thePFTauHandle->size();iPFTau++) {
    PFTauRef thePFTau(thePFTauHandle,iPFTau);

    TLorentzVector PFTauDirection((*thePFTau).px(), (*thePFTau).py(), (*thePFTau).pz());
    bool truePFTau=false;
   
    std::vector<TLorentzVector>::iterator MCjet;
    for (MCjet = TauJetsMC.begin(); MCjet != TauJetsMC.end(); MCjet++){ 
     
      if ( MCjet->DeltaR(PFTauDirection) <  matching_criteria ) {
	truePFTau=true;
	numTruePFTausCand++;
        break;
      }
    }

    // If the PFTau is matched to a MC Tau then continue

    if (truePFTau) {
      nPFTauCand_ptTauJet_->Fill(MCjet->Perp());
      nPFTauCand_etaTauJet_->Fill(MCjet->Eta());
      nPFTauCand_phiTauJet_->Fill(MCjet->Phi()*180.0/TMath::Pi());
      nPFTauCand_energyTauJet_->Fill(MCjet->E());
      nPFTauCand_ChargedHadronsSignal_->Fill((*thePFTau).signalPFChargedHadrCands().size());	
      nPFTauCand_ChargedHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFChargedHadrCands().size());
      nPFTauCand_GammasSignal_->Fill((*thePFTau).signalPFGammaCands().size());		
      nPFTauCand_GammasIsolAnnulus_->Fill((*thePFTau).isolationPFGammaCands().size());  	
      nPFTauCand_NeutralHadronsSignal_->Fill((*thePFTau).signalPFNeutrHadrCands().size());	
      nPFTauCand_NeutralHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFNeutrHadrCands().size());
      
      PFCandidateRef theLeadPFCand = (*thePFTau).leadPFChargedHadrCand();
      
      if (!theLeadPFCand) {}
      else {
	nPFTau_LeadingChargedHadron_ptTauJet_->Fill(MCjet->Perp()); 
	nPFTau_LeadingChargedHadron_etaTauJet_->Fill(MCjet->Eta());  
	nPFTau_LeadingChargedHadron_phiTauJet_->Fill(MCjet->Phi()*180.0/TMath::Pi()); 
	nPFTau_LeadingChargedHadron_energyTauJet_->Fill(MCjet->E());
	nPFTau_LeadingChargedHadron_ChargedHadronsSignal_->Fill((*thePFTau).signalPFChargedHadrCands().size());
	nPFTau_LeadingChargedHadron_ChargedHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFChargedHadrCands().size());
	nPFTau_LeadingChargedHadron_GammasSignal_->Fill((*thePFTau).signalPFGammaCands().size());		 
	nPFTau_LeadingChargedHadron_GammasIsolAnnulus_->Fill((*thePFTau).isolationPFGammaCands().size()); 
	
	//	for (unsigned int k = 0; k != (*thePFTau).isolationPFGammaCands().size(); k++) {
	// cout<<"pt of gammas: "<<(*thePFTau).isolationPFGammaCands().at(k)->pt()<<endl;
	// }
	
       
	nPFTau_LeadingChargedHadron_NeutralHadronsSignal_->Fill((*thePFTau).signalPFNeutrHadrCands().size());	 
	nPFTau_LeadingChargedHadron_NeutralHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFNeutrHadrCands().size());
	
	if ((*thePFTau).isolationPFChargedHadrCands().size()==0 && (*thePFTau).isolationPFGammaCands().size()==0) {
          N_1_NeutralHadronsSignal_->Fill((*thePFTau).signalPFNeutrHadrCands().size());	 	  
	  N_1_NeutralHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFNeutrHadrCands().size());
	}

	if ((*thePFTau).isolationPFChargedHadrCands().size()==0 && (*thePFTau).isolationPFNeutrHadrCands().size()==0) {
	   N_1_GammasSignal_->Fill((*thePFTau).signalPFGammaCands().size());	 
	   N_1_GammasIsolAnnulus_->Fill((*thePFTau).isolationPFGammaCands().size());
	}
	
	if ((*thePFTau).isolationPFGammaCands().size()==0 && (*thePFTau).isolationPFNeutrHadrCands().size()==0) {
	  N_1_ChargedHadronsSignal_->Fill((*thePFTau).signalPFChargedHadrCands().size());	  
	  N_1_ChargedHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFChargedHadrCands().size()); 
	}

	if ((*thePFTau).isolationPFChargedHadrCands().size()==0) {
	  nIsolated_NoChargedHadrons_ptTauJet_->Fill(MCjet->Perp()); 
	  nIsolated_NoChargedHadrons_etaTauJet_->Fill(MCjet->Eta());     
	  nIsolated_NoChargedHadrons_phiTauJet_->Fill(MCjet->Phi()*180.0/TMath::Pi());     
	  nIsolated_NoChargedHadrons_energyTauJet_->Fill(MCjet->E());
	  nIsolated_NoChargedHadrons_ChargedHadronsSignal_->Fill((*thePFTau).signalPFChargedHadrCands().size());	 
	  //	  nIsolated_NoChargedHadrons_ChargedHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFChargedHadrCands().size());
	  nIsolated_NoChargedHadrons_GammasSignal_->Fill((*thePFTau).signalPFGammaCands().size());		
	  nIsolated_NoChargedHadrons_GammasIsolAnnulus_->Fill((*thePFTau).isolationPFGammaCands().size());        
	  nIsolated_NoChargedHadrons_NeutralHadronsSignal_->Fill((*thePFTau).signalPFNeutrHadrCands().size());	
	  nIsolated_NoChargedHadrons_NeutralHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFNeutrHadrCands().size());

	  if ((*thePFTau).isolationPFGammaCands().size()==0) {
	    nIsolated_NoChargedNoGammas_ptTauJet_->Fill(MCjet->Perp());     
	    nIsolated_NoChargedNoGammas_etaTauJet_ ->Fill(MCjet->Eta());    
	    nIsolated_NoChargedNoGammas_phiTauJet_->Fill(MCjet->Phi()*180.0/TMath::Pi());     
	    nIsolated_NoChargedNoGammas_energyTauJet_->Fill(MCjet->E());
            nIsolated_NoChargedNoGammas_ChargedHadronsSignal_->Fill((*thePFTau).signalPFChargedHadrCands().size());	 
	    //    nIsolated_NoChargedNoGammas_ChargedHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFChargedHadrCands().size());
	    nIsolated_NoChargedNoGammas_GammasSignal_->Fill((*thePFTau).signalPFGammaCands().size());		 
	    //   nIsolated_NoChargedNoGammas_GammasIsolAnnulus_->Fill((*thePFTau).isolationPFGammaCands().size());        
	    nIsolated_NoChargedNoGammas_NeutralHadronsSignal_->Fill((*thePFTau).signalPFNeutrHadrCands().size());	 
	    nIsolated_NoChargedNoGammas_NeutralHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFNeutrHadrCands().size());

	    
	    
	    if ((*thePFTau).isolationPFNeutrHadrCands().size()==0) {
	      nChargedHadronsSignalCone_isolated_->Fill((*thePFTau).signalPFChargedHadrCands().size());
	      nGammasSignalCone_isolated_->Fill((*thePFTau).signalPFGammaCands().size());	      
	      nNeutralHadronsSignalCone_isolated_->Fill((*thePFTau).signalPFNeutrHadrCands().size());
	    }
	  }
	}
      }
    }
 
    i_PFTau++;    
  }//Closing the looping for PFTaus

  // PFJets iterativeCone5PFJets---------------------------------------------------------------------------------------------
  /*
  Handle<PFJetCollection> pfjetHandle;
  iEvent.getByLabel("iterativeCone5PFJets", pfjetHandle);
  PFJetCollection pfjetCollection = *(pfjetHandle.product());
  int PFJetsIterativeCone5 = 0;
  
  PFJetCollection::const_iterator l = pfjetCollection.begin();
  for (; l!=pfjetCollection.end(); l++){
    PFJetsIterativeCone5++;
  }
  cout<<"PFJetsIterativeCone5: "<< PFJetsIterativeCone5<<endl;*/
  
  


}
 

// ---------------------------------------------------------------------------  endJob -----------------------------------------------------------------------

void PFTauTagVal::endJob(){
  // just fill the denominator histograms for the changing cone sizes

  
  
  /* double MC_Taus = nMCTaus_etaTauJet_->getEntries();
  double CaloJets_Taus = nRecoJet_etaTauJet_->getEntries();
  double CaloJetsLeadTrack_Taus = nRecoJet_LeadingTrack_etaTauJet_->getEntries();
  double IsolatedTagged_Taus = nIsolatedJet_etaTauJet_->getEntries();
  double EMIsolatedTagged_Taus = nEMIsolatedJet_etaTauJet_->getEntries();
  
  std::streamsize prec = cout.precision();
 
  cout<<setfill('-')<<setw(110)<<"-"<<endl;
  
  cout<<setfill('-')<<setw(55)<<" TAU TAG VALIDATION SUMMARY "<<setw(55)<<"-"<<endl;
  
  cout<<setfill('-')<<setw(92)<<"-"<<setfill(' ')<<setw(9)<<"Rel.Eff."<<setw(9)<<"Tot.Eff."<<endl;
  cout<<setfill('-')<<setw(85)<<left<<" TOTAL NUMBER OF TAUS AT GENERATOR LEVEL: ";
  
  cout<<setfill(' ')<<setw(7) <<right<< nMCTaus_etaTauJet_->getEntries() <<setw(9)<<"--"<<setw(9)<<"--"<< endl;
  
  cout<<setfill('-')<<setw(85)<<left<<" Step 1. TOTAL NUMBER OF ITERATIVE CONE 5 JETS MATCHED TO MC TAUS: ";
  cout<<setfill(' ')<<setw(7) <<right<<nRecoJet_etaTauJet_->getEntries()<<setw(9)<<"--";
  if (MC_Taus > 0) 
    cout<<setw(9)<<setprecision(3)<< CaloJets_Taus/MC_Taus  << setprecision(prec)<<endl;
  else 
    cout<<setw(9)<<"--"<<endl;
  
  cout<<setfill('-')<<setw(85)<<left<<" Step 2. PLUS LEADING TRACK= 6.0 GeV MATCH CONE = 0.1: ";
  cout<<setfill(' ')<<setw(7)<<right<<nRecoJet_LeadingTrack_etaTauJet_->getEntries();
  if (CaloJets_Taus > 0) 
    cout<<setw(9)<<setprecision(3)<< CaloJetsLeadTrack_Taus/CaloJets_Taus <<setprecision(prec);
  else
    cout<<setw(9)<<"--"<<endl;
  
  if (MC_Taus > 0)
    cout<<setw(9)<<setprecision(3)<< CaloJetsLeadTrack_Taus/MC_Taus  << setprecision(prec)<<endl;
  else 
    cout<<setw(9)<<"--"<<endl;

  
 
  cout<<setfill('-')<<setw(85)<<left<<" Step 3. PLUS SIGNAL= 0.07 ISOLATION = 0.45 TRACKS IN ISO = 0 MINIMUM PT = 1.0 GeV: ";
  cout<<setfill(' ')<<setw(7) <<right<<nIsolatedJet_etaTauJet_->getEntries();
  if (CaloJetsLeadTrack_Taus > 0) 
    cout<<setw(9)<<setprecision(3)<< IsolatedTagged_Taus/CaloJetsLeadTrack_Taus <<setprecision(prec);
  else
    cout<<setw(9)<<"--"<<endl;
  
  if (MC_Taus > 0)
    cout<<setw(9)<<setprecision(3)<< IsolatedTagged_Taus/MC_Taus  << setprecision(prec)<<endl;
  else 
    cout<<setw(9)<<"--"<<endl;

  //  cout<<setfill('-')<<setw(110)<<"-"<<endl;

  cout<<setfill('-')<<setw(85)<<left<<" Step 4. PLUS ECAL ISOLATION ";
  cout<<setfill(' ')<<setw(7) <<right<<nEMIsolatedJet_etaTauJet_->getEntries();
  if (IsolatedTagged_Taus > 0) 
    cout<<setw(9)<<setprecision(3)<< EMIsolatedTagged_Taus/IsolatedTagged_Taus <<setprecision(prec);
  else
    cout<<setw(9)<<"--"<<endl;
  
  if (MC_Taus > 0)
    cout<<setw(9)<<setprecision(3)<< EMIsolatedTagged_Taus/MC_Taus  << setprecision(prec)<<endl;
  else 
    cout<<setw(9)<<"--"<<endl;

  cout<<setfill('-')<<setw(110)<<"-"<<endl;
  */

if (!outPutFile_.empty() && &*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (outPutFile_);

  
}

// Helper function  

// Get all the daughter stable particles of a particle

std::vector<TLorentzVector> PFTauTagVal::getVectorOfVisibleTauJets(HepMC::GenEvent *theEvent)
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

	  vector<HepMC::GenParticle*> TauDaught;
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
	      for(vector<HepMC::GenParticle*>::const_iterator pit=TauDaught.begin();pit!=TauDaught.end();++pit) 
		{
		  int pdg_id = abs((*pit)->pdg_id());
		  output7+=" PDG_ID = ";
		  stringstream out;
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

std::vector<HepMC::GenParticle*> PFTauTagVal::getGenStableDecayProducts(const HepMC::GenParticle* particle)
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
std::vector<TLorentzVector> PFTauTagVal::getVectorOfGenJets(Handle< GenJetCollection >& genJets ) {
int jjj=0;
  vector<TLorentzVector> GenJets;
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
