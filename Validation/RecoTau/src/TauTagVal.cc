// -*- C++ -*-
//
// Package:    TauTagVal
// Class:      TauTagVal
// 
/**\class TauTagVal TauTagVal.cc RecoTauTag/ConeIsolation/test/TauTagVal.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Simone Gennai/Ricardo Vasquez Sierra
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// $Id: TauTagVal.cc,v 1.16 2008/03/01 17:28:22 gennai Exp $
//
//
// user include files

#include "Validation/RecoTau/interface/TauTagVal.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace edm;
using namespace std;
using namespace reco;

TauTagVal::TauTagVal(const edm::ParameterSet& iConfig)
{
  dataType_ = iConfig.getParameter<string>("DataType");
  outputhistograms_ = iConfig.getParameter<string>("OutPutHistograms");
  jetTagSrc_ = iConfig.getParameter<InputTag>("JetTagProd");
  jetEMTagSrc_ = iConfig.getParameter<InputTag>("JetEMTagProd");
  genJetSrc_ = iConfig.getParameter<InputTag>("GenJetProd");
  outPutFile_ = iConfig.getParameter<string>("OutPutFile");

  rSig_ = iConfig.getParameter<double>("SignalCone");
  rMatch_ = iConfig.getParameter<double>("MatchingCone");
  rIso_ = iConfig.getParameter<double>("IsolationCone");
  ptLeadTk_ = iConfig.getParameter<double>("MinimumTransverseMomentumLeadingTrack");
  minPtIsoRing_ = iConfig.getParameter<double>("MinimumTransverseMomentumInIsolationRing");
  nTracksInIsolationRing_ = iConfig.getParameter<int>("MaximumNumberOfTracksIsolationRing");
 
 DQMStore* dbe = &*edm::Service<DQMStore>();

  if(dbe) {

    // What kind of Taus do we originally have!
    dbe->setCurrentFolder("RecoTauV/TausAtGenLevel_" + jetTagSrc_.label());    
	
    ptTauMC_    = dbe->book1D("pt_Tau_GenLevel", "pt_Tau_GenLevel", 75, 0., 150.);
    etaTauMC_   = dbe->book1D("eta_Tau_GenLevel", "eta_Tau_GenLevel", 60, -3.0, 3.0 );
    phiTauMC_   = dbe->book1D("phi_Tau_GenLevel", "phi_Tau_GenLevel", 36, -180., 180.);
    energyTauMC_= dbe->book1D("Energy_Tau_GenLevel", "Energy_Tau_GenLevel", 45, 0., 450.0);
    hGenTauDecay_DecayModes_ = dbe->book1D("genDecayMode", "DecayMode", kOther + 1, -0.5, kOther + 0.5);
    hGenTauDecay_DecayModesChosen_ = dbe->book1D("genDecayModeChosen", "DecayModeChosen", kOther + 1, -0.5, kOther + 0.5);

    nMCTaus_ptTauJet_ = dbe->book1D("nMC_Taus_vs_ptTauJet", "nMC_Taus_vs_ptTauJet", 75, 0., 150.); 
    nMCTaus_etaTauJet_ = dbe->book1D("nMC_Taus_vs_etaTauJet", "nMC_Taus_vs_etaTauJet", 60, -3.0, 3.0 );
    nMCTaus_phiTauJet_ = dbe->book1D("nMC_Taus_vs_phiTauJet", "nMC_Taus_vs_phiTauJet", 36, -180., 180.);
    nMCTaus_energyTauJet_ = dbe->book1D("nMC_Taus_vs_energyTauJet", "nMC_Taus_vs_energyTauJet", 45, 0., 450.0);

     // Leading Track Related Histograms In case the finding of the leading track is a problem ( with deltaR 0.15 and minimum pt of 1.0 GeV )
    dbe->setCurrentFolder("RecoTauV/LeadingTrackPtAndDeltaRStudies_"+jetTagSrc_.label());

    deltaRLeadTk_Jet_ = dbe->book1D("DeltaR_LeadingTrack_in_RecoJet","DeltaR_LeadingTrack_in_RecoJet",30,0.,0.15);    
    ptLeadingTrack_ = dbe->book1D("Leading_track_pt_in_RecoJet", "Leading_track_pt_in_RecoJet", 10, 0., 10.);

    // What are the number of matched IsolatedTauTagInfoCollection with MC jet
    dbe->setCurrentFolder("RecoTauV/ReconstructedJet_"+jetTagSrc_.label());

    nRecoJet_ptTauJet_ = dbe->book1D("n_RecoJet_vs_ptTauJet", "n_RecoJet_vs_ptTauJet", 75, 0., 150.);
    nRecoJet_etaTauJet_ = dbe->book1D("n_RecoJet_vs_etaTauJet", "n_RecoJet_vs_etaTauJet",60, -3.0, 3.0 );
    nRecoJet_phiTauJet_ = dbe->book1D("n_RecoJet_vs_phiTauJet", "n_RecoJet_vs_phiTauJet",36, -180.,180.);
    nRecoJet_energyTauJet_ = dbe->book1D("n_RecoJet_vs_energyTauJet", "n_RecoJet_vs_energyTauJet", 45, 0., 450.0);  
    nAssociatedTracks_ = dbe->book1D("Number_Associated_Tracks", "Number_Associated_Tracks", 10, 0., 10.);
    nSelectedTracks_ = dbe->book1D("Number_Selected_Tracks", "Number_Selected_Tracks", 10, 0., 10.);

    // What are the number of RecoJets that are matched to MC Tau with a LeadingTrack of 6.0 GeV
    dbe->setCurrentFolder("RecoTauV/ReconstructedJetWithLeadingTrack_"+jetTagSrc_.label());

    nRecoJet_LeadingTrack_ptTauJet_ = dbe->book1D("n_RecoJet+LeadingTrack_vs_ptTauJet", "n_RecoJet+LeadingTrack_vs_ptTauJet", 75, 0., 150.);
    nRecoJet_LeadingTrack_etaTauJet_ = dbe->book1D("n_RecoJet+LeadingTrack_vs_etaTauJet", "n_RecoJet+LeadingTrack_vs_etaTauJet",60, -3.0, 3.0 );
    nRecoJet_LeadingTrack_phiTauJet_ = dbe->book1D("n_RecoJet+LeadingTrack_vs_phiTauJet", "n_RecoJet+LeadingTrack_vs_phiTauJet",36,-180.,180);
    nRecoJet_LeadingTrack_energyTauJet_ = dbe->book1D("n_RecoJet+LeadingTrack_vs_energyTauJet", "n_RecoJet+LeadingTrack_vs_energyTauJet", 45, 0., 450.0); 
    nSignalTracks_ = dbe->book1D("Number_Signal_Tracks", "Number_Signal_Tracks", 10, 0., 10.); 

    // What are the numbers of Tagged and matched IsolatedTauTagInfoCollection with  MC Jet
    dbe->setCurrentFolder("RecoTauV/TauTaggedJets_"+jetTagSrc_.label());

    nIsolatedJet_ptTauJet_ =       dbe->book1D("n_IsolatedTauTaggedJets_vs_ptTauJet","n_IsolatedTauTaggedJets_vs_ptTauJet", 75, 0., 150.);
    nIsolatedJet_etaTauJet_ =      dbe->book1D("n_IsolatedTauTaggedJets_vs_etaTauJet","n_IsolatedTauTaggedJets_vs_etaTauJet", 60, -3.0, 3.0 );
    nIsolatedJet_phiTauJet_ =      dbe->book1D("n_IsolatedTauTaggedJets_vs_phiTauJet","n_IsolatedTauTaggedJets_vs_phiTauJets", 36, -180., 180);
    nIsolatedJet_energyTauJet_ =   dbe->book1D("n_IsolatedTauTaggedJets_vs_energyTauJet", "n_IsolatedTauTaggedJets_vs_energyTauJet", 45, 0., 450.0);    
    nSignalTracksAfterIsolation_ = dbe->book1D("Signal_Tks_After_Isolation", "Signal_Tks_After_Isolation", 10, 0., 10.);
    nIsolatedTausLeadingTrackPt_ = dbe->book1D("LeadingTrackPt_After_Isolation", "LeadingTrackPt_After_Isolation",  75, 0., 150.);
    nIsolatedTausDeltaR_LTandJet_= dbe->book1D("DeltaR_LT_and_Jet_After_Isolation","DeltaR_LT_and_Jet_After_Isolation", 22, 0.,0.11);
    nAssociatedTracks_of_IsolatedTaus_ = dbe->book1D("Associated_Tks_After_Isolation","Associated_Tks_After_Isolation", 10, 0., 10.);
    nSelectedTracks_of_IsolatedTaus_ = dbe->book1D("Selected_Tks_After_Isolation","Selected_Tks_After_Isolation", 10, 0., 10.); 


  // What are the numbers of Tagged and matched EM IsolatedTauTagInfoCollection with  MC Jet
    dbe->setCurrentFolder("RecoTauV/TauEMTaggedJets_"+jetEMTagSrc_.label());

    nEMIsolatedJet_ptTauJet_ =       dbe->book1D("n_EMIsolatedTauTaggedJets_vs_ptTauJet","n_EMIsolatedTauTaggedJets_vs_ptTauJet", 75, 0., 150.);
    nEMIsolatedJet_etaTauJet_ =      dbe->book1D("n_EMIsolatedTauTaggedJets_vs_etaTauJet","n_EMIsolatedTauTaggedJets_vs_etaTauJet", 60, -3.0, 3.0 );
    nEMIsolatedJet_phiTauJet_ =      dbe->book1D("n_EMIsolatedTauTaggedJets_vs_phiTauJet","n_EMIsolatedTauTaggedJets_vs_phiTauJets", 36, -180., 180);
    nEMIsolatedJet_energyTauJet_ =   dbe->book1D("n_EMIsolatedTauTaggedJets_vs_energyTauJet", "n_EMIsolatedTauTaggedJets_vs_energyTauJet", 45, 0., 450.0); 


 // What is the behaviour of cone isolation size on tagging of MC Taus (CONE_MATCHING_CRITERIA) 
    dbe->setCurrentFolder("RecoTauV/TaggingStudies_"+ jetTagSrc_.label());
  
    nTausTotvsConeIsolation_ = dbe->book1D("nTaus_Tot_vs_coneIsolation", "nTaus_Tot_vs_coneIsolation", 6,0.175,0.475); // six bins centered at 0.2. 0.25. 0.3, 0.35, 0.4. 0.45
    nTausTaggedvsConeIsolation_ = dbe->book1D("nTaus_Tagged_vs_coneIsolation", "nTaus_Tagged_vs_coneIsolation", 6,0.175,0.475);

    nTausTotvsConeSignal_ = dbe->book1D("nTaus_Tot_vs_coneSignal","nTaus_Tot_vs_coneSignal", 6, 0.065, 0.125);
    nTausTaggedvsConeSignal_ = dbe->book1D("nTaus_Tagged_vs_coneSignal","nTaus_Tagged_vs_coneSignal", 6, 0.065, 0.125);  

    nTausTotvsPtLeadingTrack_ = dbe->book1D("nTaus_Tot_vs_PtLeadingTrack","nTaus_Tot_vs_PtLeadingTrack", 6, 1.5, 7.5);
    nTausTaggedvsPtLeadingTrack_ = dbe->book1D("nTaus_Tagged_vs_PtLeadingTrack","nTaus_Tagged_vs_PtLeadingTrack",6, 1.5, 7.5);

    nTausTotvsMatchingConeSize_ = dbe->book1D("nTaus_Tot_vs_MatchingConeSize","nTaus_Tot_vs_MatchingConeSize", 6, 0.065, 0.125);
    nTausTaggedvsMatchingConeSize_ = dbe->book1D("nTaus_Tagged_vs_MatchingConeSize","nTaus_Tagged_vs_MatchingConeSize", 6, 0.065, 0.125);
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

void TauTagVal::beginJob()
{ 


}

// -- method called to produce fill all the histograms --------------------
void TauTagVal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  numEvents_++;
  double  matching_criteria;
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

  if(dataType_ == "TAU"){
    TauJetsMC=getVectorOfVisibleTauJets(myGenEvent);
    matching_criteria=0.15;
  }

  if(dataType_ == "QCD")  {
    TauJetsMC=getVectorOfGenJets(genJets);
    matching_criteria=0.30;
  }
 
  //---------------------LET's See what this CaloTau has -----------------

  //Handle<CaloTauCollection> theCaloTauHandle;
  //iEvent.getByLabel("caloRecoTauProducer" ,theCaloTauHandle);
  
  //  cout<<"***"<<endl;
  //cout<<"Found "<<theCaloTauHandle->size()<<" had. tau-jet candidates"<<endl;

  //  myGenEvent->print();
  // CaloJets iterativeCone5CaloJets counting stuff-----------------------------------------------------------------------------------------------
  /*  Handle<CaloJetCollection> jetHandle;
  iEvent.getByLabel("iterativeCone5CaloJets", jetHandle);
  CaloJetCollection jetCollection = *(jetHandle.product());
 
  int CaloJetsIterativeCone5 =0;
  CaloJetCollection::const_iterator k = jetCollection.begin();
  
  for(; k!=jetCollection.end(); k++) {
    CaloJetsIterativeCone5++;
  }
  cout<<"-------------------------------------------------------------------------"<<endl;
  cout<<"CaloJetsIterativeCone5: "<<CaloJetsIterativeCone5<<endl;


  // PFJets iterativeCone5PFJets---------------------------------------------------------------------------------------------

  Handle<PFJetCollection> pfjetHandle;
  iEvent.getByLabel("iterativeCone5PFJets", pfjetHandle);
  PFJetCollection pfjetCollection = *(pfjetHandle.product());
  int PFJetsIterativeCone5 = 0;
  
  PFJetCollection::const_iterator l = pfjetCollection.begin();
  for (; l!=pfjetCollection.end(); l++){
    PFJetsIterativeCone5++;
  }
  cout<<"PFJetsIterativeCone5: "<< PFJetsIterativeCone5<<endl;*/
//---------------------------------------- IsolatedTauTagInfoCollection -------------------------------------------------
// ----------------------------------- Matching and filling up histograms -----------------------------------------------
  Handle<IsolatedTauTagInfoCollection> tauTagInfoHandle;
  iEvent.getByLabel(jetTagSrc_, tauTagInfoHandle);
  
  const IsolatedTauTagInfoCollection & tauTagInfo = *(tauTagInfoHandle.product());
  IsolatedTauTagInfoCollection::const_iterator i = tauTagInfo.begin();
  int numTauRecoJets = 0;
  int num_taujet_candidates=0;

  Handle<EMIsolatedTauTagInfoCollection> tauEMTagInfoHandle;
  iEvent.getByLabel(jetEMTagSrc_, tauEMTagInfoHandle);
  
  const EMIsolatedTauTagInfoCollection & tauEMTagInfo = *(tauEMTagInfoHandle.product());

  //bool trueTauJet=false;
  
  //  cout <<rMatch_<<" "<< rSig_ << " "<< rIso_ << " "<< ptLeadTk_ << " " << minPtIsoRing_ << " "<< nTracksInIsolationRing_ << endl;
 
  for (; i != tauTagInfo.end(); ++i) {  // Loop over all the IsolatedTauTagInfoCollection
    num_taujet_candidates++;
    TLorentzVector recoTauJet(i->jet()->px(), i->jet()->py(),i->jet()->pz(),i->jet()->energy());
    bool trueTauJet=false;
    std::vector<TLorentzVector>::iterator MCjet;
    for (MCjet = TauJetsMC.begin(); MCjet != TauJetsMC.end(); MCjet++){ 
      if ( MCjet->DeltaR(recoTauJet) < matching_criteria ) {
	trueTauJet=true;
	numTauRecoJets++;
        break;
      }
    }

    // If the IsolatedTauTagInfoCollection is matched to a MC Tau then continue
    if (trueTauJet) {

      nRecoJet_ptTauJet_->Fill(MCjet->Perp());  // Fill the histogram with the Pt, Eta, Energy of the Tau Jet at Generator level only for matched Jets
      nRecoJet_etaTauJet_->Fill(MCjet->Eta());  // How many Taus became RecoJets
      nRecoJet_phiTauJet_->Fill(MCjet->Phi()*180.0/TMath::Pi());
      nRecoJet_energyTauJet_->Fill(MCjet->E()); 

      nAssociatedTracks_->Fill(double(i->allTracks().size()));   // Fill histogram with lower kind of tracks quality kind inside of the 0.5 cone around the Jet Momentum
      nSelectedTracks_->Fill(double(i->selectedTracks().size()));// # Tracks of the following quality PixelHits >= 2, TotalHits >= 8, Chi^2/ndf < 100, Transverse IP < 0.03.
      
      if(!(i->leadingSignalTrack(rMatch_, ptLeadTk_))) {} else { // Fill in the histogram if a leading track is found how many Taus that became RecoJets have a leading track
	nRecoJet_LeadingTrack_ptTauJet_->Fill(MCjet->Perp());  // Fill the histogram with the Pt, Eta, Energy of the Tau Jet at Generator level only for matched Jets
	nRecoJet_LeadingTrack_etaTauJet_->Fill(MCjet->Eta());  // How many Tau became RecoJets
        nRecoJet_LeadingTrack_phiTauJet_->Fill(MCjet->Phi()*180.0/TMath::Pi());
	nRecoJet_LeadingTrack_energyTauJet_->Fill(MCjet->E()); 
      }	
      
      if ( i->discriminator(rMatch_, rSig_, rIso_, ptLeadTk_, minPtIsoRing_, nTracksInIsolationRing_)==1) {
	nIsolatedJet_ptTauJet_->Fill(MCjet->Perp());  // Fill the histogram with the Pt, Eta, Energy of the Tau Jet at Generator level only for matched Jets
	nIsolatedJet_etaTauJet_->Fill(MCjet->Eta()); 
        nIsolatedJet_phiTauJet_->Fill(MCjet->Phi()*180./TMath::Pi()); 
	nIsolatedJet_energyTauJet_->Fill(MCjet->E());
        nAssociatedTracks_of_IsolatedTaus_->Fill(double(i->allTracks().size()));
        nSelectedTracks_of_IsolatedTaus_->Fill(double(i->selectedTracks().size()));	
	//	nSignalTracksAfterIsolation->Fill(double (i->tracksInCone(momentum, rSig_,1.)).size()));           // Fill the histograms of the ones that get isolated

	//Filling histos for EMIsolated jets after Track Isolation
	double ecalIsolation = tauEMTagInfo [num_taujet_candidates].discriminator();
	if(ecalIsolation > 0.){
	  
	nEMIsolatedJet_ptTauJet_->Fill(MCjet->Perp());  // Fill the histogram with the Pt, Eta, Energy of the Tau Jet at Generator level only for matched Jets
	nEMIsolatedJet_etaTauJet_->Fill(MCjet->Eta()); 
        nEMIsolatedJet_phiTauJet_->Fill(MCjet->Phi()*180./TMath::Pi()); 
	nEMIsolatedJet_energyTauJet_->Fill(MCjet->E());

	}


      }
      
      // ------------------------------------------------------------------------------------------------------------------------------ 
      // Leading Track of 6 GeV just to find the number of tracks before and after Isolation ------------------------------------------

      const TrackRef leadTk6= (i->leadingSignalTrack(0.10, 6.));  // Get the leading Track for rMatch_ =0.1 and pt greater than 6.0 GeV
      if(!leadTk6)
	{
	   LogInfo("OutputInfo") << " No LeadingTrack" << endl;
	}else{
	math::XYZVector momentum = (*leadTk6).momentum();         // Get the momentum of the leading tack
	
	// Get the number of tracks greater than 1. GeV in cone of 0.07 around the leading track momentum
	nSignalTracks_->Fill(double (i->tracksInCone(momentum, rSig_,  1.).size()));                           // Fill the histogram with this number of tracks
	
	if(i->discriminator(rMatch_,rSig_, rIso_, ptLeadTk_, minPtIsoRing_, nTracksInIsolationRing_) == 1) {//  using rMatch_ = 0.1; rSig_ =0.07; ptLeadTk_ = 6.0, pT_min = 1.0
	  nSignalTracksAfterIsolation_->Fill(double (i->tracksInCone(momentum, rSig_,  1.).size()));           // Fill the histograms of the ones that get isolated
	  nIsolatedTausLeadingTrackPt_->Fill(double (momentum.rho()));
	  math::XYZVector jetMomentum(i->jet()->px(), i->jet()->py(), i->jet()->pz()); 
	  double deltaR = ROOT::Math::VectorUtil::DeltaR(jetMomentum, momentum);         // Calculate the deltaR of the two momenta
          nIsolatedTausDeltaR_LTandJet_->Fill(deltaR);
         
	}
      }
      
      // ------------------------------------------------------------------------------------------------------------------------------ 
      // Now lets study the behaviour of changing cone sizes and Pt of the Leading Track to begin with
      for(int ii=0;ii<6;ii++) {  // Six cone isolation size steps

	double ChangingIsoCone = ii*0.05 + 0.2;    // calculate the size of the Isolation Ring
	if (i->discriminator(rMatch_,rSig_, ChangingIsoCone, ptLeadTk_, minPtIsoRing_, nTracksInIsolationRing_) == 1)
	  nTausTaggedvsConeIsolation_->Fill(ChangingIsoCone);
	
	double ChangingSigCone = ii*0.01+0.07;
	if (i->discriminator(rMatch_,ChangingSigCone, rIso_, ptLeadTk_, minPtIsoRing_, nTracksInIsolationRing_) == 1)
	  nTausTaggedvsConeSignal_->Fill(ChangingSigCone);
	
	int ChangingPtLeadTk = int(ii*1.0 + 2.0);
	if (i->discriminator(rMatch_,rSig_, rIso_, ChangingPtLeadTk, minPtIsoRing_, nTracksInIsolationRing_) == 1)
	  nTausTaggedvsPtLeadingTrack_->Fill(double (ChangingPtLeadTk));

	double ChangingMatchingCone = ii*0.01 + 0.07;
 	if (i->discriminator(ChangingMatchingCone,rSig_, rIso_, ptLeadTk_, minPtIsoRing_, nTracksInIsolationRing_) == 1)
	  nTausTaggedvsMatchingConeSize_->Fill(ChangingMatchingCone);
	  
      }

      // ------------------------------------------------------------------------------------------------------------------------------ 
      // Leading track Finding much more relaxed, just to see how the Leading tracks are behaving!!!!----------------------------------

      const TrackRef leadTk1= (i->leadingSignalTrack(0.10, 1.));  // Get the leading Track for rMatch_ =0.10 and pt greater than 1.0 GeV
      
      if(!leadTk1)
	{
	  LogInfo("OutputInfo") << " No LeadingTrack 2" << endl;
	}else{
	ptLeadingTrack_->Fill((*leadTk1).pt());                    // Fill the ptLeadingTrack histogram with the pt of the leadTrack found
      }

      //To be fixed!!!
      const TrackRef leadTk2= (i->leadingSignalTrack(0.15,6.0));  //   Get the leading Track for rMatch_ =0.15 and pt greater than 6.0 GeV

      if(!leadTk2)
	{
	  LogInfo("OutputInfo") << " No LeadingTrack 2" << endl;
	}else{	
	math::XYZVector momentum = (*leadTk2).momentum();                           // Get the momentum from the leadingTrack
	math::XYZVector jetMomentum(i->jet()->px(), i->jet()->py(), i->jet()->pz());  // Get the momentum from the Jet
	double deltaR = ROOT::Math::VectorUtil::DeltaR(jetMomentum, momentum);         // Calculate the deltaR of the two momenta
	deltaRLeadTk_Jet_->Fill(deltaR);                                              // Fill the histogram with the value	
	}

    }// end of if (trueTauJet)
  }// end of the IsolatedTauTagInfoCollection loop
  
  //  std::cout<<" Number of Matched Reconstructed Tau Jets: "<<numTauRecoJets_<<endl;
  //std::cout<<"num_taujet_candidates: " <<num_taujet_candidates << endl; 
  delete myGenEvent;  
}

// ---------------------------------------------------------------------------  endJob -----------------------------------------------------------------------

void TauTagVal::endJob(){
  // just fill the denominator histograms for the changing cone sizes

  
  
  double MC_Taus = nMCTaus_etaTauJet_->getEntries();
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


  if (!outPutFile_.empty() && &*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (outPutFile_);

}

// Helper function  

// Get all the daughter particles of a particle
std::vector<TLorentzVector> TauTagVal::getVectorOfVisibleTauJets(HepMC::GenEvent *theEvent)
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
		   tauDecayMode == kThreeProng1pi0 ) {
		   //||		   tauDecayMode == kOther) { 
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
		  for (int jj =0; jj != 6; jj++){
		    double ChangingIsoCone = jj*0.05 + 0.2;
		    nTausTotvsConeIsolation_->Fill(ChangingIsoCone);
		    double ChangingSigCone = jj*0.01+0.07;
		    nTausTotvsConeSignal_->Fill(ChangingSigCone);
		    int ChangingPtLeadTk = int(jj*1.0 + 2.0);
		    nTausTotvsPtLeadingTrack_->Fill(double (ChangingPtLeadTk));
		    double ChangingMatchingCone = jj*0.01 + 0.07;
		    nTausTotvsMatchingConeSize_->Fill(ChangingMatchingCone);
		  }
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

std::vector<HepMC::GenParticle*> TauTagVal::getGenStableDecayProducts(const HepMC::GenParticle* particle)
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
std::vector<TLorentzVector> TauTagVal::getVectorOfGenJets(Handle< GenJetCollection >& genJets ) {
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
//Get a list of visible Tau Jets

