// -*- C++ -*-
//
// Package:    TauTagValidation
// Class:      TauTagValidation
// 
/**\class TauTagValidation TauTagValidation.cc

 Description: <one line class summary>
 
 Class used to do the Validation of the TauTag

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ricardo Vasquez Sierra
//         Created:  October 8, 2008
// $Id: TauTagValidation.cc,v 1.19 2011/01/21 14:45:43 mverzett Exp $
//
//
// user include files

#include "Validation/RecoTau/interface/TauTagValidation.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

using namespace edm;
using namespace std;
using namespace reco;

TauTagValidation::TauTagValidation(const edm::ParameterSet& iConfig)
{
  // What do we want to use as source Leptons or Jets (the only difference is the matching criteria)
  dataType_ = iConfig.getParameter<string>("DataType");

  // We need different matching criteria if we talk about leptons or jets
  matchDeltaR_Leptons_ = iConfig.getParameter<double>("MatchDeltaR_Leptons");
  matchDeltaR_Jets_    = iConfig.getParameter<double>("MatchDeltaR_Jets");

  // The output histograms can be stored or not
  saveoutputhistograms_ = iConfig.getParameter<bool>("SaveOutputHistograms");

  // Here it can be pretty much anything either a lepton or a jet 
  refCollectionInputTag_ = iConfig.getParameter<InputTag>("RefCollection");
  refCollection_ = refCollectionInputTag_.label();
  
  // The extension name has information about the Reference collection used
  extensionName_ = iConfig.getParameter<string>("ExtensionName");
  
  // Here is the reconstructed product of interest.
  TauProducerInputTag_ = iConfig.getParameter<InputTag>("TauProducer");
  TauProducer_ = TauProducerInputTag_.label();

  // The vector of Tau Discriminators to be monitored
  // TauProducerDiscriminators_ = iConfig.getUntrackedParameter<std::vector<string> >("TauProducerDiscriminators");
  
  // The cut on the Discriminators
  //  TauDiscriminatorCuts_ = iConfig.getUntrackedParameter<std::vector<double> > ("TauDiscriminatorCuts");

  // Get the discriminators and their cuts
  discriminators_ = iConfig.getParameter< std::vector<edm::ParameterSet> >( "discriminators" ); 

  //  cout << " RefCollection: " << refCollection_.label() << " "<< refCollection_ << endl;

  tversion = edm::getReleaseVersion();
  //    cout<<endl<<"-----------------------*******************************Version: " << tversion<<endl;
  
  if (!saveoutputhistograms_) {
    LogInfo("OutputInfo") << " TauVisible histograms will NOT be saved";
  } else {
    outPutFile_ = TauProducer_;
    outPutFile_.append("_");    
    tversion.erase(0,1);
    tversion.erase(tversion.size()-1,1);
    outPutFile_.append(tversion);
    outPutFile_.append("_"+ refCollection_);
    if ( ! extensionName_.empty()){
      outPutFile_.append("_"+ extensionName_);
    }
    outPutFile_.append(".root");
    
    //    cout<<endl<< outPutFile_<<endl;
    LogInfo("OutputInfo") << " TauVisiblehistograms will be saved to file:" << outPutFile_;
  }

  //---- book-keeping information ---
  numEvents_ = 0 ;

}

// ------------ method called once each job just before starting event loop  ------------
void TauTagValidation::beginJob()
{ 

  dbeTau = &*edm::Service<DQMStore>();

  if(dbeTau) {

    MonitorElement * ptTemp,* etaTemp,* phiTemp, *energyTemp;
    
    dbeTau->setCurrentFolder("RecoTauV/" + TauProducer_ + extensionName_ + "_ReferenceCollection" );

    // What kind of Taus do we originally have!
    
    ptTemp    =  dbeTau->book1D("nRef_Taus_vs_ptTauVisible", "nRef_Taus_vs_ptTauVisible", 75, 0., 150.);
    etaTemp   =  dbeTau->book1D("nRef_Taus_vs_etaTauVisible", "nRef_Taus_vs_etaTauVisible", 60, -3.0, 3.0 );
    phiTemp   =  dbeTau->book1D("nRef_Taus_vs_phiTauVisible", "nRef_Taus_vs_phiTauVisible", 36, -180., 180.);
    energyTemp =  dbeTau->book1D("nRef_Taus_vs_energyTauVisible", "nRef_Taus_vs_energyTauVisible", 45, 0., 450.0);

    ptTauVisibleMap.insert( std::make_pair( refCollection_,ptTemp));
    etaTauVisibleMap.insert( std::make_pair(refCollection_,etaTemp));
    phiTauVisibleMap.insert( std::make_pair(refCollection_,phiTemp));
    energyTauVisibleMap.insert( std::make_pair(refCollection_,energyTemp));

    // Number of Tau Candidates matched to MC Taus    

    dbeTau->setCurrentFolder("RecoTauV/"+ TauProducer_ + extensionName_ + "_Matched");

    ptTemp    =  dbeTau->book1D(TauProducer_ +"Matched_vs_ptTauVisible", TauProducer_ +"Matched_vs_ptTauVisible", 75, 0., 150.);
    etaTemp   =  dbeTau->book1D(TauProducer_ +"Matched_vs_etaTauVisible", TauProducer_ +"Matched_vs_etaTauVisible", 60, -3.0, 3.0 );
    phiTemp   =  dbeTau->book1D(TauProducer_ +"Matched_vs_phiTauVisible", TauProducer_ +"Matched_vs_phiTauVisible", 36, -180., 180.);
    energyTemp =  dbeTau->book1D(TauProducer_ +"Matched_vs_energyTauVisible", TauProducer_ +"Matched_vs_energyTauVisible", 45, 0., 450.0);

    ptTauVisibleMap.insert( std::make_pair( TauProducer_+"Matched" ,ptTemp));
    etaTauVisibleMap.insert( std::make_pair(TauProducer_+"Matched" ,etaTemp));
    phiTauVisibleMap.insert( std::make_pair(TauProducer_+"Matched" ,phiTemp));
    energyTauVisibleMap.insert( std::make_pair(TauProducer_+"Matched" ,energyTemp));  

    for ( std::vector< edm::ParameterSet >::iterator it = discriminators_.begin(); it!= discriminators_.end();  it++)
      {
	string DiscriminatorLabel = it->getParameter<string>("discriminator");

	dbeTau->setCurrentFolder("RecoTauV/" +  TauProducer_ + extensionName_ + "_" +  DiscriminatorLabel );
	
	ptTemp    =  dbeTau->book1D(DiscriminatorLabel + "_vs_ptTauVisible", DiscriminatorLabel +"_vs_ptTauVisible", 75, 0., 150.);
	etaTemp   =  dbeTau->book1D(DiscriminatorLabel + "_vs_etaTauVisible", DiscriminatorLabel + "_vs_etaTauVisible", 60, -3.0, 3.0 );
	phiTemp   =  dbeTau->book1D(DiscriminatorLabel + "_vs_phiTauVisible", DiscriminatorLabel + "_vs_phiTauVisible", 36, -180., 180.);
	energyTemp =  dbeTau->book1D(DiscriminatorLabel + "_vs_energyTauVisible", DiscriminatorLabel + "_vs_energyTauVisible", 45, 0., 450.0);
		
	ptTauVisibleMap.insert( std::make_pair(DiscriminatorLabel,ptTemp));
	etaTauVisibleMap.insert( std::make_pair(DiscriminatorLabel,etaTemp));
	phiTauVisibleMap.insert( std::make_pair(DiscriminatorLabel,phiTemp));
	energyTauVisibleMap.insert( std::make_pair(DiscriminatorLabel,energyTemp));

	//	if ( TauProducer_.find("PFTau") != string::npos) 
	// {
	
	if ( DiscriminatorLabel.find("LeadingTrackPtCut") != string::npos){
	  if ( TauProducer_.find("PFTau") != string::npos)
	    {
	      nPFJet_LeadingChargedHadron_ChargedHadronsSignal_	        =dbeTau->book1D(DiscriminatorLabel + "_ChargedHadronsSignal",DiscriminatorLabel + "_ChargedHadronsSignal", 21, -0.5, 20.5);		 
	      nPFJet_LeadingChargedHadron_ChargedHadronsIsolAnnulus_    =dbeTau->book1D(DiscriminatorLabel + "_ChargedHadronsIsolAnnulus",DiscriminatorLabel + "_ChargedHadronsIsolAnnulus", 21, -0.5, 20.5);	 
	      nPFJet_LeadingChargedHadron_GammasSignal_		        =dbeTau->book1D(DiscriminatorLabel + "_GammasSignal",DiscriminatorLabel + "_GammasSignal",21, -0.5, 20.5);				 
	      nPFJet_LeadingChargedHadron_GammasIsolAnnulus_ 	        =dbeTau->book1D(DiscriminatorLabel + "_GammasIsolAnnulus",DiscriminatorLabel + "_GammasIsolAnnulus",21, -0.5, 20.5);  		 
	      nPFJet_LeadingChargedHadron_NeutralHadronsSignal_	        =dbeTau->book1D(DiscriminatorLabel + "_NeutralHadronsSignal",DiscriminatorLabel + "_NeutralHadronsSignal",21, -0.5, 20.5);		 
	      nPFJet_LeadingChargedHadron_NeutralHadronsIsolAnnulus_	=dbeTau->book1D(DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",21, -0.5, 20.5);   	      
	    }
	  else if (TauProducer_.find("caloRecoTau") != string::npos)
	    {
	      nCaloJet_LeadingTrack_signalTracksInvariantMass_   =dbeTau->book1D(DiscriminatorLabel + "_signalTracksInvariantMass",DiscriminatorLabel + "_signalTracksInvariantMass",  75, 0., 150.); 
	      nCaloJet_LeadingTrack_signalTracks_		 =dbeTau->book1D(DiscriminatorLabel + "_signalTracks", DiscriminatorLabel + "_signalTracks" , 15, -0.5, 14.5);         	     
	      nCaloJet_LeadingTrack_isolationTracks_	    	 =dbeTau->book1D(DiscriminatorLabel + "_isolationTracks", DiscriminatorLabel + "_isolationTracks",  15, -0.5, 14.5);      		 
	      nCaloJet_LeadingTrack_isolationECALhitsEtSum_      =dbeTau->book1D(DiscriminatorLabel + "_isolationECALhitsEtSum", DiscriminatorLabel + "_isolationECALhitsEtSum", 75, 0., 75.);       
	    }
	}
	
	if ( DiscriminatorLabel.find("ByIsolationLater") != string::npos ){
	  if ( TauProducer_.find("PFTau") != string::npos)
	    {
	      nIsolated_NoChargedHadrons_ChargedHadronsSignal_	      =dbeTau->book1D(DiscriminatorLabel + "_ChargedHadronsSignal",DiscriminatorLabel + "_ChargedHadronsSignal", 21, -0.5, 20.5);	 	      
	      nIsolated_NoChargedHadrons_GammasSignal_		      =dbeTau->book1D(DiscriminatorLabel + "_GammasSignal",DiscriminatorLabel + "_GammasSignal",21, -0.5, 20.5);			   
	      nIsolated_NoChargedHadrons_GammasIsolAnnulus_           =dbeTau->book1D(DiscriminatorLabel + "_GammasIsolAnnulus",DiscriminatorLabel + "_GammasIsolAnnulus",21, -0.5, 20.5);  		   
	      nIsolated_NoChargedHadrons_NeutralHadronsSignal_	      =dbeTau->book1D(DiscriminatorLabel + "_NeutralHadronsSignal",DiscriminatorLabel + "_NeutralHadronsSignal",21, -0.5, 20.5);	   
	      nIsolated_NoChargedHadrons_NeutralHadronsIsolAnnulus_   =dbeTau->book1D(DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",21, -0.5, 20.5); 
	    }
	  else if (TauProducer_.find("caloRecoTau") != string::npos)
	    {
	      nTrackIsolated_isolationECALhitsEtSum_      =dbeTau->book1D(DiscriminatorLabel + "_isolationECALhitsEtSum", DiscriminatorLabel + "_isolationECALhitsEtSum", 75, 0., 75.);	  
	      nTrackIsolated_signalTracksInvariantMass_	  =dbeTau->book1D(DiscriminatorLabel + "_signalTracksInvariantMass",DiscriminatorLabel + "_signalTracksInvariantMass", 75, 0., 150.);
	      nTrackIsolated_signalTracks_		  =dbeTau->book1D(DiscriminatorLabel + "_signalTracks",DiscriminatorLabel + "_signalTracks", 15, -0.5, 14.5);                        
	    }
	}

	if ( DiscriminatorLabel.find("ByIsolation") != string::npos ){
	  if ( TauProducer_.find("PFTau") != string::npos)
	    {
	      nIsolated_NoChargedNoGammas_ChargedHadronsSignal_        =dbeTau->book1D(DiscriminatorLabel + "_ChargedHadronsSignal",DiscriminatorLabel + "_ChargedHadronsSignal", 21, -0.5, 20.5);	  
	      nIsolated_NoChargedNoGammas_GammasSignal_                =dbeTau->book1D(DiscriminatorLabel + "_GammasSignal",DiscriminatorLabel + "_GammasSignal",21, -0.5, 20.5);	 
	      nIsolated_NoChargedNoGammas_NeutralHadronsSignal_	       =dbeTau->book1D(DiscriminatorLabel + "_NeutralHadronsSignal",DiscriminatorLabel + "_NeutralHadronsSignal",21, -0.5, 20.5);	   	 
	      nIsolated_NoChargedNoGammas_NeutralHadronsIsolAnnulus_   =dbeTau->book1D(DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",21, -0.5, 20.5); 

	    }
	  else if (TauProducer_.find("caloRecoTau") != string::npos)
	    {
	      nEMIsolated_signalTracksInvariantMass_  =dbeTau->book1D(DiscriminatorLabel+"_signalTracksInvariantMass",DiscriminatorLabel+"_signalTracksInvariantMass", 75, 0., 150.);
	      nEMIsolated_signalTracks_               =dbeTau->book1D(DiscriminatorLabel+"_signalTracks",DiscriminatorLabel+"_signalTracks", 15, -0.5, 14.5);    
	    }
	}

      }
  }

  for ( std::vector< edm::ParameterSet >::iterator it = discriminators_.begin(); it!= discriminators_.end();  it++) 
    {
      cerr<< " "<< it->getParameter<string>("discriminator") << " "<< it->getParameter<double>("selectionCut") << endl;
      
    }
}

// -- method called to produce fill all the histograms --------------------
void TauTagValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  numEvents_++;
  double matching_criteria = -1.0;

  typedef edm::View<reco::Candidate> genCandidateCollection;
  //  typedef edm::Vector<reco::PFTau> pfCandidateCollection;
  //  typedef edm::Vector<reco::CaloTau> caloCandidateCollection;

  //  std::cout << "--------------------------------------------------------------"<<endl;
  //std::cout << " RunNumber: " << iEvent.id().run() << ", EventNumber: " << iEvent.id().event() << std:: endl;
  //std::cout << "Event number: " << ++numEvents_ << endl;
  //std::cout << "--------------------------------------------------------------"<<endl;

  // ----------------------- Reference product -----------------------------------------------------------------------

  Handle<genCandidateCollection> ReferenceCollection;
  bool isGen = iEvent.getByLabel(refCollectionInputTag_, ReferenceCollection);    // get the product from the event

  if (!isGen) {
    std::cerr << " Reference collection: " << refCollection_ << " not found while running TauTagValidation.cc " << std::endl;
    return;
  }

  if(dataType_ == "Leptons"){
    matching_criteria = matchDeltaR_Leptons_;
  }
  else
  {
    matching_criteria = matchDeltaR_Jets_;
  }

  // ------------------------------ PFTauCollection Matched and other discriminators ---------------------------------------------------------

  if ( TauProducer_.find("PFTau") != string::npos)
    {
      Handle<PFTauCollection> thePFTauHandle;
      iEvent.getByLabel(TauProducerInputTag_,thePFTauHandle);
      
      const PFTauCollection  *pfTauProduct;
      pfTauProduct = thePFTauHandle.product();

      PFTauCollection::size_type thePFTauClosest;      

      for (genCandidateCollection::const_iterator RefJet= ReferenceCollection->begin() ; RefJet != ReferenceCollection->end(); RefJet++ ){ 

	
	ptTauVisibleMap.find(refCollection_)->second->Fill(RefJet->pt());
	etaTauVisibleMap.find(refCollection_)->second->Fill(RefJet->eta());
	phiTauVisibleMap.find(refCollection_)->second->Fill(RefJet->phi()*180.0/TMath::Pi());
	energyTauVisibleMap.find(refCollection_)->second->Fill(RefJet->energy());
	
	const reco::Candidate *gen_particle = &(*RefJet);
	
	double delta=TMath::Pi();

	thePFTauClosest = pfTauProduct->size();

	for (PFTauCollection::size_type iPFTau=0 ; iPFTau <  pfTauProduct->size() ; iPFTau++) 
	  {		    
	    if (algo_->deltaR(gen_particle, & pfTauProduct->at(iPFTau)) < delta){
	      delta = algo_->deltaR(gen_particle, & pfTauProduct->at(iPFTau));
	      thePFTauClosest = iPFTau;
	    }
	  }
	
	// Skip if there is no reconstructed Tau matching the Reference 
	if (thePFTauClosest == pfTauProduct->size()) continue;
	
	double deltaR = algo_->deltaR(gen_particle, & pfTauProduct->at(thePFTauClosest));

	// Skip if the delta R difference is larger than the required criteria
	if (deltaR > matching_criteria && matching_criteria != -1.0) continue;
	
	ptTauVisibleMap.find( TauProducer_+"Matched")->second->Fill(RefJet->pt());
	etaTauVisibleMap.find( TauProducer_+"Matched" )->second->Fill(RefJet->eta());
	phiTauVisibleMap.find( TauProducer_+"Matched" )->second->Fill(RefJet->phi()*180.0/TMath::Pi());
	energyTauVisibleMap.find(  TauProducer_+"Matched")->second->Fill(RefJet->energy());
	
	PFTauRef thePFTau(thePFTauHandle, thePFTauClosest);
	Handle<PFTauDiscriminator> currentDiscriminator;
	
	for ( std::vector< edm::ParameterSet >::iterator it = discriminators_.begin(); it!= discriminators_.end();  it++) 
	  {
	    string currentDiscriminatorLabel = it->getParameter<string>("discriminator");	      
	    iEvent.getByLabel(currentDiscriminatorLabel, currentDiscriminator);
	    
	    if ((*currentDiscriminator)[thePFTau] >= it->getParameter<double>("selectionCut")){
	      ptTauVisibleMap.find(  currentDiscriminatorLabel )->second->Fill(RefJet->pt());
	      etaTauVisibleMap.find(  currentDiscriminatorLabel )->second->Fill(RefJet->eta());
	      phiTauVisibleMap.find(  currentDiscriminatorLabel )->second->Fill(RefJet->phi()*180.0/TMath::Pi());
	      energyTauVisibleMap.find(  currentDiscriminatorLabel )->second->Fill(RefJet->energy());
	      
	      if ( currentDiscriminatorLabel.find("LeadingTrackPtCut") != string::npos){
		nPFJet_LeadingChargedHadron_ChargedHadronsSignal_->Fill((*thePFTau).signalPFChargedHadrCands().size());
		nPFJet_LeadingChargedHadron_ChargedHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFChargedHadrCands().size());
		nPFJet_LeadingChargedHadron_GammasSignal_->Fill((*thePFTau).signalPFGammaCands().size());		 
		nPFJet_LeadingChargedHadron_GammasIsolAnnulus_->Fill((*thePFTau).isolationPFGammaCands().size()); 
		nPFJet_LeadingChargedHadron_NeutralHadronsSignal_->Fill((*thePFTau).signalPFNeutrHadrCands().size());	 
		nPFJet_LeadingChargedHadron_NeutralHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFNeutrHadrCands().size());
	      }	      
	      else if ( currentDiscriminatorLabel.find("ByIsolation") != string::npos ){
		nIsolated_NoChargedNoGammas_ChargedHadronsSignal_->Fill((*thePFTau).signalPFChargedHadrCands().size());	 
		nIsolated_NoChargedNoGammas_GammasSignal_->Fill((*thePFTau).signalPFGammaCands().size());		 
		nIsolated_NoChargedNoGammas_NeutralHadronsSignal_->Fill((*thePFTau).signalPFNeutrHadrCands().size());	 
		nIsolated_NoChargedNoGammas_NeutralHadronsIsolAnnulus_->Fill((*thePFTau).isolationPFNeutrHadrCands().size());		  
	      }
	    }
	    else {
	      break; 
	    }
	  }
      }
    }
  // ------------------------------ CaloTauCollection Matched and other discriminators ---------------------------------------------------------
  else if (TauProducer_.find("caloRecoTau") != string::npos)
    {
      
      Handle<CaloTauCollection> theCaloTauHandle;
      iEvent.getByLabel(TauProducer_,theCaloTauHandle);
      
      const CaloTauCollection *caloTauProduct;
      caloTauProduct = theCaloTauHandle.product();
      
      for (genCandidateCollection::const_iterator RefJet= ReferenceCollection->begin() ; RefJet != ReferenceCollection->end(); RefJet++ ){ 
	
	ptTauVisibleMap.find(refCollection_)->second->Fill(RefJet->pt());
	etaTauVisibleMap.find(refCollection_)->second->Fill(RefJet->eta());
	phiTauVisibleMap.find(refCollection_)->second->Fill(RefJet->phi()*180.0/TMath::Pi());
	energyTauVisibleMap.find(refCollection_)->second->Fill(RefJet->energy());

	const reco::Candidate *gen_particle = &(*RefJet);

	bool trueCaloTau = false;
	double delta=TMath::Pi();
	CaloTauCollection::size_type  theCaloTauClosest=caloTauProduct->size();
	
	for (CaloTauCollection::size_type iCaloTau=0 ; iCaloTau <  caloTauProduct->size() ; iCaloTau++) 
	  {	
	    
	    if (algo_->deltaR(gen_particle, & caloTauProduct->at(iCaloTau)) < delta){
	      delta = algo_->deltaR(gen_particle, & caloTauProduct->at(iCaloTau));
	      theCaloTauClosest = iCaloTau;
	    }
	    if ( delta <  matching_criteria ) {
	      trueCaloTau=true;
	    }
	  }

	// Skip if there is no reconstructed Tau matching the Reference 
	if (theCaloTauClosest == caloTauProduct->size()) continue;
	
	double deltaR = algo_->deltaR(gen_particle, & caloTauProduct->at(theCaloTauClosest));
	
	// Skip if the delta R difference is larger than the required criteria
	if (deltaR > matching_criteria && matching_criteria != -1.0) continue;
	
	ptTauVisibleMap.find( TauProducer_+"Matched")->second->Fill(RefJet->pt());
	etaTauVisibleMap.find( TauProducer_+"Matched" )->second->Fill(RefJet->eta());
	phiTauVisibleMap.find( TauProducer_+"Matched" )->second->Fill(RefJet->phi()*180.0/TMath::Pi());
	energyTauVisibleMap.find(  TauProducer_+"Matched")->second->Fill(RefJet->energy());
	
	CaloTauRef theCaloTau(theCaloTauHandle,theCaloTauClosest);
	Handle<CaloTauDiscriminator> currentDiscriminator;
	
	for (  std::vector< edm::ParameterSet >::iterator it = discriminators_.begin(); it!= discriminators_.end();  it++)
	  {
	    
	    string currentDiscriminatorLabel = it->getParameter<string>("discriminator");
	    iEvent.getByLabel(currentDiscriminatorLabel, currentDiscriminator);	      
	    
	    if((*currentDiscriminator)[theCaloTau] >= it->getParameter<double>("selectionCut") )
	      {
		ptTauVisibleMap.find( currentDiscriminatorLabel)->second->Fill(RefJet->pt());
		etaTauVisibleMap.find( currentDiscriminatorLabel )->second->Fill(RefJet->eta());
		phiTauVisibleMap.find( currentDiscriminatorLabel )->second->Fill(RefJet->phi()*180.0/TMath::Pi());
		energyTauVisibleMap.find( currentDiscriminatorLabel )->second->Fill(RefJet->energy());
		
		if ( currentDiscriminatorLabel.find("LeadingTrackPtCut") != string::npos){
		  nCaloJet_LeadingTrack_signalTracksInvariantMass_->Fill((*theCaloTau).signalTracksInvariantMass());
		  nCaloJet_LeadingTrack_signalTracks_->Fill((*theCaloTau).signalTracks().size()); 
		  nCaloJet_LeadingTrack_isolationTracks_->Fill((*theCaloTau).isolationTracks().size());
		  nCaloJet_LeadingTrack_isolationECALhitsEtSum_->Fill((*theCaloTau).isolationECALhitsEtSum());
		}
		
		else if ( currentDiscriminatorLabel.find("ByIsolation") != string::npos ){
		  nEMIsolated_signalTracksInvariantMass_->Fill((*theCaloTau).signalTracksInvariantMass());
		  nEMIsolated_signalTracks_->Fill((*theCaloTau).signalTracks().size());     
		}
	      }
	    else {
	      break; 
	    }
	  }	
      }
    }
  //------------------------------- hpsTanc (why do I need this? I don't know, the program is made this way) -----------------------------------
  else if ( TauProducer_.find("hpsTancTaus") != string::npos)
    {
      //cout<<"entering the hpsTancTaus section\n"<<endl;
      Handle<PFTauCollection> thePFTauHandle;
      iEvent.getByLabel(TauProducerInputTag_,thePFTauHandle);
      
      const PFTauCollection  *pfTauProduct;
      pfTauProduct = thePFTauHandle.product();

      PFTauCollection::size_type thePFTauClosest;      

      for (genCandidateCollection::const_iterator RefJet= ReferenceCollection->begin() ; RefJet != ReferenceCollection->end(); RefJet++ ){ 

	
	ptTauVisibleMap.find(refCollection_)->second->Fill(RefJet->pt());
	etaTauVisibleMap.find(refCollection_)->second->Fill(RefJet->eta());
	phiTauVisibleMap.find(refCollection_)->second->Fill(RefJet->phi()*180.0/TMath::Pi());
	energyTauVisibleMap.find(refCollection_)->second->Fill(RefJet->energy());
	
	const reco::Candidate *gen_particle = &(*RefJet);
	
	double delta=TMath::Pi();

	thePFTauClosest = pfTauProduct->size();

	for (PFTauCollection::size_type iPFTau=0 ; iPFTau <  pfTauProduct->size() ; iPFTau++) 
	  {		    
	    if (algo_->deltaR(gen_particle, & pfTauProduct->at(iPFTau)) < delta){
	      delta = algo_->deltaR(gen_particle, & pfTauProduct->at(iPFTau));
	      thePFTauClosest = iPFTau;
	    }
	  }
	
	// Skip if there is no reconstructed Tau matching the Reference 
	if (thePFTauClosest == pfTauProduct->size()) continue;
	
	double deltaR = algo_->deltaR(gen_particle, & pfTauProduct->at(thePFTauClosest));

	// Skip if the delta R difference is larger than the required criteria
	if (deltaR > matching_criteria && matching_criteria != -1.0) continue;
	
	ptTauVisibleMap.find( TauProducer_+"Matched")->second->Fill(RefJet->pt());
	etaTauVisibleMap.find( TauProducer_+"Matched" )->second->Fill(RefJet->eta());
	phiTauVisibleMap.find( TauProducer_+"Matched" )->second->Fill(RefJet->phi()*180.0/TMath::Pi());
	energyTauVisibleMap.find(  TauProducer_+"Matched")->second->Fill(RefJet->energy());
	
	PFTauRef thePFTau(thePFTauHandle, thePFTauClosest);
	Handle<PFTauDiscriminator> currentDiscriminator;
	
	for ( std::vector< edm::ParameterSet >::iterator it = discriminators_.begin(); it!= discriminators_.end();  it++) 
	  {
	    string currentDiscriminatorLabel = it->getParameter<string>("discriminator");	      
	    iEvent.getByLabel(currentDiscriminatorLabel, currentDiscriminator);
	    
	    if ((*currentDiscriminator)[thePFTau] >= it->getParameter<double>("selectionCut")){
	      ptTauVisibleMap.find(  currentDiscriminatorLabel )->second->Fill(RefJet->pt());
	      etaTauVisibleMap.find(  currentDiscriminatorLabel )->second->Fill(RefJet->eta());
	      phiTauVisibleMap.find(  currentDiscriminatorLabel )->second->Fill(RefJet->phi()*180.0/TMath::Pi());
	      energyTauVisibleMap.find(  currentDiscriminatorLabel )->second->Fill(RefJet->energy());
	      
	    }
	    else {
	      break; 
	    }
	  }
      }
    }    
}


 

// ---------------------------------------------------------------------------  endJob -----------------------------------------------------------------------

void TauTagValidation::endJob(){

  if(saveoutputhistograms_) //USED for debugging. I keep it here in case of need ;)
    {
      cout << "dumping entries for hpsTanc"<<endl;
      for(std::map<std::string,MonitorElement*>::iterator mapEntry = ptTauVisibleMap.begin(); mapEntry != ptTauVisibleMap.end(); mapEntry++)
	if( mapEntry->first.find("hpsTancTaus") !=string::npos)    
	  cout << mapEntry->first << "      entries:   " <<  mapEntry->second->getTH1()->GetEntries() << endl;
    }

  // just fill the denominator histograms for the changing cone sizes
  /*  
  double Denominator_Taus = nRefTaus_etaTauVisible_->getEntries();
  double First_Taus = nPFJet_etaTauVisible_->getEntries();
  double Second_Taus = nPFJet_LeadingChargedHadron_etaTauVisible_->getEntries();
  double Third_Taus = nIsolated_NoChargedHadrons_etaTauVisible_->getEntries();
  double Forth_Taus = nIsolated_NoChargedNoGammas_etaTauVisible_->getEntries();
   
  std::streamsize prec = cout.precision();
 
  cout<<setfill('-')<<setw(110)<<"-"<<endl;
  
  cout<<setfill('-')<<setw(55)<<" TAU TAG VALIDATION SUMMARY "<<setw(55)<<"-"<<endl;
  
  cout<<setfill('-')<<setw(92)<<"-"<<setfill(' ')<<setw(9)<<"Rel.Eff."<<setw(9)<<"Tot.Eff."<<endl;
  cout<<setfill('-')<<setw(85)<<left<<" TOTAL NUMBER OF REF OBJECTS LEVEL: ";
  
  cout<<setfill(' ')<<setw(7) <<right<< Denominator_Taus <<setw(9)<<"--"<<setw(9)<<"--"<< endl;
  
  cout<<setfill('-')<<setw(85)<<left<<" Step 1. TOTAL NUMBER OF PFJETS MATCHED TO REF COLLECTION: ";
  cout<<setfill(' ')<<setw(7) <<right<< First_Taus <<setw(9)<<"--";
  if (Denominator_Taus > 0) 
    cout<<setw(9)<<setprecision(3)<< First_Taus/Denominator_Taus  << setprecision(prec)<<endl;
  else 
    cout<<setw(9)<<"--"<<endl;
  
  cout<<setfill('-')<<setw(85)<<left<<" Step 2. PLUS LEADING CHARGED HADRON= 5.0 GeV: ";
  cout<<setfill(' ')<<setw(7)<<right<<Second_Taus;
  if (First_Taus > 0) 
    cout<<setw(9)<<setprecision(3)<< Second_Taus/First_Taus <<setprecision(prec);
  else
    cout<<setw(9)<<"--"<<endl;
  
  if (Denominator_Taus > 0)
    cout<<setw(9)<<setprecision(3)<< Second_Taus/Denominator_Taus  << setprecision(prec)<<endl;
  else 
    cout<<setw(9)<<"--"<<endl;

  
 
  cout<<setfill('-')<<setw(85)<<left<<" Step 3. PLUS CHARGED HADRON ISOLATION: ";
  cout<<setfill(' ')<<setw(7) <<right<<Third_Taus;
  if (Second_Taus > 0) 
    cout<<setw(9)<<setprecision(3)<< Third_Taus/Second_Taus <<setprecision(prec);
  else
    cout<<setw(9)<<"--"<<endl;
  
  if (Denominator_Taus > 0)
    cout<<setw(9)<<setprecision(3)<< Third_Taus/Denominator_Taus  << setprecision(prec)<<endl;
  else 
    cout<<setw(9)<<"--"<<endl;

  //  cout<<setfill('-')<<setw(110)<<"-"<<endl;

  cout<<setfill('-')<<setw(85)<<left<<" Step 4. PLUS GAMMA ISOLATION: ";
  cout<<setfill(' ')<<setw(7) <<right<<Forth_Taus;
  if (Third_Taus > 0) 
    cout<<setw(9)<<setprecision(3)<< Forth_Taus/Third_Taus <<setprecision(prec);
  else
    cout<<setw(9)<<"--"<<endl;
  
  if (Denominator_Taus > 0)
    cout<<setw(9)<<setprecision(3)<< Forth_Taus/Denominator_Taus  << setprecision(prec)<<endl;
  else 
    cout<<setw(9)<<"--"<<endl;

    cout<<setfill('-')<<setw(110)<<"-"<<endl; 
  */

  if (!outPutFile_.empty() && &*edm::Service<DQMStore>() && saveoutputhistograms_) dbeTau->save (outPutFile_);
  
}


