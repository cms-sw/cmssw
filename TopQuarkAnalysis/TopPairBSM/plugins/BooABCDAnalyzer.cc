// -*- C++ -*-
//
// Package:    BooABCDAnalyzer
// Class:      BooABCDAnalyzer
// 
/**\class BostedTopPair/BooABCDAnalyzer.cc

 Description:

 Implementation:
     <Notes on implementation>

	 Author: Francisco Yumiceva
*/
//
// $Id: BooABCDAnalyzer.cc,v 1.1.2.10.2.1 2009/07/29 22:05:35 jengbou Exp $
//
//


// system include files
#include <memory>

#include "DataFormats/VertexReco/interface/Vertex.h"

// user include files
#include "TopQuarkAnalysis/TopPairBSM/interface/BooABCDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"
//#include "PhysicsTools/Utilities/interface/deltaR.h"
//#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "TopQuarkAnalysis/TopPairBSM/interface/METzCalculator.h"
#include "TopQuarkAnalysis/TopPairBSM/interface/BooTools.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "TLorentzVector.h"
#include "TVector3.h"

#include "Math/VectorUtil.h"
//
// constructors and destructor
//
BooABCDAnalyzer::BooABCDAnalyzer(const edm::ParameterSet& iConfig)
{
	
  debug             = iConfig.getParameter<bool>   ("debug");
  fwriteAscii       = iConfig.getParameter<bool>   ("writeAscii");
  fasciiFileName    = iConfig.getParameter<std::string> ("asciiFilename");
  //fApplyWeights     = iConfig.getUntrackedParameter<bool>   ("applyWeights");
  rootFileName      = iConfig.getParameter<std::string> ("rootFilename");

  // Collections
  genEvnSrc         = iConfig.getParameter<edm::InputTag> ("genEventSource");
  muonSrc           = iConfig.getParameter<edm::InputTag> ("muonSource");
  electronSrc       = iConfig.getParameter<edm::InputTag> ("electronSource");
  metSrc            = iConfig.getParameter<edm::InputTag> ("metSource");
  jetSrc            = iConfig.getParameter<edm::InputTag> ("jetSource");
  //jetSrc1           = iConfig.getParameter<edm::InputTag> ("jetSource1");
  //jetSrc2           = iConfig.getParameter<edm::InputTag> ("jetSource2");

  fIsMCTop          = iConfig.getParameter<bool>  ("IsMCTop");

  fMinMuonPt        = iConfig.getParameter<edm::ParameterSet>("muonCuts").getParameter<double>("MinPt");
  fMaxMuonEta       = iConfig.getParameter<edm::ParameterSet>("muonCuts").getParameter<double>("MaxEta");
  
  fMuonRelIso       = iConfig.getParameter<edm::ParameterSet>("muonIsolation").getParameter<double>("RelIso");
  fMaxMuonEm        = iConfig.getParameter<edm::ParameterSet>("muonIsolation").getParameter<double>("MaxVetoEm");
  fMaxMuonHad       = iConfig.getParameter<edm::ParameterSet>("muonIsolation").getParameter<double>("MaxVetoHad");

  fMinElectronPt    = iConfig.getParameter<edm::ParameterSet>("electronCuts").getParameter<double>("MinPt");
  fMaxElectronEta   = iConfig.getParameter<edm::ParameterSet>("electronCuts").getParameter<double>("MaxEta");
  fElectronRelIso      = iConfig.getParameter<edm::ParameterSet>("electronCuts").getParameter<double>("RelIso");
  
  fMinJetPt         = iConfig.getParameter<edm::ParameterSet>("jetCuts").getParameter<double>("MinJetPt");
  fMaxJetEta        = iConfig.getParameter<edm::ParameterSet>("jetCuts").getParameter<double>("MaxJetEta");

  fUsebTagging      = iConfig.getParameter<bool>  ("UsebTagging");
  
  //fMinHt           = iConfig.getParameter<edm::ParameterSet>("jetCuts").getParameter<double>("MinHt");
  //fMinMET           = iConfig.getParameter<edm::ParameterSet>("METCuts").getParameter<double>("MinMET");
    
  feventToProcess   = iConfig.getParameter<int> ("processOnlyEvent");
  fdisplayJets      = iConfig.getParameter<bool>   ("makeJetLegoPlots");


  // write ascii output
  if (fwriteAscii) {
	  edm::LogWarning ( "BooABCDAnalyzer" ) << " Results will also be saved into an ascii file: " << fasciiFileName;
	  fasciiFile.open(fasciiFileName.c_str());
  }

  // Create a root file
  theFile = new TFile(rootFileName.c_str(), "RECREATE");

  // create tree
  ftree = new TTree("top","top");
  ftree->AutoSave();

  fntuple = new BooEventNtuple();
  ftree->Branch("top.","BooEventNtuple",&fntuple,64000,1); 
  
  // make diretories
  theFile->mkdir("Generator");
  theFile->cd();
  theFile->mkdir("Muons");
  theFile->cd();
  theFile->mkdir("Electrons");
  theFile->cd();
  theFile->mkdir("MET");
  theFile->cd();
  theFile->mkdir("Jets");
  theFile->cd();
  theFile->mkdir("Mass");
  if (fdisplayJets) {
	  theFile->cd();
	  theFile->mkdir("DisplayJets");
  }
  theFile->cd();
  
  

  // initialize histogram manager
  hcounter= new BooHistograms();
  hmuons_ = new BooHistograms();
  helectrons_= new BooHistograms();
  hmet_   = new BooHistograms();
  hjets_  = new BooHistograms();
  hgen_   = new BooHistograms();
  hmass_  = new BooHistograms();
  
  if (fdisplayJets) hdisp_  = new BooHistograms();

 
  hcounter->Init("counter");

  hmuons_->Init("Muons","nohlt");
  hmuons_->Init("Muons","cut0");
  hmuons_->Init("Muons","cut1");
  hmuons_->Init("Muons","cut2");
  hmuons_->Init("Muons","cut3");
  hmuons_->Init("Muons","cut4");

  helectrons_->Init("Electrons","cut0");
  helectrons_->Init("Electrons","cut1");
  helectrons_->Init("Electrons","cut2");
  helectrons_->Init("Electrons","cut3");
  
  hgen_->Init("generator");

  hmass_->Init("Mass","cut0");
  hmass_->Init("Mass","cut1");
  hmass_->Init("Mass","cut2");
  hmass_->Init("Mass","cut3");
  hmass_->Init("Mass","cut4");
  hmass_->Init("Mass","cut5");
  hmass_->Init("Mass","cut6");

  hjets_->Init("Jets","cut0");
  hjets_->Init("Jets","cut1");
  hjets_->Init("Jets","cut2");
  hjets_->Init("Jets","cut3");
  hjets_->Init("Jets","cut4");
  hjets_->Init("Jets","cut5");
  hjets_->Init("Jets","cut6");

  hmet_->Init("MET","cut0");
  hmet_->Init("MET","cut1");

  if (fdisplayJets) hdisp_->Init("DisplayJets","cut0");
  
  nevents = 0;
  nbadmuons = 0;
  nWcomplex = 0;
  MCAllmatch_sumEt_ = 0;
  MCAllmatch_chi2_ = 0;
  
}


BooABCDAnalyzer::~BooABCDAnalyzer()
{

	if (debug) std::cout << "BooABCDAnalyzer Destructor called" << std::endl;

	// print out useful informantion
	std::cout << "BooABCDAnalyzer Total events analyzed = " << nevents << std::endl;
	std::cout << "BooABCDAnalyzer Number of bad muon events = " << nbadmuons << std::endl;
	std::cout << "BooABCDAnalyzer Number of complex solutions = " << nWcomplex <<std::endl;
	std::cout << "BooABCDAnalyzer Number of solutions with unambigous jet-parton matching, for sumEt case = " << MCAllmatch_sumEt_ << std::endl;
	std::cout << "BooABCDAnalyzer Number of solutions with unambigous jet-parton matching, for chi2 case  = " << MCAllmatch_chi2_ << std::endl;
	
	   
	if (fwriteAscii) fasciiFile.close();

	// save all histograms
	theFile->cd();

	ftree->Write();

	hcounter->Save();
	
	theFile->cd();
	theFile->cd("Muons");
	hmuons_->Save();

	theFile->cd();
	theFile->cd("Electrons");
	helectrons_->Save();
   
	theFile->cd();
	theFile->cd("MET");
	hmet_->Save();

	theFile->cd();
	theFile->cd("Jets");
	hjets_->Save();
   
	theFile->cd();
	theFile->cd("Generator");
	hgen_->Save();

	theFile->cd();
	theFile->cd("Mass");
	hmass_->Save();

	if (fdisplayJets) {
		theFile->cd();
		theFile->cd("DisplayJets");
		hdisp_->Save();
	}
   
   //Release the memory
      
   //Close the Root file
   theFile->Close();

   delete fntuple;
   
   if (debug) std::cout << "************* Finished writing histograms to file in destructor" << std::endl;

}

double
BooABCDAnalyzer::PtRel(TLorentzVector p, TLorentzVector paxis) {

	TVector3 p3 = p.Vect();
	TVector3 p3axis = paxis.Vect();

	return p3.Perp(p3axis);

}


void
BooABCDAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{	

	using namespace edm;
       	
	if (debug) std::cout << " nevents = " << nevents << std::endl;

	fntuple->Reset();
	fntuple->event = iEvent.id().event();
	fntuple->run = iEvent.id().run();

	//check if we process one event
	bool processthis = false;
    // for debugging specific event
	if (feventToProcess>-1 && nevents==feventToProcess) processthis = true;
	else if (feventToProcess==-1) processthis = true;
	
	if (!processthis) { nevents++; return; }

    /////////////////////////////////////////// 
	//
	// C O L L E C T I O N S
	//

	// beam spot
	reco::BeamSpot beamSpot;
	edm::Handle<reco::BeamSpot> beamSpotHandle;
	iEvent.getByLabel("offlineBeamSpot", beamSpotHandle);

	if ( beamSpotHandle.isValid() )
	  {
	    beamSpot = *beamSpotHandle;

	  } else
	    {
	      edm::LogInfo("MyAnalyzer")
		<< "No beam spot available from EventSetup \n";
	    }
      
	// Muons
	Handle< View<pat::Muon> > muonHandle;
	iEvent.getByLabel(muonSrc, muonHandle);
	const View<pat::Muon> &muons = *muonHandle;
	if (debug) std::cout << "got muon collection" << std::endl;

    // Electrons
	Handle< View<pat::Electron> > electronHandle;
	iEvent.getByLabel(electronSrc, electronHandle);
	const View<pat::Electron> &electrons = *electronHandle;
	if (debug) std::cout << "got electron collection" << std::endl;
	
	// MET
	Handle< View<pat::MET> > metHandle;
	iEvent.getByLabel(metSrc, metHandle);
	const View<pat::MET> &met = *metHandle;
	if (debug) std::cout << "got MET collection" << std::endl;
	// Jets
	Handle< View<pat::Jet> > jetHandle;
	iEvent.getByLabel(jetSrc, jetHandle);
	const View<pat::Jet> &jets = *jetHandle;
	if (debug) std::cout << "got jets collection" << std::endl;
	// primary vertices
	Handle< View<reco::Vertex> > PVHandle;
	iEvent.getByLabel("offlinePrimaryVertices", PVHandle);
	const View<reco::Vertex> &PVs = *PVHandle;
	// Generator
	Handle<TtGenEvent > genEvent;
	iEvent.getByLabel(genEvnSrc, genEvent);
	// Trigger
	Handle<TriggerResults> hltresults;
    iEvent.getByLabel("TriggerResults", hltresults);
    int ntrigs=hltresults->size();
    edm::TriggerNames triggerNames_;
    triggerNames_.init(*hltresults);
    bool acceptHLT = false;

	// count events
	hcounter->Counter("Processed");
	
	/////////////////////////////////////////
	//
	// G E N E R A T O R
	//
	//
	if (debug) {
		//std::cout << "GenEvent = " << genEvent << std::endl;
		TtGenEvent gevent=*genEvent;
		std::cout << "generator, isSemiLeptonic = " << gevent.isSemiLeptonic() << std::endl;
		if(gevent.lepton())  std::cout << "LeptonId = " << gevent.lepton()->pdgId() << std::endl;
		if (gevent.leptonBar()) std::cout << "LeptonBarId = " << gevent.leptonBar()->pdgId() << std::endl;
	}
	
	const reco::Candidate *genMuon   = genEvent->singleLepton();
	const reco::Candidate *genNu     = genEvent->singleNeutrino();
	const reco::Candidate *genTop    = genEvent->top();
	const reco::Candidate *genTopBar = genEvent->topBar();
	const reco::Candidate *genHadWp  = genEvent->hadronicDecayQuark();
	const reco::Candidate *genHadWq  = genEvent->hadronicDecayQuarkBar();
	const reco::Candidate *genHadb   = genEvent->hadronicDecayB();
	const reco::Candidate *genLepb   = genEvent->leptonicDecayB();

	if (debug) std::cout << "got generator candidates" << std::endl;
	double genNupz = 0;
   


	// count events
	hcounter->Counter("Generator");
	
	// check relevant collection are not empty
	// ignore electrons
	bool emptymuons = false;
	bool emptyjets  = false;
	bool emptyMET   = false;
	
   if ( muons.size()==0 ) { emptymuons = true; edm::LogWarning ( "BooABCDAnalyzer" ) << " Muon collection: " << muonSrc << " is EMPTY.";}
   if ( jets.size() ==0 ) { emptyjets  = true; edm::LogWarning ( "BooABCDAnalyzer" ) << " Jets collection: " << jetSrc << " is EMPTY.";}
   if ( met.size()  ==0 ) { emptyMET   = true; edm::LogWarning ( "BooABCDAnalyzer" ) << " MET collection: " << metSrc << " is EMPTY.";}
   if ( emptyjets || emptyMET ) {std::cout << " skipping this event. empty jet or MET collection" << std::endl; return;}

   /////////////////////////////////////////
   //
   // H L T
   //
   for (int itrig = 0; itrig < ntrigs; ++itrig) {
	   if (triggerNames_.triggerName(itrig) == "HLT_Mu9") {
		   acceptHLT = hltresults->accept(itrig);
	   }
   }

   for( size_t imu=0; imu != muons.size(); ++imu) {

	   // require Global muons
	   if ( ! muons[imu].isGlobalMuon() ) continue;

	   
	   double muonpt = muons[imu].pt(); // innerTrack()->pt(); // to test tracker muons
	   double muoneta= muons[imu].eta();
	   hmuons_->Fill1d("muon_pt_nohlt", muonpt );
	   hmuons_->Fill1d("muon_eta_nohlt", muoneta );
   }
   
   if ( !acceptHLT ) return;
   hcounter->Counter("HLT_Mu9");
   
	   
   /////////////////////////////////////////
   //
   // D E F I N E    4 - M O M E N T U M
   //

   // fixme move from array to vector
   TLorentzVector jetP4[6];// for the moment select only up to 6 leading jets
   TLorentzVector myMETP4;
   TLorentzVector METP4;
   TLorentzVector muonP4;
   TLorentzVector nuP4;
   TLorentzVector topPairP4;
   TLorentzVector hadTopP4;
   TLorentzVector lepTopP4;
   
   std::vector< double > vect_bdiscriminators;

   ////////////
   // Tools

   BooTools tools;
   
   ////////////////////////////////////////
   //
   // S E L E C T    J E T S
   //
   
   if (debug) std::cout << " number of jets = " << jets.size() << std::endl;
   hjets_->Fill1d(TString("jets")+"_"+"cut0", jets.size());

   int NgoodJets = 0;
   
   for( size_t ijet=0; ijet != jets.size(); ++ijet) {

	   // jet cuts
	   if (jets[ijet].pt() <= fMinJetPt || fabs(jets[ijet].eta()) >= fMaxJetEta ) continue;

	   NgoodJets++;
	   	   
	   TLorentzVector tmpP4;
	   tmpP4.SetPxPyPzE(jets[ijet].px(),jets[ijet].py(),jets[ijet].pz(),jets[ijet].energy());

	   myMETP4 = myMETP4 + TLorentzVector(jets[ijet].px(),jets[ijet].py(),0,jets[ijet].pt());

	   if ( NgoodJets < 7 ) {
		   jetP4[NgoodJets-1] = tmpP4;
		   vect_bdiscriminators.push_back( jets[ijet].bDiscriminator("trackCountingHighEffBJetTags") );
	   }
	   
	   	   
	   // all jets histograms
	   hjets_->Fill1d(TString("jet_et")+"_"+"cut0", jets[ijet].et());
	   hjets_->Fill1d(TString("jet_eta")+"_"+"cut0", jets[ijet].eta());
	   hjets_->Fill1d(TString("jet_phi")+"_"+"cut0", jets[ijet].phi());
	   hjets_->Fill2d(TString("jet_ptVseta")+"_cut0", jets[ijet].et(),jets[ijet].eta());
	   hjets_->Fill1d(TString("jet_emFrac_cut0"), jets[ijet].emEnergyFraction());
	   //float jetcorr = jets[ijet].jetCorrFactors().scaleDefault();
	   //hjets_->Fill1d(TString("jet_nocorr_et")+"_cut0", jets[ijet].et() /jetcorr);
	   
   }
   
   if (debug) std::cout << "Jet section done. Number of good jets: " << NgoodJets << std::endl;
   
   //if ( NgoodJets < 4 ) { nevents++; return; }

   if ( NgoodJets >= 1 ) hcounter->Counter("Njets>1");
   if ( NgoodJets >= 4 ) hcounter->Counter("Njets>3");

   std::vector< TLorentzVector > vectorjets;
   size_t cutNgoodJets = NgoodJets;
   if (NgoodJets > 6) cutNgoodJets = 6; // use only 6 good jets
   for( size_t ijet=0; ijet != cutNgoodJets; ++ijet) {
	   vectorjets.push_back(jetP4[ijet]);
	   fntuple->jet_e.push_back( jetP4[ijet].E() );
	   fntuple->jet_pt.push_back( jetP4[ijet].Pt() );
	   fntuple->jet_eta.push_back( jetP4[ijet].Eta() );
	   fntuple->jet_phi.push_back( jetP4[ijet].Phi() );
   }

   fntuple->njets = cutNgoodJets;
   
   ////////////////////////////////////////
   //
   // P R I M A R Y   V E R T E X
   //

   // use the first vertex in the collection
   // which is the vertex of the highest Pt of the associated tracks

   TVector3 thePV;

   if ( PVs.size() != 0 ) {

	   thePV = TVector3(PVs[0].x(),PVs[0].y(),PVs[0].z());

   }
   
   
   ////////////////////////////////////////
   //
   // S E L E C T     M U O N S
   //

   int NGlobalMuons = 0;
   int NgoodMuons = 0;
   int NgoodMuonsID = 0;
   int NgoodIsoMuons = 0;
   int NlooseMuonsID = 0;

   int TotalMuons = muons.size();
   hmuons_->Fill1d(TString("muons")+"_cut0",TotalMuons);
   hmuons_->FillvsJets2d(TString("muons_vsJets")+"_cut0",TotalMuons, jets);   
   int muonCharge = 0;
   double muonRelIso = 0;
   double muonVetoEm = 0;
   double muonVetoHad = 0;

   double muonIPsig = 0;

   size_t ithleadingmuon = 0;
   size_t ithisolatedmuon = 0;
   
   for( size_t imu=0; imu != muons.size(); ++imu) {

	   // require Global muons
	   if ( ! muons[imu].isGlobalMuon() ) continue;

	   NGlobalMuons++;
	   
	   double muonpt = muons[imu].pt(); // innerTrack()->pt(); // to test tracker muons
	   double muoneta= muons[imu].eta();
	   hmuons_->Fill1d("muon_pt_cut0", muonpt );
	   hmuons_->Fill1d("muon_eta_cut0", muoneta );

	   double oldRelIso = ( muonpt/(muonpt + muons[imu].caloIso() + muons[imu].trackIso()) );
	   // Loose muons
	   if (muonpt > 10. && fabs(muons[imu].eta()) < 2.5 && oldRelIso>0.8 )
		   NlooseMuonsID++;
			   
	   if ( (muonpt > fMinMuonPt) && fabs(muons[imu].eta()) < fMaxMuonEta ) {

		   NgoodMuons++;
		   hmuons_->Fill1d("muon_pt_cut1", muonpt );
		   hmuons_->Fill1d("muon_eta_cut1", muoneta );
	   
		   // Muon ID
		   int nhit = muons[imu].innerTrack()->numberOfValidHits();
		   double normChi2 = muons[imu].globalTrack()->chi2() / muons[imu].globalTrack()->ndof();
		   // math::XYZPoint point(bSpot.x0()+bSpot.dxdz()*eleDZ0,bSpot.y0()+bSpot.dydz()*eleDZ0, bSpot.z0());
		   math::XYZPoint point(beamSpot.x0(),beamSpot.y0(), beamSpot.z0());
		   //double d0 = muons[imu].innerTrack()->d0();
		   double d0 = -1.* muons[imu].innerTrack()->dxy(point);
		   hmuons_->Fill2d("muon_phi_vs_d0_cut0", muons[imu].innerTrack()->phi(), muons[imu].innerTrack()->d0() );
		   hmuons_->Fill2d("muon_phi_vs_d0_cut1", muons[imu].innerTrack()->phi(), d0 );

		   double d0sigma = sqrt( muons[imu].innerTrack()->d0Error() * muons[imu].innerTrack()->d0Error() + beamSpot.BeamWidthX()*beamSpot.BeamWidthX());
		   
		   if ( nhit >= 11 && normChi2 < 10 ) {

			   NgoodMuonsID++;
			   hmuons_->Fill1d("muon_pt_cut2", muonpt );
			   //hmuons_->Fill1d("muon_d0_cut2", d0 );
			   
			   // ISOLATION	   
			   //double oldRelIso = ( muonpt/(muonpt + muons[imu].caloIso() + muons[imu].trackIso()) );
			   double RelIso = muons[imu].caloIso()/muons[imu].et() + muons[imu].trackIso()/muonpt;
			   double newRelIso = RelIso;

			   
			   hmuons_->Fill1d("muon_RelIso_cut2", RelIso);

			   			   			   
			   if ( oldRelIso > fMuonRelIso  && fabs(d0/d0sigma)<3 ) {

				   NgoodIsoMuons++;
				   if ( NgoodIsoMuons==1 ) {
					   ithisolatedmuon = imu;
					   // energy in veto cone
					   muonVetoEm = muons[imu].ecalIsoDeposit()->candEnergy();
					   muonVetoHad = muons[imu].hcalIsoDeposit()->candEnergy();
				   }
				   hmuons_->Fill1d("muon_pt_cut3", muonpt );
			   
				   //double energymu = sqrt(muons[imu].innerTrack()->px()*muons[imu].innerTrack()->px() +
				   //					  muons[imu].innerTrack()->py()*muons[imu].innerTrack()->py() +
				   //				  muons[imu].innerTrack()->pz()*muons[imu].innerTrack()->pz() + 0.1057*0.1057);
			   }

			   // which muon should I pick ??
			   // pick the leading muon
			   if ( NgoodMuonsID == 1 ) {
				   
					   double energymu = muons[imu].energy();
					   muonP4.SetPxPyPzE(muons[imu].px(),muons[imu].py(),muons[imu].pz(),energymu );
					   muonCharge = muons[imu].charge();
					   muonRelIso = RelIso;
					   muonIPsig = d0/d0sigma;

					   hmuons_->Fill1d("muon_vetoEm_cut3", muonVetoEm);
					   hmuons_->Fill1d("muon_vetoHad_cut3", muonVetoHad);

					 
					   
					   hmuons_->Fill1d("muon_pt_cut4", muonpt );

					   ithleadingmuon = imu;
					   // energy in veto cone
					   muonVetoEm = muons[imu].ecalIsoDeposit()->candEnergy();
					   muonVetoHad = muons[imu].hcalIsoDeposit()->candEnergy();
					   
					   
			   }

		   }
	   
	   }
   }

   if ( NgoodIsoMuons == 1 ) {
	   if ( NlooseMuonsID == 1 ) {
	   size_t imu = ithisolatedmuon;
	   math::XYZPoint point(beamSpot.x0(),beamSpot.y0(), beamSpot.z0());
	   double d0 = -1.* muons[imu].innerTrack()->dxy(point);
	   double d0sigma = sqrt( muons[imu].innerTrack()->d0Error() * muons[imu].innerTrack()->d0Error() + beamSpot.BeamWidthX()*beamSpot.BeamWidthX());
	   fntuple->muon_px.push_back( muons[imu].px() );
	   fntuple->muon_py.push_back( muons[imu].py() );
	   fntuple->muon_pz.push_back( muons[imu].pz() );
	   fntuple->muon_e.push_back( muons[imu].energy() );
	   fntuple->muon_old_reliso.push_back( muons[imu].pt()/(muons[imu].pt() + muons[imu].caloIso() + muons[imu].trackIso()) );
	   fntuple->muon_new_reliso.push_back( muons[imu].caloIso()/muons[imu].et() + muons[imu].trackIso()/muons[imu].pt() );
	   fntuple->muon_d0.push_back( d0 );
	   fntuple->muon_d0Error.push_back( d0sigma );
	   }
	   else { return; }
   } else if ( NgoodMuonsID >= 1 ) {
	   size_t imu = ithleadingmuon;
	   math::XYZPoint point(beamSpot.x0(),beamSpot.y0(), beamSpot.z0());
	   double d0 = -1.* muons[imu].innerTrack()->dxy(point);
	   double d0sigma = sqrt( muons[imu].innerTrack()->d0Error() * muons[imu].innerTrack()->d0Error() + beamSpot.BeamWidthX()*beamSpot.BeamWidthX());
	   fntuple->muon_px.push_back( muons[imu].px() );
	   fntuple->muon_py.push_back( muons[imu].py() );
	   fntuple->muon_pz.push_back( muons[imu].pz() );
	   fntuple->muon_e.push_back( muons[imu].energy() );
	   fntuple->muon_old_reliso.push_back( muons[imu].pt()/(muons[imu].pt() + muons[imu].caloIso() + muons[imu].trackIso()) );
	   fntuple->muon_new_reliso.push_back( muons[imu].caloIso()/muons[imu].et() + muons[imu].trackIso()/muons[imu].pt() );
	   fntuple->muon_d0.push_back( d0 );
	   fntuple->muon_d0Error.push_back( d0sigma );

   }


   //count muons and jets
   if ( NgoodJets >= 4 ) {

	   if ( NGlobalMuons > 0 ) hcounter->Counter("GlobalMuons");
	   if ( NgoodMuons > 0   ) hcounter->Counter("GoodMuons");
	   if ( NgoodMuonsID > 0 ) hcounter->Counter("GoodMuonsID");
	   if ( NgoodIsoMuons > 0) hcounter->Counter("GoodIsoMuons");
	   if ( NgoodIsoMuons ==1) hcounter->Counter("GoodOneIsoMuon");
   }

   // number of muons vs number of jets
   hmuons_->FillvsJets2d(TString("muons_vsJets")+"_cut3", NgoodIsoMuons, vectorjets);
   hmuons_->Fill1d(TString("muons")+"_cut0", NGlobalMuons);
   hmuons_->Fill1d(TString("muons")+"_cut1", NgoodMuons);
   hmuons_->Fill1d(TString("muons")+"_cut2", NgoodMuonsID);
   hmuons_->Fill1d(TString("muons")+"_cut3", NgoodIsoMuons);
   
   if (debug) std::cout << "Muon section done. Number of good muons: " << NgoodIsoMuons << std::endl;

	
   // select events with at least one good muon.
   if ( NgoodMuonsID == 0 ) {
	   nbadmuons++;
	   //edm::LogWarning ("BooABCDAnalyzer") << "Event with number of good muons: "<< NgoodIsoMuons << ", skip this event since we request one good muon.";
	   return;
   }
   // reject events with more than one isolated good muon.
   if ( NgoodIsoMuons > 1 ) {

	   return;
   }

   /////////////////////////////////////
   //
   // C L E A N   M U O N   M I Ps
   //


   
   double minDeltaR_muon_jet = 9e9;
   int theJetClosestMu = -1;
   double closestEMFrac = -1;
   TLorentzVector closestJet;
   //TLorentzVector closestJet2;
   
   for( size_t ijet=0; ijet < jets.size(); ++ijet ) {

	   if (jets[ijet].pt() <= fMinJetPt || fabs(jets[ijet].eta()) >= fMaxJetEta ) continue;
	   
	   TLorentzVector tmpP4;
	   tmpP4.SetPxPyPzE(jets[ijet].px(),jets[ijet].py(),jets[ijet].pz(),jets[ijet].energy());
	   TLorentzVector tmpP4raw;
	   //float jetcorr = jets[ijet].jetCorrFactors().scaleDefault();
	   //tmpP4raw.SetPxPyPzE(jets[ijet].px()/jetcorr,jets[ijet].py()/jetcorr,jets[ijet].pz()/jetcorr,jets[ijet].energy()/jetcorr);

	   double tmpclosestEMFrac = jets[ijet].emEnergyFraction();
	   
	   //double aDeltaR_muon_jet = ROOT::Math::VectorUtil::DeltaR( tmpP4.Vect(), muonP4.Vect() );
	   double aDeltaR_muon_jet = deltaR(tmpP4.Eta(), tmpP4.Phi(), muonP4.Eta(), muonP4.Phi() );
	   hjets_->Fill1d(TString("jet_deltaR_muon")+"_cut0", aDeltaR_muon_jet);
	   if ( aDeltaR_muon_jet < minDeltaR_muon_jet ) {
		   theJetClosestMu = (int) ijet;
		   minDeltaR_muon_jet = aDeltaR_muon_jet;
		   closestJet = tmpP4;
		   //closestJet2= tmpP4raw;
		   closestEMFrac = tmpclosestEMFrac;
	   }
   }
   
   hjets_->Fill1d(TString("jet_deltaR_muon")+"_cut1", minDeltaR_muon_jet);
   fntuple->muon_minDeltaR.push_back( minDeltaR_muon_jet);
   fntuple->muon_closestJet_px.push_back( closestJet.Px() );
   fntuple->muon_closestJet_px.push_back( closestJet.Py() );
   fntuple->muon_closestJet_px.push_back( closestJet.Pz() );
   fntuple->muon_closestJet_px.push_back( closestJet.E() );
   
   if ( muonVetoEm < fMaxMuonEm  && muonVetoHad < fMaxMuonHad ) hjets_->Fill1d("jet_deltaR_muon_cut2", minDeltaR_muon_jet);
   
   //if (closestJet2.Et() > 10. ) hjets_->Fill1d(TString("jet_deltaR_muon")+"_cut2", minDeltaR_muon_jet);
   //if (closestEMFrac > 0.1 && closestEMFrac<0.9 ) hjets_->Fill1d(TString("jet_deltaR_muon")+"_cut3", minDeltaR_muon_jet );

   
   hjets_->Fill2d(TString("jet_deltaR_muon_vs_RelIso")+"_cut1", minDeltaR_muon_jet, muonRelIso);

   if (theJetClosestMu != -1 ) {
//	   hjets_->Fill1d(TString("jet_pTrel_muon")+"_cut0",PtRel(muonP4,muonP4+closestJet));
	   fntuple->muon_ptrel.push_back( PtRel(muonP4, closestJet) );//muonP4+closestJet));
	   
	   hjets_->Fill1d(TString("jet_pT_closest_muon")+"_cut0", closestJet.Pt());
	   //hjets_->Fill1d(TString("jet_pT_closest_muon")+"_cut2", closestJet2.Pt());
	   hjets_->Fill1d(TString("jet_emFrac_cut1"), closestEMFrac );
	   
	   //if (closestJet2.Et() > 10. ) {
	   //	   hjets_->Fill1d(TString("jet_pT_closest_muon")+"_cut3", closestJet.Pt());
	   //	   hjets_->Fill1d(TString("jet_emFrac_cut2"), closestEMFrac );
	   //}
	   
	   int flavour = abs(jets[theJetClosestMu].partonFlavour());
	   hjets_->Fill1d(TString("jet_flavor_closest_muon")+"_cut0", flavour);
//	   if ( flavour == 5 ) hjets_->Fill1d(TString("jet_pTrel_muon_b")+"_cut0",PtRel(muonP4,muonP4+closestJet));
	   //   if ( flavour == 4 ) hjets_->Fill1d(TString("jet_pTrel_muon_c")+"_cut0",PtRel(muonP4,muonP4+closestJet));
	   //if ( (flavour < 4 && flavour != 0 )||(flavour == 21 ) ) hjets_->Fill1d(TString("jet_pTrel_muon_udsg")+"_cut0",PtRel(muonP4,muonP4+closestJet));

	   
   }
        
   //if ( minDeltaR_muon_jet <= 0.3 ) {
	   //return;
   //}

   if ( NgoodJets >= 4 && minDeltaR_muon_jet > 0.3 ) hcounter->Counter("DeltaR");

   if ( muonVetoEm >= fMaxMuonEm || muonVetoHad >= fMaxMuonHad ) return;

   hjets_->Fill1d("jets_cut1",NgoodJets);
   
   if ( NgoodJets >= 4 && muonVetoEm < fMaxMuonEm  && muonVetoHad < fMaxMuonHad ) hcounter->Counter("muonVetoCone");

   
   if (debug) std::cout << "deltaR(muon,near jet) cut survive" << std::endl;

   
  
   //////////////////////////////////////
   //
   // E L E C T R O N   R E M O V A L
   //
   
   int NlooseElectrons = 0;
   int NgoodElectrons = 0;
   helectrons_->Fill1d("electrons_cut0", electrons.size() );
   
   for( size_t ie=0; ie != electrons.size(); ++ie) {

	   double ept = electrons[ie].pt();
	   helectrons_->Fill1d("electron_pt_cut0", ept );
	   helectrons_->Fill1d("electron_eta_cut0", electrons[ie].eta() );

	   double relIso = electrons[ie].trackIso()/ept + electrons[ie].caloIso()/electrons[ie].et();
	   
	   // loose
	   if ( ept > 15 && fabs(electrons[ie].eta()) < 2.5 && relIso > 0.8 ) NlooseElectrons++;
	   
	   if ( ept > fMinElectronPt && fabs(electrons[ie].eta()) < fMaxElectronEta &&
		   electrons[ie].electronID("eidTight")>0) {

		   //double relIso = electrons[ie].trackIso()/ept + electrons[ie].caloIso()/electrons[ie].et();

		   if ( relIso < fElectronRelIso ) {

			   NgoodElectrons++;
		   }
	   }	
   }

   helectrons_->Fill1d("electrons_cut1", NgoodElectrons );
   if ( NgoodElectrons > 0 ) return;

   if ( NgoodJets >= 4 ) hcounter->Counter("NoElectrons");
   
   if ( NlooseElectrons > 1 ) return;
   
   ////////////////////////////////////
   //
   // M E T
   //

   double Ht = 0;
   
   // plot my MET
   hmet_->Fill1d(TString("myMET")+"_cut0", myMETP4.Pt());
   // correct my MET
   myMETP4 = myMETP4 + TLorentzVector(muonP4.Px(),muonP4.Py(),0,muonP4.Pt());
   hmet_->Fill1d(TString("myMET")+"_cut1", myMETP4.Pt());
   
   // met is corrected by muon momentum, how about muon energy?
   if (met.size() != 1 ) edm::LogWarning ("BooABCDAnalyzer") << "MET collection has size different from ONE! size: "<< met.size() << std::endl;
      
   for( size_t imet=0; imet != met.size(); ++imet) {
	   Ht = met[imet].sumEt();
	   
	   hmet_->Fill1d(TString("MET")+"_"+"cut0", met[imet].et());
	   hmet_->Fill1d(TString("MET_eta")+"_"+"cut0", met[imet].eta());
	   hmet_->Fill1d(TString("MET_phi")+"_"+"cut0", met[imet].phi());
	   hmet_->Fill1d(TString("MET_deltaR_muon")+"_"+"cut0", DeltaR<reco::Candidate>()( met[imet] , muons[0] ));
	   hmet_->FillvsJets2d(TString("MET_vsJets")+"_cut0",met[imet].et(), vectorjets);
	   hmet_->Fill1d(TString("Ht")+"_cut0", met[imet].sumEt());
	   hmet_->FillvsJets2d(TString("Ht_vsJets")+"_cut0",met[imet].sumEt(), vectorjets);
   }

   METP4.SetPxPyPzE(met[0].px(), met[0].py(), met[0].pz(), met[0].energy());
   myMETP4 = (-1)*myMETP4;
   
   if (debug) std::cout << "MET section done" << std::endl;
      
   //if (NgoodJets >=4 ) {
	   //hmuons_->Fill2d("muon_RelIso_vs_MET_cut4", muonRelIso, METP4.Et() );
     double Htl = muonP4.Pt();
     for( size_t ijet=0; ijet != NgoodJets; ++ijet) Htl += jetP4[ijet].Et();

	 //hmuons_->Fill2d("muon_RelIso_vs_Htl_cut4", muonRelIso, Htl );

	 //hmuons_->Fill2d("muon_RelIso_vs_Ht_cut4", muonRelIso, Ht );

	 //hmuons_->Fill2d("muon_ptrel_vs_Ht_cut4", muonptrel, Ht );

	 //hmuons_->Fill2d("muon_RelIso_vs_IPsig_cut4", muonRelIso, muonIPsig );

	 fntuple->MET.push_back( METP4.Et() );
	 fntuple->Ht.push_back( Ht );
	 
	 //}

   if (debug) std::cout << "done." << std::endl;
	   
    
   // count all events
   nevents++;

   // fill tree
   ftree->Fill();

}


//define this as a plug-in
DEFINE_FWK_MODULE(BooABCDAnalyzer);
