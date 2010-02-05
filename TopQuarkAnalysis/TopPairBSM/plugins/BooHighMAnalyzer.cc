// -*- C++ -*-
//
// Package:    BooHighMAnalyzer
// Class:      BooHighMAnalyzer
// 
/**\class BostedTopPair/BooHighMAnalyzer.cc

 Description:

 Implementation:
     <Notes on implementation>

	 Author: Francisco Yumiceva
*/
//
// $Id: BooHighMAnalyzer.cc,v 1.2 2009/07/30 06:32:25 jengbou Exp $
//
//


// system include files
#include <memory>

#include "DataFormats/VertexReco/interface/Vertex.h"

// user include files
#include "TopQuarkAnalysis/TopPairBSM/interface/BooHighMAnalyzer.h"

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
BooHighMAnalyzer::BooHighMAnalyzer(const edm::ParameterSet& iConfig)
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
  fApplyJetAsymmetricCuts = iConfig.getParameter<edm::ParameterSet>("jetCuts").getParameter<bool>("ApplyAsymmetricCuts");

  fUsebTagging      = iConfig.getParameter<bool>  ("UsebTagging");
  
  //fMinHt           = iConfig.getParameter<edm::ParameterSet>("jetCuts").getParameter<double>("MinHt");
  fMinMET           = iConfig.getParameter<edm::ParameterSet>("METCuts").getParameter<double>("MinMET");
  fUseMyMET         = iConfig.getParameter<edm::ParameterSet>("METCuts").getParameter<bool>("Recalculate");
    
  feventToProcess   = iConfig.getParameter<int> ("processOnlyEvent");
  fdisplayJets      = iConfig.getParameter<bool>   ("makeJetLegoPlots");


  // write ascii output
  if (fwriteAscii) {
	  edm::LogWarning ( "BooHighMAnalyzer" ) << " Results will also be saved into an ascii file: " << fasciiFileName;
	  fasciiFile.open(fasciiFileName.c_str());
  }

  // Create a root file
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
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
  
  // initialize cuts // Do we want this?
  cut_map["cut0"] = "Initial selection";
  //  cut_map["cut1"] = "cut1";
  //cut_map["cut2"] = "Min Jet Et";
  //cut_map["cut3"] = "Min MET";
  //cut_map["cut4"] = "";

  // initialize histogram manager
  hcounter= new BooHistograms();
  hmuons_ = new BooHistograms();
  helectrons_= new BooHistograms();
  hmet_   = new BooHistograms();
  hjets_  = new BooHistograms();
  hgen_   = new BooHistograms();
  hmass_  = new BooHistograms();
  
  if (fdisplayJets) hdisp_  = new BooHistograms();

  // clean this later
  for (std::map<TString, TString>::const_iterator imap=cut_map.begin();
	   imap!=cut_map.end(); ++imap) {
	  
	  TString acut = imap->first;
	  
	  //hjets_->Init("Jets",acut);
	  //h_->Init("jets",acut,"MC");
	  //hmuons_->Init("Muons",acut);
	  //h_->Init("muons",acut,"MC");
	  //hmet_->Init("MET",acut);
	  
  }

  hcounter->Init("counter");

  hmuons_->Init("Muons","nohlt");
  hmuons_->Init("Muons","nohltgood");
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


BooHighMAnalyzer::~BooHighMAnalyzer()
{

	if (debug) std::cout << "BooHighMAnalyzer Destructor called" << std::endl;

	// print out useful informantion
	std::cout << "BooHighMAnalyzer Total events analyzed = " << nevents << std::endl;
	std::cout << "BooHighMAnalyzer Number of bad muon events = " << nbadmuons << std::endl;
	std::cout << "BooHighMAnalyzer Number of complex solutions = " << nWcomplex <<std::endl;
	std::cout << "BooHighMAnalyzer Number of solutions with unambigous jet-parton matching, for sumEt case = " << MCAllmatch_sumEt_ << std::endl;
	std::cout << "BooHighMAnalyzer Number of solutions with unambigous jet-parton matching, for chi2 case  = " << MCAllmatch_chi2_ << std::endl;
	
	   
	if (fwriteAscii) fasciiFile.close();

	// save all histograms
	theFile->cd();
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

   if (debug) std::cout << "************* Finished writing histograms to file in destructor" << std::endl;

}

double
BooHighMAnalyzer::dij(TLorentzVector p1, TLorentzVector p2, double mass, bool min) {

	TLorentzVector ptot = p1 + p2;
	Double_t theta1 = TMath::ACos( (p1.Vect().Dot(ptot.Vect()))/(p1.P()*ptot.P()) );
	Double_t theta2 = TMath::ACos( (p2.Vect().Dot(ptot.Vect()))/(p2.P()*ptot.P()) );
	double theta = theta1+theta2;
	double Emin2ij = p1.E() * p1.E();
	double Emax2ij = p2.E() * p2.E();
	if ( p1.E()>p2.E() ) { Emin2ij = p2.E() * p2.E(); Emax2ij = p1.E() * p1.E();}

	double result = TMath::Sin(theta/2)*TMath::Sin(theta/2)*Emin2ij/(mass*mass);
	if (!min) result = TMath::Sin(theta/2)*TMath::Sin(theta/2)*Emax2ij/(mass*mass);

	return result;
	
}


double
BooHighMAnalyzer::Psi(TLorentzVector p1, TLorentzVector p2, double mass) {

	TLorentzVector ptot = p1 + p2;
	Double_t theta1 = TMath::ACos( (p1.Vect().Dot(ptot.Vect()))/(p1.P()*ptot.P()) );
	Double_t theta2 = TMath::ACos( (p2.Vect().Dot(ptot.Vect()))/(p2.P()*ptot.P()) );
	Double_t min = 0;
	if (p1.Pt() > p2.Pt() ) min = TMath::Sqrt(TMath::Sqrt(p2.P()/p1.P()));
	else min = TMath::Sqrt(TMath::Sqrt(p1.P()/p2.P()));
	double th1th2 = theta1 + theta2;
	double psi = (p1.P()+p2.P())*TMath::Sin((th1th2)/2)*min/mass;	

/*
	TLorentzVector ptot = p1 + p2;
	Double_t theta1 = TMath::ACos( (p1.Vect().Dot(ptot.Vect()))/(p1.P()*ptot.P()) );
	Double_t theta2 = TMath::ACos( (p2.Vect().Dot(ptot.Vect()))/(p2.P()*ptot.P()) );
	//Double_t sign = 1.;
	//if ( (theta1+theta2) > (TMath::Pi()/2) ) sign = -1.;
	double th1th2 = theta1 + theta2;
	double psi = (p1.P()+p2.P())*TMath::Abs(TMath::Sin(th1th2))/(2.* mass );
	if ( th1th2 > (TMath::Pi()/2) )
		psi = (p1.P()+p2.P())*( 1. + TMath::Abs(TMath::Cos(th1th2)))/(2.* mass );

*/
	return psi;
}

double
BooHighMAnalyzer::PtRel(TLorentzVector p, TLorentzVector paxis) {

	TVector3 p3 = p.Vect();
	TVector3 p3axis = paxis.Vect();

	return p3.Perp(p3axis);

}

bool
BooHighMAnalyzer::IsTruthMatch( Combo acombo, const edm::View<pat::Jet> jets, TtGenEvent genEvt, bool MatchFlavor ) {

	bool match = false;
	
	// get back pat jets from a combo
	pat::Jet patWp = jets[ acombo.GetIdWp() ];
	pat::Jet patWq = jets[ acombo.GetIdWq() ];
	pat::Jet patHadb = jets[ acombo.GetIdHadb() ];
	pat::Jet patLepb = jets[ acombo.GetIdLepb() ];

	// get gen partons
	const reco::Candidate* genWp  = genEvt.hadronicDecayQuark();
	const reco::Candidate* genWq  = genEvt.hadronicDecayQuarkBar();
	const reco::Candidate* genHadb = genEvt.hadronicDecayB();
	const reco::Candidate* genLepb = genEvt.leptonicDecayB();

	// first check if we have all partons 
	if ( !genWp || !genWq || !genHadb || !genLepb ) {
	  //std::cout << "parton-jet Matching: all null pointers" << std::endl;
	  return match;
	}

	//std::cout << " GenMatching: Hadb = " << genHadb->pdgId() << " q = " << genHadb->threeCharge() << " Mother = " << genHadb->mother()->pdgId() << std::endl;
	//std::cout << " GenMatching: Lepb = " << genLepb->pdgId() << " q = " << genLepb->threeCharge() << " Mother = " << genHadb->mother()->pdgId() << std::endl;
	//std::cout << " GenMatching: Wp = " << genWp->pdgId() << " q = " << genWp->threeCharge() << " Mother = " << genHadb->mother()->pdgId() << std::endl;
	//std::cout << " GenMatching: Wq = " << genWq->pdgId() << " q = " << genWq->threeCharge() << " Mother = " << genHadb->mother()->pdgId() << std::endl;

	
	std::vector<const reco::Candidate*> partons;
	partons.push_back( genWp );
	partons.push_back( genWq );
	partons.push_back( genHadb );
	partons.push_back( genLepb );

	std::vector< pat::Jet > pjets;
	pjets.push_back( patWp );
	pjets.push_back( patWq );
	pjets.push_back( patHadb );
	pjets.push_back( patLepb );
	std::vector< int > pjetsId;
	pjetsId.push_back( 4 );// let's set light jets (u,d,s,c) Id=4
	pjetsId.push_back( 4 );// let's set light jets (u,d,s,c) Id=4
	pjetsId.push_back( 5 );// b jet
	pjetsId.push_back( 5 );// b jet
	
	// sort partons in descendent pt
	std::sort(partons.begin(), partons.end(), partonMaxPt() );
		
	
	// now check correct association
	int nWdau = 0;
	int nbjets = 0;
	
	for ( unsigned int ip=0; ip < partons.size(); ++ip ) {

		double minDelta = 999.;
		int ithjet = -1;
		
		for ( unsigned int ij=0; ij < pjets.size(); ++ij ) {

			double adelta = ROOT::Math::VectorUtil::DeltaR( partons[ip]->p4() ,  pjets[ij].p4() );
			if ( adelta < 0.3 && adelta <= minDelta ) {
				minDelta = adelta;
				ithjet = (int) ij;
			}
		}
		if ( ( abs( partons[ip]->pdgId() ) <= 4 ) && ( pjetsId[ithjet] == 4 ) ) nWdau++;
		if ( ( abs( partons[ip]->pdgId() ) == 5 ) && ( pjetsId[ithjet] == 5 ) ) nbjets++;
			
		// remove matched jet
		if (ithjet > -1 ) {
			pjets.erase(pjets.begin() + ithjet );
			pjetsId.erase(pjetsId.begin() + ithjet );
		}
		
	}

	//std::cout << " GenMatching: number of no matched jets " << pjets.size() << std::endl;
	//std::cout << " GenMatching: nWdau = " << nWdau << " nbjets = " << nbjets << std::endl;
	
	// check if we match all the jets
	if ( int(pjets.size()) == 0 ) {

		if ( !MatchFlavor )	match = true;
		
		else if ( nWdau==2 && nbjets ==2 ) match = true;
			
	}

	partons.clear();
	pjets.clear();
	pjetsId.clear();

	return match;
	
}


void
BooHighMAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{	

	using namespace edm;
       	
	if (debug) std::cout << " nevents = " << nevents << std::endl;

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
	bool IsEventReconstructable = false;
	
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
   
	if (genTop && genTopBar) {
		LorentzVector gentoppairP4;
		gentoppairP4 = genTop->p4() + genTopBar->p4();
		hgen_->Fill1d("gen_toppair_mass", gentoppairP4.M() );
		hgen_->Fill1d("gen_toppair_pt", gentoppairP4.pt() );
		
		hgen_->Fill1d("gen_top_pt", genTop->pt() );
		hgen_->Fill1d("gen_top_eta", genTop->eta() );
		hgen_->Fill1d("gen_top_rapidity", genTop->y() );	
			
		if ( genTop->pt() > 50 ) hgen_->Fill1d("gen_top_eta1", genTop->eta() );
		if ( genTop->pt() > 100 ) hgen_->Fill1d("gen_top_eta2", genTop->eta() );
		if ( genTop->pt() > 150 ) hgen_->Fill1d("gen_top_eta3", genTop->eta() );
		
		
		if (debug) std::cout << " gen top rapidities: t_y = " << genTop->y()
							 << " tbar_y = " << genTopBar->y() << std::endl;

		// run next block only for muon+jets channel
		if ( genEvent->isSemiLeptonic() && genEvent->semiLeptonicChannel()==2 ) {

		if (genHadWp && genHadWq && genMuon && genHadb && genLepb && genNu) {
			hgen_->Fill2d("gentop_rapidities", genTop->y(), genTopBar->y() );
			genNupz = genNu->p4().Pz();
			hgen_->Fill1d("gen_nu_pz", genNupz );
			hgen_->Fill2d("gen_nu_pt_vs_pz", genNu->p4().Pt(), genNupz );
			hgen_->Fill1d("gen_deltaR_qb", DeltaR<reco::Candidate>()(*genHadWq, *genHadb ) );
			hgen_->Fill1d("gen_deltaR_pb", DeltaR<reco::Candidate>()(*genHadWp, *genHadb ) );
			hgen_->Fill1d("gen_deltaR_pq", DeltaR<reco::Candidate>()(*genHadWp, *genHadWq ) );
			hgen_->Fill1d("gen_deltaR_lb", DeltaR<reco::Candidate>()(*genMuon , *genLepb ) );
			hgen_->Fill1d("gen_deltaR_qLepb", DeltaR<reco::Candidate>()(*genHadWq , *genLepb ) );
			hgen_->Fill1d("gen_deltaR_qmu", DeltaR<reco::Candidate>()(*genHadWq , *genMuon ) );
			hgen_->Fill1d("gen_deltaR_muLepb", DeltaR<reco::Candidate>()(*genMuon, *genLepb ) );
			
			// i should fill it twice with p and q jets....
			hgen_->Fill2d("gen_deltaR_pq_vs_toppt", DeltaR<reco::Candidate>()(*genHadWp, *genHadWq ), genEvent->hadronicDecayTop()->pt()  );
			hgen_->Fill2d("gen_deltaR_qb_vs_toppt", DeltaR<reco::Candidate>()(*genHadWq, *genHadb ), genEvent->hadronicDecayTop()->pt()  );
			hgen_->Fill2d("gen_deltaR_muLepb_vs_toppt", DeltaR<reco::Candidate>()(*genMuon, *genLepb ), genEvent->leptonicDecayTop()->pt()  );

			hgen_->Fill2d("gen_muonpt_vs_lepbpt", genMuon->pt(), genLepb->pt() );
			
			LorentzVector tmpgentop = genEvent->hadronicDecayTop()->p4();
			LorentzVector tmpgenW = genEvent->hadronicDecayW()->p4();
			LorentzVector tmpgenWp= genEvent->hadronicDecayQuark()->p4();
			LorentzVector tmpgenWq= genEvent->hadronicDecayQuarkBar()->p4();
			TVector3 genp1(tmpgenWp.Px(),tmpgenWp.Py(),tmpgenWp.Pz());
			TVector3 genp2(tmpgenWq.Px(),tmpgenWq.Py(),tmpgenWq.Pz());
			TVector3 genptot(tmpgenW.Px(),tmpgenW.Py(),tmpgenW.Pz());
			//Double_t tmptheta1 = TMath::ACos( (genp1.Dot(genptot))/(tmpgenWp.P()*tmpgenW.P()) );
			//Double_t tmptheta2 = TMath::ACos( (genp2.Dot(genptot))/(tmpgenWq.P()*tmpgenW.P()) );
			
			hgen_->Fill2d("gen_toprapidity_vs_psi_pq",tmpgentop.Rapidity(), Psi(TLorentzVector(tmpgenWp.Px(),tmpgenWp.Py(),tmpgenWp.Pz(),tmpgenWp.E()),TLorentzVector(tmpgenWq.Px(),tmpgenWq.Py(),tmpgenWq.Pz(),tmpgenWq.E()),tmpgentop.M()));
//(tmpgenWp.P()+tmpgenWq.P())*TMath::Sin(tmptheta1+tmptheta2)/(2.*tmpgentop.M() ));
			
			hgen_->Fill2d("gen_toprapidity_vs_deltaR_pq",tmpgentop.Rapidity(),DeltaR<reco::Candidate>()(*genHadWp, *genHadWq )  );
			TLorentzVector tmpgenP4Wp = TLorentzVector(tmpgenWp.Px(),tmpgenWp.Py(),tmpgenWp.Pz(),tmpgenWp.E());
			TLorentzVector tmpgenP4Wq = TLorentzVector(tmpgenWq.Px(),tmpgenWq.Py(),tmpgenWq.Pz(),tmpgenWq.E());
			
			hgen_->Fill2d("gen_toprapidity_vs_dminij_pq",tmpgentop.Rapidity(), dij(tmpgenP4Wp,tmpgenP4Wq,tmpgentop.M()) );
			hgen_->Fill2d("gen_toprapidity_vs_dmaxij_pq",tmpgentop.Rapidity(), dij(tmpgenP4Wp,tmpgenP4Wq,tmpgentop.M(), false) );
			
			double tmppL= (tmpgenW.Px()*tmpgentop.Px()+tmpgenW.Py()*tmpgentop.Py()+tmpgenW.Pz()*tmpgentop.Pz()) / tmpgentop.P();
			double tmppT= TMath::Sqrt(tmpgenW.P()*tmpgenW.P() - tmppL*tmppL);
			
			hgen_->Fill2d("gen_HadW_pT_vs_pL", tmppT, tmppL );
			
			LorentzVector tmpgenhadb = genHadb->p4();
			tmppL= (tmpgenhadb.Px()*tmpgentop.Px()+tmpgenhadb.Py()*tmpgentop.Py()+tmpgenhadb.Pz()*tmpgentop.Pz()) / tmpgentop.P();
			tmppT= TMath::Sqrt(tmpgenhadb.P()*tmpgenhadb.P() - tmppL*tmppL); 
			
			hgen_->Fill2d("gen_Hadb_pT_vs_pL", tmppT, tmppL );
			
			LorentzVector tmpgenmu = genMuon->p4();
			LorentzVector tmpgenLepW = genEvent->leptonicDecayW()->p4();
			tmppL= (tmpgenmu.Px()*tmpgenLepW.Px()+tmpgenmu.Py()*tmpgenLepW.Py()+tmpgenmu.Pz()*tmpgenLepW.Pz()) / tmpgenLepW.P();
			tmppT= TMath::Sqrt(tmpgenmu.P()*tmpgenmu.P() - tmppL*tmppL); 
			
			double tmpcosCM = TMath::Cos(TMath::ASin(2.*tmppT/tmpgenLepW.M()));
			
			LorentzVector tmpgennu = genNu->p4();
			hgen_->Fill2d("gen_cosCM_vs_psi",tmpcosCM,Psi(TLorentzVector(tmpgenmu.Px(),tmpgenmu.Py(),tmpgenmu.Pz(),tmpgenmu.E()),
														  TLorentzVector(tmpgennu.Px(),tmpgennu.Py(),tmpgennu.Pz(),tmpgennu.E()),
														  tmpgenLepW.M()));

			double theminGenDeltaR = 999.;

			if ( DeltaR<reco::Candidate>()(*genHadWp, *genHadWq ) < theminGenDeltaR ) theminGenDeltaR = DeltaR<reco::Candidate>()(*genHadWp, *genHadWq );
			if ( DeltaR<reco::Candidate>()(*genHadWp, *genHadb ) < theminGenDeltaR ) theminGenDeltaR = DeltaR<reco::Candidate>()(*genHadWp, *genHadb );
			if ( DeltaR<reco::Candidate>()(*genHadWp, *genLepb ) < theminGenDeltaR ) theminGenDeltaR = DeltaR<reco::Candidate>()(*genHadWp, *genLepb );
			if ( DeltaR<reco::Candidate>()(*genHadWq, *genHadb ) < theminGenDeltaR ) theminGenDeltaR = DeltaR<reco::Candidate>()(*genHadWq, *genHadb );
			if ( DeltaR<reco::Candidate>()(*genHadWq, *genLepb ) < theminGenDeltaR ) theminGenDeltaR = DeltaR<reco::Candidate>()(*genHadWq, *genLepb );
			if ( DeltaR<reco::Candidate>()(*genHadb, *genLepb ) < theminGenDeltaR ) theminGenDeltaR = DeltaR<reco::Candidate>()(*genHadb, *genLepb );
			
				 
			// count gen events in the fidutial region
			if ( ( genHadWp->pt() > fMinJetPt ) &&
				 ( genHadWq->pt() > fMinJetPt ) &&
				 ( genHadb->pt()  > fMinJetPt ) &&
				 ( genLepb->pt()  > fMinJetPt ) ) {

				hcounter->Counter("GenJetPt");

				if ( ( fabs(genHadWp->eta()) < fMaxJetEta ) &&
					 ( fabs(genHadWq->eta()) < fMaxJetEta ) &&
					 ( fabs(genHadb->eta() ) < fMaxJetEta ) &&
					 ( fabs(genLepb->eta() ) < fMaxJetEta ) ) {

					hcounter->Counter("GenJetEta");

					// deltaR
					if ( theminGenDeltaR > 0.5 ) {

						hcounter->Counter("GenJetDeltaR");
										
						if ( ( fabs(genMuon->eta()) < fMaxMuonEta ) &&
							 ( genMuon->et()  > fMinMuonPt) ) {
							hcounter->Counter("GenJetDeltaRMuon");
							IsEventReconstructable = true;
						}
					}
					if ( ( fabs(genMuon->eta()) < fMaxMuonEta ) &&
						 ( genMuon->et()  > fMinMuonPt) ) hcounter->Counter("GenMuon");
				}
			}
						 
				 
		} else {
			edm::LogWarning ( "BooHighMAnalyzer" ) << " No top decay generator info, what happen here?";
			if (!genHadWp) edm::LogWarning ( "BooHighMAnalyzer" ) << " no genHadWp";
			if (!genHadWq) edm::LogWarning ( "BooHighMAnalyzer" ) << " no genHadWq";
			if (!genHadb)  edm::LogWarning ( "BooHighMAnalyzer" ) << " no genHadb";
			if (!genLepb)  edm::LogWarning ( "BooHighMAnalyzer" ) << " no genLepb";
			if (!genMuon)  edm::LogWarning ( "BooHighMAnalyzer" ) << " no genMuon";
			if (!genNu)    edm::LogWarning ( "BooHighMAnalyzer" ) << " no genNu";
			edm::LogWarning ( "BooHighMAnalyzer" ) << " skipping event, no counting this event ";
			//return;
		}
		}
		if (debug) std::cout << "done gen histo" << std::endl;
	} else {
	  if (fIsMCTop)
	    edm::LogWarning ( "BooHighMAnalyzer" ) << "no ttbar pair in generator";
	}

	// count events
	hcounter->Counter("Generator");
	
	// check relevant collection are not empty
	// ignore electrons
	bool emptymuons = false;
	bool emptyjets  = false;
	bool emptyMET   = false;
	
   if ( muons.size()==0 ) { emptymuons = true; edm::LogWarning ( "BooHighMAnalyzer" ) << " Muon collection: " << muonSrc << " is EMPTY.";}
   if ( jets.size() ==0 ) { emptyjets  = true; edm::LogWarning ( "BooHighMAnalyzer" ) << " Jets collection: " << jetSrc << " is EMPTY.";}
   if ( met.size()  ==0 ) { emptyMET   = true; edm::LogWarning ( "BooHighMAnalyzer" ) << " MET collection: " << metSrc << " is EMPTY.";}
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
	    // Muon ID
	   int nhit = 0;
	   if ( muons[imu].isTrackerMuon() ) nhit =  muons[imu].innerTrack()->numberOfValidHits();
	   double normChi2 = muons[imu].globalTrack()->chi2() / muons[imu].globalTrack()->ndof();
	   if ( nhit > 10 && normChi2 < 10 ) {

		   hmuons_->Fill1d("muon_pt_nohltgood", muonpt );
		   hmuons_->Fill1d("muon_eta_nohltgood", muoneta );
	   }
   }

   if (debug) std::cout << " HLT done" << std::endl;
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
   
   bool gotLeadingJet = false;

   for( size_t ijet=0; ijet != jets.size(); ++ijet) {

	   TLorentzVector tmpP4;
	   tmpP4.SetPxPyPzE(jets[ijet].px(),jets[ijet].py(),jets[ijet].pz(),jets[ijet].energy());

	   // recalculate my MET, remove soft jets that could be junk
	   if (tmpP4.Pt() > 20)
		   myMETP4 = myMETP4 + TLorentzVector(jets[ijet].px(),jets[ijet].py(),0,jets[ijet].pt());
	   
	   // jet cuts
	   if (jets[ijet].pt() <= fMinJetPt || fabs(jets[ijet].eta()) >= fMaxJetEta ) continue;

	   NgoodJets++;
	   	   
	   

	   if ( NgoodJets < 7 ) {
		   jetP4[NgoodJets-1] = tmpP4;
		   vect_bdiscriminators.push_back( jets[ijet].bDiscriminator("trackCountingHighEffBJetTags") );
	   }
	   
	   if( NgoodJets == 1 ) {
	     
	     if (jets[ijet].pt() > 240) gotLeadingJet = true;

		   if (debug) std::cout << "leading jet et: " << jets[ijet].et() << std::endl;

		   hjets_->Fill1d(TString("jet0_et")+"_"+"cut0", jets[ijet].et());
		   hjets_->Fill1d(TString("jet0_eta")+"_"+"cut0", jets[ijet].eta());
		   
		   // display jets//
		   /*
		   if (fdisplayJets) {
			   std::vector<CaloTowerPtr> jetCaloRefs = jets[ijet].getCaloConstituents();
			   if (debug) std::cout << " jetCaloRefs size = " << jetCaloRefs.size() << std::endl;
			   for ( size_t icalo=0; icalo != jetCaloRefs.size(); ++icalo ) {
				   if (debug) std::cout << " got icalo: " << "energy: "<< jetCaloRefs[icalo]->energy() << " eta: " << jetCaloRefs[icalo]->eta() <<  std::endl;
				   //if ( jetCaloRefs[icalo]->eta() < 1.740 )
				   hdisp_->Fill2d(TString("jet0_calotowerI")+"_cut0", jetCaloRefs[icalo]->eta(), jetCaloRefs[icalo]->phi(), jetCaloRefs[icalo]->energy() );
				   //else
				   //hjets_->Fill2d(TString("jet0_calotowerII")+"_cut0", jetCaloRefs[icalo]->eta(), jetCaloRefs[icalo]->phi(), jetCaloRefs[icalo]->energy() );				
			   }
		   }
		   */
	   }
	   if( NgoodJets == 2 ) {
		   
	
		   hjets_->Fill1d(TString("jet1_et")+"_"+"cut0", jets[ijet].et() );
		   hjets_->Fill1d(TString("jet1_eta")+"_"+"cut0", jets[ijet].eta());
		   
		   // display jets//
		   /*
		   if (fdisplayJets) {
			   std::vector<CaloTowerPtr> jetCaloRefs = jets[ijet].getCaloConstituents();
			   for ( size_t icalo=0; icalo != jetCaloRefs.size(); ++icalo ) {
				   hdisp_->Fill2d(TString("jet1_calotowerI")+"_cut0", jetCaloRefs[icalo]->eta(), jetCaloRefs[icalo]->phi(), jetCaloRefs[icalo]->energy() );
			   }
		   }
		   */
	   }
	   if( NgoodJets == 3 ) {

			   
		   hjets_->Fill1d(TString("jet2_et")+"_"+"cut0", jets[ijet].et() );
		   hjets_->Fill1d(TString("jet2_eta")+"_"+"cut0", jets[ijet].eta());
		   
		   /*
		   if (fdisplayJets) {
			   std::vector<CaloTowerPtr> jetCaloRefs = jets[ijet].getCaloConstituents();
			   for ( size_t icalo=0; icalo != jetCaloRefs.size(); ++icalo ) {
				   hdisp_->Fill2d(TString("jet2_calotowerI")+"_cut0", jetCaloRefs[icalo]->eta(), jetCaloRefs[icalo]->phi(), jetCaloRefs[icalo]->energy() );
			   }
		   }
		   */
	   }
	   if( NgoodJets == 4 ) {

			   
		   hjets_->Fill1d(TString("jet3_et")+"_"+"cut0", jets[ijet].et() );
		   hjets_->Fill1d(TString("jet3_eta")+"_"+"cut0", jets[ijet].eta());
		   
		   /*
		   if (fdisplayJets) {
			   std::vector<CaloTowerPtr> jetCaloRefs = jets[ijet].getCaloConstituents();
			   for ( size_t icalo=0; icalo != jetCaloRefs.size(); ++icalo ) {
				   hdisp_->Fill2d(TString("jet3_calotowerI")+"_cut0", jetCaloRefs[icalo]->eta(), jetCaloRefs[icalo]->phi(), jetCaloRefs[icalo]->energy() );
			   }
		   }
		   */
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

   if ( ! gotLeadingJet ) return;

   if (debug) std::cout << "Jet section done. Number of good jets: " << NgoodJets << std::endl;

   // remove this constraint for the moment to check Z and W +jets muon spectrum
   //if ( NgoodJets == 0 ) { nevents++; return; } 

   if ( NgoodJets >= 1 ) hcounter->Counter("Njets>1");
   if ( NgoodJets >= 4 ) hcounter->Counter("Njets>3");

   std::vector< TLorentzVector > vectorjets;
   size_t cutNgoodJets = NgoodJets;
   if (NgoodJets > 6) cutNgoodJets = 6; // use only 6 good jets
   std::vector< float > cutptjets;
   cutptjets.push_back(60); cutptjets.push_back(50); cutptjets.push_back(40); cutptjets.push_back(30);

   for( size_t ijet=0; ijet != cutNgoodJets; ++ijet) {

     if (fApplyJetAsymmetricCuts) {

       if ( (int) ijet < 4 ) {
	 if (jetP4[ijet].Pt() > cutptjets[ijet] ) vectorjets.push_back(jetP4[ijet]);
       } else {
	 vectorjets.push_back(jetP4[ijet]);
       }
     } else {
       vectorjets.push_back(jetP4[ijet]);
     }

   }
   // fix NgoodJets in case of asymmetric cuts

   NgoodJets = (int) vectorjets.size();

   ////////////////////////////////////////
   //
   // P R I M A R Y   V E R T E X
   //

   // use the first vertex in the collection
   // which is the vertex of the highest Pt of the associated tracks
   /*
   TVector3 thePV;

   if ( PVs.size() != 0 ) {

	   thePV = TVector3(PVs[0].x(),PVs[0].y(),PVs[0].z());

   }
   */
   
   ////////////////////////////////////////
   //
   // S E L E C T     M U O N S
   //

   int NGlobalMuons = 0;
   int NgoodMuons = 0;
   int NgoodMuonsID = 0;
   int NgoodIsoMuons = 0;
   
   int TotalMuons = muons.size();
   hmuons_->Fill1d(TString("muons")+"_cut0",TotalMuons);
   hmuons_->FillvsJets2d(TString("muons_vsJets")+"_cut0",TotalMuons, vectorjets);   
   int muonCharge = 0;
   double muonRelIso = 0;
   double muonVetoEm = 0;
   double muonVetoHad = 0;
    
   for( size_t imu=0; imu != muons.size(); ++imu) {

	   // require Global muons
	   if ( ! muons[imu].isGlobalMuon() ) continue;

	   NGlobalMuons++;
	   
	   double muonpt = muons[imu].pt(); // innerTrack()->pt(); // to test tracker muons
	   double muoneta= muons[imu].eta();
	   hmuons_->Fill1d("muon_pt_cut0", muonpt );
	   hmuons_->Fill1d("muon_eta_cut0", muoneta );

	   // Muon ID
	   int nhit = muons[imu].innerTrack()->numberOfValidHits();
	   double normChi2 = muons[imu].globalTrack()->chi2() / muons[imu].globalTrack()->ndof();
	   if ( nhit > 10 && normChi2 < 10 ) {
		   hmuons_->Fill1d("muon_pt_cut1", muonpt );
		   hmuons_->Fill1d("muon_eta_cut1", muoneta );
	   }
	   
	   if ( (muonpt > fMinMuonPt) && fabs(muons[imu].eta()) < fMaxMuonEta ) {

		   NgoodMuons++;
		   
	   
		   
		   // math::XYZPoint point(bSpot.x0()+bSpot.dxdz()*eleDZ0,bSpot.y0()+bSpot.dydz()*eleDZ0, bSpot.z0());
		   math::XYZPoint point(beamSpot.x0(),beamSpot.y0(), beamSpot.z0());
		   //double d0 = muons[imu].innerTrack()->d0();
		   double d0 = -1.* muons[imu].innerTrack()->dxy(point);
		   hmuons_->Fill2d("muon_phi_vs_d0_cut0", muons[imu].innerTrack()->phi(), muons[imu].innerTrack()->d0() );
		   hmuons_->Fill2d("muon_phi_vs_d0_cut1", muons[imu].innerTrack()->phi(), d0 );
		   double d0sigma = sqrt( muons[imu].innerTrack()->d0Error() * muons[imu].innerTrack()->d0Error() + beamSpot.BeamWidthX()*beamSpot.BeamWidthX());

		   hmuons_->Fill1d("muon_IPS_cut1", d0/d0sigma );

		   if ( nhit >= 11 && normChi2 < 10 && fabs(d0/d0sigma)<3 ) {

			   NgoodMuonsID++;
			   hmuons_->Fill1d("muon_pt_cut2", muonpt );
			   hmuons_->FillvsJets2d("muon_pt_vsJets_cut2",muonpt, vectorjets);   
			   //hmuons_->Fill1d("muon_d0_cut2", d0 );
			   
			   // ISOLATION	   
			   double RelIso = ( muonpt/(muonpt + muons[imu].caloIso() + muons[imu].trackIso()) );
			   //double RelIso = (muons[imu].caloIso())/muons[imu].et() + (muons[imu].trackIso())/muonpt;
			   hmuons_->Fill1d("muon_RelIso_cut2", RelIso);
			   
			   if ( RelIso <= fMuonRelIso ) {

				   NgoodIsoMuons++;
				   hmuons_->Fill1d("muon_pt_cut3", muonpt );
				   hmuons_->FillvsJets2d("muon_pt_vsJets_cut3",muonpt, vectorjets);
				   
				   //double energymu = sqrt(muons[imu].innerTrack()->px()*muons[imu].innerTrack()->px() +
				   //					  muons[imu].innerTrack()->py()*muons[imu].innerTrack()->py() +
				   //				  muons[imu].innerTrack()->pz()*muons[imu].innerTrack()->pz() + 0.1057*0.1057);

				   // pick the leading muon
				   if ( NgoodIsoMuons == 1 ) {
					   double energymu = muons[imu].energy();
					   muonP4.SetPxPyPzE(muons[imu].px(),muons[imu].py(),muons[imu].pz(),energymu );
					   muonCharge = muons[imu].charge();
					   muonRelIso = RelIso;

					   muonVetoEm = muons[imu].ecalIsoDeposit()->candEnergy();
					   muonVetoHad = muons[imu].hcalIsoDeposit()->candEnergy();

					   hmuons_->Fill1d("muon_vetoEm_cut3", muonVetoEm);
					   hmuons_->Fill1d("muon_vetoHad_cut3", muonVetoHad);

					 
					   if (muonVetoEm < fMaxMuonEm  && muonVetoHad < fMaxMuonHad ) {
						   hmuons_->Fill1d("muon_pt_cut4", muonpt );
						   hmuons_->FillvsJets2d("muon_pt_vsJets_cut4",muonpt, vectorjets);
					   }
					   
				   }

			   }
	   
		   }
	   }
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

	
   // select events with only ONE muon otherwise skip the event
   if ( NgoodIsoMuons != 1 ) {
	   nbadmuons++;
	   //edm::LogWarning ("BooHighMAnalyzer") << "Event with number of good muons: "<< NgoodIsoMuons << ", skip this event since we request one good muon.";
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
   
   if ( muonVetoEm < fMaxMuonEm  && muonVetoHad < fMaxMuonHad ) hjets_->Fill1d("jet_deltaR_muon_cut2", minDeltaR_muon_jet);
   
   //if (closestJet2.Et() > 10. ) hjets_->Fill1d(TString("jet_deltaR_muon")+"_cut2", minDeltaR_muon_jet);
   //if (closestEMFrac > 0.1 && closestEMFrac<0.9 ) hjets_->Fill1d(TString("jet_deltaR_muon")+"_cut3", minDeltaR_muon_jet );

   
   hjets_->Fill2d(TString("jet_deltaR_muon_vs_RelIso")+"_cut1", minDeltaR_muon_jet, muonRelIso);

   if (theJetClosestMu != -1 ) {
	   hjets_->Fill1d(TString("jet_pTrel_muon")+"_cut0",PtRel(muonP4,muonP4+closestJet));
	   hjets_->Fill1d(TString("jet_pT_closest_muon")+"_cut0", closestJet.Pt());
	   //hjets_->Fill1d(TString("jet_pT_closest_muon")+"_cut2", closestJet2.Pt());
	   hjets_->Fill1d(TString("jet_emFrac_cut1"), closestEMFrac );
	   
	   //if (closestJet2.Et() > 10. ) {
	   //	   hjets_->Fill1d(TString("jet_pT_closest_muon")+"_cut3", closestJet.Pt());
	   //	   hjets_->Fill1d(TString("jet_emFrac_cut2"), closestEMFrac );
	   //}
	   
	   int flavour = abs(jets[theJetClosestMu].partonFlavour());
	   hjets_->Fill1d(TString("jet_flavor_closest_muon")+"_cut0", flavour);
	   if ( flavour == 5 ) hjets_->Fill1d(TString("jet_pTrel_muon_b")+"_cut0",PtRel(muonP4,muonP4+closestJet));
	   if ( flavour == 4 ) hjets_->Fill1d(TString("jet_pTrel_muon_c")+"_cut0",PtRel(muonP4,muonP4+closestJet));
	   if ( (flavour < 4 && flavour != 0 )||(flavour == 21 ) ) hjets_->Fill1d(TString("jet_pTrel_muon_udsg")+"_cut0",PtRel(muonP4,muonP4+closestJet));

	   
   }
        
   //if ( minDeltaR_muon_jet <= 0.3 ) {
	   //return;
   //}

   if ( NgoodJets >= 4 && minDeltaR_muon_jet > 0.3 ) hcounter->Counter("DeltaR");

   //if ( muonVetoEm >= fMaxMuonEm || muonVetoHad >= fMaxMuonHad ) return;

   hjets_->Fill1d("jets_cut1",NgoodJets);
   
   if ( NgoodJets >= 4 && muonVetoEm < fMaxMuonEm  && muonVetoHad < fMaxMuonHad ) hcounter->Counter("muonVetoCone");

   
   if (debug) std::cout << "deltaR(muon,near jet) cut survive" << std::endl;

   
   // find pTRel
   if (theJetClosestMu != -1 ) {
	   hjets_->Fill1d(TString("jet_pTrel_muon")+"_cut1",PtRel(muonP4,muonP4+closestJet));
	   hjets_->Fill1d(TString("jet_pT_closest_muon")+"_cut1", closestJet.Pt());
	   int flavour = abs(jets[theJetClosestMu].partonFlavour());
	   hjets_->Fill1d(TString("jet_flavor_closest_muon")+"_cut1", flavour);
   }
   

   //////////////////////////////////////
   //
   // E L E C T R O N   R E M O V A L
   //
   /*
   int NgoodElectrons = 0;
   helectrons_->Fill1d("electrons_cut0", electrons.size() );
   
   for( size_t ie=0; ie != electrons.size(); ++ie) {

	   double ept = electrons[ie].pt();

	   //math::XYZPoint point(beamSpot.x0(),beamSpot.y0(), beamSpot.z0());
	   double ed0 = 0.;//-1.* electrons[ie].track()->dxy(point);
		   
	   helectrons_->Fill1d("electron_pt_cut0", ept );
	   helectrons_->Fill1d("electron_eta_cut0", electrons[ie].eta() );
	   //helectrons_->Fill2d("electron_phi_vs_d0_cut1", electrons[ie].track()->phi(), ed0 );

	   if ( ept > fMinElectronPt && fabs(electrons[ie].eta()) < fMaxElectronEta &&
		   electrons[ie].electronID("eidTight")>0) {

	     double relIso = ( 1./(1. + electrons[ie].caloIso()/electrons[ie].et() + electrons[ie].trackIso()/ept) );

	     //double relIso = electrons[ie].trackIso() /ept + electrons[ie].caloIso()/electrons[ie].et();
	     
		   if ( relIso > fElectronRelIso ) {

			   NgoodElectrons++;
		   }
	   }	
   }

   helectrons_->Fill1d("electrons_cut1", NgoodElectrons );
   if ( NgoodElectrons > 0 ) return;

   if ( NgoodJets >= 4 ) hcounter->Counter("NoElectrons");
   
   */

   ////////////////////////////////////
   //
   // M E T
   //
   
   // plot my MET
   hmet_->Fill1d(TString("myMET")+"_cut0", myMETP4.Pt());
   // correct my MET
   myMETP4 = myMETP4 + TLorentzVector(muonP4.Px(),muonP4.Py(),0,muonP4.Pt());
   hmet_->Fill1d(TString("myMET")+"_cut1", myMETP4.Pt());
   
   // met is corrected by muon momentum, how about muon energy?
   if (met.size() != 1 ) edm::LogWarning ("BooHighMAnalyzer") << "MET collection has size different from ONE! size: "<< met.size() << std::endl;
      
   for( size_t imet=0; imet != met.size(); ++imet) {
	   hmet_->Fill1d(TString("MET")+"_"+"cut0", met[imet].et());
	   hmet_->Fill1d(TString("MET_eta")+"_"+"cut0", met[imet].eta());
	   hmet_->Fill1d(TString("MET_phi")+"_"+"cut0", met[imet].phi());
	   hmet_->Fill1d(TString("MET_deltaR_muon")+"_"+"cut0", DeltaR<reco::Candidate>()( met[imet] , muons[0] ));
	   hmet_->FillvsJets2d(TString("MET_vsJets")+"_cut0",met[imet].et(), vectorjets);
	   hmet_->Fill1d(TString("Ht")+"_cut0", met[imet].sumEt());
	   hmet_->FillvsJets2d(TString("Ht_vsJets")+"_cut0",met[imet].sumEt(), vectorjets);
	   hmet_->Fill1d("MET_resolution_cut0",(met[imet].et() - met[imet].genMET()->et())/ met[imet].genMET()->et() );
   }

   METP4.SetPxPyPzE(met[0].px(), met[0].py(), met[0].pz(), met[0].energy());
   myMETP4 = (-1)*myMETP4;

   if (fUseMyMET) METP4 = myMETP4;
   
   if (debug) std::cout << "MET section done" << std::endl;
      
   if (NgoodJets >=4 ) {
     hmuons_->Fill2d("muon_RelIso_vs_MET_cut4", muonRelIso, METP4.Pt() );
     double Htl = muonP4.Pt();
     for( int ijet=0; ijet != NgoodJets; ++ijet) Htl += jetP4[ijet].Et();
     hmuons_->Fill2d("muon_RelIso_vs_Htl_cut4", muonRelIso, Htl );
   }

   ///////////////////////////////////////////////
   //
   // C A L C U L A T E   N E U T R I N O   Pz
   //
   
   double neutrinoPz = -999999.;
   bool found_nu = false;
   bool found_goodMET = false;
  
   if ( met.size()>0 ) {
	   
	   // Solving for neutrino Pz from W->mu+nu
	   METzCalculator zcalculator;
	   
	   zcalculator.SetMET( METP4 );
	   //zcalculator.SetMET(myMTP4);
	   zcalculator.SetMuon( muonP4 );//muons[0] );

	   if (debug) zcalculator.Print();
	   
	   neutrinoPz = zcalculator.Calculate(1);// 1 = closest to the lepton Pz, 3 = largest cosineCM

	   if (zcalculator.IsComplex()) {
		   nWcomplex += 1;
		   hcounter->Counter("ComplexPz");
	   }
	   
	   if (debug) std::cout << " reconstructed neutrino Pz = " << neutrinoPz << std::endl;

	   nuP4.SetPxPyPzE(met[0].px(), met[0].py(), neutrinoPz,
					   sqrt(met[0].px()*met[0].px()+met[0].py()*met[0].py()+neutrinoPz*neutrinoPz) );
	   //nuP4 = myMETP4 + TLorentzVector(0,0,neutrinoPz,neutrinoPz);
	   
	   hmuons_->Fill1d(TString("muon_deltaR_nu")+"_cut0", deltaR( muonP4.Eta(), muonP4.Phi(), nuP4.Eta(), nuP4.Phi() ));
	   hmuons_->Fill1d(TString("muon_deltaPhi_nu")+"_cut0", deltaPhi( muonP4.Phi(), nuP4.Phi() ));
	   hmet_->Fill1d(TString("nu_pz")+"_cut0",neutrinoPz);
	   hmet_->Fill1d(TString("nu_eta")+"_cut0",nuP4.Eta());
	   //hmet_->Fill1d(TString("delta_nu_pz")+"_cut0",(neutrinoPz - genNupz));
	   found_nu = true;

	   hmass_->Fill1d(TString("LeptonicW_mass")+"_cut1",(muonP4+nuP4).M());
	   if (zcalculator.IsComplex()) hmass_->Fill1d(TString("LeptonicW_mass")+"_cut2",(muonP4+nuP4).M());

	   hmet_->Fill1d(TString("LeptonicW_dij")+"_cut0",dij(muonP4, nuP4, 80.4));

	   
	   // apply psi cut to MET
	   //double LepW_psi = Psi(muonP4, nuP4, 80.4);
	   //hmet_->Fill1d(TString("LeptonicW_psi")+"_cut0",LepW_psi);	   
	   // select good MET
	   // first try using deltaR:
	   //if ( ROOT::Math::VectorUtil::DeltaPhi( muonP4.Vect(), nuP4.Vect() ) < 0.75 ) found_goodMET = true;
	   // second, using psi:
	   //if ( (muonP4+nuP4).M() < 100. && LepW_psi < 2 ) {

	   if ( (muonP4+nuP4).M() < 150. ) {
		   found_goodMET = true;
		   //hmet_->Fill1d(TString("delta_nu_pz")+"_cut1",(neutrinoPz - genNupz));
	   }

	   
   }
   
   if (debug) std::cout << "got neutrino? " << found_nu << std::endl;
   if (debug) std::cout << "got good MET? " << found_goodMET << std::endl;

   
   
   // write ASCII file if requested
   if (fwriteAscii) {
	   std::string dummyzero = " 0 ";
	   // dump generated neutrino
	   if (genNu) fasciiFile << "-1 " << genNu->p4().Px() <<" "<< genNu->p4().Py() <<" "<< genNu->p4().Pz() << dummyzero << std::endl;
	   else fasciiFile << "-1 0 0 0 0" << std::endl;
	   // dump Muon
	   fasciiFile << "-15 " << muonP4.Px() <<" "<< muonP4.Py() <<" "<< muonP4.Pz() << dummyzero << std::endl;
	   // dump Neutrino
	   fasciiFile << "-5 " << nuP4.Px() <<" "<< nuP4.Py() <<" "<< nuP4.Pz() << dummyzero << std::endl;
	   // dump jets with pt>20
	   for( size_t ijet=0; ijet != jets.size(); ++ijet) {

		   if (jets[ijet].pt() <= fMinJetPt || fabs(jets[ijet].eta()) >= fMaxJetEta ) continue;
		   
		   fasciiFile << jets[ijet].energy() <<" "<< jets[ijet].px() <<" "<<jets[ijet].py()<<" "<<jets[ijet].pz()<< " " << jets[ijet].bDiscriminator("trackCountingHighEffBJetTags") << std::endl;
		       
	   }
   }

   // find delta R of leptonic W with other jets
   //bool found_leadingJet = false;
   //size_t ith_leadingJet = 0;
   //TLorentzVector leadingP4;
     

   /////////////////////////////////////////////////////
   //
   //  T O P    R E C O N S T R U C T I O N
   //

   int topdecay = 0;
   if ( genEvent->isFullHadronic() ) topdecay =1;
   if ( genEvent->isFullLeptonic() ) topdecay =2;
   if ( genEvent->isSemiLeptonic() && genEvent->semiLeptonicChannel()==1 ) topdecay = 3;
   if ( genEvent->isSemiLeptonic() && genEvent->semiLeptonicChannel()==2 ) topdecay = 4;
   if ( genEvent->isSemiLeptonic() && genEvent->semiLeptonicChannel()==3 ) topdecay = 5;
   
   //hgen_->Fill1d("gen_top_decays", topdecay );
   hgen_->FillvsJets2d("gen_top_decays_vsJets", topdecay, vectorjets);
	   
   // OK now let's repeat the TOP loose selection
   if (NgoodJets >=4 ) {

	   hcounter->Counter("M3Selection");
	   
	   TLorentzVector lepWP4 = muonP4 + nuP4;
	   TLorentzVector hadWP4;

	   // do combinatorics
	   if (debug ) std::cout << "number of good jets " << NgoodJets << std::endl;

	   //ymyCombi0_.SetMaxNJets(4); // only 4 jets
	   myCombi0_.SetLeptonicW( lepWP4 );
	   //myCombi0_.Verbose();
	   myCombi0_.FourJetsCombinations(vectorjets,vect_bdiscriminators );
	   		   
	   Combo bestCombo = myCombi0_.GetCombinationSumEt(0);
	   //std::cout << "got BEST combination w/SumEt: " << std::endl;
	   //bestCombo.Print();
	   
	   hadWP4 = bestCombo.GetHadW();
	   hadTopP4 = bestCombo.GetHadTop();
	   lepTopP4 = bestCombo.GetLepTop();

	   if (debug) std::cout << " got best combination using SumEt" << std::endl;
	   
	   // fill plots, old prescription
	   //hjets_->Fill1d(TString("jet_combinations_ProbChi2_cut1"), TMath::Prob(bestCombo.GetChi2(),3));
	   //hjets_->Fill1d(TString("jet_combinations_NormChi2_cut1"), bestCombo.GetChi2()/3.);
	   //std::cout << "fill histos 1" << std::endl;
	   hmass_->Fill1d(TString("LeptonicTop_mass")+"_cut1", lepTopP4.M());
	   hmass_->Fill1d(TString("HadronicTop_mass")+"_cut1", hadTopP4.M());
	   hmass_->Fill1d(TString("HadronicW_mass")+"_cut1", hadWP4.M());
	   hmass_->Fill2d(TString("LepTop_vs_LepW")+"_cut1", lepTopP4.M(), lepWP4.M());
	   hmass_->Fill2d(TString("HadTop_vs_HadW")+"_cut1", hadTopP4.M(), hadWP4.M());

	   // number oftag jets MediumOP
	   int nbtags =0;
	   if (jets[ bestCombo.GetIdHadb() ].bDiscriminator("trackCountingHighEffBJetTags") > 4.38 ) nbtags++;
	   if (jets[ bestCombo.GetIdLepb() ].bDiscriminator("trackCountingHighEffBJetTags") > 4.38 ) nbtags++;
	   if (jets[ bestCombo.GetIdWq() ].bDiscriminator("trackCountingHighEffBJetTags") > 4.38 ) nbtags++;
	   if (jets[ bestCombo.GetIdWp() ].bDiscriminator("trackCountingHighEffBJetTags") > 4.38 ) nbtags++;
	   hjets_->Fill1d("number_bjets_cut1", nbtags );

	   // check MC truth
	   //std::cout << "check gen matching" << std::endl;
	   if (fIsMCTop) {
	     if ( IsTruthMatch(bestCombo, jets, *genEvent, true ) ) {
	       MCAllmatch_sumEt_++;
	       hcounter->Counter("M3MatchedAllJets");
	     }

		 if ( IsEventReconstructable ) {

			 hmass_->Fill1d("recLeptonicTop_mass_cut1", lepTopP4.M());
			 hmass_->Fill1d("recHadronicTop_mass_cut1", hadTopP4.M());
			 hmass_->Fill1d("recHadronicW_mass_cut1", hadWP4.M());
			 hmass_->Fill2d("recLepTop_vs_LepW_cut1", lepTopP4.M(), lepWP4.M());
			 hmass_->Fill2d("recHadTop_vs_HadW_cut1", hadTopP4.M(), hadWP4.M());
		 }

	     //std::cout << "check gen matching2"<< std::endl;
	     if ( IsTruthMatch(bestCombo, jets, *genEvent ) ) {
	       hjets_->Fill1d(TString("MCjet_combinations_ProbChi2_cut1"), TMath::Prob(bestCombo.GetChi2(),3));
	       hjets_->Fill1d(TString("MCjet_combinations_NormChi2_cut1"), bestCombo.GetChi2()/3.);
	       hmass_->Fill1d(TString("MCLeptonicTop_mass")+"_cut1", lepTopP4.M());
	       hmass_->Fill1d(TString("MCHadronicTop_mass")+"_cut1", hadTopP4.M());
	       hmass_->Fill1d(TString("MCHadronicW_mass")+"_cut1", hadWP4.M());
	       hmass_->Fill2d(TString("MCLepTop_vs_LepW")+"_cut1", lepTopP4.M(), lepWP4.M());
	       hmass_->Fill2d(TString("MCHadTop_vs_HadW")+"_cut1", hadTopP4.M(), hadWP4.M());
	       hcounter->Counter("M3MatchedJets");
	     }
	   }
	   
   }
   
   // now do my magic
   if ( found_goodMET) {
	   
	   if (NgoodJets >= 4 ) {

		   hcounter->Counter("M3Prime");
		   
		   TLorentzVector lepWP4 = muonP4 + nuP4;
		   TLorentzVector hadWP4;

		   // do combinatorics
		   if (debug ) std::cout << "number of good jets " << NgoodJets << std::endl;

		   myCombi_.SetLeptonicW( lepWP4 );
		   // remove top mass constraint
		   //myCombi_.UseMtopConstraint(false);
		   double Ndof = 3.;
		   // apply btagging
		   if (fUsebTagging) {
			   myCombi_.UsebTagging();
			   myCombi_.SetbTagPdf("/uscms/home/yumiceva/work/CMSSW_2_2_3/src/TopQuarkAnalysis/TopPairBSM/data/bdiscriminator.root");
			   Ndof = 7.;//btagging
		   }

		   myCombi_.FourJetsCombinations(vectorjets, vect_bdiscriminators ); // pass the b-tag dicriminators
		  		   
		   myCombi2_.SetLeptonicW( lepWP4 );
		   myCombi2_.SetMaxMassHadW( 110.);
		   myCombi2_.SetMaxMassLepW( 150.);

		   //extra
		   myCombi2_.SetMinMassLepTop( 150.);
		   myCombi2_.SetMaxMassLepTop( 230.);
		   // remove top mass constraint
		   //myCombi2_.UseMtopConstraint(false);

		   // btagging
		   //myCombi2_.UsebTagging();
		   //myCombi2_.SetbTagPdf("/uscms/home/yumiceva/work/CMSSW_2_2_3/src/TopQuarkAnalysis/TopPairBSM/data/bdiscriminator.root");
		   
		   
		   myCombi2_.FourJetsCombinations(vectorjets , vect_bdiscriminators );
		   
                   
		   //myCombi3_.SetLeptonicW( lepWP4 );
		   //myCombi3_.FourJetsCombinations(vectorjets);
		   //myCombi3_.SetMaxMassHadW( 110.);
		   //myCombi3_.SetMaxMassLepW( 150.);
		   //myCombi3_.SetMinMassLepTop( 150.);
		   //myCombi3_.SetMaxMassLepTop( 220.);
		   
		   Combo bestCombo;
		   		   
		   for ( int icombo=0; icombo< myCombi_.GetNumberOfCombos(); ++icombo) {

			   if (debug) std::cout << " combination # " << icombo << std::endl;
			   
			   if (icombo==0) bestCombo = myCombi_.GetCombination(0);
			   			   
			   Combo tmpCombo = myCombi_.GetCombination(icombo);
			   
			   hadWP4 = tmpCombo.GetHadW();
			   hadTopP4 = tmpCombo.GetHadTop();
			   lepTopP4 = tmpCombo.GetLepTop();

			   // fill plots, all solutions
			   if (debug ) {
				   std::cout << "combination chi2 = " << tmpCombo.GetChi2()
							 << " hadWP4.M() = " << hadWP4.M() << " hadTopP4.M() = " << hadTopP4.M() << std::endl;
			   }
			   hjets_->Fill1d(TString("jet_combinations_ProbChi2_cut0"), TMath::Prob(tmpCombo.GetChi2(),3));
			   hjets_->Fill1d(TString("jet_combinations_NormChi2_cut0"), tmpCombo.GetChi2()/Ndof);
			   hmass_->Fill1d(TString("LeptonicTop_mass")+"_cut0", lepTopP4.M());
			   hmass_->Fill1d(TString("HadronicTop_mass")+"_cut0", hadTopP4.M());
			   hmass_->Fill1d(TString("HadronicW_mass")+"_cut0", hadWP4.M());
			   hmass_->Fill2d(TString("LepTop_vs_LepW")+"_cut0", lepTopP4.M(), lepWP4.M());
			   hmass_->Fill2d(TString("HadTop_vs_HadW")+"_cut0", hadTopP4.M(), hadWP4.M());
		   }

		   			   
		   hadWP4 = bestCombo.GetHadW();
		   hadTopP4 = bestCombo.GetHadTop();
		   lepTopP4 = bestCombo.GetLepTop();

		   // debug
		   //bestCombo.Print();
		   
		   if (hadWP4.M() < 5. ) {
			   std::cout << " had. W mass !! " << hadWP4.M() << std::endl;
			   std::cout << "best sol. had W mass = " << hadWP4.M() << " for Ngoodjets = " << NgoodJets << std::endl;
			   std::cout << "number of combinations = " << myCombi_.GetNumberOfCombos() << std::endl;
		   }
		   // fill plots, best chi-square
		   hjets_->Fill1d(TString("jet_combinations_ProbChi2_cut2"), TMath::Prob(bestCombo.GetChi2(),3));
		   hjets_->Fill1d(TString("jet_combinations_NormChi2_cut2"), bestCombo.GetChi2()/Ndof);
		   hmass_->Fill1d(TString("LeptonicTop_mass")+"_cut2", lepTopP4.M());
		   hmass_->Fill1d(TString("HadronicTop_mass")+"_cut2", hadTopP4.M());
		   hmass_->Fill1d(TString("HadronicW_mass")+"_cut2", hadWP4.M());
		   hmass_->Fill2d(TString("LepTop_vs_LepW")+"_cut2", lepTopP4.M(), lepWP4.M());
		   hmass_->Fill2d(TString("HadTop_vs_HadW")+"_cut2", hadTopP4.M(), hadWP4.M());

		   hjets_->Fill1d("jet_Hadb_disc_cut2", jets[ bestCombo.GetIdHadb() ].bDiscriminator("trackCountingHighEffBJetTags") );
		   hjets_->Fill1d("jet_Lepb_disc_cut2", jets[ bestCombo.GetIdLepb() ].bDiscriminator("trackCountingHighEffBJetTags") );
		   hjets_->Fill1d("jet_Hadb_flavor_cut2", jets[ bestCombo.GetIdHadb() ].partonFlavour() );
		   hjets_->Fill1d("jet_Lepb_flavor_cut2", jets[ bestCombo.GetIdLepb() ].partonFlavour() );

		   // number of tag jets Medium OP
		   int nbtags = 0;
		   if ( jets[ bestCombo.GetIdHadb() ].bDiscriminator("trackCountingHighEffBJetTags") > 4.38 ) nbtags++;
		   if (jets[ bestCombo.GetIdLepb() ].bDiscriminator("trackCountingHighEffBJetTags") > 4.38 ) nbtags++;
		   if (jets[ bestCombo.GetIdWq() ].bDiscriminator("trackCountingHighEffBJetTags") > 4.38 ) nbtags++;
		   if (jets[ bestCombo.GetIdWp() ].bDiscriminator("trackCountingHighEffBJetTags") > 4.38 ) nbtags++;
		   hjets_->Fill1d("number_bjets_cut2", nbtags );

		   ///////////////
		   // Wiggle jets four-momentum
		   TLorentzVector *newWp = new TLorentzVector(bestCombo.GetWp().Px(),
													  bestCombo.GetWp().Py(),
													  bestCombo.GetWp().Pz(),
													  bestCombo.GetWp().E() );
		   TLorentzVector *newWq = new TLorentzVector(bestCombo.GetWq().Px(),
													  bestCombo.GetWq().Py(),
													  bestCombo.GetWq().Pz(),
													  bestCombo.GetWq().E() );
		   //std::cout << "newWp pt = " << newWp->Pt() << std::endl;
		   //std::cout << "newWq pt = " << newWq->Pt() << std::endl;
		   
		   double resolution1 = sqrt( 1.11*1.11/newWp->Pt() + 0.03*0.03 )*(newWp->Pt());
		   double resolution2 = sqrt( 1.11*1.11/newWq->Pt() + 0.03*0.03 )*(newWq->Pt());
		   double Wsigmas = tools.fix4VectorsForMass( *newWp, *newWq, 79.8, resolution1, resolution2,
													 resolution1, resolution2);
		   
		   
		   //std::cout << "newWp pt = " << newWp->Pt() << std::endl;
		   //std::cout << "newWq pt = " << newWq->Pt() << std::endl;

		   TLorentzVector fitHadW = TLorentzVector(newWp->Px(),newWp->Py(),newWp->Pz(),newWp->E()) +
			   TLorentzVector(newWq->Px(),newWq->Py(),newWq->Pz(),newWq->E());
		   TLorentzVector fitHadTop = fitHadW + bestCombo.GetHadb();

		   hmass_->Fill1d(TString("fitHadronicW_mass")+"_cut2", fitHadW.M());
		   hmass_->Fill1d(TString("fitHadronicTop_mass")+"_cut2", fitHadTop.M());
		   hmass_->Fill1d(TString("fittopPair_cut2"), (fitHadTop+lepTopP4).M() );
		   
		   delete newWp;
		   delete newWq;

		   if ( fIsMCTop ) {
		     if ( IsTruthMatch(bestCombo, jets, *genEvent, true) ) {
		       MCAllmatch_chi2_++;
		       hcounter->Counter("M3PrimeMatchedAllJets");
		      
		     }

			 if ( IsEventReconstructable ) {

				 hmass_->Fill1d("recLeptonicTop_mass_cut2", lepTopP4.M());
				 hmass_->Fill1d("recHadronicTop_mass_cut2", hadTopP4.M());
				 hmass_->Fill1d("recHadronicW_mass_cut2", hadWP4.M());
				 hmass_->Fill2d("recLepTop_vs_LepW_cut2", lepTopP4.M(), lepWP4.M());
				 hmass_->Fill2d("recHadTop_vs_HadW_cut2", hadTopP4.M(), hadWP4.M());

				 hmass_->Fill1d("res_fitHadronicTop-recHadronicTop_mass_cut2", fitHadTop.M() - hadTopP4.M() );

			 }

		     if ( IsTruthMatch(bestCombo, jets, *genEvent) ) {
		     
		       hjets_->Fill1d(TString("MCjet_combinations_ProbChi2_cut2"), TMath::Prob(bestCombo.GetChi2(),3));
		       hjets_->Fill1d(TString("MCjet_combinations_NormChi2_cut2"), bestCombo.GetChi2()/Ndof);
		       hmass_->Fill1d(TString("MCLeptonicTop_mass")+"_cut2", lepTopP4.M());
		       hmass_->Fill1d(TString("MCHadronicTop_mass")+"_cut2", hadTopP4.M());
		       hmass_->Fill1d(TString("MCHadronicW_mass")+"_cut2", hadWP4.M());
		       hmass_->Fill2d(TString("MCLepTop_vs_LepW")+"_cut2", lepTopP4.M(), lepWP4.M());
		       hmass_->Fill2d(TString("MCHadTop_vs_HadW")+"_cut2", hadTopP4.M(), hadWP4.M());

			   hjets_->Fill1d("jet_Wmass_sigmas_cut2", Wsigmas );
			   hmass_->Fill1d("res_fitHadronicTop-MCHadronicTop_mass_cut2", fitHadTop.M() - hadTopP4.M() );

		       hcounter->Counter("M3PrimeMatchedJets");
		     }
		   }
		   
		   //Combinatorics - 3rd chi-square
		   int theNthCombinatorics = 2; // counting from zero
		   Combo Combo3 = myCombi_.GetCombination(theNthCombinatorics);

		   hadWP4 = Combo3.GetHadW();
		   hadTopP4 = Combo3.GetHadTop();
		   lepTopP4 = Combo3.GetLepTop();

		   // fill plots for combinatorics, 3rd chi-square solution
		   hjets_->Fill1d(TString("jet_combinations_ProbChi2_cut3"), TMath::Prob(Combo3.GetChi2(),3));
		   hjets_->Fill1d(TString("jet_combinations_NormChi2_cut3"), Combo3.GetChi2()/Ndof);
		   hmass_->Fill1d(TString("LeptonicTop_mass")+"_cut3", lepTopP4.M());
		   hmass_->Fill1d(TString("HadronicTop_mass")+"_cut3", hadTopP4.M());
		   hmass_->Fill1d(TString("HadronicW_mass")+"_cut3", hadWP4.M());
		   hmass_->Fill2d(TString("LepTop_vs_LepW")+"_cut3", lepTopP4.M(), lepWP4.M());
		   hmass_->Fill2d(TString("HadTop_vs_HadW")+"_cut3", hadTopP4.M(), hadWP4.M());

		   if ( fIsMCTop ) {
		     if ( IsTruthMatch(Combo3, jets, *genEvent) ) {

		       hjets_->Fill1d(TString("MCjet_combinations_ProbChi2_cut3"), TMath::Prob(Combo3.GetChi2(),3));
		       hjets_->Fill1d(TString("MCjet_combinations_NormChi2_cut3"), Combo3.GetChi2()/Ndof);
		       hmass_->Fill1d(TString("MCLeptonicTop_mass")+"_cut3", lepTopP4.M());
		       hmass_->Fill1d(TString("MCHadronicTop_mass")+"_cut3", hadTopP4.M());
		       hmass_->Fill1d(TString("MCHadronicW_mass")+"_cut3", hadWP4.M());
		       hmass_->Fill2d(TString("MCLepTop_vs_LepW")+"_cut3", lepTopP4.M(), lepWP4.M());
		       hmass_->Fill2d(TString("MCHadTop_vs_HadW")+"_cut3", hadTopP4.M(), hadWP4.M());

		     }
		   }

		   // apply a cut on W had and lep mass

		   // best chi-square + cuts
		   Combo bestComboCut = myCombi2_.GetCombination(0);

		   hadWP4 = bestComboCut.GetHadW();
		   hadTopP4 = bestComboCut.GetHadTop();
		   lepTopP4 = bestComboCut.GetLepTop();

		   hjets_->Fill1d(TString("jet_combinations_ProbChi2_cut4"), TMath::Prob(bestComboCut.GetChi2(),3));
		   hjets_->Fill1d(TString("jet_combinations_NormChi2_cut4"), bestComboCut.GetChi2()/Ndof);
		   hmass_->Fill1d(TString("LeptonicTop_mass")+"_cut4", lepTopP4.M());
		   hmass_->Fill1d(TString("HadronicTop_mass")+"_cut4", hadTopP4.M());
		   hmass_->Fill1d(TString("HadronicW_mass")+"_cut4", hadWP4.M());
		   hmass_->Fill2d(TString("LepTop_vs_LepW")+"_cut4", lepTopP4.M(), lepWP4.M());
		   hmass_->Fill2d(TString("HadTop_vs_HadW")+"_cut4", hadTopP4.M(), hadWP4.M());

		   if ( fIsMCTop ) {
		     if ( IsTruthMatch(bestComboCut, jets, *genEvent) ) {
		       hjets_->Fill1d(TString("MCjet_combinations_ProbChi2_cut4"), TMath::Prob(bestComboCut.GetChi2(),3));
		       hjets_->Fill1d(TString("MCjet_combinations_NormChi2_cut4"), bestComboCut.GetChi2()/3.);
		       hmass_->Fill1d(TString("MCLeptonicTop_mass")+"_cut4", lepTopP4.M());
		       hmass_->Fill1d(TString("MCHadronicTop_mass")+"_cut4", hadTopP4.M());
		       hmass_->Fill1d(TString("MCHadronicW_mass")+"_cut4", hadWP4.M());
		       hmass_->Fill2d(TString("MCLepTop_vs_LepW")+"_cut4", lepTopP4.M(), lepWP4.M());
		       hmass_->Fill2d(TString("MCHadTop_vs_HadW")+"_cut4", hadTopP4.M(), hadWP4.M());

		     }
		   }

		   hmass_->Fill1d(TString("topPair_cut4"), (hadTopP4+lepTopP4).M() );

		   //Combinatorics - 3rd chi-squre solution + cuts

		   Combo Combo3Cut = myCombi2_.GetCombination(theNthCombinatorics);

		   hadWP4 = Combo3Cut.GetHadW();
		   hadTopP4 = Combo3Cut.GetHadTop();
		   lepTopP4 = Combo3Cut.GetLepTop();

		   hjets_->Fill1d(TString("jet_combinations_ProbChi2_cut5"), TMath::Prob(Combo3Cut.GetChi2(),3));
		   hjets_->Fill1d(TString("jet_combinations_NormChi2_cut5"), Combo3Cut.GetChi2()/3.);
		   hmass_->Fill1d(TString("LeptonicTop_mass")+"_cut5", lepTopP4.M());
		   hmass_->Fill1d(TString("HadronicTop_mass")+"_cut5", hadTopP4.M());
		   hmass_->Fill1d(TString("HadronicW_mass")+"_cut5", hadWP4.M());
		   hmass_->Fill2d(TString("LepTop_vs_LepW")+"_cut5", lepTopP4.M(), lepWP4.M());
		   hmass_->Fill2d(TString("HadTop_vs_HadW")+"_cut5", hadTopP4.M(), hadWP4.M());

		   if ( fIsMCTop ) {
		     if ( IsTruthMatch(Combo3Cut, jets, *genEvent) ) {
		       hjets_->Fill1d(TString("MCjet_combinations_ProbChi2_cut5"), TMath::Prob(Combo3Cut.GetChi2(),3));
		       hjets_->Fill1d(TString("MCjet_combinations_NormChi2_cut5"), Combo3Cut.GetChi2()/3.);
		       hmass_->Fill1d(TString("MCLeptonicTop_mass")+"_cut5", lepTopP4.M());
		       hmass_->Fill1d(TString("MCHadronicTop_mass")+"_cut5", hadTopP4.M());
		       hmass_->Fill1d(TString("MCHadronicW_mass")+"_cut5", hadWP4.M());
		       hmass_->Fill2d(TString("MCLepTop_vs_LepW")+"_cut5", lepTopP4.M(), lepWP4.M());
		       hmass_->Fill2d(TString("MCHadTop_vs_HadW")+"_cut5", hadTopP4.M(), hadWP4.M());
		       
		     }
		   }
	   }


	   /*

	   TLorentzVector lepWP4 = muonP4 + nuP4;
	   lepTopP4 = lepTopP4 + lepWP4;
	   
	   for( size_t ijet=0; ijet != jets.size(); ++ijet) {

		   if (jets[ijet].pt() <= 20. ) continue;
		   
		   TLorentzVector tmpP4;
		   tmpP4.SetPxPyPzE(jets[ijet].px(),jets[ijet].py(),jets[ijet].pz(),jets[ijet].energy());

		   double deltaR_jLepW = ROOT::Math::VectorUtil::DeltaR( tmpP4.Vect(), lepWP4.Vect() );
		   hjets_->Fill1d(TString("jet_deltaR_LeptonicW")+"_cut0", deltaR_jLepW);

		   if ( !found_leadingJet && deltaR_jLepW>2. ) {
			   found_leadingJet = true;
			   leadingP4 = tmpP4;
			   ith_leadingJet = ijet;
			   if (debug) std::cout << "found leading jet" <<std::endl;
		   }

	   }

	   if (debug) std::cout << "leading jet ("<<leadingP4.Px() <<","<<leadingP4.Py()<<","<<leadingP4.Pz()<<","<<leadingP4.E()<<")"<<std::endl;
	   if (debug) std::cout << "lepW jet ("<<lepWP4.Px() <<","<<lepWP4.Py()<<","<<lepWP4.Pz()<<","<<lepWP4.E()<<")"<<std::endl;
	   
	   if ( found_leadingJet ) {
	     //add leading jet to hadronic top
	     hadTopP4 += leadingP4;

		   for( size_t ijet=0; ijet != jets.size(); ++ijet) {

			   if (jets[ijet].pt() <= 20. ) continue;
			   
		     if (ith_leadingJet != ijet) {
			   TLorentzVector tmpP4;
			   tmpP4.SetPxPyPzE(jets[ijet].px(),jets[ijet].py(),jets[ijet].pz(),jets[ijet].energy());
			   
			   double psi_LepTop = Psi(tmpP4, lepWP4, 175.0 );
			   if (debug) std::cout << "psi_LepTop= " << psi_LepTop << std::endl;
			   double psi_HadTop = Psi(tmpP4, leadingP4, 175.0 );
			   if (debug) std::cout << "psi_HadTop= " << psi_HadTop << std::endl;
			   
			   hjets_->Fill1d(TString("LeptonicTop_psi")+"_cut0", psi_LepTop);
			   hjets_->Fill1d(TString("HadronicTop_psi")+"_cut0", psi_HadTop);

			   if ( psi_LepTop < psi_HadTop ) lepTopP4 += tmpP4;
			   else hadTopP4 += tmpP4;
			   
			   //if ( psi_LepTop>=0 && psi_HadTop>=0) {
			   // if ( psi_LepTop < psi_HadTop ) lepTopP4 += tmpP4;
			   // else hadTopP4 += tmpP4;
			   //}
			   //if ( psi_LepTop < 0 && psi_HadTop >= 0 ) hadTopP4 += tmpP4;
			   //if ( psi_LepTop >= 0 && psi_HadTop < 0 ) lepTopP4 += tmpP4;
		     }
		   }
	   }

	   if (debug) {
		   std::cout << "leptonic top pt= " << lepTopP4.Pt() << " mass= " << lepTopP4.M() << std::endl;
		   std::cout << "hadronic top pt= " << hadTopP4.Pt() << " mass= " << hadTopP4.M() <<std::endl;
	   }
	   
	   hjets_->Fill1d(TString("LeptonicTop_pt")+"_cut0", lepTopP4.Pt());
	   hjets_->Fill1d(TString("HadronicTop_pt")+"_cut0", hadTopP4.Pt());
	   if (debug) std::cout << "done pt" << std::endl;
	   
	   hmass_->Fill1d(TString("LeptonicTop_mass")+"_cut1", lepTopP4.M());
	   hmass_->Fill1d(TString("HadronicTop_mass")+"_cut1", hadTopP4.M());
	   if (debug) std::cout << "done mass" << std::endl;
	   
	   topPairP4 = hadTopP4+lepTopP4;
	   
	   hmass_->Fill1d(TString("topPair")+"_cut1", topPairP4.M());
	   */
	   
	   if (debug) std::cout << "done." << std::endl;
	   
   }// found good MET
  
 
	// count all events
   nevents++;
}


//define this as a plug-in
DEFINE_FWK_MODULE(BooHighMAnalyzer);
