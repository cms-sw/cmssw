// Producer for validation histograms for PFJet objects
// J. Weng, based on F. Ratnikov, Sept. 7, 2006

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "RecoJets/JetAlgorithms/interface/JetMatchingTools.h"

#include "PFJetTester.h"

using namespace edm;
using namespace reco;
using namespace std;
namespace {
  bool is_B (const reco::Jet& fJet) {return fabs (fJet.eta()) < 1.4;}
  bool is_E (const reco::Jet& fJet) {return fabs (fJet.eta()) >= 1.4 && fabs (fJet.eta()) < 3.;}
  bool is_F (const reco::Jet& fJet) {return fabs (fJet.eta()) >= 3.;}
}

PFJetTester::PFJetTester(const edm::ParameterSet& iConfig)
  : mInputCollection (iConfig.getParameter<edm::InputTag>( "src" )), 
    mInputGenCollection (iConfig.getParameter<edm::InputTag>( "srcGen" )),
    mOutputFile (iConfig.getUntrackedParameter<string>("outputFile", "")),
    mMatchGenPtThreshold (iConfig.getParameter<double>("genPtThreshold")),
    mGenEnergyFractionThreshold (iConfig.getParameter<double>("genEnergyFractionThreshold")),
    mReverseEnergyFractionThreshold (iConfig.getParameter<double>("reverseEnergyFractionThreshold")),
    mRThreshold (iConfig.getParameter<double>("RThreshold"))
{

  //**** @@@
  mEta = mPhi = mE = mP = mPt = mMass = mConstituents
    = mEtaFirst = mPhiFirst = mEFirst = mPtFirst 
    =mChargedHadronEnergy
    = mNeutralHadronEnergy = mChargedEmEnergy = mChargedMuEnergy
    = mNeutralEmEnergy =  mChargedMultiplicity = mNeutralMultiplicity
    = mMuonMultiplicity =   mAllGenJetsPt = mMatchedGenJetsPt = mAllGenJetsEta = mMatchedGenJetsEta 
    = mGenJetMatchEnergyFraction = mReverseMatchEnergyFraction = mRMatch
    = mDeltaEta = mDeltaPhi = mEScale = mDeltaE= 0;
  //cout<<" PFJetTester -----------------------------------------------------> 1"<<endl;//////////////////////
  //cout<<"setting all variables zero -----------------------------------------------------> 2"<<endl;//////////////////////
  DQMStore* dbe = &*edm::Service<DQMStore>();
  if (dbe) {
    //cout<<" Creating the histograms -----------------------------------------------------> 3"<<endl;//////////////////////
    dbe->setCurrentFolder("RecoJetsV/PFJetTask_" + mInputCollection.label());
    mEta = dbe->book1D("Eta", "Eta", 100, -5, 5); 
    mPhi = dbe->book1D("Phi", "Phi", 70, -3.5, 3.5); 
    mE = dbe->book1D("E", "E", 100, 0, 500); 
    mP = dbe->book1D("P", "P", 100, 0, 500); 
    mPt = dbe->book1D("Pt", "Pt", 100, 0, 50); 
    mMass = dbe->book1D("Mass", "Mass", 100, 0, 25); 
    mConstituents = dbe->book1D("Constituents", "# of Constituents", 100, 0, 100); 
    //
    mEtaFirst = dbe->book1D("EtaFirst", "EtaFirst", 100, -5, 5); 
    mPhiFirst = dbe->book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5); 
    mEFirst = dbe->book1D("EFirst", "EFirst", 100, 0, 1000); 
    mPtFirst = dbe->book1D("PtFirst", "PtFirst", 100, 0, 500); 
    //
    mChargedHadronEnergy = dbe->book1D("mChargedHadronEnergy", "mChargedHadronEnergy", 100, 0, 100); 
    mNeutralHadronEnergy = dbe->book1D("mNeutralHadronEnergy", "mNeutralHadronEnergy", 100, 0, 100);  
    mChargedEmEnergy= dbe->book1D("mChargedEmEnergy ", "mChargedEmEnergy ", 100, 0, 100); 
    mChargedMuEnergy = dbe->book1D("mChargedMuEnergy", "mChargedMuEnergy", 100, 0, 100); 
    mNeutralEmEnergy= dbe->book1D("mNeutralEmEnergy", "mNeutralEmEnergy", 100, 0, 100);   
    mChargedMultiplicity= dbe->book1D("mChargedMultiplicity ", "mChargedMultiplicity ", 100, 0, 100);     
    mNeutralMultiplicity = dbe->book1D(" mNeutralMultiplicity", "mNeutralMultiplicity", 100, 0, 100);
    mMuonMultiplicity= dbe->book1D("mMuonMultiplicity", "mMuonMultiplicity", 100, 0, 100);
    double log10PtMin = 0.5;
    double log10PtMax = 4.;
    int log10PtBins = 14;
    double etaMin = -5.;
    double etaMax = 5.;
    int etaBins = 50;
    mAllGenJetsPt = dbe->book1D("GenJetLOGpT", "GenJet LOG(pT_gen)", 
				log10PtBins, log10PtMin, log10PtMax);
    mMatchedGenJetsPt = dbe->book1D("MatchedGenJetLOGpT", "MatchedGenJet LOG(pT_gen)", 
				    log10PtBins, log10PtMin, log10PtMax);
    mAllGenJetsEta = dbe->book2D("GenJetEta", "GenJet Eta vs LOG(pT_gen)", 
				 log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax);
    mMatchedGenJetsEta = dbe->book2D("MatchedGenJetEta", "MatchedGenJet Eta vs LOG(pT_gen)", 
				     log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax);
    mGenJetMatchEnergyFraction  = dbe->book3D("GenJetMatchEnergyFraction", "GenJetMatchEnergyFraction vs LOG(pT_gen) vs eta", 
					      log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 101, 0, 1.01);
    mReverseMatchEnergyFraction  = dbe->book3D("ReverseMatchEnergyFraction", "ReverseMatchEnergyFraction vs LOG(pT_gen) vs eta", 
					       log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 101, 0, 1.01);
    mRMatch  = dbe->book3D("RMatch", "delta(R)(Gen-Calo) vs LOG(pT_gen) vs eta", 
			   log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 60, 0, 3);
    mDeltaEta = dbe->book3D("DeltaEta", "DeltaEta vs LOG(pT_gen) vs eta", 
			    log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 100, -1, 1);
    mDeltaPhi = dbe->book3D("DeltaPhi", "DeltaPhi vs LOG(pT_gen) vs eta", 
			    log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 100, -1, 1);
    mEScale = dbe->book3D("EScale", "EnergyScale vs LOG(pT_gen) vs eta", 
			  log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 100, 0, 2);
    mDeltaE = dbe->book3D("DeltaE", "DeltaE vs LOG(pT_gen) vs eta", 
			  log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 2000, -200, 200);

    //__________________________________________________

    mNeutralFraction = dbe->book1D("NeutralFraction","Neutral Fraction",100,0,1);
    mEEffNeutralFraction = dbe->book1D("EEffNeutralFraction","E Efficiency vs Neutral Fraction",100,0,1);    //_________________(*)
    mEEffChargedFraction = dbe->book1D("EEffChargedFraction","E Efficiency vs Charged Fraction",100,0,1); 
    mEResNeutralFraction = dbe->book1D("EResNeutralFraction","E Resolution vs Neutral Fraction",100,0,1);
    mEResChargedFraction = dbe->book1D("EResChargedFraction","E Resolution vs Charged Fraction",100,0,1); 
    nEEff = dbe->book1D("nEEff","EEff",100,0,1);
    mNeutralFraction2 = dbe->book1D("NeutralFraction2","Neutral Fraction 2",100,0,1);
    
  }
  if (mOutputFile.empty ()) {
    LogInfo("OutputInfo") << " PFJet histograms will NOT be saved";
  } 
  else {
    LogInfo("OutputInfo") << " PFJethistograms will be saved to file:" << mOutputFile;
  }
}
   
PFJetTester::~PFJetTester()
{
}

void PFJetTester::beginJob(const edm::EventSetup& c){
}

void PFJetTester::endJob() {

  if (!mOutputFile.empty() && &*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (mOutputFile);
  //cout<<" endjob ----------------------------------------------------->"<<endl;//////////////////////
}


void PFJetTester::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup)
{
  //cout<<" start analyze -----------------------------------------------------> 4"<<endl;//////////////////////
  Handle<PFJetCollection> pfJets;
  //cout<<"  -----------------------------------------------------> 4a"<<endl;//////////////////////
  mEvent.getByLabel(mInputCollection, pfJets);
  //  cout <<"!!!!!!!!!!! " <<mInputCollection << " " <<collection->size() <<endl;
  PFJetCollection::const_iterator jet = pfJets->begin ();

  //cout<<"filling the histogramms ------------------------------------------------------------------> 5"<<endl;
  for (; jet != pfJets->end (); jet++) {
    if (mEta) mEta->Fill (jet->eta());
    //cout<<jet->eta()<<" ------------------------------------------------------------------> 6"<<endl;
    if (mPhi) mPhi->Fill (jet->phi());
    if (mE) mE->Fill (jet->energy());
    if (mP) mP->Fill (jet->p());
    if (mPt) mPt->Fill (jet->pt());
    if (mMass) mMass->Fill (jet->mass());
    if (mConstituents) mConstituents->Fill (jet->nConstituents());
    if (jet == pfJets->begin ()) { // first jet
      if (mEtaFirst) mEtaFirst->Fill (jet->eta());
      if (mPhiFirst) mPhiFirst->Fill (jet->phi());
      if (mEFirst) mEFirst->Fill (jet->energy());
      if (mPtFirst) mPtFirst->Fill (jet->pt());
    }
    if (mChargedHadronEnergy)  mChargedHadronEnergy->Fill (jet->chargedHadronEnergy());
    if (mNeutralHadronEnergy)  mNeutralHadronEnergy->Fill (jet->neutralHadronEnergy());
    if (mChargedEmEnergy) mChargedEmEnergy->Fill(jet->chargedEmEnergy());
    if (mChargedMuEnergy) mChargedMuEnergy->Fill (jet->chargedMuEnergy ());
    if (mNeutralEmEnergy) mNeutralEmEnergy->Fill(jet->neutralEmEnergy()); 
    if (mChargedMultiplicity ) mChargedMultiplicity->Fill(jet->chargedMultiplicity());
    if (mNeutralMultiplicity ) mNeutralMultiplicity->Fill(jet->neutralMultiplicity());
    if (mMuonMultiplicity )mMuonMultiplicity->Fill (jet-> muonMultiplicity());
    //_______________________________________________________
    if (mNeutralFraction) mNeutralFraction->Fill (jet->neutralMultiplicity()/jet->nConstituents());
  } 

  // now match PFJets to GenJets
  JetMatchingTools jetMatching (mEvent);
  if (!(mInputGenCollection.label().empty())) {
    Handle<GenJetCollection> genJets;
    mEvent.getByLabel(mInputGenCollection, genJets);


    for (unsigned iGenJet = 0; iGenJet < genJets->size(); ++iGenJet) {
      const GenJet& genJet = (*genJets) [iGenJet];
      double genJetPt = genJet.pt();
      if (fabs(genJet.eta()) > 5.) continue; // out of detector 
      if (genJetPt < mMatchGenPtThreshold) continue; // no low momentum 
      double logPtGen = log10 (genJetPt);
      mAllGenJetsPt->Fill (logPtGen);
      mAllGenJetsEta->Fill (logPtGen, genJet.eta());
      if (pfJets->size() <= 0) continue; // no PFJets - nothing to match
      if (mRThreshold > 0) {
	unsigned iPFJetBest = 0;
	double deltaRBest = 999.;
	for (unsigned iPFJet = 0; iPFJet < pfJets->size(); ++iPFJet) {
	  double dR = deltaR (genJet.eta(), genJet.phi(), (*pfJets) [iPFJet].eta(), (*pfJets) [iPFJet].phi());
	  if (deltaRBest < mRThreshold && dR < mRThreshold && genJet.pt() > 5.) {
	    std::cout << "Yet another matched jet for GenJet pt=" << genJet.pt()
		      << " previous PFJet pt/dr: " << (*pfJets) [iPFJetBest].pt() << '/' << deltaRBest
		      << " new PFJet pt/dr: " << (*pfJets) [iPFJet].pt() << '/' << dR
		      << std::endl;
	  }
	  if (dR < deltaRBest) {
	    iPFJetBest = iPFJet;
	    deltaRBest = dR;
	  }
	}
	mRMatch->Fill (logPtGen, genJet.eta(), deltaRBest);
	if (deltaRBest < mRThreshold) { // Matched
	  fillMatchHists (genJet, (*pfJets) [iPFJetBest]);
	}
      }
      else {}
	/*	{
	unsigned iPFJetBest = 0;
	double energyFractionBest = 0.;
	for (unsigned iPFJet = 0; iPFJet < pfJets->size(); ++iPFJet) {
	  double energyFraction = jetMatching.overlapEnergyFraction (genJetConstituents [iGenJet], 
								     pfJetConstituents [iPFJet]);
	  if (energyFraction > energyFractionBest) {
	    iPFJetBest = iPFJet;
	    energyFractionBest = energyFraction;
	  }
	}
	mGenJetMatchEnergyFraction->Fill (logPtGen, genJet.eta(), energyFractionBest);
	if (energyFractionBest > mGenEnergyFractionThreshold) { // good enough
	  double reverseEnergyFraction = jetMatching.overlapEnergyFraction (pfJetConstituents [iPFJetBest], 
									    genJetConstituents [iGenJet]);
	  mReverseMatchEnergyFraction->Fill (logPtGen, genJet.eta(), reverseEnergyFraction);
	  if (reverseEnergyFraction > mReverseEnergyFractionThreshold) { // Matched
	    fillMatchHists (genJet, (*pfJets) [iPFJetBest]);
	  }
	}
	}*/
    }
  }
}




void PFJetTester::fillMatchHists (const reco::GenJet& fGenJet, const reco::PFJet& fPFJet) {
  double logPtGen = log10 (fGenJet.pt());
  std::cout << "Filling matchingn" << std::cout;
  mMatchedGenJetsPt->Fill (logPtGen);
  mMatchedGenJetsEta->Fill (logPtGen, fGenJet.eta());
  mDeltaEta->Fill (logPtGen, fGenJet.eta(), fPFJet.eta()-fGenJet.eta());
  mDeltaPhi->Fill (logPtGen, fGenJet.eta(), fPFJet.phi()-fGenJet.phi());
  mEScale->Fill (logPtGen, fGenJet.eta(), fPFJet.energy()/fGenJet.energy());
  mDeltaE->Fill (logPtGen, fGenJet.eta(), fPFJet.energy()-fGenJet.energy());

  //_______________________________________________________________________________________________________________

  //mEEffNeutralFraction->Fill (fPFJet.energy()/fGenJet.energy());
  mEEffNeutralFraction->Fill (fPFJet.neutralMultiplicity()/fPFJet.nConstituents(), fPFJet.energy()/fGenJet.energy());
  nEEff->Fill(fPFJet.energy()/fGenJet.energy());
  mNeutralFraction2->Fill (fPFJet.neutralMultiplicity()/fPFJet.nConstituents());
  mEEffChargedFraction->Fill (fPFJet.chargedMultiplicity()/fPFJet.nConstituents(), fPFJet.energy()/fGenJet.energy());
  mEResNeutralFraction->Fill (fPFJet.neutralMultiplicity()/fPFJet.nConstituents(), (fPFJet.energy()-fGenJet.energy())/fGenJet.energy());
  mEResChargedFraction->Fill (fPFJet.chargedMultiplicity()/fPFJet.nConstituents(), (fPFJet.energy()-fGenJet.energy())/fGenJet.energy());
}
