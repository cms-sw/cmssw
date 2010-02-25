#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
//#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

//#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include <memory>
#include <string>
#include <iostream>

#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TDirectory.h>

#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
//#include "PhysicsTools/UtilAlgos/interface/TFileService.h" // Framework service for histograms
//#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <Math/GenVector/VectorUtil.h>

// get rid of this damn TLorentzVector!
//#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Math/interface/deltaR.h"
//#include "TLorentzVector.h"

// Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;
using namespace reco; 
using namespace std;

typedef Particle::LorentzVector LorentzVector;

class PFTauHLTTest : public EDAnalyzer {
public:
  explicit PFTauHLTTest(const ParameterSet&);
  ~PFTauHLTTest() {}
  virtual void analyze(const Event& iEvent,const EventSetup& iSetup);
  virtual void beginJob();
  virtual void endJob();
private:

  DeltaR<LorentzVector> deltaRComputer_;

  string PFTauProducer_, MatchedPFTauProducer_,PFJetProducer_;

  
  int nEvent;
  int nTauMatchPFTau;
  int nElecMatchPFTau;
  int nTauElecPreID;
  int nElecElecPreID;
  int nTauNonElecPreID;
  int nElecNonElecPreID;

  // files
  std::string outPutFile_;
  TFile *_file;
  TDirectory *_dir;

  // Efficiency plots
  MonitorElement* h_PFTau_Eta_;
  MonitorElement* h_NumberOfJets_;
  MonitorElement* h_MatchedPFTau_Eta_;
  MonitorElement* h_PFTau_Phi_;
  MonitorElement* h_PFTau_Pt_;
  MonitorElement* h_Tau_Pt_;
  MonitorElement* h_MatchedPFTau_Pt_;
  MonitorElement* h_PFTau_SignalChargedHadron_;
  MonitorElement* h_PFTau_SignalNeutralHadron_;
  MonitorElement* h_PFTau_LeadingTrackPt_;
  MonitorElement* h_PFTau_SignalTracks_;
  MonitorElement* h_PFTau_SignalGammas_;
  MonitorElement* h_PFTauEt_MCTauEt_;
  DQMStore* dbeTau;
};

PFTauHLTTest::PFTauHLTTest(const ParameterSet& iConfig){
  PFTauProducer_                         = iConfig.getParameter<string>("PFTauProducer");
  PFJetProducer_                         = iConfig.getParameter<string>("PFJetProducer");
  MatchedPFTauProducer_                         = iConfig.getParameter<string>("MatchedPFTauProducer");

  nEvent=0;

  nTauMatchPFTau=0;
  nElecMatchPFTau=0;
  nTauElecPreID=0;
  nElecElecPreID=0;
  nTauNonElecPreID=0;
  nElecNonElecPreID=0;

  outPutFile_ = "PFTauTest.root";
}
void PFTauHLTTest::beginJob(){
  dbeTau = &*edm::Service<DQMStore>();
  dbeTau->setCurrentFolder("RecoPFTau"); 

  // Book histograms
  h_PFTau_Eta_ = dbeTau->book1D("PFTau_Eta","PFTau_Eta",50,0.,5.0);
  h_MatchedPFTau_Eta_ = dbeTau->book1D("MatchedPFTau_Eta","MatchedPFTau_Eta",50,0.,5.0);
  h_PFTau_Phi_ = dbeTau->book1D("PFTau_Phi","PFTau_Phi",50,-3.15,3.15);
  h_PFTau_Pt_ = dbeTau->book1D("PFTau_Pt","PFTau_Pt",50,0.,50.);
  h_Tau_Pt_ = dbeTau->book1D("Tau_Pt","Tau_Pt",50,0.,50.);
  h_MatchedPFTau_Pt_ = dbeTau->book1D("MatchedPFTau_Pt","MatchedPFTau_Pt",50,0.,50.);
  h_PFTau_SignalChargedHadron_ = dbeTau->book1D("PFTau_NumberChargedHadrons","PFTau_NumberChargedHadrons",10,0.,10.);
  h_PFTau_SignalNeutralHadron_ = dbeTau->book1D("PFTau_NumberNeutralHadrons","PFTau_NumberNeutralHadrons",10,0.,10.);
  h_PFTau_LeadingTrackPt_ = dbeTau->book1D("PFTau_LeadingTrackPt","PFTau_LeadingTrackPt",10,0.,10.);
  h_PFTau_SignalTracks_ = dbeTau->book1D("PFTau_NumberTracks","PFTau_NumberTracks",10,0.,10.);
  h_PFTau_SignalGammas_= dbeTau->book1D("PFTau_NumberPhotons","PFTau_NumberPhotons",10,0.,10.); 

  h_PFTauEt_MCTauEt_  = dbeTau->book2D("PFTauEt_MCTauEt","PFTau_Et Vs MCTau_Et", 50,0.,50.,50,0.,50.);

  h_NumberOfJets_ = dbeTau->book1D("NumberOfJets","NumberOfJets",20,0.,20.);

}


void PFTauHLTTest::analyze(const Event& iEvent, const EventSetup& iSetup){
  //cout<<"********"<<endl;
  //cout<<"Event number "<<nEvent++<<endl;



  ////////////////////////////////////////////////////////  

   
  
  Handle<PFTauCollection> thePFTauHandle;
  iEvent.getByLabel(PFTauProducer_,thePFTauHandle);
  
  Handle<PFTauCollection> thePFJetHandle;
  iEvent.getByLabel(PFJetProducer_,thePFJetHandle);
  double numberOfJets = thePFJetHandle->size()*1.;
  h_NumberOfJets_->Fill(numberOfJets);
  
  Handle<View<Candidate> > theMatchedPFTauHandle;
  iEvent.getByLabel(MatchedPFTauProducer_,theMatchedPFTauHandle);

  //int n = 0;
  // Tau Loop
  for (unsigned int iPFTau=0;iPFTau<thePFTauHandle->size();iPFTau++) { 
    const PFTauRef thePFTau(thePFTauHandle, iPFTau);
    if(thePFTau->pt() > 0.){
      h_PFTau_Eta_->Fill(fabs((*thePFTau).eta()));
      h_PFTau_Pt_->Fill((*thePFTau).pt());
      h_PFTau_Phi_->Fill((*thePFTau).phi());
    }
    bool matched = false;
    //loop over the matched jets
    for (View<Candidate>::size_type iMPFTau=0;iMPFTau<theMatchedPFTauHandle->size();iMPFTau++) { 
      const Candidate *theMPFTau = &(*theMatchedPFTauHandle)[iMPFTau];
      double deltaR = ROOT::Math::VectorUtil::DeltaR(thePFTau->p4().Vect(), (theMPFTau->p4()).Vect());
      if(deltaR < 0.5) {
	matched = true;
	h_PFTauEt_MCTauEt_->Fill(theMPFTau->pt(),thePFTau->pt());
	break;
      }

    }

    if(matched) {
      h_MatchedPFTau_Pt_->Fill((*thePFTau).pt());
      h_MatchedPFTau_Eta_->Fill(fabs((*thePFTau).eta()));
      cout <<"Particle type "<<(*thePFTau).leadPFChargedHadrCand()->particleId()<<endl;
      if((*thePFTau).hasMuonReference()){
	MuonRef muonref = (*thePFTau).leadPFChargedHadrCand()->muonRef();
	cout <<"Muon segments " <<muonref->numberOfMatches()<<endl;
      }
    }
    if(!matched){
            h_PFTau_SignalChargedHadron_->Fill((*thePFTau).signalPFChargedHadrCands().size()*1.);
            h_PFTau_SignalNeutralHadron_->Fill((*thePFTau).signalPFNeutrHadrCands().size()*1.);
           if((*thePFTau).leadPFChargedHadrCand().isNonnull())
		h_PFTau_LeadingTrackPt_->Fill((*thePFTau).leadPFChargedHadrCand()->pt());

	      h_PFTau_SignalTracks_->Fill((*thePFTau).signalTracks().size()*1.);
	      h_PFTau_SignalGammas_->Fill((*thePFTau).signalPFGammaCands().size()*1.);
    }

  }

   for (View<Candidate>::size_type iMPFTau=0;iMPFTau<theMatchedPFTauHandle->size();iMPFTau++) { 
      const Candidate *theMPFTau = &(*theMatchedPFTauHandle)[iMPFTau];
      h_Tau_Pt_->Fill(theMPFTau->pt());
    }
  
}

void PFTauHLTTest::endJob(){
 if (!outPutFile_.empty() && &*edm::Service<DQMStore>()) dbeTau->save (outPutFile_);
}


DEFINE_FWK_MODULE(PFTauHLTTest);
