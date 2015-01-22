// -*- C++ -*-
//
// Package:    MyPrimaryVertexAnalyzer4PU
// Class:      MyPrimaryVertexAnalyzer4PU
// 
/**\class PrimaryVertexAnalyzer4PU PrimaryVertexAnalyzer4PU.cc Validation/RecoVertex/src/PrimaryVertexAnalyzer4PU.cc

 Description: primary vertex analyzer for events with pile-up

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Wolfram Erdmann


// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

// generator level
#include "HepMC/SimpleVector.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// math
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

// reco track
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// reco vertex
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// simulated track
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

// simulated vertex
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// pile-up
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

// tracking
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

// vertexing
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"

// ROOT
#include <TH1.h>

// ROOT forward declarations
class TFile;

// class declaration
class PrimaryVertexAnalyzer4PU : public edm::EDAnalyzer {

 typedef math::XYZTLorentzVector LorentzVector;

typedef reco::TrackBase::ParameterVector ParameterVector;
struct SimPart{
  ParameterVector par;
  int type; // 0 = primary
  double zdcap;   // z@dca' (closest approach to the beam
  double ddcap;
  double zvtx;    // z of the vertex
  double xvtx;    // x of the vertex
  double yvtx;    // y of the vertex
  int pdg;        // particle pdg id
  int rec;
};

// auxiliary class holding simulated primary vertices
class simPrimaryVertex {
public:
  
  simPrimaryVertex(double x1,double y1,double z1):x(x1),y(y1),z(z1),ptsq(0),nGenTrk(0){
    ptot.setPx(0);
    ptot.setPy(0);
    ptot.setPz(0);
    ptot.setE(0);
    cluster=-1;
    nclutrk=0;
    p4=LorentzVector(0,0,0,0) ;
    //    event=0;
  };
  double x,y,z;
  HepMC::FourVector ptot;
  LorentzVector p4;
  double ptsq;
  int nGenTrk;
  int nMatchedTracks;
  int cluster;
  EncodedEventId eventId;
  double nclutrk;
  std::vector<int> finalstateParticles;
  std::vector<int> simTrackIndex;
  std::vector<int> matchedRecTrackIndex;
  std::vector<int> genVertex;
  std::vector<reco::Track> reconstructedTracks;
  const reco::Vertex *recVtx;
};


// auxiliary class holding simulated events
class SimEvent {
public:
  
  SimEvent(){
    nrecTrack=0;
    z=-99;
    zfit=-99;
    sumpt2rec=0.;
    sumpt2=0;
    sumpt=0;
    Tc=-1;
    dzmax=0;
    dztrim=0;
    chisq=0;
  };
  double x,y,z;
  double xfit,yfit,zfit;
  int nrecTrack;
  EncodedEventId eventId;
  std::vector<const TrackingParticle*> tp;
  std::vector<reco::TransientTrack> tk;
  std::vector<reco::TransientTrack> tkprim;
  std::vector<reco::TransientTrack> tkprimsel;
  double sumpt2rec;
  double sumpt2,sumpt;
  double Tc,chisq,dzmax,dztrim,m4m2;
  // rec vertex matching
  double zmatch;
  int nmatch;
  std::map<double, int> ntInRecVz;  // number of tracks in recvtx at z
};



public:
  explicit PrimaryVertexAnalyzer4PU(const edm::ParameterSet&);
  ~PrimaryVertexAnalyzer4PU();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob();
  virtual void endJob();

private:
  void printPVTrks(const edm::Handle<reco::TrackCollection> &recTrks, 
		   const edm::Handle<reco::VertexCollection> &recVtxs,  
		   std::vector<SimPart>& tsim,
		   std::vector<SimEvent>& simEvt,
		   const bool selectedOnly=true);

  std::vector<int> supf(std::vector<SimPart>& simtrks, const reco::TrackCollection & trks);
  static bool match(const ParameterVector  &a, const ParameterVector &b);
  std::vector<SimPart> getSimTrkParameters( edm::Handle<edm::SimTrackContainer> & simTrks,
					    edm::Handle<edm::SimVertexContainer> & simVtcs,
					    double simUnit=1.0);
  void getTc(const std::vector<reco::TransientTrack>&,double &, double &, double &, double &, double&);
  void add(std::map<std::string, TH1*>& h, TH1* hist){  h[hist->GetName()]=hist; hist->StatOverflows(kTRUE);}

  void Fill(std::map<std::string, TH1*>& h, std::string s, double x){
    if(h.count(s)==0){
      std::cout << "Trying to fill non-exiting Histogram named " << s << std::endl;
      return;
    }
    h[s]->Fill(x);
  }

  void Fill(std::map<std::string, TH1*>& h, std::string s, double x, double y){
    if(h.count(s)==0){
      std::cout << "Trying to fill non-exiting Histogram named " << s << std::endl;
      return;
    }
    h[s]->Fill(x,y);
  }

  void Fill(std::map<std::string, TH1*>& h, std::string s, double x, bool signal){
    if(h.count(s)==0){
      std::cout << "Trying to fill non-exiting Histogram named " << s << std::endl;
      return;
    }

    h[s]->Fill(x);
    if(signal){
      if(h.count(s+"Signal")==0){
	std::cout << "Trying to fill non-exiting Histogram named " << s+"Signal" << std::endl;
	return;
      }
      h[s+"Signal"]->Fill(x);
    }else{
      if(h.count(s+"PU")==0){
	std::cout << "Trying to fill non-exiting Histogram named " << s+"PU" << std::endl;
	return;
      }
      h[s+"PU"]->Fill(x);
    }
  }

  void Fill(std::map<std::string, TH1*>& h, std::string s, bool yesno, bool signal){
    if (yesno){
      Fill(h, s,1.,signal);
    }else{
      Fill(h, s,0.,signal);
    }
  }

  void Cumulate(TH1* h){
    
    if((h->GetEntries()==0) || (h->Integral()<=0) ){
      std::cout << "DEBUG : Cumulate called with empty histogram " << h->GetTitle() << std::endl;
      return;
    }
    std::cout << "DEBUG : cumulating  " << h->GetTitle() << std::endl;
    try{
      h->ComputeIntegral();
      Double_t * integral=h->GetIntegral();
      h->SetContent(integral);
    }catch(...){
      std::cout << "DEBUG : an error occurred cumulating  " << h->GetTitle()  <<  std::endl;
    }
    std::cout << "DEBUG : cumulating  " << h->GetTitle() << "done " <<  std::endl;
  }

  std::map<std::string, TH1*> bookVertexHistograms();

  bool matchVertex(const simPrimaryVertex  &vsim, 
		   const reco::Vertex       &vrec);
  bool isResonance(const HepMC::GenParticle * p);
  bool isFinalstateParticle(const HepMC::GenParticle * p);
  bool isCharged(const HepMC::GenParticle * p);
  void fillTrackHistos(std::map<std::string, TH1*> & h, const std::string & ttype, const reco::Track & t, const reco::Vertex *v = NULL);
  void dumpHitInfo(const reco::Track & t);
  void printRecTrks( const edm::Handle<reco::TrackCollection> & recTrks);
  void printRecVtxs(const edm::Handle<reco::VertexCollection> recVtxs,  std::string title="Reconstructed Vertices");
  void printSimVtxs(const edm::Handle<edm::SimVertexContainer> simVtxs);
  void printSimTrks(const edm::Handle<edm::SimTrackContainer> simVtrks);
  std::vector<simPrimaryVertex> getSimPVs(const edm::Handle<edm::HepMCProduct> evtMC);
  std::vector<simPrimaryVertex> getSimPVs(const edm::Handle<edm::HepMCProduct> evt, 
					  const edm::Handle<edm::SimVertexContainer> simVtxs, 
					  const edm::Handle<edm::SimTrackContainer> simTrks);
  std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> getSimPVs(const edm::Handle<TrackingVertexCollection>);
  double getTrueSeparation(float, const std::vector<float> &);
  double getTrueSeparation(SimEvent, std::vector<SimEvent> &);
  std::vector<int>* vertex_match(float, const edm::Handle<reco::VertexCollection>);

  bool truthMatchedTrack( edm::RefToBase<reco::Track>, TrackingParticleRef &  );
  std::vector< edm::RefToBase<reco::Track> >  getTruthMatchedVertexTracks(
				       const reco::Vertex&
				       );

  std::vector<PrimaryVertexAnalyzer4PU::SimEvent> getSimEvents(
							      edm::Handle<TrackingParticleCollection>, 
							      edm::Handle<TrackingVertexCollection>,
							      edm::Handle<edm::View<reco::Track> >
							      );

  void matchRecTracksToVertex(simPrimaryVertex & pv, 
			      const std::vector<SimPart > & tsim,
			      const edm::Handle<reco::TrackCollection> & recTrks);

  void analyzeVertexCollection(std::map<std::string, TH1*> & h,
			       const edm::Handle<reco::VertexCollection> recVtxs,
			       const edm::Handle<reco::TrackCollection> recTrks, 
			       std::vector<simPrimaryVertex> & simpv,
			       const std::vector<float> & pui_z, 
			       const std::string message="");

  void analyzeVertexCollectionTP(std::map<std::string, TH1*> & h,
			       const edm::Handle<reco::VertexCollection> recVtxs,
			       const edm::Handle<reco::TrackCollection> recTrks, 
			       std::vector<SimEvent> & simEvt,
			       reco::RecoToSimCollection rsC, 
				   const std::string message="");

  void printEventSummary(std::map<std::string, TH1*> & h,
			 const edm::Handle<reco::VertexCollection> recVtxs,
			       const edm::Handle<reco::TrackCollection> recTrks, 
			 std::vector<SimEvent> & simEvt,
			 const std::string message);

  void history(const edm::Handle<edm::View<reco::Track> > & tracks,const size_t trackindex=10000);
  std::string particleString(int) const;
  std::string vertexString(
    const TrackingParticleRefVector&,
    const TrackingParticleRefVector&
  ) const;
  std::string vertexString(
    HepMC::GenVertex::particles_in_const_iterator,
    HepMC::GenVertex::particles_in_const_iterator,
    HepMC::GenVertex::particles_out_const_iterator,
    HepMC::GenVertex::particles_out_const_iterator
  ) const;


  // ----------member data ---------------------------
  bool verbose_;
  bool doMatching_;
  bool printXBS_;
  bool dumpThisEvent_;
  bool dumpPUcandidates_;
  bool DEBUG_;

  // local counters
  int eventcounter_;
  int dumpcounter_;
  int ndump_;

  // from the event setup
  edm::RunNumber_t run_;
  edm::LuminosityBlockNumber_t luminosityBlock_;
  edm::EventNumber_t event_;
  int bunchCrossing_;
  int orbitNumber_;

  double fBfield_;
  double simUnit_;     
  double zmatch_;
  double wxy2_;

  math::XYZPoint myBeamSpot;
  reco::RecoToSimCollection r2s_;
  TrackFilterForPVFinding theTrackFilter;
  reco::BeamSpot vertexBeamSpot_;

  edm::ESHandle< ParticleDataTable > pdt_;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle_;
  edm::ESHandle<TransientTrackBuilder> theB_;

  TFile* rootFile_;             
  const reco::TrackToTrackingParticleAssociator * associatorByHits_;

  std::string recoTrackProducer_;
  std::string outputFile_;       // output file
  std::vector<std::string> vtxSample_;        // make this a a vector to keep cfg compatibility with PrimaryVertexAnalyzer

  std::map<std::string, TH1*> hBS;
  std::map<std::string, TH1*> hnoBS;
  std::map<std::string, TH1*> hDA;
  std::map<std::string, TH1*> hPIX;
  std::map<std::string, TH1*> hMVF;
  std::map<std::string, TH1*> hsimPV;

  std::map<double, TrackingParticleRef> z2tp_;
  
  edm::EDGetTokenT< std::vector<PileupSummaryInfo> > vecPileupSummaryInfoToken_;
  edm::EDGetTokenT<reco::VertexCollection> recoVertexCollectionToken_, recoVertexCollection_BS_Token_, recoVertexCollection_DA_Token_;
  edm::EDGetTokenT<reco::TrackCollection> recoTrackCollectionToken_;
  edm::EDGetTokenT<reco::BeamSpot> recoBeamSpotToken_;
  edm::EDGetTokenT< edm::View<reco::Track> > edmView_recoTrack_Token_;
  edm::EDGetTokenT<edm::SimVertexContainer> edmSimVertexContainerToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> edmSimTrackContainerToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;
  edm::EDGetTokenT<TrackingVertexCollection> trackingVertexCollectionToken_;
  edm::EDGetTokenT<edm::HepMCProduct> edmHepMCProductToken_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> recoTrackToTrackingParticleAssociatorToken_;
};
