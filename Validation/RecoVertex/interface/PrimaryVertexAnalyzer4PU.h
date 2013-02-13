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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

//generator level
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"

// vertex stuff
/////#include <DataFormats/VertexReco/interface/Vertex.h>
#include <DataFormats/VertexReco/interface/VertexFwd.h>
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

// simulated vertices,..., add <use name=SimDataFormats/Vertex> and <../Track>
#include <SimDataFormats/Vertex/interface/SimVertex.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>
#include <SimDataFormats/Track/interface/SimTrack.h>
#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
//#include "DataFormats/Math/interface/LorentzVector.h"
#include <SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h>
#include <SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h>
#include <SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h>
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

//Track et al
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

// Root
#include <TH1.h>
#include <TFile.h>

#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"




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
  //int event;
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
    //event=-1;
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
  //int event;
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
  //virtual void beginJob(edm::EventSetup const&);
  virtual void beginJob();
  virtual void endJob();

private:
  void printPVTrks(const edm::Handle<reco::TrackCollection> &recTrks, 
		   const edm::Handle<reco::VertexCollection> &recVtxs,  
		   std::vector<SimPart>& tsim,
		   std::vector<SimEvent>& simEvt,
		   const bool selectedOnly=true);

  int* supf(std::vector<SimPart>& simtrks, const reco::TrackCollection & trks);
  static bool match(const ParameterVector  &a, const ParameterVector &b);
  std::vector<SimPart> getSimTrkParameters( edm::Handle<edm::SimTrackContainer> & simTrks,
					    edm::Handle<edm::SimVertexContainer> & simVtcs,
					    double simUnit=1.0);
  void getTc(const std::vector<reco::TransientTrack>&,double &, double &, double &, double &, double&);
  void add(std::map<std::string, TH1*>& h, TH1* hist){  h[hist->GetName()]=hist; hist->StatOverflows(kTRUE);}

  void Fill(std::map<std::string, TH1*>& h, std::string s, double x){
    //    cout << "Fill1 " << s << endl;
    if(h.count(s)==0){
      std::cout << "Trying to fill non-exiting Histogram named " << s << std::endl;
      return;
    }
    h[s]->Fill(x);
  }

  void Fill(std::map<std::string, TH1*>& h, std::string s, double x, double y){
    //    cout << "Fill2 " << s << endl;
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
			       const std::string message="");

  void analyzeVertexCollectionTP(std::map<std::string, TH1*> & h,
			       const edm::Handle<reco::VertexCollection> recVtxs,
			       const edm::Handle<reco::TrackCollection> recTrks, 
			       std::vector<SimEvent> & simEvt,
				 const std::string message="");

  void printEventSummary(std::map<std::string, TH1*> & h,
			 const edm::Handle<reco::VertexCollection> recVtxs,
			       const edm::Handle<reco::TrackCollection> recTrks, 
			 std::vector<SimEvent> & simEvt,
			 const std::string message);

  void history(const edm::Handle<edm::View<reco::Track> > & tracks,const size_t trackindex=10000);
  std::string particleString(int) const;
  std::string vertexString(
    TrackingParticleRefVector,
    TrackingParticleRefVector
  ) const;
  std::string vertexString(
    HepMC::GenVertex::particles_in_const_iterator,
    HepMC::GenVertex::particles_in_const_iterator,
    HepMC::GenVertex::particles_out_const_iterator,
    HepMC::GenVertex::particles_out_const_iterator
  ) const;


  // ----------member data ---------------------------
  std::string recoTrackProducer_;
  std::string outputFile_;       // output file
  std::vector<std::string> vtxSample_;        // make this a a vector to keep cfg compatibility with PrimaryVertexAnalyzer
  double fBfield_;
  TFile*  rootFile_;             
  bool verbose_;
  bool doMatching_;
  bool printXBS_;
  edm::InputTag simG4_;
  double simUnit_;     
  double zmatch_;
  edm::ESHandle < ParticleDataTable > pdt_;
  math::XYZPoint myBeamSpot;
  // local counters
  int eventcounter_;
  int dumpcounter_;
  int ndump_;
  bool dumpThisEvent_;
  bool dumpPUcandidates_;

  // from the event setup
  int run_;
  int luminosityBlock_;
  int event_;
  int bunchCrossing_;
  int orbitNumber_;

  bool DEBUG_;
  

  std::map<std::string, TH1*> hBS;
  std::map<std::string, TH1*> hnoBS;
  std::map<std::string, TH1*> hDA;
  std::map<std::string, TH1*> hPIX;
  std::map<std::string, TH1*> hMVF;
  std::map<std::string, TH1*> hsimPV;

  TrackAssociatorBase * associatorByHits_;
  reco::RecoToSimCollection r2s_;
  std::map<double, TrackingParticleRef> z2tp_;

  TrackFilterForPVFinding theTrackFilter;
  reco::BeamSpot vertexBeamSpot_;
  double wxy2_;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle_;
  edm::ESHandle<TransientTrackBuilder> theB_;
  edm::InputTag beamSpot_;
};

