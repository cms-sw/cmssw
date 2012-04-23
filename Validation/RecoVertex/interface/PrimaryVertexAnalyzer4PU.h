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

// AOD
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

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
#include <TObjString.h>
#include <TString.h>

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
    type=0;
    sumpT=0;
  };
  int type;                // 0=not defined, 1=full,  2 = from PileupSummaryInfo
  double x,y,z;
  HepMC::FourVector ptot;
  LorentzVector p4;
  double ptsq;
  double sumpT;
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
    type=0;
    nChTP=0;
    z=-99;
    zfit=-99;
    sumpt2rec=0.;
    sumpt2=0;
    sumpt=0;
    Tc=-1;
    dzmax=0;
    dztrim=0;
    chisq=0;
    trkidx.clear();
    nwosmatch=0;
    nntmatch=0;
    recvnt.clear();
    wos.clear();
    matchQuality=0;
    rec=-1;
  };
  int type;         // 0=not filled, 1=full (e.g. from TrackingParticles), 2=partially filled (from PileUpSummary)
  double x,y,z;
  double xfit,yfit,zfit;
  int nChTP;
   //int event;
  unsigned int key;  // =index
  EncodedEventId eventId;
  std::vector<const TrackingParticle*> tp;
  std::vector<reco::TransientTrack> tk;
  std::vector<reco::TransientTrack> tkprim;
  std::vector<reco::TransientTrack> tkprimsel;
  std::vector<unsigned int> trkidx;
  double sumpt2rec;
  double sumpt2,sumpt;
  double Tc,chisq,dzmax,dztrim,m4m2;
  // rec vertex matching
  int nmatch, nmatch2;
  double zmatchn, zmatchn2;
  double pmatchn, pmatchn2;
  double wmatch;
  double zmatchw;

  unsigned int nwosmatch;  // number of recvertices dominated by this simevt (by wos) 
  unsigned int nntmatch;  // number of recvertices dominated by this simevt  (by nt)

  std::map<double, int> ntInRecVz;  // number of tracks in recvtx at z

  std::map<unsigned int, unsigned int> recvnt;  // number of tracks in recvtx (by index)
  std::map<unsigned int, unsigned int> wos;  // sum of wos in recvtx (by index)
  unsigned int matchQuality;
  int rec;
    
  

  void addTrack(unsigned int irecv, double twos){
    if (recvnt.find(irecv)==recvnt.end()){
      recvnt[irecv]=1;
    }else{
      recvnt[irecv]++;
    }
    if (wos.find(irecv)==wos.end()){
      wos[irecv]=twos;
    }else{
      wos[irecv]+=twos;
    }
  };
  
};


/* helper class holding recvertex -> simvertex matching information */
class RSmatch {
public:
  
  RSmatch(){
    sumwos=0;
    wos.clear();
    nt.clear();
    truthMatchedVertexTracks.clear();
    wosmatch=0;
    ntmatch=0;
    maxwos=-1;
    maxnt=0;
    sumnt=0;

    matchQuality=0;
    sim=-1;
  }

  void addTrack(unsigned int iev, double twos){
    sumnt++;
    if( nt.find(iev)==nt.end() ){
      nt[iev]=1;
    }else{
      nt[iev]++;
    }

    sumwos+=twos;
    if( wos.find(iev)==wos.end() ){
      wos[iev]=twos;
    }else{
      wos[iev]+=twos;
    }
      
  }

  std::vector< edm::RefToBase<reco::Track> > truthMatchedVertexTracks; // =getTruthMatchedVertexTracks(*v)
  std::map<unsigned int, double> wos;   // simevent -> wos
  std::map<unsigned int, unsigned int> nt;  // simevent -> number of truth matched tracks
  unsigned int wosmatch;  // index of the simevent providing the largest contribution tp wos
  unsigned int ntmatch;   // index of the simevent providing the highest number of tracks
  double sumwos;          // total sum of wos of all truth matched tracks
  unsigned int sumnt;     // toal number of truth matchted tracks
  double maxwos;          // largest wos sum from one sim event (wosmatch)
  double maxnt;           // largest number of tracks from one sim event (ntmatch)

  int sim;                // best match  (<0 = no match
  unsigned int matchQuality; // quality flag
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
		   const reco::VertexCollection * recVtxs,  
		   std::vector<SimPart>& tsim,
		   std::vector<SimEvent>& simEvt,
		   const bool selectedOnly=true);

  int* supf(std::vector<SimPart>& simtrks, const reco::TrackCollection & trks);




  static bool match(const ParameterVector  &a, const ParameterVector &b);
  std::vector<SimPart> getSimTrkParameters( edm::Handle<edm::SimTrackContainer> & simTrks,
					    edm::Handle<edm::SimVertexContainer> & simVtcs,
					    double simUnit=1.0);
  std::vector<SimPart> getSimTrkParameters( const edm::Handle<reco::GenParticleCollection>);
  void getTc(const std::vector<reco::TransientTrack>&,double &, double &, double &, double &, double&);
  void add(std::map<std::string, TH1*>& h, TH1* hist){  
    //std::cout << "adding histogram " << hist->GetName() << std::endl;
    h[hist->GetName()]=hist; 
    hist->StatOverflows(kTRUE);
  }



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
      //std::cout << "DEBUG : Cumulate called with empty histogram " << h->GetTitle() << std::endl;
      return;
    }
    try{
      h->ComputeIntegral();
      Double_t * integral=h->GetIntegral();
      h->SetContent(integral);
    }catch(...){
      std::cout << "DEBUG : an error occurred cumulating  " << h->GetTitle()  <<  std::endl;
    }
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
  void printRecVtxs(const reco::VertexCollection * recVtxs,  std::string title="Reconstructed Vertices");
  void printSimVtxs(const edm::Handle<edm::SimVertexContainer> simVtxs);
  void printSimTrks(const edm::Handle<edm::SimTrackContainer> simVtrks);
  std::vector<simPrimaryVertex> getSimPVs(const edm::Handle<edm::HepMCProduct> evtMC);
  std::vector<simPrimaryVertex> getSimPVs(const edm::Handle<reco::GenParticleCollection>);
  std::vector<simPrimaryVertex> getSimPVs(const edm::Handle<edm::SimVertexContainer> simVtxs, 
					  const edm::Handle<edm::SimTrackContainer> simTrks);
  std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> getSimPVs(const edm::Handle<TrackingVertexCollection>);

  Int_t getAssociatedRecoTrackIndex(const edm::Handle<reco::TrackCollection> &recTrks, TrackingParticleRef tpr );
  bool truthMatchedTrack( edm::RefToBase<reco::Track>, TrackingParticleRef &  );
  std::vector< edm::RefToBase<reco::Track> >  getTruthMatchedVertexTracks( const reco::Vertex& );

  std::vector<PrimaryVertexAnalyzer4PU::SimEvent> getSimEvents(
							      edm::Handle<TrackingParticleCollection>, 
							      edm::Handle<TrackingVertexCollection>,
							      edm::Handle<edm::View<reco::Track> >
							      );

  void matchRecTracksToVertex(simPrimaryVertex & pv, 
			      const std::vector<SimPart > & tsim,
			      const edm::Handle<reco::TrackCollection> & recTrks);

  void analyzeVertexCollection(std::map<std::string, TH1*> & h,
			       const reco::VertexCollection * recVtxs,
			       const edm::Handle<reco::TrackCollection> recTrks, 
			       std::vector<simPrimaryVertex> & simpv,
			       const std::string message="");

  void analyzeVertexCollectionTP(std::map<std::string, TH1*> & h,
			       const reco::VertexCollection * recVtxs,
			       const edm::Handle<reco::TrackCollection> recTrks, 
			       std::vector<SimEvent> & simEvt,
				 std::vector<RSmatch> & recvmatch,
				 const std::string message="");

  std::vector<RSmatch> tpmatch(std::map<std::string, TH1*> & h,
	       const reco::VertexCollection * recVtxs,
	       const edm::Handle<reco::TrackCollection> recTrks, 
	       std::vector<SimEvent> & simEvt,
	       const std::string message="");
  

  void printEventSummary(std::map<std::string, TH1*> & h,
			 const reco::VertexCollection * recVtxs,
			       const edm::Handle<reco::TrackCollection> recTrks, 
			 std::vector<SimEvent> & simEvt,
			 std::vector<RSmatch>& recvmatch,
			 const std::string message);

  void printEventSummary(std::map<std::string, TH1*> & h,
			 const reco::VertexCollection * recVtxs,
			 const edm::Handle<reco::TrackCollection> recTrks, 
			 std::vector<simPrimaryVertex> & simpv,
			 const std::string message);

  reco::VertexCollection * vertexFilter( edm::Handle<reco::VertexCollection> , bool filter);

  void compareCollections(std::vector<SimEvent> & simEvt, std::vector<simPrimaryVertex> & simpv);

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
  std::string trackAssociatorLabel_;
  double trackAssociatorMin_;
  std::string outputFile_;       // output file
  TObjString * info_;
  std::vector<std::string> vtxSample_;        // make this a a vector to keep cfg compatibility with PrimaryVertexAnalyzer
  double fBfield_;
  TFile*  rootFile_;             
  bool verbose_;
  bool veryverbose_;
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
  int emptyeventcounter_;
  int autoDumpCounter_;
  int ndump_;
  bool dumpThisEvent_;
  bool dumpPUcandidates_;
  bool dumpSignalVsTag_;
  int eventSummaryCounter_;
  int nEventSummary_;
  int nCompareCollections_;

  // from the event setup
  int run_;
  int luminosityBlock_;
  int event_;
  int bunchCrossing_;
  int orbitNumber_;
  double sigmaZ_;
  unsigned int nPUmin_;
  unsigned int nPUmax_;
  double sigmaZoverride_;
  bool useVertexFilter_;
  int bxFilter_;
  float instBunchLumi_;

  bool DEBUG_;
  int nfake_;
  int npair_;
  int currentLS_;
  bool MC_;
  
  std::vector<std::string> vertexCollectionLabels_;
  std::map<std::string, std::map<std::string, TH1*> > histograms_;
  std::map<std::string,  reco::VertexCollection * > recVtxs_;
  std::map<std::string,  std::vector<RSmatch> > recvmatch_;


  std::map<std::string, TH1*> hsimPV;
  std::map<std::string, TH1*> hTrk;
  std::map<std::string, TH1*> hEvt;

  TrackAssociatorBase * associator_;
  reco::RecoToSimCollection r2s_; 
  //reco::SimToRecoCollection s2r_;

  std::map<double, TrackingParticleRef> z2tp_;   // map reco::track.vz() --> tracking particle
  std::map<unsigned int, TrackingParticleRef> trkidx2tp_;  // reco::track index    --> tracking particle
  std::map<unsigned int, unsigned int> trkidx2simevt_;
  std::map<unsigned int, unsigned int> trkidx2recvtx_;



  TrackFilterForPVFinding theTrackFilter;
  reco::BeamSpot vertexBeamSpot_;
  double wxy2_, wx_,wy_;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle_;
  edm::ESHandle<TransientTrackBuilder> theB_;
  bool RECO_;
  double instBXLumi_;
  int nDigiPix_;

};

