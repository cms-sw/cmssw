// -*- C++ -*-
//
// Package:    PrimaryVertexAnalyzer
// Class:      PrimaryVertexAnalyzer
// 
/**\class PrimaryVertexAnalyzer PrimaryVertexAnalyzer.cc Validation/RecoVertex/src/PrimaryVertexAnalyzer.cc

 Description: simple primary vertex analyzer

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Wolfram Erdmann
//         Created:  Fri Jun  2 10:54:05 CEST 2006
//
//


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

// reco track
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// reco vertex
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// simulated track
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

// simulated vertex
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// ROOT forward declarations
class TDirectory;
class TFile;
class TH1;

// class declaration
class PrimaryVertexAnalyzer : public edm::EDAnalyzer {


// auxiliary class holding simulated primary vertices
class simPrimaryVertex {
public:
  simPrimaryVertex(double x1,double y1,double z1):x(x1),y(y1),z(z1),ptsq(0),nGenTrk(0){};
  double x,y,z;
   HepMC::FourVector ptot;
  //HepLorentzVector ptot;
  double ptsq;
  int nGenTrk;
  std::vector<int> finalstateParticles;
  std::vector<int> simTrackIndex;
  std::vector<int> genVertex;
  const reco::Vertex *recVtx;
};



public:
  explicit PrimaryVertexAnalyzer(const edm::ParameterSet&);
  ~PrimaryVertexAnalyzer();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob();
  virtual void endJob();

private:

  bool matchVertex(const simPrimaryVertex  &vsim, 
		   const reco::Vertex       &vrec);
  bool isResonance(const HepMC::GenParticle * p);
  bool isFinalstateParticle(const HepMC::GenParticle * p);
  bool isCharged(const HepMC::GenParticle * p);
 
  void printRecVtxs(const edm::Handle<reco::VertexCollection> & recVtxs);
  void printSimVtxs(const edm::Handle<edm::SimVertexContainer> & simVtxs);
  void printSimTrks(const edm::Handle<edm::SimTrackContainer> & simVtrks);
  std::vector<simPrimaryVertex> getSimPVs(const edm::Handle<edm::HepMCProduct> & evtMC, const std::string & suffix = "");
  std::vector<simPrimaryVertex> getSimPVs(const edm::Handle<edm::HepMCProduct> & evt, 
					  const edm::Handle<edm::SimVertexContainer> & simVtxs, 
					  const edm::Handle<edm::SimTrackContainer> & simTrks);
  // ----------member data ---------------------------
  bool verbose_;
  double simUnit_;     
  edm::ESHandle < ParticleDataTable > pdt;
  edm::EDGetTokenT< reco::TrackCollection > recoTrackCollectionToken_;
  edm::EDGetTokenT< edm::SimVertexContainer > edmSimVertexContainerToken_;
  edm::EDGetTokenT< edm::SimTrackContainer > edmSimTrackContainerToken_;
  edm::EDGetTokenT< edm::HepMCProduct > edmHepMCProductToken_;
  std::vector< edm::EDGetTokenT< reco::VertexCollection > > recoVertexCollectionTokens_;
  std::string outputFile_;       // output file
  std::vector<std::string> suffixSample_; // which vertices to analyze
  TFile*  rootFile_;
  std::map<std::string, TH1*> h;
  std::map<std::string, TDirectory*> hdir;
	
};

