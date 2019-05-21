// -*- C++ -*-
//
// Package:    TrackParameterAnalyzer
// Class:      TrackParameterAnalyzer
//
/**\class TrackParameterAnalyzer TrackParameterAnalyzer.cc Validation/RecoVertex/src/TrackParameterAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Wolfram Erdmann
//         Created:  Fri Jun  2 10:54:05 CEST 2006
//
//

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// simulated vertex
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// simulated track
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

// track
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// ROOT forward declarations
class TFile;
class TH1;
class TH2;

// class declaration
//
typedef reco::TrackBase::ParameterVector ParameterVector;

class TrackParameterAnalyzer : public edm::EDAnalyzer {
public:
  explicit TrackParameterAnalyzer(const edm::ParameterSet&);
  ~TrackParameterAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  bool match(const ParameterVector& a, const ParameterVector& b);
  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::SimVertexContainer> edmSimVertexContainerToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> edmSimTrackContainerToken_;
  edm::EDGetTokenT<reco::TrackCollection> recoTrackCollectionToken_;
  // root file to store histograms
  std::string outputFile_;  // output file
  TFile* rootFile_;
  TH1* h1_pull0_;
  TH1* h1_pull1_;
  TH1* h1_pull2_;
  TH1* h1_pull3_;
  TH1* h1_pull4_;
  TH1* h1_res0_;
  TH1* h1_res1_;
  TH1* h1_res2_;
  TH1* h1_res3_;
  TH1* h1_res4_;
  TH1* h1_Beff_;
  TH2* h2_dvsphi_;
  TH1* h1_par0_;
  TH1* h1_par1_;
  TH1* h1_par2_;
  TH1* h1_par3_;
  TH1* h1_par4_;
  double simUnit_;
  bool verbose_;
};
