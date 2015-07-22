#ifndef testVertexAssociator_h
#define testVertexAssociator_h

#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TH1F.h"
#include "TFile.h"

#include <iostream>
#include <string>
#include <map>
#include <set>

#include "TTree.h"
#include "TMath.h"
#include <vector>
#include "TROOT.h"

#include <Math/GenVector/PxPyPzE4D.h>
#include <Math/GenVector/PxPyPzM4D.h>
#include "DataFormats/Math/interface/LorentzVector.h"
#include <cmath>

namespace reco {
  class TrackToTrackingParticleAssociator;
  class VertexToTrackingVertexAssociator;
}

class testVertexAssociator : public edm::EDAnalyzer {

 public:
  testVertexAssociator(const edm::ParameterSet& conf);
  virtual ~testVertexAssociator();
  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  const reco::TrackToTrackingParticleAssociator*  associatorByChi2;
  const reco::TrackToTrackingParticleAssociator*  associatorByHits;
  const reco::VertexToTrackingVertexAssociator* associatorByTracks;

  edm::InputTag vertexCollection_;
  edm::EDGetTokenT<reco::VertexToTrackingVertexAssociator> associatorByTracksToken;

  int n_event_;
  int n_rs_vertices_;
  int n_rs_vtxassocs_;
  int n_sr_vertices_;
  int n_sr_vtxassocs_;

  //--------- RecoToSim Histos -----
 
  TH1F*  rs_resx;
  TH1F*  rs_resy;
  TH1F*  rs_resz;  
  TH1F*  rs_pullx;
  TH1F*  rs_pully;
  TH1F*  rs_pullz; 
  TH1F*  rs_dist;
  TH1F*  rs_simz;
  TH1F*  rs_recz;
  TH1F*  rs_nrectrk;
  TH1F*  rs_nsimtrk;
  TH1F*  rs_qual;
  TH1F*  rs_chi2norm;
  TH1F*  rs_chi2prob;

  //--------- SimToReco Histos -----

  TH1F*  sr_resx;
  TH1F*  sr_resy;
  TH1F*  sr_resz;  
  TH1F*  sr_pullx;
  TH1F*  sr_pully;
  TH1F*  sr_pullz; 
  TH1F*  sr_dist;
  TH1F*  sr_simz;
  TH1F*  sr_recz;
  TH1F*  sr_nrectrk;
  TH1F*  sr_nsimtrk;
  TH1F*  sr_qual;
  TH1F*  sr_chi2norm;
  TH1F*  sr_chi2prob;

};

#endif
