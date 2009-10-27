#ifndef BeamFitter_H
#define BeamFitter_H

/**_________________________________________________________________
   class:   BeamFitter.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)
 
 version $Id: BeamFitter.h,v 1.8 2009/10/27 14:38:20 yumiceva Exp $

 ________________________________________________________________**/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"

#include<fstream>

class BeamFitter {
 public:
  BeamFitter() {}
  BeamFitter(const edm::ParameterSet& iConfig);
  virtual ~BeamFitter();

  void readEvent(const edm::Event& iEvent);

  bool runFitter(); 
  void runAllFitter();
  void resetTrkVector() { fBSvector.clear(); }
  void resetTotTrk() { ftotal_tracks=0; }
  void dumpTxtFile();
  void write2DB();
  reco::BeamSpot getBeamSpot() { return fbeamspot; }
  std::vector<BSTrkParameters> getBSvector() { return fBSvector; }
  
 private:

  std::vector<BSTrkParameters> fBSvector;
  reco::BeamSpot fbeamspot;
  BSFitter *fmyalgo;
  std::ofstream fasciiFile;

  bool debug_;
  edm::InputTag tracksLabel_;
  bool writeTxt_;
  std::string outputTxt_;
  double trk_MinpT_;
  double trk_MaxZ_;
  double trk_MaxEta_;
  double trk_MaxIP_;
  int trk_MinNTotLayers_;
  int trk_MinNPixLayers_;
  double trk_MaxNormChi2_;
  std::vector<std::string> trk_Algorithm_;
  std::vector<std::string> trk_Quality_;
  std::vector<reco::TrackBase::TrackQuality> quality_;
  std::vector<reco::TrackBase::TrackAlgorithm> algorithm_;
  double inputBeamWidth_;
  double convergence_;
  int ftotal_tracks;
  int min_Ntrks_;
  bool isMuon_;
  
  // ntuple
  TH1F* h1z;
  bool saveNtuple_;
  std::string outputfilename_;
  TFile* file_;
  TTree* ftree_;
  double ftheta;
  double fpt;
  double feta;
  int    fcharge;
  double fnormchi2;
  double fphi0;
  double fd0;
  double fsigmad0;
  double fz0;
  double fsigmaz0;
  int fnTotLayerMeas;
  int fnPixelLayerMeas;
  int fnStripLayerMeas;
  int fnTIBLayerMeas;
  int fnTIDLayerMeas;
  int fnTOBLayerMeas;
  int fnTECLayerMeas;
  int fnPXBLayerMeas;
  int fnPXFLayerMeas;
  double fd0phi_chi2;
  double fd0phi_d0;
  double fcov[7][7];
  double fvx;
  double fvy;
  int frun;
  int flumi;
  
};

#endif
