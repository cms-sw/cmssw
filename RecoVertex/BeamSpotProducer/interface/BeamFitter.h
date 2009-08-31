#ifndef BeamFitter_H
#define BeamFitter_H

/**_________________________________________________________________
   class:   BeamFitter.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)
 
 version $Id: BeamFitter.h,v 1.4 2009/08/26 22:06:01 yumiceva Exp $

 ________________________________________________________________**/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"

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
  
};

#endif
