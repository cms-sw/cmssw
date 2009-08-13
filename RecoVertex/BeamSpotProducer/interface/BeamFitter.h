#ifndef BeamFitter_H
#define BeamFitter_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"


class BeamFitter {
 public:
  BeamFitter() {}
  BeamFitter(const edm::ParameterSet& iConfig);
  virtual ~BeamFitter();

  void readEvent(const edm::Event& iEvent);

  void runFitter(); 
  void resetTrkVector() { fBSvector.clear(); }
  reco::BeamSpot getBeamSpot() { return fbeamspot; }

 private:

  std::vector< BSTrkParameters > fBSvector;
  reco::BeamSpot fbeamspot;
  BSFitter *fmyalgo;

  bool debug_;
  edm::InputTag tracksLabel_;
  bool writeTxt_;
  std::string outputTxt_;
  double trk_MinpT_;
  double trk_MaxEta_;
  int trk_MinNTotLayers_;
  int trk_MinNPixLayers_;
  double trk_MinNormChi2_;
  int trk_Algorithm_;


};

#endif
