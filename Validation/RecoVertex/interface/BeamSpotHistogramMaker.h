#ifndef Validation_RecoVertex_BeamSpotHistogramMaker_H
#define Validation_RecoVertex_BeamSpotHistogramMaker_H

#include <string>
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

namespace edm {
  class ParameterSet;
}

namespace reco {
  class BeamSpot;
}

class TH1F;
class TProfile;
class TFileDirectory;

class BeamSpotHistogramMaker {

 public:
  BeamSpotHistogramMaker(edm::ConsumesCollector&& iC);
  BeamSpotHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);

  ~BeamSpotHistogramMaker();

  void book(const std::string dirname="");
  void beginRun(const unsigned int nrun);
  void fill(const unsigned int orbit, const reco::BeamSpot& bs);

 private:

  TFileDirectory* _currdir;
  const edm::ParameterSet _histoParameters;

  RunHistogramManager _rhm;
  TH1F** _hbsxrun;
  TH1F** _hbsyrun;
  TH1F** _hbszrun;
  TH1F** _hbssigmaxrun;
  TH1F** _hbssigmayrun;
  TH1F** _hbssigmazrun;
  TProfile** _hbsxvsorbrun;
  TProfile** _hbsyvsorbrun;
  TProfile** _hbszvsorbrun;
  TProfile** _hbssigmaxvsorbrun;
  TProfile** _hbssigmayvsorbrun;
  TProfile** _hbssigmazvsorbrun;


};


#endif //  Validation_RecoVertex_BeamSpotHistogramMaker_H
