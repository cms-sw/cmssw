#ifndef Validation_RecoVertex_BSvsPVHistogramMaker_H
#define Validation_RecoVertex_BSvsPVHistogramMaker_H

#include <string>
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

namespace edm {
  class ParameterSet;
  class Event;
}

namespace reco {
  class BeamSpot;
}


class TH1F;
class TH2F;
class TProfile;
class TFileDirectory;

class BSvsPVHistogramMaker {

 public:
  BSvsPVHistogramMaker(edm::ConsumesCollector&& iC);
  BSvsPVHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);

  ~BSvsPVHistogramMaker();

  void book(const std::string dirname="");
  void beginRun(const unsigned int nrun);
  void fill(const unsigned int orbit, const int bx, const reco::VertexCollection& vertices, const reco::BeamSpot& bs);
  void fill(const edm::Event& iEvent, const reco::VertexCollection& vertices, const reco::BeamSpot& bs);

  double x(const reco::BeamSpot& bs, const double z) const;
  double y(const reco::BeamSpot& bs, const double z) const;

 private:

  TFileDirectory* _currdir;
  const unsigned int m_maxLS;
  const bool useSlope_;
  const bool _trueOnly;
  const bool _runHisto;
  const bool _runHistoProfile;
  const bool _runHistoBXProfile;
  const bool _runHistoBX2D;
  const edm::ParameterSet _histoParameters;

  RunHistogramManager _rhm;
  TH1F* _hdeltax;
  TH1F* _hdeltay;
  TH1F* _hdeltaz;
  TProfile* _hdeltaxvsz;
  TProfile* _hdeltayvsz;
  TH1F** _hdeltaxrun;
  TH1F** _hdeltayrun;
  TH1F** _hdeltazrun;
  TProfile** _hdeltaxvszrun;
  TProfile** _hdeltayvszrun;
  TProfile** _hdeltaxvsorbrun;
  TProfile** _hdeltayvsorbrun;
  TProfile** _hdeltazvsorbrun;

  TProfile** _hdeltaxvsbxrun;
  TProfile** _hdeltayvsbxrun;
  TProfile** _hdeltazvsbxrun;

  TH2F** _hdeltaxvsbx2drun;
  TH2F** _hdeltayvsbx2drun;
  TH2F** _hdeltazvsbx2drun;

};


#endif //  Validation_RecoVertex_BSvsPVHistogramMaker_H
