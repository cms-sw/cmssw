#ifndef Validation_RecoVertex_VertexHistogramMaker_H
#define Validation_RecoVertex_VertexHistogramMaker_H

#include <string>
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

namespace edm {
  class ParameterSet;
  class Event;
}

class TH1F;
class TH2F;
class TProfile;
class TFileDirectory;

class VertexHistogramMaker {

 public:
  VertexHistogramMaker();
  VertexHistogramMaker(const edm::ParameterSet& iConfig);
 
  ~VertexHistogramMaker();

  void book(const std::string dirname="");
  void beginRun(const unsigned int nrun);
  void fill(const unsigned int orbit, const int bx, const reco::VertexCollection& vertices, const double weight=1.);
  void fill(const edm::Event& iEvent, const reco::VertexCollection& vertices, const double weight=1.);

 private:

  TFileDirectory* m_currdir;
  const double m_weightThreshold;
  const bool m_trueOnly;
  const bool m_runHisto;
  const bool m_runHistoProfile;
  const bool m_runHistoBXProfile;
  const bool m_runHisto2D;
  const bool m_bsConstrained;
  const edm::ParameterSet m_histoParameters;

  RunHistogramManager m_rhm;
  TH1F* m_hnvtx;
  TH1F* m_hntruevtx;
  TH1F* m_hntracks;
  TH1F* m_hsqsumptsq;
  TH1F* m_hsqsumptsqheavy;
  TH1F* m_hnheavytracks;
  TH1F* m_hndof;
  TH1F* m_haveweight;
  TH2F* m_hndofvstracks;
  TProfile* m_hndofvsvtxz; 
  TProfile* m_hntracksvsvtxz; 
  TProfile* m_haveweightvsvtxz; 
  TH1F* m_hweights;
  TH1F* m_hvtxx;
  TH1F* m_hvtxy;
  TH1F* m_hvtxz;
  TH1F** m_hvtxxrun;
  TH1F** m_hvtxyrun;
  TH1F** m_hvtxzrun;
  TProfile** m_hvtxxvsorbrun;
  TProfile** m_hvtxyvsorbrun;
  TProfile** m_hvtxzvsorbrun;
  TProfile** m_hnvtxvsorbrun;
  TProfile2D** m_hnvtxvsbxvsorbrun;

  TProfile** m_hvtxxvsbxrun;
  TProfile** m_hvtxyvsbxrun;
  TProfile** m_hvtxzvsbxrun;
  TProfile** m_hnvtxvsbxrun;

  TH2F** m_hvtxxvsbx2drun;
  TH2F** m_hvtxyvsbx2drun;
  TH2F** m_hvtxzvsbx2drun;

};


#endif //  Validation_RecoVertex_VertexHistogramMaker_H
