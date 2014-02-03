#ifndef PFJETFILTER_H
#define PFJETFILTER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

class PFJetFilter: public edm::EDFilter {
 public:

  explicit PFJetFilter(const edm::ParameterSet&);
  ~PFJetFilter();

 private:

  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  double resolution(double, bool);
  double response(double, bool);

  double recPt_cut;
  double genPt_cut;
  double deltaEt_min;
  double deltaEt_max;
  double deltaR_min;
  double deltaR_max;
  double eta_min;
  double eta_max;
  edm::InputTag inputTruthLabel_;
  edm::InputTag inputRecoLabel_;
  unsigned int entry;
  bool verbose;
  
  PFBenchmarkAlgo *algo_;

};


#endif //PFJETBENCHMARKFILTER_H
