#ifndef PFJETFILTER_H
#define PFJETFILTER_H

#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

class PFJetFilter : public edm::EDFilter {
public:
  explicit PFJetFilter(const edm::ParameterSet &);
  ~PFJetFilter() override;

private:
  void beginJob() override;
  bool filter(edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

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
  edm::EDGetTokenT<edm::View<reco::Candidate>> inputTruthLabel_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> inputRecoLabel_;

  unsigned int entry;
  bool verbose;

  PFBenchmarkAlgo *algo_;
};

#endif  // PFJETBENCHMARKFILTER_H
