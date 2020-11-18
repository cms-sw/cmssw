#ifndef _HGCalMaskResolutionAna_h_
#define _HGCalMaskResolutionAna_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "Validation/HGCalValidation/interface/validHit.h"

#include "TTree.h"
#include "TH1F.h"

#include <iostream>
#include <string>

struct ValidRecHits {
  std::vector<float> energy;
  std::vector<float> time;
  std::vector<float> timeError;
  std::vector<unsigned int> detid;
  std::vector<unsigned int> flagBits;
  std::vector<float> son;
};

class HeterogeneousHGCalRecHitsValidator : public edm::EDAnalyzer {
public:
  explicit HeterogeneousHGCalRecHitsValidator(const edm::ParameterSet&);
  ~HeterogeneousHGCalRecHitsValidator() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  static const unsigned int nsubdetectors = 3;      //ce-e, ce-h-fine, ce-h-coarse
  static const unsigned int ncomputingdevices = 2;  //cpu, gpu
  //cpu amd gpu tokens and handles for the 3 subdetectors, cpu and gpu
  std::array<std::array<edm::EDGetTokenT<HGChefRecHitCollection>, ncomputingdevices>, nsubdetectors> tokens_;
  std::array<std::array<edm::Handle<HGChefRecHitCollection>, ncomputingdevices>, nsubdetectors> handles_;
  std::array<std::string, nsubdetectors> handles_str_ = {
      {"HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"}};
  hgcal::RecHitTools recHitTools_;

  std::array<TTree*, nsubdetectors> trees_;
  std::array<std::string, nsubdetectors> treenames_;
  std::array<validHitCollection, nsubdetectors> cpuValidRecHits, gpuValidRecHits, diffsValidRecHits;
  //std::vector< TH1F* > zhist;

  void set_geometry_(const edm::EventSetup&, const unsigned int&);
};

#endif
