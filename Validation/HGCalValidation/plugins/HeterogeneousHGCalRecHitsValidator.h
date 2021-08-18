#ifndef _HGCalMaskResolutionAna_h_
#define _HGCalMaskResolutionAna_h_

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Validation/HGCalValidation/interface/ValidHit.h"

#include "TTree.h"
#include "TH1F.h"

#include <iostream>
#include <string>

struct ValidRecHits {
  std::vector<float> energy;
  std::vector<float> time;
  std::vector<float> timeError;
  std::vector<unsigned> detid;
  std::vector<unsigned> flagBits;
  std::vector<float> son;
};

class HeterogeneousHGCalRecHitsValidator : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HeterogeneousHGCalRecHitsValidator(const edm::ParameterSet&);
  ~HeterogeneousHGCalRecHitsValidator() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  static const unsigned nsubdetectors = 3;      //ce-e, ce-h-fine, ce-h-coarse
  static const unsigned ncomputingdevices = 2;  //cpu, gpu
  //cpu amd gpu tokens and handles for the 3 subdetectors, cpu and gpu
  std::array<std::array<edm::EDGetTokenT<HGChefRecHitCollection>, ncomputingdevices>, nsubdetectors> tokens_;
  std::array<edm::ESGetToken<HGCalGeometry, IdealGeometryRecord>, nsubdetectors> estokens_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> estokenGeom_;

  std::array<std::string, nsubdetectors> handles_str_ = {
      {"HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"}};
  hgcal::RecHitTools recHitTools_;

  std::array<TTree*, nsubdetectors> trees_;
  std::array<std::string, nsubdetectors> treenames_;
  std::array<ValidHitCollection, nsubdetectors> cpuValidRecHits, gpuValidRecHits, diffsValidRecHits;
  //std::vector< TH1F* > zhist;

  void set_geometry_(const edm::EventSetup&, const unsigned&);
};

#endif
