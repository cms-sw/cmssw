#include "Validation/HGCalValidation/plugins/HeterogeneousHGCalRecHitsValidator.h"

#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

HeterogeneousHGCalRecHitsValidator::HeterogeneousHGCalRecHitsValidator(const edm::ParameterSet &ps)
    : tokens_({{{{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("cpuRecHitsEEToken")),
                  consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("gpuRecHitsEEToken"))}},
                {{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("cpuRecHitsHSiToken")),
                  consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("gpuRecHitsHSiToken"))}},
                {{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("cpuRecHitsHSciToken")),
                  consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("gpuRecHitsHSciToken"))}}}}),
      treenames_({{"CEE", "CHSi", "CHSci"}}) {
  edm::Service<TFileService> fs;
  for (unsigned int i = 0; i < nsubdetectors; ++i) {
    trees_[i] = fs->make<TTree>(treenames_[i].c_str(), treenames_[i].c_str());
    trees_[i]->Branch("cpu", "ValidHitCollection", &cpuValidRecHits[i]);
    trees_[i]->Branch("gpu", "ValidHitCollection", &gpuValidRecHits[i]);
    trees_[i]->Branch("diffs", "ValidHitCollection", &diffsValidRecHits[i]);
  }
}

HeterogeneousHGCalRecHitsValidator::~HeterogeneousHGCalRecHitsValidator() {}

void HeterogeneousHGCalRecHitsValidator::endJob() {}

void HeterogeneousHGCalRecHitsValidator::set_geometry_(const edm::EventSetup &setup, const unsigned int &detidx) {
  edm::ESHandle<HGCalGeometry> handle;
  setup.get<IdealGeometryRecord>().get(handles_str_[detidx], handle);
}

void HeterogeneousHGCalRecHitsValidator::analyze(const edm::Event &event, const edm::EventSetup &setup) {
  edm::ESHandle<CaloGeometry> baseGeom;
  setup.get<CaloGeometryRecord>().get(baseGeom);
  recHitTools_.setGeometry(*baseGeom);

  //future subdetector loop
  for (size_t idet = 0; idet < nsubdetectors; ++idet) {
    set_geometry_(setup, idet);

    //get hits produced with the CPU
    event.getByToken(tokens_[idet][0], handles_[idet][0]);
    const auto &cpuhits = *handles_[idet][0];

    //get hits produced with the GPU
    event.getByToken(tokens_[idet][1], handles_[idet][1]);
    const auto &gpuhits = *handles_[idet][1];

    size_t nhits = cpuhits.size();
    assert(nhits == gpuhits.size());
    float sum_cpu = 0.f;
    float sum_gpu = 0.f;
    float sum_son_cpu = 0.f;
    float sum_son_gpu = 0.f;
    for (unsigned int i = 0; i < nhits; i++) {
      const HGCRecHit &cpuHit = cpuhits[i];
      const HGCRecHit &gpuHit = gpuhits[i];

      const float cpuEn = cpuHit.energy();
      sum_cpu += cpuEn;
      const float gpuEn = gpuHit.energy();
      sum_gpu += gpuEn;

      const float cpuTime = cpuHit.time();
      const float gpuTime = gpuHit.time();
      const float cpuTimeErr = cpuHit.timeError();
      const float gpuTimeErr = gpuHit.timeError();
      const HGCalDetId cpuDetId = cpuHit.detid();
      const HGCalDetId gpuDetId = gpuHit.detid();
      const float cpuFB = cpuHit.flagBits();
      const float gpuFB = gpuHit.flagBits();
      const float cpuSoN = cpuHit.signalOverSigmaNoise();
      sum_son_cpu += cpuSoN;
      const float gpuSoN = gpuHit.signalOverSigmaNoise();
      sum_son_gpu += gpuSoN;

      ValidHit vCPU(cpuEn, cpuTime, cpuTimeErr, cpuDetId, cpuFB, cpuSoN);
      ValidHit vGPU(gpuEn, gpuTime, gpuTimeErr, gpuDetId, gpuFB, gpuSoN);
      ValidHit vDiffs(cpuEn - gpuEn,
                      cpuTime - gpuTime,
                      cpuTimeErr - gpuTimeErr,
                      cpuDetId - gpuDetId,
                      cpuFB - gpuFB,
                      cpuSoN - gpuSoN);

      cpuValidRecHits[idet].push_back(vCPU);
      gpuValidRecHits[idet].push_back(vGPU);
      diffsValidRecHits[idet].push_back(vDiffs);
    }
    trees_[idet]->Fill();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HeterogeneousHGCalRecHitsValidator);
