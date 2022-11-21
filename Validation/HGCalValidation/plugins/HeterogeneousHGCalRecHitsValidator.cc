#include "Validation/HGCalValidation/plugins/HeterogeneousHGCalRecHitsValidator.h"

HeterogeneousHGCalRecHitsValidator::HeterogeneousHGCalRecHitsValidator(const edm::ParameterSet &ps)
    : tokens_({{{{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("cpuRecHitsEEToken")),
                  consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("gpuRecHitsEEToken"))}},
                {{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("cpuRecHitsHSiToken")),
                  consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("gpuRecHitsHSiToken"))}},
                {{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("cpuRecHitsHSciToken")),
                  consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("gpuRecHitsHSciToken"))}}}}),
      treenames_({{"CEE", "CHSi", "CHSci"}}) {
  usesResource(TFileService::kSharedResource);
  estokenGeom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  edm::Service<TFileService> fs;
  for (unsigned i(0); i < nsubdetectors; ++i) {
    estokens_[i] = esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", handles_str_[i]});
    trees_[i] = fs->make<TTree>(treenames_[i].c_str(), treenames_[i].c_str());
    trees_[i]->Branch("cpu", "ValidHitCollection", &cpuValidRecHits[i]);
    trees_[i]->Branch("gpu", "ValidHitCollection", &gpuValidRecHits[i]);
    trees_[i]->Branch("diffs", "ValidHitCollection", &diffsValidRecHits[i]);
  }
}

HeterogeneousHGCalRecHitsValidator::~HeterogeneousHGCalRecHitsValidator() {}

void HeterogeneousHGCalRecHitsValidator::endJob() {}

void HeterogeneousHGCalRecHitsValidator::set_geometry_(const edm::EventSetup &setup, const unsigned &detidx) {
  edm::ESHandle<HGCalGeometry> handle = setup.getHandle(estokens_[detidx]);
}

void HeterogeneousHGCalRecHitsValidator::analyze(const edm::Event &event, const edm::EventSetup &setup) {
  recHitTools_.setGeometry(setup.getData(estokenGeom_));

  //future subdetector loop
  for (size_t idet = 0; idet < nsubdetectors; ++idet) {
    set_geometry_(setup, idet);

    //get hits produced with the CPU
    const auto &cpuhits = event.get(tokens_[idet][0]);

    //get hits produced with the GPU
    const auto &gpuhits = event.get(tokens_[idet][1]);

    size_t nhits = cpuhits.size();
    std::cout << nhits << ", " << gpuhits.size() << std::endl;
    assert(nhits == gpuhits.size());
    //float sum_cpu = 0.f, sum_gpu = 0.f, sum_son_cpu = 0.f, sum_son_gpu = 0.f;
    for (unsigned i(0); i < nhits; i++) {
      const HGCRecHit &cpuHit = cpuhits[i];
      const HGCRecHit &gpuHit = gpuhits[i];

      const float cpuEn = cpuHit.energy();
      const float gpuEn = gpuHit.energy();
      //sum_cpu += cpuEn; sum_gpu += gpuEn;

      const float cpuTime = cpuHit.time();
      const float gpuTime = gpuHit.time();
      const float cpuTimeErr = cpuHit.timeError();
      const float gpuTimeErr = gpuHit.timeError();
      const DetId cpuDetId = cpuHit.detid();
      const DetId gpuDetId = gpuHit.detid();
      const float cpuFB = cpuHit.flagBits();
      const float gpuFB = gpuHit.flagBits();
      const float cpuSoN = cpuHit.signalOverSigmaNoise();
      const float gpuSoN = gpuHit.signalOverSigmaNoise();
      //sum_son_cpu += cpuSoN; sum_son_gpu += gpuSoN;

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
