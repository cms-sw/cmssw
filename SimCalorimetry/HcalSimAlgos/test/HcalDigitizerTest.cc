#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapes.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"
#include "CondFormats/DataRecord/interface/HcalTimeSlewRecord.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "CLHEP/Random/JamesRandom.h"
#include <vector>
#include <iostream>
#include <iterator>

class HcalTimeSlew;

class HcalDigitizerTest : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HcalDigitizerTest(const edm::ParameterSet& iConfig);
  ~HcalDigitizerTest() override;

private:
  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  HcalDbHardcode dbHardcode;
  std::vector<PCaloHit> hits;
  std::vector<DetId> hcalDetIds, hoDetIds, hfDetIds, hzdcDetIds, allDetIds;
  std::vector<HcalDetId> outerHcalDetIds;

  const HcalTimeSlew* hcalTimeSlew_delay_;
};

HcalDigitizerTest::HcalDigitizerTest(const edm::ParameterSet& iConfig) {
  //DB helper preparation
  dbHardcode.setHB(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("hb")));
  dbHardcode.setHE(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("he")));
  dbHardcode.setHF(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("hf")));
  dbHardcode.setHO(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("ho")));
  dbHardcode.setHBUpgrade(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("hbUpgrade")));
  dbHardcode.setHEUpgrade(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("heUpgrade")));
  dbHardcode.setHFUpgrade(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("hfUpgrade")));
  dbHardcode.useHBUpgrade(iConfig.getParameter<bool>("useHBUpgrade"));
  dbHardcode.useHEUpgrade(iConfig.getParameter<bool>("useHEUpgrade"));
  dbHardcode.useHFUpgrade(iConfig.getParameter<bool>("useHFUpgrade"));
  dbHardcode.testHFQIE10(iConfig.getParameter<bool>("testHFQIE10"));

  hcalTimeSlew_delay_ = nullptr;
}

HcalDigitizerTest::~HcalDigitizerTest() {}

void HcalDigitizerTest::beginJob() {
  // make a silly little hit in each subdetector, which should
  // correspond to a 100 GeV particle

  for (int phi = 1; phi < 50; ++phi) {
    HcalDetId detId(HcalBarrel, 1, phi, 1);
    PCaloHit barrelHit(detId.rawId(), 0.085 * phi, 0.);
    hcalDetIds.push_back(detId);
    hits.push_back(barrelHit);
  }

  HcalDetId endcapDetId(HcalEndcap, 17, 1, 1);
  PCaloHit endcapHit(endcapDetId.rawId(), 0.9, 0.);
  hcalDetIds.push_back(endcapDetId);
  hits.push_back(endcapHit);

  HcalDetId outerDetId(HcalOuter, 1, 1, 4);
  PCaloHit outerHit(outerDetId.rawId(), 0.45, 0.);
  hoDetIds.push_back(outerDetId);
  outerHcalDetIds.push_back(outerDetId);
  hits.push_back(outerHit);

  HcalDetId forwardDetId1(HcalForward, 30, 1, 1);
  PCaloHit forwardHit1(forwardDetId1.rawId(), 35.2, 0.);
  hfDetIds.push_back(forwardDetId1);
  hits.push_back(forwardHit1);

  HcalDetId forwardDetId2(HcalForward, 30, 1, 2);
  PCaloHit forwardHit2(forwardDetId2.rawId(), 47.8, 0.);
  hfDetIds.push_back(forwardDetId2);
  hits.push_back(forwardHit2);

  HcalZDCDetId zdcDetId(HcalZDCDetId::Section(1), true, 1);
  PCaloHit zdcHit(zdcDetId.rawId(), 50.0, 0.123);
  hzdcDetIds.push_back(zdcDetId);
  hits.push_back(zdcHit);

  std::cout << zdcDetId << std::endl;
  std::cout << zdcHit << std::endl;

  allDetIds.insert(allDetIds.end(), hcalDetIds.begin(), hcalDetIds.end());
  allDetIds.insert(allDetIds.end(), hoDetIds.begin(), hoDetIds.end());
  allDetIds.insert(allDetIds.end(), hfDetIds.begin(), hfDetIds.end());
  allDetIds.insert(allDetIds.end(), hzdcDetIds.begin(), hzdcDetIds.end());
}

void HcalDigitizerTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<HcalRecNumberingRecord>().get(htopo);
  HcalTopology topology = (*htopo);

  edm::ESHandle<HcalTimeSlew> delay;
  iSetup.get<HcalTimeSlewRecord>().get("HBHE", delay);
  hcalTimeSlew_delay_ = &*delay;

  std::string hitsName = "HcalHits";
  std::vector<std::string> caloDets;
  CrossingFrame<PCaloHit> crossingFrame(-5, 5, 25, hitsName, 0);
  crossingFrame.addSignals(&hits, iEvent.id());

  // make 1 GeV pileup hit
  HcalDetId barrelDetId(HcalBarrel, 1, 1, 1);
  PCaloHit barrelPileup(barrelDetId.rawId(), 0.00855, 0.);
  // 10 GeV pileup hit
  HcalDetId forwardDetId1(HcalForward, 30, 1, 1);
  PCaloHit forwardPileup(forwardDetId1.rawId(), 3.52, 0.);
  HcalZDCDetId zdcDetId(HcalZDCDetId::Section(1), true, 1);
  PCaloHit zdcPileup(zdcDetId.rawId(), 3.52, 0.);

  std::vector<PCaloHit> pileups;
  pileups.push_back(barrelPileup);
  pileups.push_back(forwardPileup);
  pileups.push_back(zdcPileup);
  ///TODO fix once the new crossingframe is released
  //crossingFrame.addPileupCaloHits(-3, hitsName, &pileups);
  // -or -
  // crossingFrame.addPileupCaloHits(-3, hitsName, &pileups, 0);
  HcalSimParameterMap parameterMap;
  HcalSimParameterMap siPMParameterMap = parameterMap;
  siPMParameterMap.setHOZecotekDetIds(outerHcalDetIds);
  HcalShape hcalShape;
  HcalSiPMShape sipmShape;
  HFShape hfShape;
  ZDCShape zdcShape;

  CaloShapeIntegrator hcalShapeIntegrator(new HcalShape());
  CaloShapeIntegrator sipmShapeIntegrator(new HcalSiPMShape());
  CaloShapeIntegrator hfShapeIntegrator(new HFShape());
  CaloShapeIntegrator zdcShapeIntegrator(new ZDCShape());
  CaloShapes sipmShapes(&sipmShapeIntegrator);
  //for(float t = -25; t < 200; t += 5) {
  //  std::cout <<  t << " " << hcalShape(t) << "  " << sipmShape(t) << "  " << hcalShapeIntegrator(t) << "  "<< sipmShapeIntegrator(t) << std::endl;
  //}

  CaloHitResponse hbheResponse(&parameterMap, &hcalShapeIntegrator);
  CaloHitResponse hoResponse(&parameterMap, &hcalShapeIntegrator);
  CaloHitResponse hfResponse(&parameterMap, &hfShapeIntegrator);
  CaloHitResponse zdcResponse(&parameterMap, &zdcShapeIntegrator);
  HcalSiPMHitResponse hoSiPMResponse(&siPMParameterMap, &sipmShapes);

  CLHEP::HepJamesRandom randomEngine;

  HBHEHitFilter hbheHitFilter;
  HOHitFilter hoHitFilter;
  HFHitFilter hfHitFilter;
  ZDCHitFilter zdcHitFilter;

  hbheResponse.setHitFilter(&hbheHitFilter);
  hoSiPMResponse.setHitFilter(&hoHitFilter);
  hoResponse.setHitFilter(&hoHitFilter);
  hfResponse.setHitFilter(&hfHitFilter);
  zdcResponse.setHitFilter(&zdcHitFilter);

  HcalPedestals pedestals(&topology);
  HcalPedestalWidths pedestalWidths(&topology);
  HcalGains gains(&topology);
  HcalGainWidths gainWidths(&topology);
  // make a calibration service by hand
  for (auto detItr = allDetIds.begin(); detItr != allDetIds.end(); ++detItr) {
    pedestals.addValues(dbHardcode.makePedestal(*detItr, false, false, NULL, 0.0));
    pedestalWidths.addValues(dbHardcode.makePedestalWidth(*detItr, false, NULL, 0.0));
    gains.addValues(dbHardcode.makeGain(*detItr));
    gainWidths.addValues(dbHardcode.makeGainWidth(*detItr));
  }

  //pedestals.sort();
  //pedestalWidths.sort();
  //gains.sort();
  //gainWidths.sort();

  //std::cout << "TEST Pedestal " << pedestals.getValues(barrelDetId,  1) << std::endl;
  //std::cout << "ZDC pedestal " << pedestals.getValue(zdcDetId,  1) << std::endl;
  //std::cout << "ZDC pedestal width " << pedestalWidths.getWidth(zdcDetId,  1) << std::endl;
  //std::cout << "ZDC gain " << gains.getValue(zdcDetId,  1) << std::endl;
  //std::cout << "ZDC gain width " << gainWidths.getValue(zdcDetId,  1) << std::endl;

  HcalDbService calibratorHandle;
  calibratorHandle.setData(&pedestals);
  calibratorHandle.setData(&pedestalWidths);
  calibratorHandle.setData(&gains);
  calibratorHandle.setData(&gainWidths);

  bool addNoise = false;
  bool PM1 = false;
  bool PM2 = false;
  HcalAmplifier amplifier(&parameterMap, addNoise, PM1, PM2);
  HcalCoderFactory coderFactory(HcalCoderFactory::NOMINAL);
  HcalElectronicsSim electronicsSim(&parameterMap, &amplifier, &coderFactory, PM1);
  amplifier.setDbService(&calibratorHandle);
  amplifier.setTimeSlew(hcalTimeSlew_delay_);
  parameterMap.setDbService(&calibratorHandle);
  siPMParameterMap.setDbService(&calibratorHandle);

  CaloTDigitizer<HBHEDigitizerTraits> hbheDigitizer(&hbheResponse, &electronicsSim, addNoise);
  CaloTDigitizer<HODigitizerTraits> hoDigitizer(&hoResponse, &electronicsSim, addNoise);
  CaloTDigitizer<HFDigitizerTraits> hfDigitizer(&hfResponse, &electronicsSim, addNoise);
  CaloTDigitizer<ZDCDigitizerTraits> zdcDigitizer(&zdcResponse, &electronicsSim, addNoise);

  hbheDigitizer.setDetIds(hcalDetIds);
  hfDigitizer.setDetIds(hfDetIds);
  hoDigitizer.setDetIds(hoDetIds);
  zdcDigitizer.setDetIds(hzdcDetIds);

  std::unique_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection);
  std::unique_ptr<HODigiCollection> hoResult(new HODigiCollection);
  std::unique_ptr<HFDigiCollection> hfResult(new HFDigiCollection);
  std::unique_ptr<ZDCDigiCollection> zdcResult(new ZDCDigiCollection);

  MixCollection<PCaloHit> hitCollection(&crossingFrame);

  std::cout << "HBHE " << std::endl;
  hbheResponse.run(hitCollection, &randomEngine);
  std::cout << "SIPM " << std::endl;
  hoSiPMResponse.run(hitCollection, &randomEngine);
  //hbheDigitizer.run(hitCollection, *hbheResult);
  //hoDigitizer.run(hitCollection, *hoResult);
  //hfDigitizer.run(hitCollection, *hfResult);
  //zdcDigitizer.run(hitCollection, *zdcResult);

  // print out all the digis
  std::cout << "HBHE Frames" << std::endl;
  std::copy(hbheResult->begin(), hbheResult->end(), std::ostream_iterator<HBHEDataFrame>(std::cout, "\n"));

  std::cout << "HF Frames" << std::endl;
  std::copy(hfResult->begin(), hfResult->end(), std::ostream_iterator<HFDataFrame>(std::cout, "\n"));

  std::cout << "ZDC Frames" << std::endl;
  std::copy(zdcResult->begin(), zdcResult->end(), std::ostream_iterator<ZDCDataFrame>(std::cout, "\n"));
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDigitizerTest);
