#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalNoisifier.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"

#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"

#include <vector>
#include<iostream>
#include<iterator>
using namespace std;
using namespace cms;

int main() {
  // make a silly little hit in each subdetector, which should
  // correspond to a 100 GeV particle
  HcalDetId barrelDetId(HcalBarrel, 1, 1, 1);
  PCaloHit barrelHit(barrelDetId.rawId(),  0.855, 0.);

  HcalDetId endcapDetId(HcalEndcap, 17, 1, 1);
  PCaloHit endcapHit(endcapDetId.rawId(), 0.9, 0.);

  HcalDetId outerDetId(HcalOuter, 1, 1, 4);
  PCaloHit outerHit(outerDetId.rawId(), 0.45, 0.);

  HcalDetId forwardDetId1(HcalForward, 30, 1, 1);
  PCaloHit forwardHit1(forwardDetId1.rawId(), 35., 0.);

  HcalDetId forwardDetId2(HcalForward, 30, 1, 2);
  PCaloHit forwardHit2(forwardDetId2.rawId(), 48., 0.);

  vector<DetId> hcalDetIds, hoDetIds, hfDetIds;
  hcalDetIds.push_back(barrelDetId);
  hcalDetIds.push_back(endcapDetId);
  hoDetIds.push_back(outerDetId);
  hfDetIds.push_back(forwardDetId1);
  hfDetIds.push_back(forwardDetId2);

  vector<PCaloHit> hbheHits, hoHits, hfHits;
  hbheHits.push_back(barrelHit);
  hbheHits.push_back(endcapHit);
  hoHits.push_back(outerHit);
  hfHits.push_back(forwardHit1);
  hfHits.push_back(forwardHit2);

  HcalSimParameterMap parameterMap;
  HcalShape hcalShape;
  HFShape hfShape;

  CaloShapeIntegrator hcalShapeIntegrator(&hcalShape);
  CaloShapeIntegrator hfShapeIntegrator(&hfShape);

  CaloHitResponse hcalResponse(&parameterMap, &hcalShapeIntegrator);
  CaloHitResponse hfResponse(&parameterMap, &hfShapeIntegrator);

  HcalDbServiceHardcode dbService;
  HcalDbService calibratorHandle(&dbService);
  HcalNoisifier noisifier;
  noisifier.setDbService(&calibratorHandle);
  HcalNominalCoder coder; 
  HcalElectronicsSim electronicsSim(&noisifier, &coder);

  CaloTDigitizer<HBHEDigitizerTraits> hbheDigitizer(&hcalResponse, &electronicsSim);
  CaloTDigitizer<HODigitizerTraits> hoDigitizer(&hcalResponse, &electronicsSim);
  CaloTDigitizer<HFDigitizerTraits> hfDigitizer(&hfResponse, &electronicsSim);
  hbheDigitizer.setDetIds(hcalDetIds);
  hfDigitizer.setDetIds(hfDetIds);
  hoDigitizer.setDetIds(hoDetIds);

  auto_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection);
  auto_ptr<HODigiCollection> hoResult(new HODigiCollection);
  auto_ptr<HFDigiCollection> hfResult(new HFDigiCollection);

  hbheDigitizer.run(hbheHits, *hbheResult);
  hoDigitizer.run(hoHits, *hoResult);
  hfDigitizer.run(hfHits, *hfResult);

  // print out all the digis
  cout << "HBHE Frames" << endl;
  copy(hbheResult->begin(), hbheResult->end(), std::ostream_iterator<HBHEDataFrame>(std::cout, "\n"));

  cout << "HF Frames" << endl;
  copy(hfResult->begin(), hfResult->end(), std::ostream_iterator<HFDataFrame>(std::cout, "\n"));

  cout << "SHAPES" << endl;
  for(unsigned i = 0; i < 25; ++i) {
     cout << i << " " << hcalShape(i) << " " << hfShape(i) << " " << hcalShapeIntegrator(i) << " " << hfShapeIntegrator(i) << endl;
  }

return 0;
}


