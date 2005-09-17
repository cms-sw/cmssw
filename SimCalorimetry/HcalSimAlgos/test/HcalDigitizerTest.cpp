#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHit.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"
#include "CalibFormats/HcalObjects/interface/HcalDbServiceHandle.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalNoisifier.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"

#include <vector>
#include<iostream>
#include<iterator>
using namespace std;
using namespace cms;

int main() {
  // make a silly little hit in each subdetector, which should
  // correspond to a 100 GeV particle
  HcalDetId barrelDetId(HcalBarrel, 1, 1, 1);
  CaloHit barrelHit(barrelDetId, 0.855, 0., 0);

  HcalDetId endcapDetId(HcalEndcap, 17, 1, 1);
  CaloHit endcapHit(endcapDetId, 0.9, 0., 0);

  HcalDetId outerDetId(HcalOuter, 1, 1, 4);
  CaloHit outerHit(outerDetId, 0.45, 0., 0);

  HcalDetId forwardDetId1(HcalForward, 30, 1, 1);
  CaloHit forwardHit1(forwardDetId1, 35., 0., 0);

  HcalDetId forwardDetId2(HcalForward, 30, 1, 2);
  CaloHit forwardHit2(forwardDetId2, 48., 0., 0);

  vector<DetId> hcalDetIds, hoDetIds, hfDetIds;
  hcalDetIds.push_back(barrelDetId);
  hcalDetIds.push_back(endcapDetId);
  hoDetIds.push_back(outerDetId);
  hfDetIds.push_back(forwardDetId1);
  hfDetIds.push_back(forwardDetId2);

  vector<CaloHit> hbheHits, hoHits, hfHits;
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
  HcalDbServiceHandle calibratorHandle(&dbService);
  HcalNoisifier noisifier(&calibratorHandle);
  HcalNominalCoder coder; 
  HcalElectronicsSim electronicsSim(&noisifier, &coder);

  CaloTDigitizer<HBHEDataFrame, HcalElectronicsSim> hbheDigitizer(&hcalResponse, &electronicsSim, hcalDetIds);
  CaloTDigitizer<HODataFrame, HcalElectronicsSim> hoDigitizer(&hcalResponse, &electronicsSim, hoDetIds);
  CaloTDigitizer<HFDataFrame, HcalElectronicsSim> hfDigitizer(&hfResponse, &electronicsSim, hfDetIds);

  auto_ptr<vector<HBHEDataFrame> > hbheResult(new vector<HBHEDataFrame>);
  auto_ptr<vector<HODataFrame> > hoResult(new vector<HODataFrame>);
  auto_ptr<vector<HFDataFrame> > hfResult(new vector<HFDataFrame>);

  hbheDigitizer.run(hbheHits, hbheResult);
  hoDigitizer.run(hoHits, hoResult);
  hfDigitizer.run(hfHits, hfResult);

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


