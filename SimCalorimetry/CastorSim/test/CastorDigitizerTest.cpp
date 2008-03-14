#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameterMap.h"
#include "SimCalorimetry/CastorSim/src/CastorShape.h"
#include "SimCalorimetry/CastorSim/src/CastorElectronicsSim.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "SimCalorimetry/CastorSim/src/CastorAmplifier.h"
#include "SimCalorimetry/CastorSim/src/CastorCoderFactory.h"
#include "SimCalorimetry/CastorSim/src/CastorHitFilter.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimCalorimetry/CastorSim/src/CastorDigitizerTraits.h"
#include "SimCalorimetry/CastorSim/src/CastorHitCorrection.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include "Geometry/ForwardGeometry/interface/CastorHardcodeGeometryLoader.h"
#include "CLHEP/Random/JamesRandom.h"
#include <vector>
#include<iostream>
#include<iterator>
using namespace std;
using namespace cms;

void testHitCorrection(CastorHitCorrection * correction, MixCollection<PCaloHit> & hits)
{
  correction->fillChargeSums(hits);
  for(MixCollection<PCaloHit>::MixItr hitItr = hits.begin();
      hitItr != hits.end(); ++hitItr)
    {
      DetId detId((*hitItr).id());
      if (detId.det()==DetId::Calo && detId.subdetId()==HcalCastorDetId::SubdetectorId){
	std::cout<<"Castor -- ";
    } 
      std::cout << "HIT charge " << correction->charge(*hitItr) << " delay " 
		//<< correction->delay(*hitItr)
		<< " Timebin " << correction->timeBin(*hitItr) <<std::endl;
    }
}


int main() {
  // make a silly little hit in each subdetector, which should
  // correspond to a 100 GeV particle

  HcalCastorDetId castorDetId(HcalCastorDetId::Section(2), true, 1, 1);
  PCaloHit castorHit(castorDetId.rawId(), 50.0, 0.123);
 
  std::cout<<castorDetId<<std::endl;
  std::cout<<castorHit<<std::endl;

  vector<DetId> hcastorDetIds;
  hcastorDetIds.push_back(castorDetId);

  vector<DetId> allDetIds;
  allDetIds.insert(allDetIds.end(), hcastorDetIds.begin(), hcastorDetIds.end());

  vector<PCaloHit> hits;
  hits.push_back(castorHit);

  string hitsName = "HcalHits";
  vector<string> caloDets;

  CrossingFrame<PCaloHit> crossingFrame(-5, 5, 25,  hitsName, 1);
  edm::EventID eventId;
  crossingFrame.addSignals(&hits, eventId);

  // make 1 GeV pileup hit
  PCaloHit castorPileup(castorDetId.rawId(), 3.52, 0.);

  vector<PCaloHit> pileups;
  pileups.push_back(castorPileup);
  ///TODO fix once the new crossingframe is released
  //crossingFrame.addPileupCaloHits(-3, hitsName, &pileups);
  // -or -
  // crossingFrame.addPileupCaloHits(-3, hitsName, &pileups, 0);
  CastorSimParameterMap parameterMap;
  CastorShape castorShape;

  CaloShapeIntegrator castorShapeIntegrator(&castorShape);

  CaloHitResponse castorResponse(&parameterMap, &castorShapeIntegrator);

  CastorHitCorrection hitCorrection(&parameterMap);
  castorResponse.setHitCorrection(&hitCorrection);

  CLHEP::HepJamesRandom randomEngine;
  castorResponse.setRandomEngine(randomEngine);

  CastorHitFilter castorHitFilter;

  castorResponse.setHitFilter(&castorHitFilter);

  HcalPedestals pedestals;
  HcalPedestalWidths pedestalWidths;
  HcalGains gains;
  HcalGainWidths gainWidths;
  // make a calibration service by hand
  for(vector<DetId>::const_iterator detItr = allDetIds.begin(); detItr != allDetIds.end(); ++detItr) {
    pedestals.addValue(*detItr, HcalDbHardcode::makePedestal(*detItr).getValues ());
    *pedestalWidths.setWidth(*detItr) = HcalDbHardcode::makePedestalWidth(*detItr);
    gains.addValue(*detItr, HcalDbHardcode::makeGain(*detItr).getValues ());
    gainWidths.addValue(*detItr, HcalDbHardcode::makeGainWidth(*detItr).getValues ());
  }
  
  pedestals.sort();
  pedestalWidths.sort();
  gains.sort();
  gainWidths.sort();

//  std::cout << "TEST Pedestal " << pedestals.getValue(barrelDetId,  1) << std::endl;
  std::cout << "Castor pedestal " << pedestals.getValue(castorDetId,  1) << std::endl;
  std::cout << "Castor pedestal width " << pedestalWidths.getWidth(castorDetId,  1) << std::endl;
  std::cout << "Castor gain " << gains.getValue(castorDetId,  1) << std::endl;
  std::cout << "Castor gain width " << gainWidths.getValue(castorDetId,  1) << std::endl;

  HcalDbService calibratorHandle;
  calibratorHandle.setData(&pedestals);
  calibratorHandle.setData(&pedestalWidths);
  calibratorHandle.setData(&gains);
  calibratorHandle.setData(&gainWidths);

  bool addNoise = false;
  CastorAmplifier amplifier(&parameterMap, addNoise);
  CastorCoderFactory coderFactory(CastorCoderFactory::NOMINAL);
  CastorElectronicsSim electronicsSim(&amplifier, &coderFactory);
  amplifier.setDbService(&calibratorHandle);
  amplifier.setRandomEngine(randomEngine);
  electronicsSim.setRandomEngine(randomEngine);
  parameterMap.setDbService(&calibratorHandle);

  CaloTDigitizer<CastorDigitizerTraits> castorDigitizer(&castorResponse, &electronicsSim, addNoise);

  castorDigitizer.setDetIds(hcastorDetIds);

  auto_ptr<CastorDigiCollection> castorResult(new CastorDigiCollection);

//  MixCollection<PCaloHit> hitCollection(&crossingFrame, hitsName);
  MixCollection<PCaloHit> hitCollection(&crossingFrame);

  testHitCorrection(&hitCorrection, hitCollection);

  castorDigitizer.run(hitCollection, *castorResult);

  cout << "Castor Frames" << endl;
  copy(castorResult->begin(), castorResult->end(), std::ostream_iterator<CastorDataFrame>(std::cout, "\n"));
  
  return 0;
}


 
