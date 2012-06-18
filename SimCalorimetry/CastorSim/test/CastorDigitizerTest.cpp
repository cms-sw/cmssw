#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameterMap.h"
#include "SimCalorimetry/CastorSim/src/CastorShape.h"
#include "SimCalorimetry/CastorSim/src/CastorElectronicsSim.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorDbHardcode.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
//#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "SimCalorimetry/CastorSim/src/CastorAmplifier.h"
#include "SimCalorimetry/CastorSim/src/CastorCoderFactory.h"
#include "SimCalorimetry/CastorSim/src/CastorHitFilter.h"
#include "CondFormats/CastorObjects/interface/CastorPedestals.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidths.h"
#include "CondFormats/CastorObjects/interface/CastorGains.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidths.h"
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

void testHitCorrection(CastorHitCorrection * correction, std::vector<PCaloHit> & hits)
{
  correction->fillChargeSums(hits);
  for(PCaloHit& hit : hits)
    {
      DetId detId(hit.id());
      if (detId.det()==DetId::Calo && detId.subdetId()==HcalCastorDetId::SubdetectorId){
	std::cout<<"Castor -- ";
    } 
      std::cout << "HIT charge " << correction->charge(hit) << " delay " 
		//<< correction->delay(*hitItr)
		<< " Timebin " << correction->timeBin(hit) <<std::endl;
    }
}

int main() {
  // make a silly little hit in each subdetector, which should
  // correspond to a 100 GeV particle

  HcalCastorDetId castorDetId(HcalCastorDetId::Section(2), true, 1, 1);
  PCaloHit castorHit(castorDetId.rawId(), 50.0, 0.123);

  //assert(castorDetId.zside() == true);
   
  std::cout<<castorDetId<<std::endl;
  std::cout<<castorHit<<std::endl;

  vector<DetId> hcastorDetIds;
  hcastorDetIds.push_back(castorDetId);

  /*
    DEBUG
  HcalGenericDetId genericId(castorDetId);

  std::cout<<"generic Id "<< genericId <<std::endl;
  std::cout<< genericId.det() <<" "<< genericId.subdetId() <<std::endl;
 
  std::cout<<"castor id "<< castorDetId <<std::endl;
  std::cout<< castorDetId.det() <<" "<< castorDetId.subdetId() <<std::endl;

  std::cout<<"should be: "<< DetId::Calo <<" "<< HcalCastorDetId::SubdetectorId <<std::endl;
  */
  /*
    DEBUG
  vector<HcalCastorDetId> hcastorDetIds;
  hcastorDetIds.push_back(castorDetId);

  vector<HcalCastorDetId>::iterator testDetId = hcastorDetIds.begin();
  std::cout<< (*testDetId).zside() <<" "
	   << (*testDetId).sector() <<" "
	   << (*testDetId).module() <<std::endl;
  */

  vector<DetId> allDetIds;
  allDetIds.insert(allDetIds.end(), hcastorDetIds.begin(), hcastorDetIds.end());
  vector<PCaloHit> hits;
  hits.push_back(castorHit);

  string hitsName = "CastorHits";
  vector<string> caloDets;

  CrossingFrame<PCaloHit> crossingFrame(-5, 5, 25,  hitsName, 0);
  edm::EventID eventId;
  crossingFrame.addSignals(&hits, eventId);

  // make 1 GeV pileup hit
  PCaloHit castorPileup(castorDetId.rawId(), 3.52, 0.);

  vector<PCaloHit> pileups;
  pileups.push_back(castorPileup);
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

  CastorPedestals pedestals;
  CastorPedestalWidths pedestalWidths;
  CastorGains gains;
  CastorGainWidths gainWidths;

  // make a calibration service by hand
  for(vector<DetId>::const_iterator detItr = allDetIds.begin(); 
      detItr != allDetIds.end(); ++detItr) 
  {
/* check CastorCondObjectContainer!
      pedestals.addValues(*detItr, CastorDbHardcode::makePedestal(*detItr).getValues ());
      *pedestalWidths.setWidth(*detItr) = CastorDbHardcode::makePedestalWidth(*detItr);
      gains.addValues(*detItr, CastorDbHardcode::makeGain(*detItr).getValues ());
      gainWidths.addValues(*detItr, CastorDbHardcode::makeGainWidth(*detItr).getValues ());
*/
      pedestals.addValues(CastorDbHardcode::makePedestal(*detItr));
      pedestalWidths.addValues(CastorDbHardcode::makePedestalWidth(*detItr));
      gains.addValues(CastorDbHardcode::makeGain(*detItr));
      gainWidths.addValues(CastorDbHardcode::makeGainWidth(*detItr));

  }

/* obsolete stuff 

  pedestals.sort();
  pedestalWidths.sort();
  gains.sort();
  gainWidths.sort();


//  std::cout << "TEST Pedestal " << pedestals.getValue(barrelDetId,  1) << std::endl;
  std::cout << "Castor pedestals " << pedestals.getValue(castorDetId,  1) << std::endl;
  std::cout << "Castor pedestal widths " << pedestalWidths.getWidth(castorDetId,  1) << std::endl;
  std::cout << "Castor gains " << gains.getValue(castorDetId,  1) << std::endl;
  std::cout << "Castor gain widths " << gainWidths.getValue(castorDetId,  1) << std::endl;

*/

edm::ParameterSet emptyPSet;
CastorDbService calibratorHandle(emptyPSet);

//  CastorDbService calibratorHandle;
  calibratorHandle.setData(&pedestals);
  calibratorHandle.setData(&pedestalWidths);
  calibratorHandle.setData(&gains);
  calibratorHandle.setData(&gainWidths);
  cout << "set data" << std::endl;
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
  cout << "setDetIds" << std::endl;
  auto_ptr<CastorDigiCollection> castorResult(new CastorDigiCollection);
  cout << "castorResult" << std::endl;
  cout << "test hit correction" << std::endl;
  //something breaks here!
  testHitCorrection(&hitCorrection, hits);
  castorDigitizer.add(hits, 0);
  // TODO Add pileups
  //testHitCorrection(&hitCorrection, pileups);
  //castorDigitizer.add(pileups, -3);
 

  cout << "castordigitizer.run" << std::endl;
  castorDigitizer.run(*castorResult);

  cout << "Castor Frames" << std::endl;
  copy(castorResult->begin(), castorResult->end(), std::ostream_iterator<CastorDataFrame>(std::cout, "\n"));
  
  return 0;
}


 
