#ifndef ECALTBDDIGIPRODUCER_H
#define ECALTBDDIGIPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

#include "SimCalorimetry/EcalTestBeamAlgos/interface/EcalTBReadout.h"

#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCSample.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"
#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRecInfoAlgo.h"

#include <string>

class EcalTBDigiProducer : public edm::EDProducer
{
public:

  // The following is not yet used, but will be the primary
  // constructor when the parameter set system is available.
  //
  explicit EcalTBDigiProducer(const edm::ParameterSet& params);
  virtual ~EcalTBDigiProducer();

  /**Produces the EDM products,*/
  void produce(edm::Event& event, const edm::EventSetup& eventSetup);

private:

  void checkGeometry(const edm::EventSetup & eventSetup);

  void updateGeometry();

  void checkCalibrations(const edm::EventSetup & eventSetup);

  void setPhaseShift(const DetId & detId);

  void fillTBTDCRawInfo(EcalTBTDCRawInfo & theTBTDCRawInfo);

  /** Reconstruction algorithm*/
  typedef EcalTDigitizer<EBDigitizerTraits> EBDigitizer;
  //typedef CaloTDigitizer<EBDigitizerTraits> EBDigitizer;

  EBDigitizer * theBarrelDigitizer;

  const EcalSimParameterMap * theParameterMap;
  const CaloVShape * theEcalShape;

  CaloHitResponse * theEcalResponse;

  CorrelatedNoisifier * theCorrNoise;
  EcalCorrelatedNoiseMatrix * theNoiseMatrix;

  EcalElectronicsSim * theElectronicsSim;
  EcalCoder * theCoder;

  const CaloGeometry * theGeometry;
  std::vector<DetId> theBarrelDets;

  const EcalTrigTowerConstituentsMap * theTTmap;

  EcalTBReadout * theTBReadout;

  std::string ecalTBInfoLabel;

  bool doPhaseShift;
  double thisPhaseShift;

  bool doReadout;

  std::vector<EcalTBTDCRecInfoAlgo::EcalTBTDCRanges> tdcRanges;
  bool use2004OffsetConvention_;

  double tunePhaseShift;

  std::string EBdigiCollection_;

  double EBs25notCont;
  double EEs25notCont;  
};

#endif 
