#ifndef ECALTBDIGIPRODUCER_H
#define ECALTBDIGIPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
//TB#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
//TB#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSimFast.h"
//TB#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
//TB#include "SimCalorimetry/EcalSimAlgos/interface/ESFastTDigitizer.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/Math/interface/Error.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

//For TB ----------------------------
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "SimCalorimetry/EcalTestBeamAlgos/interface/EcalTBReadout.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCSample.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"
#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRecInfoAlgo.h"
//For TB ----------------------------

class EcalTBDigiProducer : public edm::EDProducer
{
public:

  // The following is not yet used, but will be the primary
  // constructor when the parameter set system is available.
  //
  explicit EcalTBDigiProducer(const edm::ParameterSet& params);
  virtual ~EcalTBDigiProducer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);

private:

  void checkGeometry(const edm::EventSetup & eventSetup);

  void updateGeometry();

  void checkCalibrations(const edm::EventSetup & eventSetup);

  void setPhaseShift(const DetId & detId);

  void fillTBTDCRawInfo(EcalTBTDCRawInfo & theTBTDCRawInfo);

  /** Reconstruction algorithm*/
  typedef EcalTDigitizer<EBDigitizerTraits> EBDigitizer;
  typedef EcalTDigitizer<EEDigitizerTraits> EEDigitizer;
//TB  typedef CaloTDigitizer<ESDigitizerTraits> ESDigitizer;

  EBDigitizer * theBarrelDigitizer;
  EEDigitizer * theEndcapDigitizer;
//TB  ESDigitizer * theESDigitizer;
//TB  ESFastTDigitizer * theESDigitizerFast;

  const EcalSimParameterMap * theParameterMap;
  const CaloVShape * theEcalShape;
//TB  const ESShape * theESShape;

  CaloHitResponse * theEcalResponse;
//TB  CaloHitResponse * theESResponse;

  CorrelatedNoisifier<EcalCorrMatrix> * theCorrNoise;
  EcalCorrelatedNoiseMatrix * theNoiseMatrix;

  EcalElectronicsSim * theElectronicsSim;
//TB  ESElectronicsSim * theESElectronicsSim;
//TB  ESElectronicsSimFast * theESElectronicsSimFast;
  EcalCoder * theCoder;

  const CaloGeometry * theGeometry;

  std::string EBdigiCollection_;
  std::string EEdigiCollection_;
//TB  std::string ESdigiCollection_;

  std::string hitsProducer_;

  double EBs25notCont;
  double EEs25notCont;

//For TB -------------------------------------------

  const EcalTrigTowerConstituentsMap * theTTmap;

  EcalTBReadout * theTBReadout;

  std::string ecalTBInfoLabel;

  bool doPhaseShift;
  double thisPhaseShift;

  bool doReadout;

  std::vector<EcalTBTDCRecInfoAlgo::EcalTBTDCRanges> tdcRanges;
  bool use2004OffsetConvention_;

  double tunePhaseShift;
//For TB ---------------------

//TB  bool cosmicsPhase;
//TB  double cosmicsShift;

//TB  bool doFast; 
};

#endif 
