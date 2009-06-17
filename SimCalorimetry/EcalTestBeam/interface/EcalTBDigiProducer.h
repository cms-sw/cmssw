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

  bool cosmicsPhase;
  double cosmicsShift;

  bool doFast; 
};

#endif 
