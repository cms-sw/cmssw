#ifndef ECALDDIGIPRODUCER_H
#define ECALDDIGIPRODUCER_H

using namespace std;
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Provenance.h"

#include "SimCalorimetry/CaloSimAlgos/interface/CaloDigiCollectionSorter.h"

class EcalDigiProducer : public edm::EDProducer
{
public:

  // The following is not yet used, but will be the primary
  // constructor when the parameter set system is available.
  //
  explicit EcalDigiProducer(const edm::ParameterSet& params);
  virtual ~EcalDigiProducer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);

private:
  // some hits in each subdetector, just for testing purposes
  void fillFakeHits();

  void checkGeometry(const edm::EventSetup & eventSetup);
  void updateGeometry();

  void checkCalibrations(const edm::EventSetup & eventSetup);

  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<EBDigitizerTraits> EBDigitizer;
  typedef CaloTDigitizer<EEDigitizerTraits> EEDigitizer;
  typedef CaloTDigitizer<ESDigitizerTraits> ESDigitizer;

  EBDigitizer * theBarrelDigitizer;
  EEDigitizer * theEndcapDigitizer;
  ESDigitizer * theESDigitizer;

  const EcalSimParameterMap * theParameterMap;
  const CaloVShape * theEcalShape;
  const ESShape * theESShape;

  CaloHitResponse * theEcalResponse;
  CaloHitResponse * theESResponse;

  EcalElectronicsSim * theElectronicsSim;
  ESElectronicsSim * theESElectronicsSim;
  EcalCoder * theCoder;

  const CaloGeometry * theGeometry;
  std::vector<DetId> theBarrelDets;
  std::vector<DetId> theEndcapDets;
  std::vector<DetId> theESDets;
  EcalPedestals thePedestals;
  void setupFakePedestals();

};

#endif 
