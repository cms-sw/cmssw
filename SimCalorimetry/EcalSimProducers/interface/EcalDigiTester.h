#ifndef ECALDIGITESTER_H
#define ECALDIGITESTER_H

using namespace std ;

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
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
#include "DataFormats/Common/interface/Provenance.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "SimCalorimetry/CaloSimAlgos/interface/CaloDigiCollectionSorter.h"

class EcalDigiTester : public edm::EDAnalyzer
{
public:

  explicit EcalDigiTester (const edm::ParameterSet& params) ; 
  virtual ~EcalDigiTester () ;

  virtual void beginJob (EventSetup const&) {} ;
  virtual void endJob () {} ;
  virtual void analyze (const edm::Event& event, const edm::EventSetup& eventSetup) ;

private:
  // some hits in each subdetector, just for testing purposes
  void fillFakeHits () ;

  void checkGeometry (const edm::EventSetup & eventSetup) ;
  void checkCalibrations (const edm::EventSetup & eventSetup) ;

  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<EBDigitizerTraits> EBDigitizer ;
  typedef CaloTDigitizer<EEDigitizerTraits> EEDigitizer ;

  const EcalSimParameterMap * theParameterMap ;

  std::vector<DetId> theBarrelDets ;
  std::vector<DetId> theEndcapDets ;
  
  const CaloGeometry * theGeometry ;
  

} ;


class simpleUnit
  {
    public: 
      simpleUnit (double eta,double phi,double E) ;
      ~simpleUnit () ;
      int m_ieta ;
      int m_iphi ;
      double m_eta ;
      double m_phi ;
      double m_E ;
      bool operator< (const simpleUnit& altro) const ;
  } ;

std::ostream & operator<< (std::ostream & os, const simpleUnit & su) ;

bool compare (const simpleUnit& primo, const simpleUnit secondo) ; 

#endif
