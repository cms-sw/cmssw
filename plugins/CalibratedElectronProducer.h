// This file is imported from 

#ifndef CalibratedElectronProducer_h
#define CalibratedElectronProducer_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"


class CalibratedElectronProducer: public edm::EDProducer 
 {
  public:

    //static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit CalibratedElectronProducer( const edm::ParameterSet & ) ;
    virtual ~CalibratedElectronProducer();
    virtual void produce( edm::Event &, const edm::EventSetup & ) ;

  private:

    edm::InputTag inputElectrons_ ;
    edm::InputTag nameEnergyReg_;
    edm::InputTag nameEnergyErrorReg_;
    edm::InputTag recHitCollectionEB_ ;
    edm::InputTag recHitCollectionEE_ ;

    std::string nameNewEnergyReg_ ;
    std::string nameNewEnergyErrorReg_;

    std::string dataset ;
    bool isAOD ;
    bool isMC ;
    bool updateEnergyError ;
    int applyCorrections ;
    bool verbose ;
    bool synchronization ;

    const CaloTopology * ecalTopology_;
    const CaloGeometry * caloGeometry_;
    bool geomInitialized_;
    std::string newElectronName_;

 } ;

#endif
