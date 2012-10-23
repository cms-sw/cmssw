#ifndef RegressionEnergyPatElectronProducer_h
#define RegressionEnergyPatElectronProducer_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "EGamma/EGammaAnalysisTools/interface/ElectronEnergyRegressionEvaluate.h"

class RegressionEnergyPatElectronProducer: public edm::EDProducer 
{
 public:

  explicit RegressionEnergyPatElectronProducer( const edm::ParameterSet & ) ;
  virtual ~RegressionEnergyPatElectronProducer();
  virtual void produce( edm::Event &, const edm::EventSetup & ) ;

 private:

  // input collections
  edm::InputTag inputElectrons_ ;
  edm::InputTag rhoInputTag_ ;
  edm::InputTag verticesInputTag_ ;
  edm::InputTag recHitCollectionEB_;
  edm::InputTag recHitCollectionEE_;

  // 
  bool useReducedRecHits_;

  //output collection
  //  std::string outputCollection_;
  std::string nameEnergyReg_;
  std::string nameEnergyErrorReg_;
  
  uint32_t energyRegressionType_ ;
  uint32_t inputCollectionType_ ;
  std::string regressionInputFile_;
  bool debug_ ;
  ElectronEnergyRegressionEvaluate *regressionEvaluator_;
  bool geomInitialized_;
  bool producePatElectrons_;
  bool produceValueMaps_;
  

  const CaloTopology * ecalTopology_;
  const CaloGeometry * caloGeometry_;
  unsigned nElectrons_;
} ;

#endif
