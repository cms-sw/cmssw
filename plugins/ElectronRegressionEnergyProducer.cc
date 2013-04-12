// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyRegressionEvaluate.h"

//
// class declaration
//

using namespace std;
using namespace reco;
using namespace edm;

class ElectronRegressionEnergyProducer : public edm::EDFilter {
public:
  explicit ElectronRegressionEnergyProducer(const edm::ParameterSet&);
  ~ElectronRegressionEnergyProducer();
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
  // ----------member data ---------------------------
  bool printDebug_;
  edm::InputTag electronTag_;

  std::string regressionInputFile_;
  uint32_t energyRegressionType_;

  std::string nameEnergyReg_;
  std::string nameEnergyErrorReg_;
  
  edm::InputTag recHitCollectionEB_;
  edm::InputTag recHitCollectionEE_;

  ElectronEnergyRegressionEvaluate *regressionEvaluator;

};


ElectronRegressionEnergyProducer::ElectronRegressionEnergyProducer(const edm::ParameterSet& iConfig) {
  printDebug_  = iConfig.getUntrackedParameter<bool>("printDebug", false);
  electronTag_ = iConfig.getParameter<edm::InputTag>("electronTag");
  
  regressionInputFile_  = iConfig.getParameter<std::string>("regressionInputFile");
  energyRegressionType_ = iConfig.getParameter<uint32_t>("energyRegressionType");

  nameEnergyReg_      = iConfig.getParameter<std::string>("nameEnergyReg");
  nameEnergyErrorReg_ = iConfig.getParameter<std::string>("nameEnergyErrorReg");

  recHitCollectionEB_ = iConfig.getParameter<edm::InputTag>("recHitCollectionEB");
  recHitCollectionEE_ = iConfig.getParameter<edm::InputTag>("recHitCollectionEE");

  produces<edm::ValueMap<double> >(nameEnergyReg_);
  produces<edm::ValueMap<double> >(nameEnergyErrorReg_);

  regressionEvaluator = new ElectronEnergyRegressionEvaluate();

  //set regression type
  ElectronEnergyRegressionEvaluate::ElectronEnergyRegressionType type = ElectronEnergyRegressionEvaluate::kNoTrkVar;
  if (energyRegressionType_ == 1) type = ElectronEnergyRegressionEvaluate::kNoTrkVar;
  else if (energyRegressionType_ == 2) type = ElectronEnergyRegressionEvaluate::kWithSubCluVar;
  else if (energyRegressionType_ == 3) type = ElectronEnergyRegressionEvaluate::kWithTrkVarV1;
  else if (energyRegressionType_ == 4) type = ElectronEnergyRegressionEvaluate::kWithTrkVarV2;

  //load weights and initialize
  regressionEvaluator->initialize(regressionInputFile_.c_str(),type);

}


ElectronRegressionEnergyProducer::~ElectronRegressionEnergyProducer()
{
  delete regressionEvaluator;
}

// ------------ method called on each new Event  ------------
bool ElectronRegressionEnergyProducer::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  assert(regressionEvaluator->isInitialized());
  
  std::auto_ptr<edm::ValueMap<double> > regrEnergyMap(new edm::ValueMap<double>() );
  edm::ValueMap<double>::Filler energyFiller(*regrEnergyMap);

  std::auto_ptr<edm::ValueMap<double> > regrEnergyErrorMap(new edm::ValueMap<double>() );
  edm::ValueMap<double>::Filler energyErrorFiller(*regrEnergyErrorMap);
  
  Handle<reco::GsfElectronCollection> egCollection;
  iEvent.getByLabel(electronTag_,egCollection);
  const reco::GsfElectronCollection egCandidates = (*egCollection.product());

  std::vector<double> energyValues;
  std::vector<double> energyErrorValues;
  energyValues.reserve(egCollection->size());
  energyErrorValues.reserve(egCollection->size());

  //**************************************************************************
  //Tool for Cluster shapes
  //**************************************************************************
  EcalClusterLazyTools lazyTools(iEvent, iSetup, 
                                 recHitCollectionEB_, 
                                 recHitCollectionEE_);  

  //**************************************************************************
  //Get Number of Vertices
  //**************************************************************************
  Handle<reco::VertexCollection> hVertexProduct;
  iEvent.getByLabel(edm::InputTag("offlinePrimaryVertices"),hVertexProduct);
  const reco::VertexCollection inVertices = *(hVertexProduct.product());  

  // loop through all vertices
  Int_t nvertices = 0;
  for (reco::VertexCollection::const_iterator inV = inVertices.begin(); 
       inV != inVertices.end(); ++inV) {
    
    // pass these vertex cuts
    if (inV->ndof() >= 4
        && inV->position().Rho() <= 2.0
        && fabs(inV->z()) <= 24.0
      ) {
      nvertices++;
    }
  }

  //**************************************************************************
  //Get Rho
  //**************************************************************************
  double rho = 0;
  Handle<double> hRhoKt6PFJets;
  iEvent.getByLabel(edm::InputTag("kt6PFJets","rho"), hRhoKt6PFJets);
  rho = (*hRhoKt6PFJets);


  for ( reco::GsfElectronCollection::const_iterator egIter = egCandidates.begin(); 
        egIter != egCandidates.end(); ++egIter) {

    double energy=regressionEvaluator->calculateRegressionEnergy(&(*egIter),
                                                          lazyTools,
                                                          iSetup,
                                                          rho,nvertices,
                                                          printDebug_);
    
    double error=regressionEvaluator->calculateRegressionEnergyUncertainty(&(*egIter),
                                                                    lazyTools,
                                                                    iSetup,
                                                                    rho,nvertices,
                                                                    printDebug_);

    energyValues.push_back(energy);
    energyErrorValues.push_back(error);

  }

  energyFiller.insert( egCollection, energyValues.begin(), energyValues.end() );  
  energyFiller.fill();

  energyErrorFiller.insert( egCollection, energyErrorValues.begin(), energyErrorValues.end() );  
  energyErrorFiller.fill();
  
  iEvent.put(regrEnergyMap,nameEnergyReg_);
  iEvent.put(regrEnergyErrorMap,nameEnergyErrorReg_);
  
  return true;

}


//define this as a plug-in
DEFINE_FWK_MODULE(ElectronRegressionEnergyProducer);



