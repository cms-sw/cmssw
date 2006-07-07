/** \class EcalTrigPrimProducer
 *
 * EcalTrigPrimProducer produces a EcalTrigPrimDigiCollection
 * The barrel code does a detailed simulation
 * The code for the endcap is simulated in a rough way, due to missing strip geometry
 *
 *
 * \author Ursula Berthon, Stephanie Baffioni,  LLR Palaiseau
 *
 * \version   1st Version may 2006
 * \version   2nd Version jul 2006

 *
 ************************************************************/
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Common/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/Provenance.h"

#include "SimCalorimetry/EcalTrigPrimProducers/interface/EcalTrigPrimProducer.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalTrigPrimFunctionalAlgo.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "TFile.h"
#include "TTree.h"

  
EcalTrigPrimProducer::EcalTrigPrimProducer(const edm::ParameterSet& iConfig)
{
  //register your products
  produces <EcalTrigPrimDigiCollection >();

  valid_= iConfig.getUntrackedParameter<bool>("Validation");
  if (valid_) {
    histfile_ = new TFile("valid.root","UPDATE");
    valTree_ = new TTree("V","Validation Tree");
  } else{
    histfile_=NULL;
    valTree_=NULL;
  }
  
  label_= iConfig.getParameter<std::string>("Label");
  fgvbMinEnergy_=  iConfig.getParameter<int>("FgvbMinEnergy");
  algo_=NULL;
  //FIXME: add configuration
					      
}

void EcalTrigPrimProducer::beginJob(edm::EventSetup const& setup) {
  //FIXME add config for validation

  //get  binOfMax
    edm::Service<edm::ConstProductRegistry> reg;
    // Loop over provenance of products in registry.
    for (edm::ProductRegistry::ProductList::const_iterator it = reg->productList().begin();
	 it != reg->productList().end(); ++it) {
      edm::BranchDescription desc = it->second;
      if (!desc.friendlyClassName_.compare(0,18,"EBDataFramesSorted")) {
      edm::ParameterSet result;
      edm::pset::Registry::instance()->getParameterSet(desc.psetID(), result);
      binOfMaximum_=result.getParameter<int>("binOfMaximum");
      break;
      }
    }
    algo_ = new EcalTrigPrimFunctionalAlgo(setup, valTree_,binOfMaximum_,nrSamples_);
    edm::LogInfo("constructor") <<" EcalTrigPrimProducer built with nrSamples: "<<nrSamples_<<" found binOfMaximum = "<<binOfMaximum_<<endl;
}

EcalTrigPrimProducer::~EcalTrigPrimProducer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  delete algo_;
  if (valid_) {
    histfile_->Write();
    histfile_->Close();
  }

}


// ------------ method called to produce the data  ------------
void
EcalTrigPrimProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
{

  //  using namespace edm;
  edm::Handle<EBDigiCollection> ebDigis;
  e.getByLabel(label_,ebDigis);
  edm::Handle<EEDigiCollection> eeDigis;
  e.getByLabel(label_,eeDigis);

  LogDebug("Startproduce") <<" =================> Treating event "<<e.id()<<", Number of EBDFataFrames "<<ebDigis.product()->size() ;
  std::auto_ptr<EcalTrigPrimDigiCollection> pOut(new EcalTrigPrimDigiCollection);
  

  // invoke algorithm  //FIXME: better separation 
  //   algo_->setupES(iSetup);
  algo_->run(ebDigis.product(),eeDigis.product(),*pOut, fgvbMinEnergy_);
  for (unsigned int i=0;i<pOut->size();++i) {
    for (int isam=0;isam<(*pOut)[i].size();++isam) {
      if ((*pOut)[i][isam].raw()) LogDebug("Produced for ") <<" Tower "<<i<<", sample "<<isam<<", value "<<(*pOut)[i][isam].raw();
    }
  }
	
  // put result into the Event
  e.put(pOut);
}


