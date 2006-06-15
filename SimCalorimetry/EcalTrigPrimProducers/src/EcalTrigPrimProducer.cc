#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
  
  //FIXME: add configuration
					      
}

void EcalTrigPrimProducer::beginJob(edm::EventSetup const& setup) {
  //FIXME add config for validation
  algo_ = new EcalTrigPrimFunctionalAlgo(setup, valTree_);
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

  using namespace edm;
  edm::Handle<EBDigiCollection> ebDigis;
  e.getByType(ebDigis);

  LogDebug("Startproduce") <<" =================> Treating event "<<e.id()<<", Number of EBDFataFrames "<<ebDigis.product()->size() ;
  
  //  std::auto_ptr<std::vector <EcalTriggerPrimitiveDigi> > pOut(new std::vector <EcalTriggerPrimitiveDigi>);
  std::auto_ptr<EcalTrigPrimDigiCollection> pOut(new EcalTrigPrimDigiCollection);
  
  // invoke algorithm  //FIXME: better separation
  //   algo_->setupES(iSetup);
  algo_->run(ebDigis.product(),*pOut);
  for (unsigned int i=0;i<pOut->size();++i) {
    for (int isam=0;isam<(*pOut)[i].size();++isam) {
      if ((*pOut)[i][isam].raw()) LogDebug("Produced for ") <<" Tower "<<i<<", sample "<<isam<<", value "<<(*pOut)[i][isam].raw();
    }
  }
	
  // put result into the Event
  e.put(pOut);
}


