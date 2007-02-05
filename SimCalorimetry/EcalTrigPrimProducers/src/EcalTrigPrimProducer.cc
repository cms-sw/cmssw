/** \class EcalTrigPrimProducer
 *
 * EcalTrigPrimProducer produces a EcalTrigPrimDigiCollection
 * The barrel code does a detailed simulation
 * The code for the endcap is simulated in a rough way, due to missing  strip geometry
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
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Common/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "DataFormats/Common/interface/BranchDescription.h"

#include  "SimCalorimetry/EcalTrigPrimProducers/interface/EcalTrigPrimProducer.h"
#include  "SimCalorimetry/EcalTrigPrimAlgos/interface/DBInterface.h"
#include  "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalTrigPrimFunctionalAlgo.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "TFile.h"
#include "TTree.h"

const int EcalTrigPrimProducer::nrSamples_=5;

EcalTrigPrimProducer::EcalTrigPrimProducer(const edm::ParameterSet&  iConfig)
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
  instanceNameEB_ = iConfig.getParameter<std::string>("InstanceEB");;
  instanceNameEE_ = iConfig.getParameter<std::string>("InstanceEE");;
  databaseFileNameEB_ = iConfig.getParameter<std::string>("DatabaseFileEB");;
  databaseFileNameEE_ = iConfig.getParameter<std::string>("DatabaseFileEE");;
  //  fgvbMinEnergy_=  iConfig.getParameter<int>("FgvbMinEnergy");
  //  ttfThreshLow_ =  iConfig.getParameter<double>("TTFLowEnergy");
  //  ttfThreshHigh_=  iConfig.getParameter<double>("TTFHighEnergy");
  algo_=NULL;
  db_=NULL;
  //FIXME: add configuration
                   
}

void EcalTrigPrimProducer::beginJob(edm::EventSetup const& setup) {
  //FIXME add config for validation

  //get  binOfMax
  try {
    binOfMaximum_=0;
    edm::Service<edm::ConstProductRegistry> reg;
    // Loop over provenance of products in registry.
    for (edm::ProductRegistry::ProductList::const_iterator it =  reg->productList().begin();
     it != reg->productList().end(); ++it) {
      edm::BranchDescription desc = it->second;
      if (!desc.friendlyClassName().compare(0,18,"EBDataFramesSorted")  & desc.moduleLabel()==label_ ) {
    edm::ParameterSet result = getParameterSet(desc.psetID());
    binOfMaximum_=result.getParameter<int>("binOfMaximum");
    break;
      }
    }
  }
  catch(cms::Exception& e) {
    // segv in case product was found but not parameter..
    edm::LogWarning("")<<"Could not find parameter binOfMaximum in  product registry for EBDataFramesSorted, had to set binOfMaximum by  Hand";
    binOfMaximum_=6;
  }
  if (binOfMaximum_==0) {
    edm::LogWarning("")<<"Could not find product registry of  EBDataFramesSorted, had to set binOfMaximum by Hand";
    binOfMaximum_=6;
  }

  db_ = new DBInterface(databaseFileNameEB_,databaseFileNameEE_);
  printf("=============> Created DBinterface %p\n",db_);fflush(stdout);
  algo_ = new EcalTrigPrimFunctionalAlgo(setup,  valTree_,binOfMaximum_,nrSamples_,db_);
  edm::LogInfo("constructor") <<"EcalTrigPrimProducer will write:  "<<nrSamples_<<" samples for each digi,  binOfMaximum used:  "<<binOfMaximum_;
}

EcalTrigPrimProducer::~EcalTrigPrimProducer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  printf("========================> 1\n");fflush(stdout);
  delete algo_;
  printf("========================>2\n");fflush(stdout);
  if (valid_) {
    histfile_->Write();
    histfile_->Close();
  }
  printf("========================>3\n");fflush(stdout);
  printf("=============> destroying DBinterface %p\n",db_);fflush(stdout);
   delete db_ ;
  printf("========================>4\n");fflush(stdout);

}


// ------------ method called to produce the data  ------------
void
EcalTrigPrimProducer::produce(edm::Event& e, const edm::EventSetup&  iSetup)
{

  edm::Handle<EBDigiCollection> ebDigis;
  edm::Handle<EEDigiCollection> eeDigis;
  bool barrel=true, endcap=true;
  try{e.getByLabel(label_,instanceNameEB_,ebDigis);}
  catch(cms::Exception &e) {
    barrel=false;
    edm::LogWarning("produce") <<" Couldnt find Barrel dataframes";
  }
  try{e.getByLabel(label_,instanceNameEE_,eeDigis);}
  catch(cms::Exception &e) {
    endcap=false;
    edm::LogWarning("produce") <<" Couldnt find Endcap dataframes";
  }

  LogDebug("Startproduce") <<" =================> Treating event  "<<e.id()<<", Number of EBDFataFrames "<<ebDigis.product()->size() ;
  std::auto_ptr<EcalTrigPrimDigiCollection> pOut(new  EcalTrigPrimDigiCollection);


  // invoke algorithm  //FIXME: better separation
  const EBDigiCollection *ebdc=NULL;
  const EEDigiCollection *eedc=NULL;
  if (barrel) ebdc=ebDigis.product();
  if (endcap) eedc=eeDigis.product();
  algo_->run(ebdc,eedc,*pOut);
  for (unsigned int i=0;i<pOut->size();++i) {
    for (int isam=0;isam<(*pOut)[i].size();++isam) {
      if ((*pOut)[i][isam].raw()) LogDebug("Produced for ") <<" Tower  "<<i<<", sample "<<isam<<", value "<<(*pOut)[i][isam].raw();
    }
  }
  edm::LogInfo("produce") <<"For Barrel + Endcap, "<<pOut->size()<<" TP  Digis were produced";
    
  // put result into the Event
  e.put(pOut);
}
