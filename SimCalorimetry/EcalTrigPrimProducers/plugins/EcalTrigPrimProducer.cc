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

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "EcalTrigPrimProducer.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalTrigPrimFunctionalAlgo.h"

EcalTrigPrimProducer::EcalTrigPrimProducer(const edm::ParameterSet&  iConfig):
  barrelOnly_(iConfig.getParameter<bool>("BarrelOnly")),
  tcpFormat_(iConfig.getParameter<bool>("TcpOutput")),
  debug_(iConfig.getParameter<bool>("Debug")),ps_(iConfig)
{
  //register your products
  produces <EcalTrigPrimDigiCollection >();
  if (tcpFormat_) produces <EcalTrigPrimDigiCollection >("formatTCP");

  label_= iConfig.getParameter<std::string>("Label");
  instanceNameEB_ = iConfig.getParameter<std::string>("InstanceEB");;
  instanceNameEE_ = iConfig.getParameter<std::string>("InstanceEE");;
  algo_=NULL;
}

void EcalTrigPrimProducer::beginJob(edm::EventSetup const& setup) {

  bool famos = ps_.getParameter<bool>("Famos");

  //get  binOfMax
  try {
    binOfMaximum_=0;
    edm::Service<edm::ConstProductRegistry> reg;
    // Loop over provenance of products in registry.
    for (edm::ProductRegistry::ProductList::const_iterator it =  reg->productList().begin();
	 it != reg->productList().end(); ++it) {
      edm::BranchDescription desc = it->second;
      if (desc.friendlyClassName().find("EBDigiCollection")==0  &&
	  desc.moduleLabel()==label_ ) {
	edm::ParameterSet result = getParameterSet(desc.psetID());
        binOfMaximum_=result.getParameter<int>("binOfMaximum");
	break;
      }
    }
  }
  catch(cms::Exception& e) {
    // segv in case product was found but not parameter..
    //FIXME: something wrong, binOfMaximum comes from somewhere else
    edm::LogWarning("EcalTPG")<<"Could not find parameter binOfMaximum in  product registry for EBDigiCollection, had to set binOfMaximum by  Hand";
    binOfMaximum_=6;
  }

  if (binOfMaximum_==0) {
    binOfMaximum_=6;
    edm::LogWarning("EcalTPG")<<"Could not find product registry of EBDataFramesSorted, had to set the following parameters by Hand:  binOfMaximum="<<binOfMaximum_;
  }

  algo_ = new EcalTrigPrimFunctionalAlgo(setup,binOfMaximum_,tcpFormat_,barrelOnly_,debug_,famos);
  algo_->updateESRecord(ps_.getParameter<double>("TTFLowEnergyEB"),
                        ps_.getParameter<double>("TTFHighEnergyEB"),
                        ps_.getParameter<double>("TTFLowEnergyEE"),
                        ps_.getParameter<double>("TTFHighEnergyEE"));
  edm::LogInfo("EcalTPG") <<"EcalTrigPrimProducer will use  binOfMaximum:  "<<binOfMaximum_;//FIXME
}

EcalTrigPrimProducer::~EcalTrigPrimProducer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  delete algo_;

}


// ------------ method called to produce the data  ------------
void
EcalTrigPrimProducer::produce(edm::Event& e, const edm::EventSetup&  iSetup)
{

  // get input collections

  edm::Handle<EBDigiCollection> ebDigis;
  edm::Handle<EEDigiCollection> eeDigis;
  bool barrel=true;
  bool endcap=true;
  if (barrelOnly_) endcap=false;

  try{e.getByLabel(label_,instanceNameEB_,ebDigis);}
  catch(cms::Exception &e) {
    barrel=false;
    edm::LogWarning("EcalTPG") <<" Couldnt find Barrel dataframes with producer "<<label_<<" and label "<<instanceNameEB_<<"!!!";
  }
  if (!barrelOnly_) {
    try{e.getByLabel(label_,instanceNameEE_,eeDigis);}
    catch(cms::Exception &e) {
      endcap=false;
      edm::LogWarning("EcalTPG") <<" Couldnt find Endcap dataframes with producer "<<label_<<" and label "<<instanceNameEE_<<"!!!";
    }
  }
  if (!barrel && !endcap) {
    throw cms::Exception(" ProductNotFound") <<"No EBDataFrames(EEDataFrames) with producer "<<label_<<" and label "<<instanceNameEB_<< "found in input!!\n";
  }

  if (!barrelOnly_)   LogDebug("EcalTPG") <<" =================> Treating event  "<<e.id()<<", Number of EBDataFrames "<<ebDigis.product()->size()<<", Number of EEDataFrames "<<eeDigis.product()->size() ;
  else  LogDebug("EcalTPG") <<" =================> Treating event  "<<e.id()<<", Number of EBDataFrames "<<ebDigis.product()->size();

  std::auto_ptr<EcalTrigPrimDigiCollection> pOut(new  EcalTrigPrimDigiCollection);
  std::auto_ptr<EcalTrigPrimDigiCollection> pOutTcp(new  EcalTrigPrimDigiCollection);
 

  // invoke algorithm 

  const EBDigiCollection *ebdc=NULL;
  const EEDigiCollection *eedc=NULL;
  if (barrel) {
    ebdc=ebDigis.product();
    algo_->run(ebdc,*pOut,*pOutTcp);
  }

  if (endcap) {
    eedc=eeDigis.product();
    algo_->run(eedc,*pOut,*pOutTcp);
  }

  edm::LogInfo("produce") <<"For Barrel + Endcap, "<<pOut->size()<<" TP  Digis were produced";

  //  debug prints if TP >0

  for (unsigned int i=0;i<pOut->size();++i) {
    bool print=false;
    for (int isam=0;isam<(*pOut)[i].size();++isam) {
      if ((*pOut)[i][isam].raw()) print=true;
    }
    if (print) LogDebug("EcalTPG") <<" For tower  "<<(((*pOut)[i])).id()<<", TP is "<<(*pOut)[i];
  }
  if (barrelOnly_)  edm::LogInfo("EcalTPG") <<"\n =================> For Barrel , "<<pOut->size()<<" TP  Digis were produced (including zero ones)";
  else      edm::LogInfo("EcalTPG") <<"\n =================> For Barrel + Endcap, "<<pOut->size()<<" TP  Digis were produced (including zero ones)";

  // put result into the Event

  e.put(pOut);
  if (tcpFormat_) e.put(pOutTcp,"formatTCP");
}
