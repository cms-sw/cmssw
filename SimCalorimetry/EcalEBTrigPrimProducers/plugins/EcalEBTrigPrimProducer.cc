/** \class EcalEBTrigPrimProducer
 * For Phase II
 * EcalEBTrigPrimProducer produces a EcalEBTrigPrimDigiCollection
 * out of PhaseI Digis. This is a simple starting point to fill in the chain
 * for Phase II
 * 
 *
 *
 ************************************************************/
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
//
#include "CondFormats/DataRecord/interface/EcalTPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSlidingWindowRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSpikeRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"


/*
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainTowerEERcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainStripEERcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
*/

#include "EcalEBTrigPrimProducer.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBTrigPrimTestAlgo.h"

EcalEBTrigPrimProducer::EcalEBTrigPrimProducer(const edm::ParameterSet&  iConfig):
  barrelOnly_(iConfig.getParameter<bool>("BarrelOnly")),
  tcpFormat_(iConfig.getParameter<bool>("TcpOutput")),
  debug_(iConfig.getParameter<bool>("Debug")),
  famos_(iConfig.getParameter<bool>("Famos")),
  useRecHits_(iConfig.getParameter<bool>("UseRecHits")),
  nSamples_(iConfig.getParameter<int>("nOfSamples")),
  binOfMaximum_(iConfig.getParameter<int>("binOfMaximum"))
{  
  tokenEBrh_=consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("barrelEcalHits"));
  tokenEBdigi_=consumes<EBDigiCollection>(iConfig.getParameter<edm::InputTag>("barrelEcalDigis"));
  //register your products
  produces <EcalEBTrigPrimDigiCollection >();
  if (tcpFormat_) produces <EcalEBTrigPrimDigiCollection >("formatTCP");
}



void EcalEBTrigPrimProducer::beginRun(edm::Run const & run,edm::EventSetup const& setup) {
  //ProcessHistory is guaranteed to be constant for an entire Run
  //binOfMaximum_ = findBinOfMaximum(fillBinOfMaximumFromHistory_,binOfMaximum_,run.processHistory());

  algo_.reset( new EcalEBTrigPrimTestAlgo(setup,nSamples_,binOfMaximum_,tcpFormat_,barrelOnly_,debug_, famos_) );
 // get a first version of the records
  cacheID_=this->getRecords(setup);
  nEvent_=0;
}

unsigned long long  EcalEBTrigPrimProducer::getRecords(edm::EventSetup const& setup) {

  // get parameter records for xtals
  edm::ESHandle<EcalTPGLinearizationConst> theEcalTPGLinearization_handle;
  setup.get<EcalTPGLinearizationConstRcd>().get(theEcalTPGLinearization_handle);
  const EcalTPGLinearizationConst * ecaltpLin = theEcalTPGLinearization_handle.product();
  //
  edm::ESHandle<EcalTPGPedestals> theEcalTPGPedestals_handle;
  setup.get<EcalTPGPedestalsRcd>().get(theEcalTPGPedestals_handle);
  const EcalTPGPedestals * ecaltpPed = theEcalTPGPedestals_handle.product();
  //
  edm::ESHandle<EcalTPGCrystalStatus> theEcalTPGCrystalStatus_handle;
  setup.get<EcalTPGCrystalStatusRcd>().get(theEcalTPGCrystalStatus_handle);
  const EcalTPGCrystalStatus * ecaltpgBadX = theEcalTPGCrystalStatus_handle.product();
  //
  //for strips
  //
  edm::ESHandle<EcalTPGWeightIdMap> theEcalTPGWEightIdMap_handle;
  setup.get<EcalTPGWeightIdMapRcd>().get(theEcalTPGWEightIdMap_handle);
  const EcalTPGWeightIdMap * ecaltpgWeightMap = theEcalTPGWEightIdMap_handle.product();
  //
  edm::ESHandle<EcalTPGWeightGroup> theEcalTPGWEightGroup_handle;
  setup.get<EcalTPGWeightGroupRcd>().get(theEcalTPGWEightGroup_handle);
  const EcalTPGWeightGroup * ecaltpgWeightGroup = theEcalTPGWEightGroup_handle.product();
  // 
  edm::ESHandle<EcalTPGSlidingWindow> theEcalTPGSlidingWindow_handle;
  setup.get<EcalTPGSlidingWindowRcd>().get(theEcalTPGSlidingWindow_handle);
  const EcalTPGSlidingWindow * ecaltpgSlidW = theEcalTPGSlidingWindow_handle.product();
  //  TCP 
  edm::ESHandle<EcalTPGLutGroup> theEcalTPGLutGroup_handle;
  setup.get<EcalTPGLutGroupRcd>().get(theEcalTPGLutGroup_handle);
  const EcalTPGLutGroup * ecaltpgLutGroup = theEcalTPGLutGroup_handle.product();
  //
  edm::ESHandle<EcalTPGLutIdMap> theEcalTPGLutIdMap_handle;
  setup.get<EcalTPGLutIdMapRcd>().get(theEcalTPGLutIdMap_handle);
  const EcalTPGLutIdMap * ecaltpgLut = theEcalTPGLutIdMap_handle.product();
  //
  edm::ESHandle<EcalTPGTowerStatus> theEcalTPGTowerStatus_handle;
  setup.get<EcalTPGTowerStatusRcd>().get(theEcalTPGTowerStatus_handle);
  const EcalTPGTowerStatus * ecaltpgBadTT = theEcalTPGTowerStatus_handle.product();
  //
  edm::ESHandle<EcalTPGSpike> theEcalTPGSpike_handle;
  setup.get<EcalTPGSpikeRcd>().get(theEcalTPGSpike_handle);
  const EcalTPGSpike * ecaltpgSpike = theEcalTPGSpike_handle.product();



  ////////////////
  algo_->setPointers(ecaltpLin,ecaltpPed,ecaltpgBadX,ecaltpgWeightMap,ecaltpgWeightGroup,ecaltpgSlidW,ecaltpgLutGroup,ecaltpgLut,ecaltpgBadTT, ecaltpgSpike);
  return setup.get<EcalTPGLinearizationConstRcd>().cacheIdentifier();

 


}



void EcalEBTrigPrimProducer::endRun(edm::Run const& run,edm::EventSetup const& setup) {
  algo_.reset();
}


EcalEBTrigPrimProducer::~EcalEBTrigPrimProducer()
{}


// ------------ method called to produce the data  ------------
void
EcalEBTrigPrimProducer::produce(edm::Event& e, const edm::EventSetup&  iSetup)
{

  nEvent_++;

  // get input collections
  edm::Handle<EcalRecHitCollection> barrelHitHandle;
  edm::Handle<EBDigiCollection> barrelDigiHandle;
  
  if ( useRecHits_ ) {
    if (! e.getByToken(tokenEBrh_,barrelHitHandle)) {
      edm::EDConsumerBase::Labels labels;
      labelsForToken(tokenEBrh_, labels);
      edm::LogWarning("EcalTPG") <<" Couldnt find Barrel rechits "<<labels.module<<" and label "<<labels.productInstance<<"!!!";
    }   
  }else { 
    if (! e.getByToken(tokenEBdigi_,barrelDigiHandle)) {
      edm::EDConsumerBase::Labels labels;
      labelsForToken(tokenEBdigi_, labels);
      edm::LogWarning("EcalTPG") <<" Couldnt find Barrel digis "<<labels.module<<" and label "<<labels.productInstance<<"!!!";
    }
  }
  
 

  //LogDebug("EcalTPG") <<" =================> Treating event  "<<e.id()<<", Number of EB rechits "<<barrelHitHandle.product()->size();
  if ( useRecHits_ ) {
    if (debug_) std::cout << "EcalTPG" <<" =================> Treating event  "<< nEvent_<<", Number of EB rechits "<<barrelHitHandle.product()->size() << std::endl;
  }else{
    if (debug_) std::cout << "EcalTPG" <<" =================> Treating event  "<< nEvent_<<", Number of EB digis "<<barrelDigiHandle.product()->size() << std::endl;
  }
  auto pOut = std::make_unique<EcalEBTrigPrimDigiCollection>();
  auto pOutTcp = std::make_unique<EcalEBTrigPrimDigiCollection>();
 
  // if ( e.id().event() != 648 ) return;

  //std::cout << " Event number " << e.id().event() << std::endl;

  // invoke algorithm 

  const EcalRecHitCollection *ebrh=NULL;
  const EBDigiCollection *ebdigi=NULL;
  if ( useRecHits_ ) {
    ebrh=barrelHitHandle.product();
    algo_->run(iSetup,ebrh,*pOut,*pOutTcp);
  } else {
    ebdigi=barrelDigiHandle.product();
    algo_->run(iSetup,ebdigi,*pOut,*pOutTcp);
  }

  if (debug_ ) std::cout << "produce" << " For Barrel  "<<pOut->size()<<" TP  Digis were produced" << std::endl;

  //  debug prints if TP >0

  int nonZeroTP=0;
  for (unsigned int i=0;i<pOut->size();++i) {
   
    if (debug_  ) {
      std::cout << "EcalTPG Printing only non zero TP " <<" For tower  "<<(((*pOut)[i])).id()<<", TP is "<<(*pOut)[i];
      for (int isam=0;isam<(*pOut)[i].size();++isam) {
	
        if (  (*pOut)[i][isam].compressedEt() > 0)  {
	  nonZeroTP++;
	  std::cout << " (*pOut)[i][isam].raw() "  <<  (*pOut)[i][isam].raw() << "  (*pOut)[i][isam].compressedEt() " <<  (*pOut)[i][isam].compressedEt() <<  std::endl;
	}
      }
    }
  }
  if (debug_ ) std::cout << "EcalTPG" <<"\n =================> For Barrel , "<<pOut->size()<<" TP  Digis were produced (including zero ones)" << " Non zero primitives were " << nonZeroTP << std::endl;
 
   

  // put result into the Event
  e.put(std::move(pOut));
  if (tcpFormat_) e.put(std::move(pOutTcp),"formatTCP");
}
