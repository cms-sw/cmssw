#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

#include "SimCalorimetry/HcalTrigPrimProducers/src/HcalUpgradeTrigPrimDigiProducer.h"

//------------------------------------------------------
// Constructor, 
//   get tags for data collections
//   and instruction bools here.
//------------------------------------------------------

HcalUpgradeTrigPrimDigiProducer::HcalUpgradeTrigPrimDigiProducer(const edm::ParameterSet& iConfig):
  m_hbheDigisTag ( iConfig.getParameter<edm::InputTag>("hbheDigis")),
  m_hfDigisTag   ( iConfig.getParameter<edm::InputTag>("hfDigis"  )),  
  m_hcalUpgradeTriggerPrimitiveDigiAlgo ( new HcalUpgradeTriggerPrimitiveAlgo(iConfig.getParameter<bool>("peakFinder"),
									iConfig.getParameter<std::vector<double> >("weights"),
									iConfig.getParameter<int>("latency"),
									iConfig.getParameter<int>("FGThreshold"),
									iConfig.getParameter<int>("ZSThreshold"),
									iConfig.getParameter<int>("MinSignalThreshold"),
									iConfig.getParameter<int>("PMTNoiseThreshold"),
									iConfig.getParameter<int>("NumberOfSamples"),
									iConfig.getParameter<int>("NumberOfPresamples"),
									iConfig.getParameter<bool>("excludeDepth5")))
{ produces<HcalUpgradeTrigPrimDigiCollection>(""); }

//------------------------------------------------------
// Destructor
//------------------------------------------------------

HcalUpgradeTrigPrimDigiProducer::~HcalUpgradeTrigPrimDigiProducer(){
  if (m_hcalUpgradeTriggerPrimitiveDigiAlgo != 0) delete m_hcalUpgradeTriggerPrimitiveDigiAlgo;
}

//------------------------------------------------------
// Main production function
//------------------------------------------------------

void HcalUpgradeTrigPrimDigiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
    
  //------------------------------------------------------
  // Get the edm::Handles from the Event
  //------------------------------------------------------

  edm::Handle<HBHEDigiCollection> hbheDigis;
  edm::Handle<HFDigiCollection>   hfDigis;

  bool gotHBHE = iEvent.getByLabel ( m_hbheDigisTag , hbheDigis );
  bool gotHF   = iEvent.getByLabel ( m_hfDigisTag   , hfDigis   );

  if (!gotHBHE) { 
    edm::LogWarning("HcalUpgradeTrigPrimDigiProducer") << "Cannot get " << m_hbheDigisTag; 
    return; 
  }
  
  if (!gotHF  ) { 
    edm::LogWarning("HcalUpgradeTrigPrimDigiProducer") << "Cannot get " << m_hfDigisTag  ; 
    return; 
  }

  //------------------------------------------------------
  // Get the coders from EventSetup
  //------------------------------------------------------

  edm::ESHandle<HcalTPGCoder> inputCoder;
  edm::ESHandle<CaloTPGTranscoder> outTranscoder;

  iSetup.get<HcalTPGRecord>().get(inputCoder);  
  iSetup.get<CaloTPGRecord>().get(outTranscoder);
  
  outTranscoder->setup(iSetup,CaloTPGTranscoder::HcalTPG);

  //------------------------------------------------------
  // Create an empty collection
  //------------------------------------------------------

  std::auto_ptr<HcalUpgradeTrigPrimDigiCollection> result (new HcalUpgradeTrigPrimDigiCollection());

  //------------------------------------------------------
  // Run the algorithm
  //------------------------------------------------------

  outTranscoder->getHcalCompressor().get();

  m_hcalUpgradeTriggerPrimitiveDigiAlgo -> run(inputCoder.product(),
					       outTranscoder->getHcalCompressor().get(),
					       *hbheDigis,  *hfDigis, *result );

  //------------------------------------------------------
  // Add the final CaloTowerCollection to the event
  //------------------------------------------------------
  
  iEvent.put(result);

  outTranscoder->releaseSetup();

}

void HcalUpgradeTrigPrimDigiProducer::beginJob(){}

void HcalUpgradeTrigPrimDigiProducer::endJob() {}

DEFINE_FWK_MODULE(HcalUpgradeTrigPrimDigiProducer);
