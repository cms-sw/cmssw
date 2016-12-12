#include "SimCalorimetry/HcalSimProducers/interface/HcalDigitizer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTimeSlewSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDIonFeedbackSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalBaseSignalGenerator.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include <boost/foreach.hpp>
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDigi/interface/HcalQIENum.h"

//#define DebugLog

HcalDigitizer::HcalDigitizer(const edm::ParameterSet& ps, edm::ConsumesCollector& iC) :
  theGeometry(0),
  theRecNumber(0),
  theParameterMap(new HcalSimParameterMap(ps)),
  theShapes(new HcalShapes()),
  theHBHEResponse(new CaloHitResponse(theParameterMap, theShapes)),
  theHBHESiPMResponse(new HcalSiPMHitResponse(theParameterMap, theShapes, ps.getParameter<bool>("HcalPreMixStage1"))),
  theHOResponse(new CaloHitResponse(theParameterMap, theShapes)),   
  theHOSiPMResponse(new HcalSiPMHitResponse(theParameterMap, theShapes)),
  theHFResponse(new CaloHitResponse(theParameterMap, theShapes)),
  theHFQIE10Response(new CaloHitResponse(theParameterMap, theShapes)),
  theZDCResponse(new CaloHitResponse(theParameterMap, theShapes)),
  theHBHEAmplifier(0),
  theHFAmplifier(0),
  theHOAmplifier(0),
  theZDCAmplifier(0),
  theHFQIE10Amplifier(0),
  theHBHEQIE11Amplifier(0),
  theIonFeedback(0),
  theCoderFactory(0),
  theHBHEElectronicsSim(0),
  theHFElectronicsSim(0),
  theHOElectronicsSim(0),
  theZDCElectronicsSim(0),
  theHFQIE10ElectronicsSim(0),
  theHBHEQIE11ElectronicsSim(0),
  theHBHEHitFilter(),
  theHBHEQIE11HitFilter(),
  theHFHitFilter(),
  theHFQIE10HitFilter(),
  theHOHitFilter(),
  theHOSiPMHitFilter(),
  theZDCHitFilter(),
  theHBHEDigitizer(0),
  theHODigitizer(0),
  theHOSiPMDigitizer(0),
  theHFDigitizer(0),
  theZDCDigitizer(0),
  theHFQIE10Digitizer(0),
  theHBHEQIE11Digitizer(0),
  theRelabeller(0),
  isZDC(true),
  isHCAL(true),
  zdcgeo(true),
  hbhegeo(true),
  hogeo(true),
  hfgeo(true),
  doHFWindow_(ps.getParameter<bool>("doHFWindow")),
  killHE_(ps.getParameter<bool>("killHE")),
  debugCS_(ps.getParameter<bool>("debugCaloSamples")),
  ignoreTime_(ps.getParameter<bool>("ignoreGeantTime")),
  injectTestHits_(ps.getParameter<bool>("injectTestHits")),
  hitsProducer_(ps.getParameter<std::string>("hitsProducer")),
  theHOSiPMCode(ps.getParameter<edm::ParameterSet>("ho").getParameter<int>("siPMCode")),
  deliveredLumi(0.),
  m_HEDarkening(0),
  m_HFRecalibration(0),
  injectedHitsEnergy_(ps.getParameter<std::vector<double>>("injectTestHitsEnergy")),
  injectedHitsTime_(ps.getParameter<std::vector<double>>("injectTestHitsTime")),
  injectedHitsCells_(ps.getParameter<std::vector<int>>("injectTestHitsCells"))
{
  iC.consumes<std::vector<PCaloHit> >(edm::InputTag(hitsProducer_, "ZDCHITS"));
  iC.consumes<std::vector<PCaloHit> >(edm::InputTag(hitsProducer_, "HcalHits"));

  bool doNoise = ps.getParameter<bool>("doNoise");
 
  bool PreMix1 = ps.getParameter<bool>("HcalPreMixStage1");  // special threshold/pedestal treatment
  bool PreMix2 = ps.getParameter<bool>("HcalPreMixStage2");  // special threshold/pedestal treatment
  bool doEmpty = ps.getParameter<bool>("doEmpty");
  deliveredLumi     = ps.getParameter<double>("DelivLuminosity");
  bool agingFlagHE = ps.getParameter<bool>("HEDarkening");
  bool agingFlagHF = ps.getParameter<bool>("HFDarkening");
  double minFCToDelay= ps.getParameter<double>("minFCToDelay");

  if(PreMix1 && PreMix2) {
     throw cms::Exception("Configuration")
      << "HcalDigitizer cannot operate in PreMixing digitization and PreMixing\n"
         "digi combination modes at the same time.  Please set one mode to False\n"
         "in the configuration file.";
  }

  // need to make copies, because they might get different noise generators
  theHBHEAmplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);
  theHFAmplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);
  theHOAmplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);
  theZDCAmplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);
  theHFQIE10Amplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);
  theHBHEQIE11Amplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);

  theCoderFactory = new HcalCoderFactory(HcalCoderFactory::DB);

  theHBHEElectronicsSim = new HcalElectronicsSim(theHBHEAmplifier, theCoderFactory, PreMix1);
  theHFElectronicsSim = new HcalElectronicsSim(theHFAmplifier, theCoderFactory, PreMix1);
  theHOElectronicsSim = new HcalElectronicsSim(theHOAmplifier, theCoderFactory, PreMix1);
  theZDCElectronicsSim = new HcalElectronicsSim(theZDCAmplifier, theCoderFactory, PreMix1);
  theHFQIE10ElectronicsSim = new HcalElectronicsSim(theHFQIE10Amplifier, theCoderFactory, PreMix1); //should this use a different coder factory?
  theHBHEQIE11ElectronicsSim = new HcalElectronicsSim(theHBHEQIE11Amplifier, theCoderFactory, PreMix1); //should this use a different coder factory?

  bool doHOHPD = (theHOSiPMCode != 1);
  bool doHOSiPM = (theHOSiPMCode != 0);
  if(doHOHPD) {
    theHOResponse = new CaloHitResponse(theParameterMap, theShapes);
    theHOResponse->setHitFilter(&theHOHitFilter);
    theHODigitizer = new HODigitizer(theHOResponse, theHOElectronicsSim, doEmpty);
  }
  if(doHOSiPM) {
    theHOSiPMResponse = new HcalSiPMHitResponse(theParameterMap, theShapes);
    theHOSiPMResponse->setHitFilter(&theHOSiPMHitFilter);
    theHOSiPMDigitizer = new HODigitizer(theHOSiPMResponse, theHOElectronicsSim, doEmpty);
  }
  
  theHBHEResponse->setHitFilter(&theHBHEHitFilter);
  theHBHESiPMResponse->setHitFilter(&theHBHEQIE11HitFilter);
  
  //QIE8 and QIE11 can coexist in HBHE
  theHBHEQIE11Digitizer = new QIE11Digitizer(theHBHESiPMResponse, theHBHEQIE11ElectronicsSim, doEmpty);
  theHBHEDigitizer = new HBHEDigitizer(theHBHEResponse, theHBHEElectronicsSim, doEmpty);

  bool doTimeSlew = ps.getParameter<bool>("doTimeSlew");
  //initialize: they won't be called later if flag is set
  theTimeSlewSim = 0;
  if(doTimeSlew) {
    // no time slewing for HF
    theTimeSlewSim = new HcalTimeSlewSim(theParameterMap,minFCToDelay);
    theHBHEAmplifier->setTimeSlewSim(theTimeSlewSim);
    theHBHEQIE11Amplifier->setTimeSlewSim(theTimeSlewSim);
    theHOAmplifier->setTimeSlewSim(theTimeSlewSim);
    theZDCAmplifier->setTimeSlewSim(theTimeSlewSim);
  }

  theHFResponse->setHitFilter(&theHFHitFilter);
  theHFQIE10Response->setHitFilter(&theHFQIE10HitFilter);
  theZDCResponse->setHitFilter(&theZDCHitFilter);
  
  //QIE8 and QIE10 can coexist in HF
  theHFQIE10Digitizer = new QIE10Digitizer(theHFQIE10Response, theHFQIE10ElectronicsSim, doEmpty);
  theHFDigitizer = new HFDigitizer(theHFResponse, theHFElectronicsSim, doEmpty);

  theZDCDigitizer = new ZDCDigitizer(theZDCResponse, theZDCElectronicsSim, doEmpty);

  testNumbering_ = ps.getParameter<bool>("TestNumbering");
//  std::cout << "Flag to see if Hit Relabeller to be initiated " << testNumbering_ << std::endl;
  if (testNumbering_) theRelabeller=new HcalHitRelabeller(ps);

  if(ps.getParameter<bool>("doIonFeedback") && theHBHEResponse) {
    theIonFeedback = new HPDIonFeedbackSim(ps, theShapes);
    theHBHEResponse->setPECorrection(theIonFeedback);
    if(ps.getParameter<bool>("doThermalNoise")) {
      theHBHEAmplifier->setIonFeedbackSim(theIonFeedback);
    }
  }

  //option to save CaloSamples as event product for debugging
  if(debugCS_){
    if(theHBHEDigitizer)      theHBHEDigitizer->setDebugCaloSamples(true);
    if(theHBHEQIE11Digitizer) theHBHEQIE11Digitizer->setDebugCaloSamples(true);
    if(theHODigitizer)        theHODigitizer->setDebugCaloSamples(true);
    if(theHOSiPMDigitizer)    theHOSiPMDigitizer->setDebugCaloSamples(true);
    if(theHFDigitizer)        theHFDigitizer->setDebugCaloSamples(true);
    if(theHFQIE10Digitizer)   theHFQIE10Digitizer->setDebugCaloSamples(true);
    theZDCDigitizer->setDebugCaloSamples(true);
  }

  //option to ignore Geant time distribution in SimHits, for debugging
  if(ignoreTime_){
    theHBHEResponse->setIgnoreGeantTime(ignoreTime_);
    theHBHESiPMResponse->setIgnoreGeantTime(ignoreTime_);
    theHOResponse->setIgnoreGeantTime(ignoreTime_);
    theHOSiPMResponse->setIgnoreGeantTime(ignoreTime_);
    theHFResponse->setIgnoreGeantTime(ignoreTime_);
    theHFQIE10Response->setIgnoreGeantTime(ignoreTime_);
    theZDCResponse->setIgnoreGeantTime(ignoreTime_);
  }

  if(agingFlagHE) m_HEDarkening = new HEDarkening();
  if(agingFlagHF) m_HFRecalibration = new HFRecalibration(ps.getParameter<edm::ParameterSet>("HFRecalParameterBlock"));
}


HcalDigitizer::~HcalDigitizer() {
  if(theHBHEDigitizer)         delete theHBHEDigitizer;
  if(theHBHEQIE11Digitizer)    delete theHBHEQIE11Digitizer;
  if(theHODigitizer)           delete theHODigitizer;
  if(theHOSiPMDigitizer)       delete theHOSiPMDigitizer;
  if(theHFDigitizer)           delete theHFDigitizer;
  if(theHFQIE10Digitizer)      delete theHFQIE10Digitizer;
  delete theZDCDigitizer;
  delete theParameterMap;
  delete theHBHEResponse;
  delete theHBHESiPMResponse;
  delete theHOResponse;
  delete theHOSiPMResponse;
  delete theHFResponse;
  delete theHFQIE10Response;
  delete theZDCResponse;
  delete theHBHEElectronicsSim;
  delete theHFElectronicsSim;
  delete theHOElectronicsSim;
  delete theZDCElectronicsSim;
  delete theHFQIE10ElectronicsSim;
  delete theHBHEQIE11ElectronicsSim;
  delete theHBHEAmplifier;
  delete theHFAmplifier;
  delete theHOAmplifier;
  delete theZDCAmplifier;
  delete theHFQIE10Amplifier;
  delete theHBHEQIE11Amplifier;
  delete theCoderFactory;
  if (theRelabeller)           delete theRelabeller;
}


void HcalDigitizer::setHBHENoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHBHEElectronicsSim);
  if (theHBHEDigitizer) theHBHEDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHBHEAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setQIE11NoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHBHEQIE11ElectronicsSim);
  if(theHBHEQIE11Digitizer) theHBHEQIE11Digitizer->setNoiseSignalGenerator(noiseGenerator);
  theHBHEQIE11Amplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setHFNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHFElectronicsSim);
  if(theHFDigitizer) theHFDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHFAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setQIE10NoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHFQIE10ElectronicsSim);
  if(theHFQIE10Digitizer) theHFQIE10Digitizer->setNoiseSignalGenerator(noiseGenerator);
  theHFQIE10Amplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setHONoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHOElectronicsSim);
  if(theHODigitizer) theHODigitizer->setNoiseSignalGenerator(noiseGenerator);
  if(theHOSiPMDigitizer) theHOSiPMDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHOAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setZDCNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theZDCElectronicsSim);
  theZDCDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theZDCAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::initializeEvent(edm::Event const& e, edm::EventSetup const& eventSetup) {
  // get the appropriate gains, noises, & widths for this event
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  
  theHBHEAmplifier->setDbService(conditions.product());
  theHFAmplifier->setDbService(conditions.product());
  theHOAmplifier->setDbService(conditions.product());
  theZDCAmplifier->setDbService(conditions.product());
  theHFQIE10Amplifier->setDbService(conditions.product());
  theHBHEQIE11Amplifier->setDbService(conditions.product());
  
  theHFQIE10ElectronicsSim->setDbService(conditions.product());
  theHBHEQIE11ElectronicsSim->setDbService(conditions.product());

  theCoderFactory->setDbService(conditions.product());
  theParameterMap->setDbService(conditions.product());

  //initialize hits
  if(theHBHEDigitizer) theHBHEDigitizer->initializeHits();
  if(theHBHEQIE11Digitizer) theHBHEQIE11Digitizer->initializeHits();
  if(theHODigitizer) theHODigitizer->initializeHits();
  if(theHOSiPMDigitizer) theHOSiPMDigitizer->initializeHits();
  if(theHFQIE10Digitizer) theHFQIE10Digitizer->initializeHits();
  if(theHFDigitizer) theHFDigitizer->initializeHits();
  theZDCDigitizer->initializeHits();

}

void HcalDigitizer::accumulateCaloHits(edm::Handle<std::vector<PCaloHit> > const& hcalHandle, edm::Handle<std::vector<PCaloHit> > const& zdcHandle, int bunchCrossing, CLHEP::HepRandomEngine* engine, const HcalTopology *htopoP) {

  // Step A: pass in inputs, and accumulate digis
  if(isHCAL) {
    std::vector<PCaloHit> hcalHitsOrig = *hcalHandle.product();
    if(injectTestHits_) hcalHitsOrig = injectedHits_;
    std::vector<PCaloHit> hcalHits;
    hcalHits.reserve(hcalHitsOrig.size());

    //evaluate darkening before relabeling
    if (testNumbering_) {
      if(m_HEDarkening || m_HFRecalibration){
        darkening(hcalHitsOrig);
      }
      // Relabel PCaloHits if necessary
      edm::LogInfo("HcalDigitizer") << "Calling Relabeller";
      theRelabeller->process(hcalHitsOrig);
    }
    
    //eliminate bad hits
    for (unsigned int i=0; i< hcalHitsOrig.size(); i++) {
      DetId id(hcalHitsOrig[i].id());
      HcalDetId hid(id);
      if (!htopoP->validHcal(hid)) {
        edm::LogError("HcalDigitizer") << "bad hcal id found in digitizer. Skipping " << id.rawId() << " " << hid << std::endl;
        continue;
      }
      else if(hid.subdet()==HcalForward && !doHFWindow_ && hcalHitsOrig[i].depth()!=0){
        //skip HF window hits unless desired
        continue;
      }
      else if( killHE_ && hid.subdet()==HcalEndcap ) {
        // remove HE hits if asked for (phase 2)
        continue;
      }
      else {
#ifdef DebugLog
        std::cout << "HcalDigitizer format " << hid.oldFormat() << " for " << hid << std::endl;
#endif
        DetId newid = DetId(hid.newForm());
#ifdef DebugLog
        std::cout << "Hit " << i << " out of " << hcalHits.size() << " " << std::hex << id.rawId() << " --> " << newid.rawId() << std::dec << " " << HcalDetId(newid.rawId()) << '\n';
#endif
        hcalHitsOrig[i].setID(newid.rawId());
        hcalHits.push_back(hcalHitsOrig[i]);
      }
    }

    if(hbhegeo) {
      if(theHBHEDigitizer) theHBHEDigitizer->add(hcalHits, bunchCrossing, engine);
      if(theHBHEQIE11Digitizer) theHBHEQIE11Digitizer->add(hcalHits, bunchCrossing, engine);
    }

    if(hogeo) {
      if(theHODigitizer) theHODigitizer->add(hcalHits, bunchCrossing, engine);
      if(theHOSiPMDigitizer) theHOSiPMDigitizer->add(hcalHits, bunchCrossing, engine);
    }

    if(hfgeo) {
      if(theHFDigitizer) theHFDigitizer->add(hcalHits, bunchCrossing, engine);
      if(theHFQIE10Digitizer) theHFQIE10Digitizer->add(hcalHits, bunchCrossing, engine);
    } 
  } else {
    edm::LogInfo("HcalDigitizer") << "We don't have HCAL hit collection available ";
  }

  if(isZDC) {
    if(zdcgeo) {
      theZDCDigitizer->add(*zdcHandle.product(), bunchCrossing, engine);
    } 
  } else {
    edm::LogInfo("HcalDigitizer") << "We don't have ZDC hit collection available ";
  }
}

void HcalDigitizer::accumulate(edm::Event const& e, edm::EventSetup const& eventSetup, CLHEP::HepRandomEngine* engine) {
  // Step A: Get Inputs
  edm::InputTag zdcTag(hitsProducer_, "ZDCHITS");
  edm::Handle<std::vector<PCaloHit> > zdcHandle;
  e.getByLabel(zdcTag, zdcHandle);
  isZDC = zdcHandle.isValid();

  edm::InputTag hcalTag(hitsProducer_, "HcalHits");
  edm::Handle<std::vector<PCaloHit> > hcalHandle;
  e.getByLabel(hcalTag, hcalHandle);
  isHCAL = hcalHandle.isValid() or injectTestHits_;

  edm::ESHandle<HcalTopology> htopo;
  eventSetup.get<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology *htopoP=htopo.product();

  accumulateCaloHits(hcalHandle, zdcHandle, 0, engine, htopoP);
}

void HcalDigitizer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup, CLHEP::HepRandomEngine* engine) {
  // Step A: Get Inputs
  edm::InputTag zdcTag(hitsProducer_, "ZDCHITS");
  edm::Handle<std::vector<PCaloHit> > zdcHandle;
  e.getByLabel(zdcTag, zdcHandle);
  isZDC = zdcHandle.isValid();

  edm::InputTag hcalTag(hitsProducer_, "HcalHits");
  edm::Handle<std::vector<PCaloHit> > hcalHandle;
  e.getByLabel(hcalTag, hcalHandle);
  isHCAL = hcalHandle.isValid();

  edm::ESHandle<HcalTopology> htopo;
  eventSetup.get<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology *htopoP=htopo.product();

  accumulateCaloHits(hcalHandle, zdcHandle, e.bunchCrossing(), engine, htopoP);
}

void HcalDigitizer::finalizeEvent(edm::Event& e, const edm::EventSetup& eventSetup, CLHEP::HepRandomEngine* engine) {

  // Step B: Create empty output
  std::unique_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
  std::unique_ptr<HODigiCollection> hoResult(new HODigiCollection());
  std::unique_ptr<HFDigiCollection> hfResult(new HFDigiCollection());
  std::unique_ptr<ZDCDigiCollection> zdcResult(new ZDCDigiCollection());
  std::unique_ptr<QIE10DigiCollection> hfQIE10Result(
    new QIE10DigiCollection(
      theHFQIE10DetIds.size()>0 ? 
      theParameterMap->simParameters(theHFQIE10DetIds[0]).readoutFrameSize() : 
      QIE10DigiCollection::MAXSAMPLES
    )
  );
  std::unique_ptr<QIE11DigiCollection> hbheQIE11Result(
    new QIE11DigiCollection(
      theHBHEQIE11DetIds.size()>0 ? 
      theParameterMap->simParameters(theHBHEQIE11DetIds[0]).readoutFrameSize() : 
      QIE11DigiCollection::MAXSAMPLES
    )
  );

  // Step C: Invoke the algorithm, getting back outputs.
  if(isHCAL&&hbhegeo){
    if(theHBHEDigitizer)        theHBHEDigitizer->run(*hbheResult, engine);
    if(theHBHEQIE11Digitizer)    theHBHEQIE11Digitizer->run(*hbheQIE11Result, engine);
  }
  if(isHCAL&&hogeo) {
    if(theHODigitizer) theHODigitizer->run(*hoResult, engine);
    if(theHOSiPMDigitizer) theHOSiPMDigitizer->run(*hoResult, engine);
  }
  if(isHCAL&&hfgeo) {
    if(theHFDigitizer) theHFDigitizer->run(*hfResult, engine);
    if(theHFQIE10Digitizer) theHFQIE10Digitizer->run(*hfQIE10Result, engine);
  }
  if(isZDC&&zdcgeo) {
    theZDCDigitizer->run(*zdcResult, engine);
  }
  
  edm::LogInfo("HcalDigitizer") << "HCAL HBHE digis : " << hbheResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HO digis   : " << hoResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HF digis   : " << hfResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL ZDC digis  : " << zdcResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HF QIE10 digis : " << hfQIE10Result->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HBHE QIE11 digis : " << hbheQIE11Result->size();

#ifdef DebugLog
  std::cout << std::endl;
  std::cout << "HCAL HBHE digis : " << hbheResult->size() << std::endl;
  std::cout << "HCAL HO   digis : " << hoResult->size() << std::endl;
  std::cout << "HCAL HF   digis : " << hfResult->size() << std::endl;
  std::cout << "HCAL ZDC  digis : " << zdcResult->size() << std::endl;
  std::cout << "HCAL HF QIE10 digis : " << hfQIE10Result->size() << std::endl;
  std::cout << "HCAL HBHE QIE11 digis : " << hbheQIE11Result->size() << std::endl;
#endif

  // Step D: Put outputs into event
  e.put(std::move(hbheResult));
  e.put(std::move(hoResult));
  e.put(std::move(hfResult));
  e.put(std::move(zdcResult));
  e.put(std::move(hfQIE10Result), "HFQIE10DigiCollection");
  e.put(std::move(hbheQIE11Result), "HBHEQIE11DigiCollection");

  if(debugCS_){
    std::unique_ptr<CaloSamplesCollection> csResult(new CaloSamplesCollection());
    //smush together all the results
    if(theHBHEDigitizer)      csResult->insert(csResult->end(),theHBHEDigitizer->getCaloSamples().begin(),theHBHEDigitizer->getCaloSamples().end());
    if(theHBHEQIE11Digitizer) csResult->insert(csResult->end(),theHBHEQIE11Digitizer->getCaloSamples().begin(),theHBHEQIE11Digitizer->getCaloSamples().end());
    if(theHODigitizer)        csResult->insert(csResult->end(),theHODigitizer->getCaloSamples().begin(),theHODigitizer->getCaloSamples().end());
    if(theHOSiPMDigitizer)    csResult->insert(csResult->end(),theHOSiPMDigitizer->getCaloSamples().begin(),theHOSiPMDigitizer->getCaloSamples().end());
    if(theHFDigitizer)        csResult->insert(csResult->end(),theHFDigitizer->getCaloSamples().begin(),theHFDigitizer->getCaloSamples().end());
    if(theHFQIE10Digitizer)   csResult->insert(csResult->end(),theHFQIE10Digitizer->getCaloSamples().begin(),theHFQIE10Digitizer->getCaloSamples().end());
    csResult->insert(csResult->end(),theZDCDigitizer->getCaloSamples().begin(),theZDCDigitizer->getCaloSamples().end());
    e.put(std::move(csResult),"HcalSamples");
  }

  if(injectTestHits_){
    std::unique_ptr<edm::PCaloHitContainer> pcResult(new edm::PCaloHitContainer());
    pcResult->insert(pcResult->end(),injectedHits_.begin(),injectedHits_.end());
    e.put(std::move(pcResult),"HcalHits");
  }

#ifdef DebugLog
  std::cout << std::endl << "========>  HcalDigitizer e.put " << std::endl <<  std::endl;
#endif

}


void HcalDigitizer::beginRun(const edm::EventSetup & es) {
  checkGeometry(es);
  theShapes->beginRun(es);
}


void HcalDigitizer::endRun() {
  theShapes->endRun();
}


void HcalDigitizer::checkGeometry(const edm::EventSetup & eventSetup) {
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> geometry;
  eventSetup.get<CaloGeometryRecord>().get(geometry);
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  eventSetup.get<HcalRecNumberingRecord>().get(pHRNDC);

  // See if it's been updated
  if (&*geometry != theGeometry) {
    theGeometry = &*geometry;
    theRecNumber= &*pHRNDC;
    updateGeometry(eventSetup);
  }
}


void  HcalDigitizer::updateGeometry(const edm::EventSetup & eventSetup) {
  if(theHBHEResponse) theHBHEResponse->setGeometry(theGeometry);
  if(theHBHESiPMResponse) theHBHESiPMResponse->setGeometry(theGeometry);
  if(theHOResponse) theHOResponse->setGeometry(theGeometry);
  if(theHOSiPMResponse) theHOSiPMResponse->setGeometry(theGeometry);
  theHFResponse->setGeometry(theGeometry);
  theHFQIE10Response->setGeometry(theGeometry);
  theZDCResponse->setGeometry(theGeometry);
  if(theRelabeller) theRelabeller->setGeometry(theGeometry,theRecNumber);

  const std::vector<DetId>& hbCells = theGeometry->getValidDetIds(DetId::Hcal, HcalBarrel);
  const std::vector<DetId>& heCells = theGeometry->getValidDetIds(DetId::Hcal, HcalEndcap);
  const std::vector<DetId>& hoCells = theGeometry->getValidDetIds(DetId::Hcal, HcalOuter);
  const std::vector<DetId>& hfCells = theGeometry->getValidDetIds(DetId::Hcal, HcalForward);
  const std::vector<DetId>& zdcCells = theGeometry->getValidDetIds(DetId::Calo, HcalZDCDetId::SubdetectorId);
  //const std::vector<DetId>& hcalTrigCells = geometry->getValidDetIds(DetId::Hcal, HcalTriggerTower);
  //const std::vector<DetId>& hcalCalib = geometry->getValidDetIds(DetId::Calo, HcalCastorDetId::SubdetectorId);
//  std::cout<<"HcalDigitizer::CheckGeometry number of cells: "<<zdcCells.size()<<std::endl;
  if(zdcCells.empty()) zdcgeo = false;
  if(hbCells.empty() && heCells.empty()) hbhegeo = false;
  if(hoCells.empty()) hogeo = false;
  if(hfCells.empty()) hfgeo = false;
  // combine HB & HE

  hbheCells = hbCells;
  if( !killHE_) {
    hbheCells.insert(hbheCells.end(), heCells.begin(), heCells.end());
  }
  //handle mixed QIE8/11 scenario in HBHE
  buildHBHEQIECells(hbheCells,eventSetup);
  if(theHBHESiPMResponse)
    ((HcalSiPMHitResponse *)theHBHESiPMResponse)->setDetIds(theHBHEQIE11DetIds);
  
  if(theHOSiPMDigitizer) {
    buildHOSiPMCells(hoCells, eventSetup);
    if(theHOSiPMResponse)
      ((HcalSiPMHitResponse *)theHOSiPMResponse)->setDetIds(hoCells);
  }
  
  //handle mixed QIE8/10 scenario in HF
  buildHFQIECells(hfCells,eventSetup);
  
  theZDCDigitizer->setDetIds(zdcCells);

  //fill test hits collection if desired and empty
  if(injectTestHits_ && injectedHits_.size()==0 && injectedHitsCells_.size()>0 && injectedHitsEnergy_.size()>0){
    //make list of specified cells if desired
    std::vector<DetId> testCells;
    if(injectedHitsCells_.size()>=4){
      testCells.reserve(injectedHitsCells_.size()/4);
      for(unsigned ic = 0; ic < injectedHitsCells_.size(); ic += 4){
        if(ic+4 > injectedHitsCells_.size()) break;
        testCells.push_back(HcalDetId((HcalSubdetector)injectedHitsCells_[ic],injectedHitsCells_[ic+1],
                            injectedHitsCells_[ic+2],injectedHitsCells_[ic+3]));
      }
    }
    else{
      int testSubdet = injectedHitsCells_[0];
      if(testSubdet==HcalBarrel) testCells = hbCells;
      else if(testSubdet==HcalEndcap) testCells = heCells;
      else if(testSubdet==HcalForward) testCells = hfCells;
      else if(testSubdet==HcalOuter) testCells = hoCells;
      else throw cms::Exception("Configuration") << "Unknown subdet " << testSubdet << " for HCAL test hit injection";
    }
    bool useHitTimes = (injectedHitsTime_.size()==injectedHitsEnergy_.size());
    injectedHits_.reserve(testCells.size()*injectedHitsEnergy_.size());
    for(unsigned ih = 0; ih < injectedHitsEnergy_.size(); ++ih){
      double tmp = useHitTimes ? injectedHitsTime_[ih] : 0.;
      for(auto& aCell: testCells){
        injectedHits_.emplace_back(aCell,injectedHitsEnergy_[ih],tmp);
      }
    }
  }
}

void HcalDigitizer::buildHFQIECells(const std::vector<DetId>& allCells, const edm::EventSetup & eventSetup) {
	//if results are already cached, no need to look again
	if(theHFQIE8DetIds.size()>0 || theHFQIE10DetIds.size()>0) return;
	
	//get the QIETypes
	edm::ESHandle<HcalQIETypes> q;
    eventSetup.get<HcalQIETypesRcd>().get(q);
	edm::ESHandle<HcalTopology> htopo;
    eventSetup.get<HcalRecNumberingRecord>().get(htopo);
   
    HcalQIETypes qieTypes(*q.product());
    if (qieTypes.topo()==0) {
      qieTypes.setTopo(htopo.product());
    }
	
	for(std::vector<DetId>::const_iterator detItr = allCells.begin(); detItr != allCells.end(); ++detItr) {
      HcalQIENum qieType = HcalQIENum(qieTypes.getValues(*detItr)->getValue());
      if(qieType == QIE8) {
        theHFQIE8DetIds.push_back(*detItr);
      } else if(qieType == QIE10) {
        theHFQIE10DetIds.push_back(*detItr);
      } else { //default is QIE8
        theHFQIE8DetIds.push_back(*detItr);
      }
    }
	
	if(theHFQIE8DetIds.size()>0) theHFDigitizer->setDetIds(theHFQIE8DetIds);
	else {
		delete theHFDigitizer;
		theHFDigitizer = NULL;
	}
	
	if(theHFQIE10DetIds.size()>0) theHFQIE10Digitizer->setDetIds(theHFQIE10DetIds);
	else {
		delete theHFQIE10Digitizer;
		theHFQIE10Digitizer = NULL;
	}
}

void HcalDigitizer::buildHBHEQIECells(const std::vector<DetId>& allCells, const edm::EventSetup & eventSetup) {
	//if results are already cached, no need to look again
	if(theHBHEQIE8DetIds.size()>0 || theHBHEQIE11DetIds.size()>0) return;
	
	//get the QIETypes
	edm::ESHandle<HcalQIETypes> q;
    eventSetup.get<HcalQIETypesRcd>().get(q);
	edm::ESHandle<HcalTopology> htopo;
    eventSetup.get<HcalRecNumberingRecord>().get(htopo);
   
    HcalQIETypes qieTypes(*q.product());
    if (qieTypes.topo()==0) {
      qieTypes.setTopo(htopo.product());
    }
	
	for(std::vector<DetId>::const_iterator detItr = allCells.begin(); detItr != allCells.end(); ++detItr) {
      HcalQIENum qieType = HcalQIENum(qieTypes.getValues(*detItr)->getValue());
      if(qieType == QIE8) {
        theHBHEQIE8DetIds.push_back(*detItr);
      }
      else if(qieType == QIE11) {
        theHBHEQIE11DetIds.push_back(*detItr);
      }
      else { //default is QIE8
        theHBHEQIE8DetIds.push_back(*detItr);
      }
    }
	
	if(theHBHEQIE8DetIds.size()>0) theHBHEDigitizer->setDetIds(theHBHEQIE8DetIds);
	else {
		delete theHBHEDigitizer;
		theHBHEDigitizer = NULL;
	}
	
	if(theHBHEQIE11DetIds.size()>0) theHBHEQIE11Digitizer->setDetIds(theHBHEQIE11DetIds);
	else {
		delete theHBHEQIE11Digitizer;
		theHBHEQIE11Digitizer = NULL;
	}
	
	if(theHBHEQIE8DetIds.size()>0 && theHBHEQIE11DetIds.size()>0){
		theHBHEHitFilter.setDetIds(theHBHEQIE8DetIds);
		theHBHEQIE11HitFilter.setDetIds(theHBHEQIE11DetIds);
	}
}

void HcalDigitizer::buildHOSiPMCells(const std::vector<DetId>& allCells, const edm::EventSetup & eventSetup) {
  // all HPD

  if(theHOSiPMCode == 0) {
    theHODigitizer->setDetIds(allCells);
  } else if(theHOSiPMCode == 1) {
    theHOSiPMDigitizer->setDetIds(allCells);
    // FIXME pick Zecotek or hamamatsu?
  } else if(theHOSiPMCode == 2) {
    std::vector<HcalDetId> zecotekDetIds, hamamatsuDetIds;
    edm::ESHandle<HcalMCParams> p;
    eventSetup.get<HcalMCParamsRcd>().get(p);
    edm::ESHandle<HcalTopology> htopo;
    eventSetup.get<HcalRecNumberingRecord>().get(htopo);
   
    HcalMCParams mcParams(*p.product());
    if (mcParams.topo()==0) {
      mcParams.setTopo(htopo.product());
    }

    for(std::vector<DetId>::const_iterator detItr = allCells.begin(); detItr != allCells.end(); ++detItr) {
      int shapeType = mcParams.getValues(*detItr)->signalShape();
      if(shapeType == HcalShapes::ZECOTEK) {
        zecotekDetIds.emplace_back(*detItr);
        theHOSiPMDetIds.push_back(*detItr);
      } else if(shapeType == HcalShapes::HAMAMATSU) {
        hamamatsuDetIds.emplace_back(*detItr);
        theHOSiPMDetIds.push_back(*detItr);
      } else {
        theHOHPDDetIds.push_back(*detItr);
      }
    }

    if(theHOHPDDetIds.size()>0) theHODigitizer->setDetIds(theHOHPDDetIds);
    else {
      delete theHODigitizer;
      theHODigitizer = NULL;
    }
	
    if(theHOSiPMDetIds.size()>0) theHOSiPMDigitizer->setDetIds(theHOSiPMDetIds);
    else {
      delete theHOSiPMDigitizer;
      theHOSiPMDigitizer = NULL;
    }
	
	if(theHOHPDDetIds.size()>0 && theHOSiPMDetIds.size()>0){
      theHOSiPMHitFilter.setDetIds(theHOSiPMDetIds);
      theHOHitFilter.setDetIds(theHOHPDDetIds);
    }
	
    theParameterMap->setHOZecotekDetIds(zecotekDetIds);
    theParameterMap->setHOHamamatsuDetIds(hamamatsuDetIds);

    // make sure we don't got through this exercise again
    theHOSiPMCode = -2;
  }
}

void HcalDigitizer::darkening(std::vector<PCaloHit>& hcalHits) {

  for (unsigned int ii=0; ii<hcalHits.size(); ++ii) {
    uint32_t tmpId = hcalHits[ii].id();
    int det, z, depth, ieta, phi, lay;
    HcalTestNumbering::unpackHcalIndex(tmpId,det,z,depth,ieta,phi,lay);
	
    bool darkened = false;
    float dweight = 1.;
	
    if(det==int(HcalEndcap) && m_HEDarkening){
      //HE darkening
      dweight = m_HEDarkening->degradation(deliveredLumi,ieta,lay-2);//NB:diff. layer count
      darkened = true;
    } else if(det==int(HcalForward) && m_HFRecalibration){
      //HF darkening - approximate: invert recalibration factor
      dweight = 1.0/m_HFRecalibration->getCorr(ieta,depth,deliveredLumi);
      darkened = true;
    }
	
    //create new hit with darkened energy
    //if(darkened) hcalHits[ii] = PCaloHit(hcalHits[ii].energyEM()*dweight,hcalHits[ii].energyHad()*dweight,hcalHits[ii].time(),hcalHits[ii].geantTrackId(),hcalHits[ii].id());
	
    //reset hit energy
    if(darkened) hcalHits[ii].setEnergy(hcalHits[ii].energy()*dweight);	
  }
  
}
    

