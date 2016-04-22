#include "SimCalorimetry/HcalSimProducers/interface/HcalDigitizer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalTestHitGenerator.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitCorrection.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTimeSlewSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDIonFeedbackSim.h"
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
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrix.h"
#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrices.h"
#include <boost/foreach.hpp>
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDigi/interface/HcalQIENum.h"

//#define DebugLog

namespace HcalDigitizerImpl {

  template<typename SIPMDIGITIZER>
  void fillSiPMCells(std::vector<int> & siPMCells, SIPMDIGITIZER * siPMDigitizer)
  {
    std::vector<DetId> siPMDetIds;
    siPMDetIds.reserve(siPMCells.size());
    for(std::vector<int>::const_iterator idItr = siPMCells.begin();
        idItr != siPMCells.end(); ++idItr)
    {
      siPMDetIds.emplace_back(*idItr);
    }
    siPMDigitizer->setDetIds(siPMDetIds);
  }

  // if both exist, assume the SiPM one has cells filled, and
  // assign the rest to the HPD
  template<typename HPDDIGITIZER, typename SIPMDIGITIZER>
  void fillCells(std::vector<DetId>& allCells,
                 HPDDIGITIZER * hpdDigitizer,
                 SIPMDIGITIZER * siPMDigitizer)
  {
    // if both digitizers exist, split up the cells
    if(siPMDigitizer && hpdDigitizer)
    {
      std::vector<DetId> siPMDetIds = siPMDigitizer->detIds();
      std::sort(siPMDetIds.begin(), siPMDetIds.end());
      std::vector<DetId> sortedCells = allCells;
      std::sort(sortedCells.begin(), sortedCells.end());
      std::vector<DetId> hpdCells;
      std::set_difference(sortedCells.begin(), sortedCells.end(),
                          siPMDetIds.begin(), siPMDetIds.end(),
                          std::back_inserter(hpdCells) );
      hpdDigitizer->setDetIds(hpdCells);
    }
    else
    {
      if(siPMDigitizer) siPMDigitizer->setDetIds(allCells);
      if(hpdDigitizer) hpdDigitizer->setDetIds(allCells);
    }
  }
} // namespace HcaiDigitizerImpl


HcalDigitizer::HcalDigitizer(const edm::ParameterSet& ps, edm::ConsumesCollector& iC) :
  theGeometry(0),
  theRecNumber(0),
  theParameterMap(new HcalSimParameterMap(ps)),
  theShapes(new HcalShapes()),
  theHBHEResponse(0),
  theHBHESiPMResponse(0),
  theHOResponse(0),   
  theHOSiPMResponse(0),
  theHFResponse(new CaloHitResponse(theParameterMap, theShapes)),
  theHFQIE10Response(new CaloHitResponse(theParameterMap, theShapes)),
  theZDCResponse(new CaloHitResponse(theParameterMap, theShapes)),
  theHBHEAmplifier(0),
  theHFAmplifier(0),
  theHOAmplifier(0),
  theZDCAmplifier(0),
  theIonFeedback(0),
  theCoderFactory(0),
  theUpgradeCoderFactory(0),
  theHBHEElectronicsSim(0),
  theHFElectronicsSim(0),
  theHOElectronicsSim(0),
  theZDCElectronicsSim(0),
  theUpgradeHBHEElectronicsSim(0),
  theUpgradeHFElectronicsSim(0),
  theHFQIE10ElectronicsSim(0),
  theHBHEHitFilter(),
  theHFHitFilter(ps.getParameter<bool>("doHFWindow")),
  theHOHitFilter(),
  theHOSiPMHitFilter(HcalOuter),
  theZDCHitFilter(),
  theHitCorrection(0),
  theNoiseGenerator(0),
  theNoiseHitGenerator(0),
  theHBHEDigitizer(0),
  theHBHESiPMDigitizer(0),
  theHODigitizer(0),
  theHOSiPMDigitizer(0),
  theHFDigitizer(0),
  theZDCDigitizer(0),
  theHBHEUpgradeDigitizer(0),
  theHFUpgradeDigitizer(0),
  theHFQIE10Digitizer(0),
  theRelabeller(0),
  isZDC(true),
  isHCAL(true),
  zdcgeo(true),
  hbhegeo(true),
  hogeo(true),
  hfgeo(true),
  hitsProducer_(ps.getParameter<std::string>("hitsProducer")),
  theHOSiPMCode(ps.getParameter<edm::ParameterSet>("ho").getParameter<int>("siPMCode")),
  deliveredLumi(0.),
  m_HEDarkening(0),
  m_HFRecalibration(0)
{
  iC.consumes<std::vector<PCaloHit> >(edm::InputTag(hitsProducer_, "ZDCHITS"));
  iC.consumes<std::vector<PCaloHit> >(edm::InputTag(hitsProducer_, "HcalHits"));

  bool doNoise = ps.getParameter<bool>("doNoise");
  bool PreMix1 = ps.getParameter<bool>("HcalPreMixStage1");  // special threshold/pedestal treatment
  bool PreMix2 = ps.getParameter<bool>("HcalPreMixStage2");  // special threshold/pedestal treatment
  bool useOldNoiseHB = ps.getParameter<bool>("useOldHB");
  bool useOldNoiseHE = ps.getParameter<bool>("useOldHE");
  bool useOldNoiseHF = ps.getParameter<bool>("useOldHF");
  bool useOldNoiseHO = ps.getParameter<bool>("useOldHO");
  bool doEmpty = ps.getParameter<bool>("doEmpty");
  double HBtp = ps.getParameter<double>("HBTuningParameter");
  double HEtp = ps.getParameter<double>("HETuningParameter");
  double HFtp = ps.getParameter<double>("HFTuningParameter");
  double HOtp = ps.getParameter<double>("HOTuningParameter");
  bool doHBHEUpgrade = ps.getParameter<bool>("HBHEUpgradeQIE");
  bool doHFUpgrade   = ps.getParameter<bool>("HFUpgradeQIE");
  deliveredLumi     = ps.getParameter<double>("DelivLuminosity");
  bool agingFlagHE = ps.getParameter<bool>("HEDarkening");
  bool agingFlagHF = ps.getParameter<bool>("HFDarkening");
  double minFCToDelay= ps.getParameter<double>("minFCToDelay");
  bool doHFQIE10 = ps.getParameter<bool>("HFQIE10");
  bool doHFQIE8 = ps.getParameter<bool>("HFQIE8");

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
  theHBHEAmplifier->setHBtuningParameter(HBtp);
  theHBHEAmplifier->setHEtuningParameter(HEtp);
  theHFAmplifier->setHFtuningParameter(HFtp);
  theHOAmplifier->setHOtuningParameter(HOtp);
  theHBHEAmplifier->setUseOldHB(useOldNoiseHB);
  theHBHEAmplifier->setUseOldHE(useOldNoiseHE);
  theHFAmplifier->setUseOldHF(useOldNoiseHF);
  theHOAmplifier->setUseOldHO(useOldNoiseHO);

  theCoderFactory = new HcalCoderFactory(HcalCoderFactory::DB);
  theUpgradeCoderFactory = new HcalCoderFactory(HcalCoderFactory::UPGRADE);

//  std::cout << "HcalDigitizer: theUpgradeCoderFactory created" << std::endl;

  theHBHEElectronicsSim = new HcalElectronicsSim(theHBHEAmplifier, theCoderFactory, PreMix1);
  theHFElectronicsSim = new HcalElectronicsSim(theHFAmplifier, theCoderFactory, PreMix1);
  theHOElectronicsSim = new HcalElectronicsSim(theHOAmplifier, theCoderFactory, PreMix1);
  theZDCElectronicsSim = new HcalElectronicsSim(theZDCAmplifier, theCoderFactory, PreMix1);
  theUpgradeHBHEElectronicsSim = new HcalElectronicsSim(theHBHEAmplifier, theUpgradeCoderFactory, PreMix1);
  theUpgradeHFElectronicsSim = new HcalElectronicsSim(theHFAmplifier, theUpgradeCoderFactory, PreMix1);
  theHFQIE10ElectronicsSim = new HcalElectronicsSim(theHFAmplifier, theUpgradeCoderFactory, PreMix1); //should this use a different coder factory?

//  std::cout << "HcalDigitizer: theUpgradeElectronicsSim created" <<  std::endl; 

  // a code of 1 means make all cells SiPM
  std::vector<int> hbSiPMCells(ps.getParameter<edm::ParameterSet>("hb").getParameter<std::vector<int> >("siPMCells"));
  //std::vector<int> hoSiPMCells(ps.getParameter<edm::ParameterSet>("ho").getParameter<std::vector<int> >("siPMCells"));
  // 0 means none, 1 means all, and 2 means use hardcoded

//  std::cout << std::endl << " hbSiPMCells = " << hbSiPMCells.size() << std::endl;

  bool doHBHEHPD = hbSiPMCells.empty() || (hbSiPMCells[0] != 1);
  bool doHOHPD = (theHOSiPMCode != 1);
  bool doHBHESiPM = !hbSiPMCells.empty();
  bool doHOSiPM = (theHOSiPMCode != 0);
  if(doHBHEHPD) {
    theHBHEResponse = new CaloHitResponse(theParameterMap, theShapes);
    edm::LogInfo("HcalDigitizer") <<"Set scale for HB towers";
    theHBHEResponse->initHBHEScale();

    theHBHEResponse->setHitFilter(&theHBHEHitFilter);
    theHBHEDigitizer = new HBHEDigitizer(theHBHEResponse, theHBHEElectronicsSim, doEmpty);
    bool    changeResponse = ps.getParameter<bool>("ChangeResponse");
    edm::FileInPath fname  = ps.getParameter<edm::FileInPath>("CorrFactorFile");
    if (changeResponse) {
      std::string corrFileName = fname.fullPath();
      edm::LogInfo("HcalDigitizer") << "Set scale for HB towers from " << corrFileName;
      theHBHEResponse->setHBHEScale(corrFileName); //GMA
    }
  }
  if(doHOHPD) {
    theHOResponse = new CaloHitResponse(theParameterMap, theShapes);
    theHOResponse->setHitFilter(&theHOHitFilter);
    theHODigitizer = new HODigitizer(theHOResponse, theHOElectronicsSim, doEmpty);
  }

  if(doHBHESiPM) {
    theHBHESiPMResponse = new HcalSiPMHitResponse(theParameterMap, theShapes);
    theHBHESiPMResponse->setHitFilter(&theHBHEHitFilter);
    if (doHBHEUpgrade) {
      theHBHEUpgradeDigitizer = new UpgradeDigitizer(theHBHESiPMResponse, theUpgradeHBHEElectronicsSim, doEmpty);

//      std::cout << "HcalDigitizer: theHBHEUpgradeDigitizer created" << std::endl;

    } else {
      theHBHESiPMDigitizer = new HBHEDigitizer(theHBHESiPMResponse, theHBHEElectronicsSim, doEmpty);
    }
  }
  if(doHOSiPM) {
    theHOSiPMResponse = new HcalSiPMHitResponse(theParameterMap, theShapes);
    theHOSiPMResponse->setHitFilter(&theHOSiPMHitFilter);
    theHOSiPMDigitizer = new HODigitizer(theHOSiPMResponse, theHOElectronicsSim, doEmpty);
  }

  // if both are present, fill the SiPM cells now
  if(doHBHEHPD && doHBHESiPM) {

//    std::cout << "HcalDigitizer:  fill the SiPM cells now"  << std::endl;

    HcalDigitizerImpl::fillSiPMCells(hbSiPMCells, theHBHESiPMDigitizer);
  }

  theHFResponse->setHitFilter(&theHFHitFilter);
  theHFQIE10Response->setHitFilter(&theHFHitFilter);
  theZDCResponse->setHitFilter(&theZDCHitFilter);

  bool doTimeSlew = ps.getParameter<bool>("doTimeSlew");
  //initialize: they won't be called later if flag is set
  theTimeSlewSim = 0;
  if(doTimeSlew) {
    // no time slewing for HF
    theTimeSlewSim = new HcalTimeSlewSim(theParameterMap,minFCToDelay);
    theHBHEAmplifier->setTimeSlewSim(theTimeSlewSim);
    theHOAmplifier->setTimeSlewSim(theTimeSlewSim);
    theZDCAmplifier->setTimeSlewSim(theTimeSlewSim);
  }

  if (doHFQIE10 || doHFQIE8) { //QIE8 and QIE10 can coexist in HF
    if(doHFQIE10) theHFQIE10Digitizer = new QIE10Digitizer(theHFQIE10Response, theHFQIE10ElectronicsSim, doEmpty);
	if(doHFQIE8) theHFDigitizer = new HFDigitizer(theHFResponse, theHFElectronicsSim, doEmpty);
  }
  else if (doHFUpgrade) {
    theHFUpgradeDigitizer = new UpgradeDigitizer(theHFResponse, theUpgradeHFElectronicsSim, doEmpty);

//    std::cout << "HcalDigitizer: theHFUpgradeDigitizer created" << std::endl;

  } else {
    theHFDigitizer = new HFDigitizer(theHFResponse, theHFElectronicsSim, doEmpty);
  }
  theZDCDigitizer = new ZDCDigitizer(theZDCResponse, theZDCElectronicsSim, doEmpty);

  edm::ParameterSet ps0 = ps.getParameter<edm::ParameterSet>("HcalReLabel");
  relabel_ = ps0.getUntrackedParameter<bool>("RelabelHits");
//  std::cout << "Flag to see if Hit Relabeller to be initiated " << relabel_ << std::endl;
  if (relabel_) {
    theRelabeller=new HcalHitRelabeller(ps0.getUntrackedParameter<edm::ParameterSet>("RelabelRules"));
  }     

  bool doHPDNoise = ps.getParameter<bool>("doHPDNoise");
  if(doHPDNoise) {
    //edm::ParameterSet hpdNoisePset = ps.getParameter<edm::ParameterSet>("HPDNoiseLibrary");
    theNoiseGenerator = new HPDNoiseGenerator(ps); 
    if(theHBHEDigitizer) theHBHEDigitizer->setNoiseSignalGenerator(theNoiseGenerator);
    if(theHBHESiPMDigitizer) theHBHESiPMDigitizer->setNoiseSignalGenerator(theNoiseGenerator);
  }

  if(ps.getParameter<bool>("doIonFeedback") && theHBHEResponse) {
    theIonFeedback = new HPDIonFeedbackSim(ps, theShapes);
    theHBHEResponse->setPECorrection(theIonFeedback);
    if(ps.getParameter<bool>("doThermalNoise")) {
      theHBHEAmplifier->setIonFeedbackSim(theIonFeedback);
    }
  }

  if(ps.getParameter<bool>("injectTestHits") ) {
    theNoiseHitGenerator = new HcalTestHitGenerator(ps);
    if(theHBHEDigitizer) theHBHEDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    if(theHBHESiPMDigitizer) theHBHESiPMDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    if(theHODigitizer) theHODigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    if(theHOSiPMDigitizer) theHOSiPMDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    if(theHBHEUpgradeDigitizer) {
      theHBHEUpgradeDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);

//      std::cout << "HcalDigitizer: theHBHEUpgradeDigitizer setNoise" << std::endl; 
    }
    if(theHFDigitizer) theHFDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    if(theHFUpgradeDigitizer) { 
      theHFUpgradeDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);

//      std::cout << "HcalDigitizer: theHFUpgradeDigitizer setNoise" << std::endl;
    }
	if(theHFQIE10Digitizer) theHFQIE10Digitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    theZDCDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
  }

  if(agingFlagHE) m_HEDarkening = new HEDarkening();
  if(agingFlagHF) m_HFRecalibration = new HFRecalibration();
}


HcalDigitizer::~HcalDigitizer() {
  if(theHBHEDigitizer)         delete theHBHEDigitizer;
  if(theHBHESiPMDigitizer)     delete theHBHESiPMDigitizer;
  if(theHODigitizer)           delete theHODigitizer;
  if(theHOSiPMDigitizer)       delete theHOSiPMDigitizer;
  if(theHFDigitizer)           delete theHFDigitizer;
  delete theZDCDigitizer;
  if(theHBHEUpgradeDigitizer)  delete theHBHEUpgradeDigitizer;
  if(theHFUpgradeDigitizer)    delete theHFUpgradeDigitizer;
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
  delete theUpgradeHBHEElectronicsSim;
  delete theUpgradeHFElectronicsSim;
  delete theHBHEAmplifier;
  delete theHFAmplifier;
  delete theHOAmplifier;
  delete theZDCAmplifier;
  delete theCoderFactory;
  delete theUpgradeCoderFactory;
  delete theHitCorrection;
  delete theNoiseGenerator;
  if (theRelabeller)           delete theRelabeller;
}


void HcalDigitizer::setHBHENoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHBHEElectronicsSim);
  if (theHBHEDigitizer) theHBHEDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHBHEAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setHFNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHFElectronicsSim);
  if(theHFDigitizer) theHFDigitizer->setNoiseSignalGenerator(noiseGenerator);
  if(theHFUpgradeDigitizer) theHFUpgradeDigitizer->setNoiseSignalGenerator(noiseGenerator);
  if(theHFQIE10Digitizer) theHFQIE10Digitizer->setNoiseSignalGenerator(noiseGenerator);
  theHFAmplifier->setNoiseSignalGenerator(noiseGenerator);
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
  theUpgradeHBHEElectronicsSim->setDbService(conditions.product());
  theUpgradeHFElectronicsSim->setDbService(conditions.product());
  theHFQIE10ElectronicsSim->setDbService(conditions.product());

  theCoderFactory->setDbService(conditions.product());
  theUpgradeCoderFactory->setDbService(conditions.product());
  theParameterMap->setDbService(conditions.product());

  edm::ESHandle<HcalCholeskyMatrices> refCholesky;
  if (eventSetup.find(edm::eventsetup::EventSetupRecordKey::makeKey<HcalCholeskyMatricesRcd>())) {
    eventSetup.get<HcalCholeskyMatricesRcd>().get(refCholesky);
    const HcalCholeskyMatrices * myCholesky = refCholesky.product();
    theHBHEAmplifier->setCholesky(myCholesky);
    theHFAmplifier->setCholesky(myCholesky);
    theHOAmplifier->setCholesky(myCholesky);
  }
  edm::ESHandle<HcalPedestals> pedshandle;
  eventSetup.get<HcalPedestalsRcd>().get(pedshandle);
  const HcalPedestals *  myADCPedestals = pedshandle.product();

  theHBHEAmplifier->setADCPeds(myADCPedestals);
  theHFAmplifier->setADCPeds(myADCPedestals);
  theHOAmplifier->setADCPeds(myADCPedestals);

  if(theHitCorrection != 0) {
    theHitCorrection->clear();
  }

  //initialize hits
  if(theHBHEDigitizer) theHBHEDigitizer->initializeHits();
  if(theHBHESiPMDigitizer) theHBHESiPMDigitizer->initializeHits();
  if(theHODigitizer) theHODigitizer->initializeHits();
  if(theHOSiPMDigitizer) theHOSiPMDigitizer->initializeHits();
  if(theHBHEUpgradeDigitizer) theHBHEUpgradeDigitizer->initializeHits();
  if(theHFQIE10Digitizer) theHFQIE10Digitizer->initializeHits();
  if(theHFUpgradeDigitizer) theHFUpgradeDigitizer->initializeHits();
  if(theHFDigitizer) theHFDigitizer->initializeHits();
  theZDCDigitizer->initializeHits();

}

void HcalDigitizer::accumulateCaloHits(edm::Handle<std::vector<PCaloHit> > const& hcalHandle, edm::Handle<std::vector<PCaloHit> > const& zdcHandle, int bunchCrossing, CLHEP::HepRandomEngine* engine, const HcalTopology *htopoP) {

  // Step A: pass in inputs, and accumulate digirs
  if(isHCAL) {
    std::vector<PCaloHit> hcalHitsOrig = *hcalHandle.product();
    std::vector<PCaloHit> hcalHits;
    hcalHits.reserve(hcalHitsOrig.size());

    //evaluate darkening before relabeling
    if(m_HEDarkening || m_HFRecalibration){
      darkening(hcalHitsOrig);
    }
    // Relabel PCaloHits if necessary
    if (relabel_) {
      edm::LogInfo("HcalDigitizer") << "Calling Relabeller";
      theRelabeller->process(hcalHitsOrig);
    }
    
    //eliminate bad hits
    for (unsigned int i=0; i< hcalHitsOrig.size(); i++) {
      DetId id(hcalHitsOrig[i].id());
      HcalDetId hid(id);
      if (!htopoP->validHcal(hid)) {
	edm::LogError("HcalDigitizer") << "bad hcal id found in digitizer. Skipping " << id.rawId() << " " << hid << std::endl;
      } else {
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

    if(theHitCorrection != 0) {
      theHitCorrection->fillChargeSums(hcalHits);
    }
    if(hbhegeo) {
      if(theHBHEDigitizer) theHBHEDigitizer->add(hcalHits, bunchCrossing, engine);
      if(theHBHESiPMDigitizer) theHBHESiPMDigitizer->add(hcalHits, bunchCrossing, engine);
      if(theHBHEUpgradeDigitizer) {
        theHBHEUpgradeDigitizer->add(hcalHits, bunchCrossing, engine);
      }
    }

    if(hogeo) {
      if(theHODigitizer) theHODigitizer->add(hcalHits, bunchCrossing, engine);
      if(theHOSiPMDigitizer) theHOSiPMDigitizer->add(hcalHits, bunchCrossing, engine);
    }

    if(hfgeo) {
      if(theHFDigitizer) theHFDigitizer->add(hcalHits, bunchCrossing, engine);
      if(theHFUpgradeDigitizer) theHFUpgradeDigitizer->add(hcalHits, bunchCrossing, engine);
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
  isHCAL = hcalHandle.isValid();

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
  std::auto_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
  std::auto_ptr<HODigiCollection> hoResult(new HODigiCollection());
  std::auto_ptr<HFDigiCollection> hfResult(new HFDigiCollection());
  std::auto_ptr<ZDCDigiCollection> zdcResult(new ZDCDigiCollection());
  std::auto_ptr<HBHEUpgradeDigiCollection> hbheupgradeResult(new HBHEUpgradeDigiCollection());
  std::auto_ptr<HFUpgradeDigiCollection> hfupgradeResult(new HFUpgradeDigiCollection());
  std::auto_ptr<QIE10DigiCollection> hfQIE10Result(new QIE10DigiCollection());

  // Step C: Invoke the algorithm, getting back outputs.
  if(isHCAL&&hbhegeo){
    if(theHBHEDigitizer)        theHBHEDigitizer->run(*hbheResult, engine);
    if(theHBHESiPMDigitizer)    theHBHESiPMDigitizer->run(*hbheResult, engine);
    if(theHBHEUpgradeDigitizer) {
      theHBHEUpgradeDigitizer->run(*hbheupgradeResult, engine);
#ifdef DebugLog
      std::cout << "HcalDigitizer::finalizeEvent  theHBHEUpgradeDigitizer->run" << std::endl; 
#endif
    }
  }
  if(isHCAL&&hogeo) {
    if(theHODigitizer) theHODigitizer->run(*hoResult, engine);
    if(theHOSiPMDigitizer) theHOSiPMDigitizer->run(*hoResult, engine);
  }
  if(isHCAL&&hfgeo) {
    if(theHFDigitizer) theHFDigitizer->run(*hfResult, engine);
    if(theHFUpgradeDigitizer) theHFUpgradeDigitizer->run(*hfupgradeResult, engine);
    if(theHFQIE10Digitizer) theHFQIE10Digitizer->run(*hfQIE10Result, engine);
  }
  if(isZDC&&zdcgeo) {
    theZDCDigitizer->run(*zdcResult, engine);
  }
  
  edm::LogInfo("HcalDigitizer") << "HCAL HBHE digis : " << hbheResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HO digis   : " << hoResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HF digis   : " << hfResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL ZDC digis   : " << zdcResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HBHE upgrade digis : " << hbheupgradeResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HF upgrade digis : " << hfupgradeResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HF QIE10 digis : " << hfQIE10Result->size();

#ifdef DebugLog
  std::cout << std::endl;
  std::cout << "HCAL HBHE digis : " << hbheResult->size() << std::endl;
  std::cout << "HCAL HO   digis : " << hoResult->size() << std::endl;
  std::cout << "HCAL HF   digis : " << hfResult->size() << std::endl;
 
  std::cout << "HCAL HBHE upgrade digis : " << hbheupgradeResult->size()
	    << std::endl;
  std::cout << "HCAL HF   upgrade digis : " << hfupgradeResult->size()
	    << std::endl;
#endif

  // Step D: Put outputs into event
  e.put(hbheResult);
  e.put(hoResult);
  e.put(hfResult);
  e.put(zdcResult);
  e.put(hbheupgradeResult,"HBHEUpgradeDigiCollection");
  e.put(hfupgradeResult, "HFUpgradeDigiCollection");
  e.put(hfQIE10Result, "HFQIE10DigiCollection");

#ifdef DebugLog
  std::cout << std::endl << "========>  HcalDigitizer e.put " << std::endl <<  std::endl;
#endif

  if(theHitCorrection) {
    theHitCorrection->clear();
  }
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

  theHBHEDetIds = hbCells;
  theHBHEDetIds.insert(theHBHEDetIds.end(), heCells.begin(), heCells.end());

  HcalDigitizerImpl::fillCells(theHBHEDetIds, theHBHEDigitizer, theHBHESiPMDigitizer);
  //HcalDigitizerImpl::fillCells(hoCells, theHODigitizer, theHOSiPMDigitizer);
  buildHOSiPMCells(hoCells, eventSetup);
  //handle mixed QIE8/10 scenario in HF
  if(theHFDigitizer && theHFQIE10Digitizer) buildHFQIECells(hfCells,eventSetup);
  else if(theHFDigitizer) theHFDigitizer->setDetIds(hfCells);
  else if(theHFQIE10Digitizer) theHFQIE10Digitizer->setDetIds(hfCells);
  if(theHFUpgradeDigitizer) theHFUpgradeDigitizer->setDetIds(hfCells);
  theZDCDigitizer->setDetIds(zdcCells); 
  if(theHBHEUpgradeDigitizer) {
    theHBHEUpgradeDigitizer->setDetIds(theHBHEDetIds);
#ifdef DebugLog
    std::cout << " HcalDigitizer::updateGeometry  theHBHEUpgradeDigitizer->setDetIds(theHBHEDetIds)"<< std::endl;   
#endif
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

    for(std::vector<DetId>::const_iterator detItr = allCells.begin();
        detItr != allCells.end(); ++detItr) {
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

    assert(theHODigitizer);
    assert(theHOSiPMDigitizer);
    theHODigitizer->setDetIds(theHOHPDDetIds);
    theHOSiPMDigitizer->setDetIds(theHOSiPMDetIds);
    theHOSiPMHitFilter.setDetIds(theHOSiPMDetIds);
    // FIXME not applying a HitFilter to the HPDs, for now
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
    

