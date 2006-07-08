#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/EcalSelectiveReadoutProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <memory>

#include <fstream> //used for debugging

#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"

//#define DEBUG_SRP

using namespace std;

EcalSelectiveReadoutProducer::EcalSelectiveReadoutProducer(const edm::ParameterSet& params)
  : params_(params)
{
  //sets up parameters:
   digiProducer_ = params.getParameter<string>("digiProducer");
   ebdigiCollection_ = params.getParameter<std::string>("EBdigiCollection");
   eedigiCollection_ = params.getParameter<std::string>("EEdigiCollection");
   ebSRPdigiCollection_ = params.getParameter<std::string>("EBSRPdigiCollection");
   eeSRPdigiCollection_ = params.getParameter<std::string>("EESRPdigiCollection");
   trigPrimProducer_ = params.getParameter<string>("trigPrimProducer");
   trigPrimBypass_ = params.getParameter<bool>("trigPrimBypass");
   //instantiates the selective readout algorithm:
   suppressor_ = auto_ptr<EcalSelectiveReadoutSuppressor>(new EcalSelectiveReadoutSuppressor(params));

   //declares the products made by this producer:
   produces<EBDigiCollection>(ebSRPdigiCollection_);
   produces<EEDigiCollection>(eeSRPdigiCollection_);
}



EcalSelectiveReadoutProducer::~EcalSelectiveReadoutProducer() 
{ }


void
EcalSelectiveReadoutProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) 
{
  // check that everything is up-to-date
  checkGeometry(eventSetup);
  checkTriggerMap(eventSetup);

  //gets the trigger primitives:
  EcalTrigPrimDigiCollection emptyTPColl;
  const EcalTrigPrimDigiCollection* trigPrims =
    trigPrimBypass_?&emptyTPColl:getTrigPrims(event);

#ifdef DEBUG_SRP
  static int iEvent = 0;
  stringstream buffer;
  buffer << "TTFMap_" << event.id(); //iEvent;
  ofstream ttfFile(buffer.str().c_str());
  printTTFlags(*trigPrims, ttfFile);
#endif //DEBUG_SRP defined
  
  //gets the digis from the events:
  const EBDigiCollection* ebDigis = getEBDigis(event);
  const EEDigiCollection* eeDigis = getEEDigis(event);
  
  //runs the selective readout algorithm:
  auto_ptr<EBDigiCollection> selectedEBDigi(new EBDigiCollection);
  auto_ptr<EEDigiCollection> selectedEEDigi(new EEDigiCollection);

  suppressor_->run(eventSetup, *trigPrims, *ebDigis, *eeDigis,
		   *selectedEBDigi, *selectedEEDigi);
  
#ifdef DEBUG_SRP
  buffer.str("");
  buffer << "SRFMap_" << event.id();//iEvent;
  ofstream srfFile(buffer.str().c_str());
  suppressor_->getEcalSelectiveReadout()->printHeader(srfFile);
  suppressor_->getEcalSelectiveReadout()->print(srfFile);
  ++iEvent; //event counter
#endif //DEBUG_SRP defined
  
  //puts the selected digis into the event:
  event.put(selectedEBDigi, ebSRPdigiCollection_);
  event.put(selectedEEDigi, eeSRPdigiCollection_);
  
}

const EBDigiCollection*
EcalSelectiveReadoutProducer::getEBDigis(edm::Event& event) const
{
  edm::Handle< EBDigiCollection > hEBDigis;
  event.getByLabel(digiProducer_, hEBDigis);
  static bool firstCall= true;
  if(firstCall){
    checkWeights(event, hEBDigis.id());
    firstCall = false;
  }
  return hEBDigis.product();
}

const EEDigiCollection*
EcalSelectiveReadoutProducer::getEEDigis(edm::Event& event) const
{
  edm::Handle< EEDigiCollection > hEEDigis;
  event.getByLabel(digiProducer_, hEEDigis);
  static bool firstCall = true;
  if(firstCall){
    checkWeights(event, hEEDigis.id());
    firstCall = false;
  }
  return hEEDigis.product();
}

const EcalTrigPrimDigiCollection*
EcalSelectiveReadoutProducer::getTrigPrims(edm::Event& event) const
{
  edm::Handle<EcalTrigPrimDigiCollection> hTPDigis;
  event.getByLabel(trigPrimProducer_, hTPDigis);
  return hTPDigis.product();
}

  
void EcalSelectiveReadoutProducer::checkGeometry(const edm::EventSetup & eventSetup)
{
  edm::ESHandle<CaloGeometry> hGeometry;
  eventSetup.get<IdealGeometryRecord>().get(hGeometry);

  const CaloGeometry * pGeometry = &*hGeometry;

  // see if we need to update
  if(pGeometry != theGeometry) {
    theGeometry = pGeometry;
    suppressor_->setGeometry(theGeometry);
  }
}


void EcalSelectiveReadoutProducer::checkTriggerMap(const edm::EventSetup & eventSetup)
{

   edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap;
   eventSetup.get<IdealGeometryRecord>().get(eTTmap);

   const EcalTrigTowerConstituentsMap * pMap = &*eTTmap;
  
  // see if we need to update
  if(pMap!= theTriggerTowerMap) {
    theTriggerTowerMap = pMap;
    suppressor_->setTriggerMap(theTriggerTowerMap);
  }
}


void EcalSelectiveReadoutProducer::printTTFlags(const EcalTrigPrimDigiCollection& tp, ostream& os) const{
  const char tccFlagMarker[] = { '?', '.', 'S', '?', 'C', 'E', 'E', 'E', 'E'};
  const int nEta = EcalSelectiveReadout::nTriggerTowersInEta;
  const int nPhi = EcalSelectiveReadout::nTriggerTowersInPhi;

  //static bool firstCall=true;
  //  if(firstCall){
  //  firstCall=false;
  os << "# TCC flag map\n#\n"
    "# +-->Phi            " << tccFlagMarker[1] << ": 000 (low interest)\n"
    "# |                  " << tccFlagMarker[2] << ": 001 (mid interest)\n"
    "# |                  " << tccFlagMarker[3] << ": 010 (not valid)\n"
    "# V Eta              " << tccFlagMarker[5] << ": 011 (high interest)\n"
    "#                    " << tccFlagMarker[6] << ": 1xx forced readout (Hw error)\n"
    "#\n";
  //}
  
  vector<vector<int> > ttf(nEta, vector<int>(nPhi, -1));
  for(EcalTrigPrimDigiCollection::const_iterator it = tp.begin();
      it != tp.end(); ++it){
    const EcalTriggerPrimitiveDigi& trigPrim = *it;
    if(trigPrim.size()>0){
      int iEta0 = trigPrim.id().ieta();
      int iEta = iEta0<0?iEta0+nEta/2:iEta0+nEta/2-1;
      int iPhi = trigPrim.id().iphi() - 1;
      ttf[iEta][iPhi] = trigPrim[4].ttFlag();
    }
  }
  for(int iEta=0; iEta<nEta; ++iEta){
    for(int iPhi=0; iPhi<nPhi; ++iPhi){
      os << tccFlagMarker[ttf[iEta][iPhi]+1];
    }
    os << "\n";
  }
}

void EcalSelectiveReadoutProducer::checkWeights(const edm::Event& evt,
						const edm::ProductID& noZsDigiId) const{
  const vector<double> & weights = params_.getParameter<vector<double> >("dccNormalizedWeights");
  int nFIRTaps = EcalSelectiveReadoutSuppressor::getFIRTapCount();
  static bool warnWeightCnt = true;
  if((int)weights.size() > nFIRTaps && warnWeightCnt){
      edm::LogWarning("Configuration") << "The list of DCC zero suppression FIR "
	"weights given in parameter dccNormalizedWeights is longer "
	"than the expected depth of the FIR filter :(" << nFIRTaps << "). "
	"The last weights will be discarded.";
      warnWeightCnt = false; //it's not needed to repeat the warning.
  }
  
  if(weights.size()>0){
    int iMaxWeight = 0;
    double maxWeight = weights[iMaxWeight];
    //looks for index of maximum weight
    for(unsigned i=0; i<weights.size(); ++i){
      if(weights[i]>maxWeight){
	iMaxWeight = i;
	maxWeight = weights[iMaxWeight];
      }
    }

    //position of time sample whose maximum weight is applied:
    int maxWeightBin = params_.getParameter<int>("ecalDccZs1stSample")
      + iMaxWeight;
    
    //gets the bin of maximum (in case of raw data it will not exist)
    int binOfMax = 0;
    bool rc = getBinOfMax(evt, noZsDigiId, binOfMax);
    
    if(rc && maxWeightBin!=binOfMax){
      edm::LogWarning("Configuration")
	<< "The maximum weight of DCC zero suppression FIR filter is not "
	"applied to the expected maximum sample(" << binOfMax
	<< (binOfMax==1?"st":(binOfMax==2?"nd":(binOfMax==3?"rd":"th")))
	<< " time sample). This may indicate faulty 'dccNormalizedWeights' "
	"or 'ecalDccZs1sSample' parameters.";
    }
  }
}

bool
EcalSelectiveReadoutProducer::getBinOfMax(const edm::Event& evt,
					  const edm::ProductID& noZsDigiId,
					  int& binOfMax) const{
  bool rc;
  const edm::Provenance p=evt.getProvenance(noZsDigiId);
  edm::ParameterSet result = getParameterSet(p.psetID());
  vector<string> ebDigiParamList = result.getParameterNames();
  string bofm("binOfMaximum");
  if(find(ebDigiParamList.begin(), ebDigiParamList.end(), bofm)
     != ebDigiParamList.end()){//bofm found
    binOfMax=result.getParameter<int>("binOfMaximum");
    rc = true;
  } else{
    rc = false;
  }
  return rc;
}
