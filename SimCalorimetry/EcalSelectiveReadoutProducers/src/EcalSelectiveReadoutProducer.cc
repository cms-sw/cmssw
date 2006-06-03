#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/EcalSelectiveReadoutProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <memory>

using namespace std;

EcalSelectiveReadoutProducer::EcalSelectiveReadoutProducer(const edm::ParameterSet& params)
{
  cout << "**********************************************************************" << endl;
  cout << "in " << __PRETTY_FUNCTION__ << endl;
  cout << "**********************************************************************" << endl;
  //sets up parameters:
  digiProducer_ = params.getParameter<string>("digiProducer");
  trigPrimProducer_ = params.getParameter<string>("trigPrimProducer");
  
  //instantiates the selective readout algorithm:
  suppressor_ = auto_ptr<EcalSelectiveReadoutSuppressor>(new EcalSelectiveReadoutSuppressor(params));
  
  //declares the products made by this producer:
  produces<EBDigiCollection>();
  produces<EEDigiCollection>();
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
  const EcalTrigPrimDigiCollection* trigPrims = getTrigPrims(event);

  //gets the digis from the events:
  const EBDigiCollection* ebDigis = getEBDigis(event);
  const EEDigiCollection* eeDigis = getEEDigis(event);
  
  //runs the selective readout algorithm:
  auto_ptr<EBDigiCollection> selectedEBDigi(new EBDigiCollection);
  auto_ptr<EEDigiCollection> selectedEEDigi(new EEDigiCollection);

  suppressor_->run(*trigPrims, *ebDigis, *eeDigis,
		   *selectedEBDigi, *selectedEEDigi);
		  
  //puts the selected digis into the event:
  event.put(selectedEBDigi, "ebDigis");
  event.put(selectedEEDigi, "eeDigis");
  
  cout << "**********************************************************************" << endl;
  cout << "in " << __PRETTY_FUNCTION__ << endl;
  cout << "**********************************************************************" << endl;
}

const EBDigiCollection*
EcalSelectiveReadoutProducer::getEBDigis(edm::Event& event)
{
  edm::Handle< EBDigiCollection > hEBDigis;
  event.getByLabel(digiProducer_, hEBDigis);
  return hEBDigis.product();
}

const EEDigiCollection*
EcalSelectiveReadoutProducer::getEEDigis(edm::Event& event)
{
  edm::Handle< EEDigiCollection > hEEDigis;
  event.getByLabel(digiProducer_, hEEDigis);
  return hEEDigis.product();
}

const EcalTrigPrimDigiCollection*
EcalSelectiveReadoutProducer::getTrigPrims(edm::Event& event)
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


