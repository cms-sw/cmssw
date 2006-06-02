#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/EcalSelectiveReadoutProducer.h"

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

  
