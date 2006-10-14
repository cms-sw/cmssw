#include "SimCalorimetry/EcalZeroSuppressionProducers/interface/ESZeroSuppressionProducer.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

ESZeroSuppressionProducer::ESZeroSuppressionProducer(const edm::ParameterSet& ps) 
{
  digiProducer_   = ps.getParameter<std::string>("digiProducer");
  ESdigiCollection_ = ps.getParameter<std::string>("ESdigiCollection");
  ESZSdigiCollection_ = ps.getParameter<std::string>("ESZSdigiCollection");
 
  ESGain = ps.getUntrackedParameter<int>("ESGain", 1);
  ESBaseline = ps.getUntrackedParameter<int>("ESBaseline", 1000);
  ESMIPADC = ps.getUntrackedParameter<double>("ESMIPADC", 9);
  ESMIPkeV = ps.getUntrackedParameter<double>("ESMIPkeV", 78.47);
  ESNoiseSigma = ps.getUntrackedParameter<double>("ESNoiseSigma", 3);

  if (ESGain == 0)
    ESThreshold = 3.*1.45*ESNoiseSigma*ESMIPkeV/ESMIPADC/1000000.;
  else if (ESGain == 1)
    ESThreshold = 3.*0.9066*ESNoiseSigma*ESMIPkeV/ESMIPADC/1000000.;
  else if (ESGain == 2)
    ESThreshold = 3.*0.8815*ESNoiseSigma*ESMIPkeV/ESMIPADC/1000000.;

  algo_ = new ESRecHitSimAlgo(ESGain, ESBaseline, ESMIPADC, ESMIPkeV);

  produces<ESDigiCollection>(ESZSdigiCollection_);
}

ESZeroSuppressionProducer::~ESZeroSuppressionProducer() 
{ 
}

void ESZeroSuppressionProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) 
{
  checkGeometry(eventSetup);

  edm::Handle<ESDigiCollection> ESDigis;

  bool fullESDigis = true;

  try {
    event.getByLabel(digiProducer_, ESDigis);
  } catch ( std::exception& ex ) {
    edm::LogError("ZeroSuppressionError") << "Error! can't get the product " << ESdigiCollection_.c_str() ;
    fullESDigis = false;
  }  

  std::auto_ptr<ESDigiCollection> ESZSDigis(new ESDigiCollection());

  if (fullESDigis) {
    ESDigiCollection::const_iterator i;
    for (i=ESDigis->begin(); i!=ESDigis->end(); i++) {            
      if (algo_->EvalAmplitude(*i, false) > ESThreshold) (*ESZSDigis).push_back(*i);
    }
  }     

  event.put(ESZSDigis, ESZSdigiCollection_);  
}

void ESZeroSuppressionProducer::checkGeometry(const edm::EventSetup & eventSetup)
{
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> hGeometry;
  eventSetup.get<IdealGeometryRecord>().get(hGeometry);
   
  const CaloGeometry *pGeometry = &*hGeometry;
   
  // see if we need to update
  if(pGeometry != theGeometry) {
    theGeometry = pGeometry;
    updateGeometry();
  }
}
 
void ESZeroSuppressionProducer::updateGeometry()
{
  algo_->setGeometry(theGeometry);
}


