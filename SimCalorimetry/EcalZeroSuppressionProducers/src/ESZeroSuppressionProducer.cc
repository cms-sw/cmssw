#include "SimCalorimetry/EcalZeroSuppressionProducers/interface/ESZeroSuppressionProducer.h"

ESZeroSuppressionProducer::ESZeroSuppressionProducer(const edm::ParameterSet& ps) 
{
  ESdigiCollection_ = ps.getParameter<std::string>("ESdigiCollection");
  digiProducer_   = ps.getParameter<std::string>("digiProducer");
 
  ESGain = ps.getUntrackedParameter<int>("ESGain", 1);
  ESBaseline = ps.getUntrackedParameter<int>("ESBaseline", 1000);
  ESMIPADC = ps.getUntrackedParameter<double>("ESMIPADC", 9);
  ESMIPkeV = ps.getUntrackedParameter<double>("ESMIPkeV", 78.47);
  ESNoiseSigma = ps.getUntrackedParameter<double>("ESNoiseSigma", 3);
  ESThreshold = 1.45*ESNoiseSigma*ESMIPkeV/ESMIPADC;

  algo_ = new ESRecHitSimAlgo(ESGain, ESBaseline, ESMIPADC, ESMIPkeV);

  produces<ESDigiCollection>();
}

ESZeroSuppressionProducer::~ESZeroSuppressionProducer() 
{ 
}

void ESZeroSuppressionProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) 
{
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
      
      if (algo_->EvalAmplitude(*i) > ESThreshold) (*ESZSDigis).push_back(*i);

    }
  }   
  
  event.put(ESZSDigis);  
}

