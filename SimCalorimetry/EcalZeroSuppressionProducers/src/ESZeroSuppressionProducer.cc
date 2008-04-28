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
  ESMIPkeV = ps.getUntrackedParameter<double>("ESMIPkeV", 81.08);
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
  delete algo_;
}

void ESZeroSuppressionProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) 
{
  edm::Handle<ESDigiCollection> ESDigis;

  bool fullESDigis = true;
  event.getByLabel(digiProducer_, ESDigis);
  if (!ESDigis.isValid()) {
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

