#include "SimCalorimetry/EcalZeroSuppressionProducers/interface/ESZeroSuppressionProducer.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"

ESZeroSuppressionProducer::ESZeroSuppressionProducer(const edm::ParameterSet& ps) {

  digiProducer_   = ps.getParameter<std::string>("digiProducer");
  ESdigiCollection_ = ps.getParameter<std::string>("ESdigiCollection");
  ESZSdigiCollection_ = ps.getParameter<std::string>("ESZSdigiCollection");
 
  produces<ESDigiCollection>(ESZSdigiCollection_);
}

ESZeroSuppressionProducer::~ESZeroSuppressionProducer() { 

}

void ESZeroSuppressionProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {

  eventSetup.get<ESThresholdsRcd>().get(esthresholds_);
  const ESThresholds *thresholds = esthresholds_.product();

  eventSetup.get<ESPedestalsRcd>().get(espeds_);
  const ESPedestals *pedestals = espeds_.product();

  float ts2Threshold = thresholds->getTS2Threshold();

  edm::Handle<ESDigiCollection> ESDigis;

  bool fullESDigis = true;
  event.getByLabel(digiProducer_, ESDigis);
  if (!ESDigis.isValid()) {
    edm::LogError("ZeroSuppressionError") << "Error! can't get the product " << ESdigiCollection_.c_str() ;
    fullESDigis = false;
  }  

  std::auto_ptr<ESDigiCollection> ESZSDigis(new ESDigiCollection());
  
  if (fullESDigis) {
    for (ESDigiCollection::const_iterator i (ESDigis->begin()); 
         i!=ESDigis->end(); ++i) {            

      ESDataFrame dataframe = (*i);

      ESPedestals::const_iterator it_ped = pedestals->find(dataframe.id());

      if (dataframe.sample(1).adc() > (ts2Threshold+it_ped->getMean())) {
	//std::cout<<dataframe.sample(1).adc()<<" "<<ts2Threshold+it_ped->getMean()<<std::endl;
	(*ESZSDigis).push_back(*i);
      }
    }
  }     
  
  event.put(ESZSDigis, ESZSdigiCollection_);  
}

