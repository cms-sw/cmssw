#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimCalorimetry/EcalZeroSuppressionProducers/interface/ESZeroSuppressionProducer.h"

ESZeroSuppressionProducer::ESZeroSuppressionProducer(const edm::ParameterSet &ps)
    : digiProducer_(ps.getParameter<std::string>("digiProducer")),
      ESdigiCollection_(ps.getParameter<std::string>("ESdigiCollection")),
      ESZSdigiCollection_(ps.getParameter<std::string>("ESZSdigiCollection")),
      ES_token(consumes<ESDigiCollection>(edm::InputTag(digiProducer_))),
      esthresholdsToken_(esConsumes()),
      espedsToken_(esConsumes()) {
  produces<ESDigiCollection>(ESZSdigiCollection_);
}

ESZeroSuppressionProducer::~ESZeroSuppressionProducer() {}

void ESZeroSuppressionProducer::produce(edm::Event &event, const edm::EventSetup &eventSetup) {
  const ESThresholds &thresholds = eventSetup.getData(esthresholdsToken_);
  const ESPedestals &pedestals = eventSetup.getData(espedsToken_);

  float ts2Threshold = thresholds.getTS2Threshold();

  edm::Handle<ESDigiCollection> ESDigis;

  bool fullESDigis = true;
  event.getByToken(ES_token, ESDigis);
  if (!ESDigis.isValid()) {
    edm::LogError("ZeroSuppressionError") << "Error! can't get the product " << ESdigiCollection_.c_str();
    fullESDigis = false;
  }

  std::unique_ptr<ESDigiCollection> ESZSDigis(new ESDigiCollection());

  if (fullESDigis) {
    for (ESDigiCollection::const_iterator i(ESDigis->begin()); i != ESDigis->end(); ++i) {
      ESDataFrame dataframe = (*i);

      ESPedestals::const_iterator it_ped = pedestals.find(dataframe.id());

      if (dataframe.sample(1).adc() > (ts2Threshold + it_ped->getMean())) {
        // std::cout<<dataframe.sample(1).adc()<<"
        // "<<ts2Threshold+it_ped->getMean()<<std::endl;
        (*ESZSDigis).push_back(*i);
      }
    }
  }

  event.put(std::move(ESZSDigis), ESZSdigiCollection_);
}
