// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimCalorimetry/HcalSimAlgos/interface/HcalSignalGenerator.h"



class HcalSignalGeneratorTest : public edm::EDAnalyzer {
public:
  explicit HcalSignalGeneratorTest(const edm::ParameterSet&);
  ~HcalSignalGeneratorTest() {}


private:
  virtual void beginJob() {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() {}

  void dump(CaloVNoiseSignalGenerator * signalGenerator) const;

      // ----------member data ---------------------------
  HcalSimParameterMap theMap;
  HBHESignalGenerator theHBHESignalGenerator;
  HOSignalGenerator theHOSignalGenerator;
  HFSignalGenerator theHFSignalGenerator;
  ZDCSignalGenerator theZDCSignalGenerator;

};


HcalSignalGeneratorTest::HcalSignalGeneratorTest(const edm::ParameterSet& iConfig)
: theMap(),
  theHBHESignalGenerator(iConfig.getParameter<edm::InputTag>("HBHEdigiCollectionPile")),
  theHOSignalGenerator(iConfig.getParameter<edm::InputTag>("HOdigiCollectionPile")),
  theHFSignalGenerator(iConfig.getParameter<edm::InputTag>("HFdigiCollectionPile")),
  theZDCSignalGenerator(iConfig.getParameter<edm::InputTag>("ZDCdigiCollectionPile"))
{
  theHBHESignalGenerator.setParameterMap(&theMap);
  theHOSignalGenerator.setParameterMap(&theMap);
  theHFSignalGenerator.setParameterMap(&theMap);
  theZDCSignalGenerator.setParameterMap(&theMap);

}

void HcalSignalGeneratorTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  theHBHESignalGenerator.initializeEvent(&iEvent, &iSetup);
  theHOSignalGenerator.initializeEvent(&iEvent, &iSetup);
  theHFSignalGenerator.initializeEvent(&iEvent, &iSetup);
  theZDCSignalGenerator.initializeEvent(&iEvent, &iSetup);

  theHBHESignalGenerator.fill();
  theHOSignalGenerator.fill();
  theHFSignalGenerator.fill();
  theZDCSignalGenerator.fill();

  //dump(&theHBHESignalGenerator);
}


void HcalSignalGeneratorTest::dump(CaloVNoiseSignalGenerator * signalGenerator) const
{
  std::vector<CaloSamples> samples;
  signalGenerator->getNoiseSignals(samples);
  for(std::vector<CaloSamples>::const_iterator sampleItr = samples.begin();
      sampleItr != samples.end(); ++sampleItr)
  {
     std::cout << *sampleItr << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalSignalGeneratorTest);
