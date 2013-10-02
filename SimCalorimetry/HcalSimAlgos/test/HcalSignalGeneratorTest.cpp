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

  edm::EDGetTokenT<HBHEDigitizerTraits::DigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HODigitizerTraits::DigiCollection> tok_ho_;
  edm::EDGetTokenT<HFDigitizerTraits::DigiCollection> tok_hf_;
  edm::EDGetTokenT<ZDCDigitizerTraits::DigiCollection> tok_zdc_;

  edm::InputTag theHBHETag_, theHOTag_, theHFTag_, theZDCTag_;

};


HcalSignalGeneratorTest::HcalSignalGeneratorTest(const edm::ParameterSet& iConfig)
: theMap(),
  /*theHBHESignalGenerator(iConfig.getParameter<edm::InputTag>("HBHEdigiCollectionPile")),
  theHOSignalGenerator(iConfig.getParameter<edm::InputTag>("HOdigiCollectionPile")),
  theHFSignalGenerator(iConfig.getParameter<edm::InputTag>("HFdigiCollectionPile")),
  theZDCSignalGenerator(iConfig.getParameter<edm::InputTag>("ZDCdigiCollectionPile"))*/
  theHBHETag_(iConfig.getParameter<edm::InputTag>("HBHEdigiCollectionPile")),
  theHOTag_(iConfig.getParameter<edm::InputTag>("HOdigiCollectionPile")),
  theHFTag_(iConfig.getParameter<edm::InputTag>("HFdigiCollectionPile")),
  theZDCTag_(iConfig.getParameter<edm::InputTag>("ZDCdigiCollectionPile"))
{

  tok_hbhe_ = consumes<HBHEDigitizerTraits::DigiCollection>(theHBHETag_);
  tok_ho_ = consumes<HODigitizerTraits::DigiCollection>(theHOTag_);
  tok_hf_ = consumes<HFDigitizerTraits::DigiCollection>(theHFTag_);
  tok_zdc_ = consumes<ZDCDigitizerTraits::DigiCollection>(theZDCTag_);

  theHBHESignalGenerator = HBHESignalGenerator(theHBHETag_,tok_hbhe_);
  theHOSignalGenerator = HOSignalGenerator(theHOTag_,tok_ho_);
  theHFSignalGenerator = HFSignalGenerator(theHFTag_,tok_hf_);
  theZDCSignalGenerator = ZDCSignalGenerator(theZDCTag_,tok_zdc_);

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

  theHBHESignalGenerator.fill(nullptr);
  theHOSignalGenerator.fill(nullptr);
  theHFSignalGenerator.fill(nullptr);
  theZDCSignalGenerator.fill(nullptr);

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
