// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

#include <TFile.h>
#include <TH1F.h>
#include <TROOT.h>

class GaussianTailNoiseGeneratorTest : public edm::EDAnalyzer {
public:
  explicit GaussianTailNoiseGeneratorTest(const edm::ParameterSet &);
  ~GaussianTailNoiseGeneratorTest() override;

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;
  // ----------member data ---------------------------
  std::string filename_;
  TFile *hFile;
  TH1F *randNumber;
  GaussianTailNoiseGenerator *genNoise;
};

namespace CLHEP {
  class HepRandomEngine;
}

GaussianTailNoiseGeneratorTest::GaussianTailNoiseGeneratorTest(const edm::ParameterSet &iConfig)
    : filename_(iConfig.getParameter<std::string>("FileName")) {
  // now do what ever initialization is needed
  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration") << "GaussianTailNoiseGeneratorTest requires the "
                                             "RandomNumberGeneratorService\n"
                                             "which is not present in the configuration file.  You must add the "
                                             "service\n"
                                             "in the configuration file or remove the modules that require it.";
  }
  genNoise = new GaussianTailNoiseGenerator();
}

GaussianTailNoiseGeneratorTest::~GaussianTailNoiseGeneratorTest() { delete genNoise; }

//
// member functions
//

// ------------ method called to for each event  ------------
void GaussianTailNoiseGeneratorTest::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  float threshold = 2.;
  int numStrips = 512;
  float noiseRMS = 5 * 250;

  std::vector<std::pair<int, float>> generatedNoise;

  edm::Service<edm::RandomNumberGenerator> rng;
  genNoise->generate(numStrips, threshold, noiseRMS, generatedNoise, &rng->getEngine(iEvent.streamID()));

  typedef std::vector<std::pair<int, float>>::const_iterator VI;

  for (VI p = generatedNoise.begin(); p != generatedNoise.end(); p++) {
    randNumber->Fill((*p).second);
  }
}

// ------------ method called once each job just before starting event loop
// ------------
void GaussianTailNoiseGeneratorTest::beginJob() {
  hFile = new TFile(filename_.c_str(), "RECREATE");
  randNumber = new TH1F("randNumber", "Random Number Distribution", 200, 2000, 6000);
}

// ------------ method called once each job just after ending the event loop
// ------------
void GaussianTailNoiseGeneratorTest::endJob() {
  hFile->Write();
  hFile->Close();

  return;
}

// define this as a plug-in
DEFINE_FWK_MODULE(GaussianTailNoiseGeneratorTest);
