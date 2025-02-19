// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"

#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1F.h>

class GaussianTailNoiseGeneratorTest : public edm::EDAnalyzer {
public:
  explicit GaussianTailNoiseGeneratorTest(const edm::ParameterSet&);
  ~GaussianTailNoiseGeneratorTest();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  // ----------member data ---------------------------
  std::string filename_;
  TFile* hFile;
  TH1F* randNumber;
  CLHEP::HepRandomEngine* rndEngine;
  GaussianTailNoiseGenerator* genNoise;
};

namespace CLHEP {
  class HepRandomEngine;
}

GaussianTailNoiseGeneratorTest::GaussianTailNoiseGeneratorTest(const edm::ParameterSet& iConfig):
  filename_(iConfig.getParameter<std::string>("FileName"))
{
   //now do what ever initialization is needed
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "GaussianTailNoiseGeneratorTest requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }

  rndEngine  = &(rng->getEngine());
  genNoise = new GaussianTailNoiseGenerator((*rndEngine));

}


GaussianTailNoiseGeneratorTest::~GaussianTailNoiseGeneratorTest()
{
  delete genNoise;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
GaussianTailNoiseGeneratorTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  float threshold = 2.;
  int numStrips = 512;
  float noiseRMS = 5*250;

  std::vector<std::pair<int,float> > generatedNoise;

  genNoise->generate(numStrips,threshold,noiseRMS,generatedNoise);

  typedef std::vector<std::pair<int,float> >::const_iterator VI;

  for(VI p = generatedNoise.begin(); p != generatedNoise.end(); p++){
    randNumber->Fill((*p).second);
  }

}

// ------------ method called once each job just before starting event loop  ------------
void 
GaussianTailNoiseGeneratorTest::beginJob()
{
  hFile = new TFile (filename_.c_str(), "RECREATE" );
  randNumber = new TH1F("randNumber","Random Number Distribution",200,2000,6000); 
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GaussianTailNoiseGeneratorTest::endJob(){
  hFile->Write();
  hFile->Close();
  
  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(GaussianTailNoiseGeneratorTest);
