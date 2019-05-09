#include "CalibMuon/CSCCalibration/interface/CSCConditions.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

class CSCNoiseMatrixTest : public edm::EDAnalyzer {
public:
  CSCNoiseMatrixTest(const edm::ParameterSet &pset) : theDbConditions(pset) {
    edm::Service<edm::RandomNumberGenerator> rng;
    if (!rng.isAvailable()) {
      throw cms::Exception("Configuration") << "CSCNoiseMatrixTest requires the RandomNumberGeneratorService\n"
                                               "which is not present in the configuration file.  You must add "
                                               "the service\n"
                                               "in the configuration file or remove the modules that require it.";
    }
  }

  ~CSCNoiseMatrixTest() override {}
  // virtual void beginRun(edm::Run const& run, edm::EventSetup const&
  // eventSetup)
  void analyze(const edm::Event &e, const edm::EventSetup &eventSetup) override {
    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine *engine = &rng->getEngine(e.streamID());

    theDbConditions.initializeEvent(eventSetup);
    // fake signal is going to get noisified 200000 times!
    const int nScaBins = 8;
    const float scaBinSize = 50.;
    std::vector<float> binValues(nScaBins, 0.);

    // find the geometry & conditions for this event
    edm::ESHandle<CSCGeometry> hGeom;
    eventSetup.get<MuonGeometryRecord>().get("idealForDigi", hGeom);
    const CSCGeometry *pGeom = &*hGeom;

    // try making a noisifier and using it
    const CSCGeometry::LayerContainer &layers = pGeom->layers();
    for (auto layerItr = layers.begin(); layerItr != layers.end(); ++layerItr) {
      unsigned nstrips = (**layerItr).geometry()->numberOfStrips();
      for (unsigned istrip = 1; istrip != nstrips; ++istrip) {
        CSCAnalogSignal signal(istrip, scaBinSize, binValues, 0., 0.);
        // theDbConditions.fetchNoisifier((**layerItr).id(),istrip);
        try {
          theDbConditions.noisify((**layerItr).id(), signal, engine);
        } catch (cms::Exception &e) {
          std::cerr << "Bad Noise Matrix for " << (**layerItr).id() << "strip " << istrip << "\n";
          std::cerr << e.what() << std::endl;
        }
      }
    }
  }

  // virtual void analyze(const edm::Event&, const edm::EventSetup&) {}

private:
  CSCDbStripConditions theDbConditions;
};

DEFINE_FWK_MODULE(CSCNoiseMatrixTest);
