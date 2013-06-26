#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CalibMuon/CSCCalibration/interface/CSCConditions.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


class CSCNoiseMatrixTest : public edm::EDAnalyzer
{
public:
  CSCNoiseMatrixTest(const edm::ParameterSet & pset) : theDbConditions(pset) {}
  virtual ~CSCNoiseMatrixTest() {}
  //virtual void beginRun(edm::Run const& run, edm::EventSetup const& eventSetup)
  virtual void analyze(const edm::Event&, const edm::EventSetup&eventSetup)
  {
    theDbConditions.initializeEvent(eventSetup);
    // fake signal is going to get noisified 200000 times!
    const int nScaBins = 8;
    const float scaBinSize = 50.;
    std::vector<float> binValues(nScaBins, 0.);

    // find the geometry & conditions for this event
    edm::ESHandle<CSCGeometry> hGeom;
    eventSetup.get<MuonGeometryRecord>().get("idealForDigi",hGeom);
    const CSCGeometry *pGeom = &*hGeom;

    // try making a noisifier and using it
    std::vector<CSCLayer*> layers = pGeom->layers();
    for(std::vector<CSCLayer*>::const_iterator layerItr = layers.begin();
        layerItr != layers.end(); ++layerItr)
    {
      unsigned nstrips = (**layerItr).geometry()->numberOfStrips();
      for(unsigned istrip = 1; istrip != nstrips; ++istrip)
      {
        CSCAnalogSignal signal(istrip, scaBinSize, binValues, 0., 0.);
        //theDbConditions.fetchNoisifier((**layerItr).id(),istrip); 
        try {       
          theDbConditions.noisify((**layerItr).id(), signal);
        }
        catch(cms::Exception & e) {
           std::cerr << "Bad Noise Matrix for " << (**layerItr).id()  
                     << "strip " << istrip << "\n";
           std::cerr << e.what() << std::endl;
        }
      }
    }
  }

  //virtual void analyze(const edm::Event&, const edm::EventSetup&) {}

private:
  CSCDbStripConditions theDbConditions;
};

DEFINE_FWK_MODULE(CSCNoiseMatrixTest);

