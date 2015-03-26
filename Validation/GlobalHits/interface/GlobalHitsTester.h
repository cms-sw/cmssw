#ifndef GlobalHitsTester_h
#define GlobalHitsTester_h

/** \class GlobalHitsAnalyzer
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
//#include "DataFormats/DetId/interface/DetId.h"
#include "TRandom.h"
#include "TRandom3.h"

//DQM services
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// tracker info
//#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
//#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

// muon info
//#include "Geometry/Records/interface/MuonGeometryRecord.h"
//#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
//#include "Geometry/DTGeometry/interface/DTGeometry.h"
//#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
//#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
//#include "DataFormats/MuonDetId/interface/RPCDetId.h"
//#include "DataFormats/MuonDetId/interface/DTWireId.h"

// calorimeter info
//#include "Geometry/Records/interface/IdealGeometryRecord.h"
//#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
//#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
//#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
//#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
//#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

// data in edm::event
//#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
//#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//#include "SimDataFormats/Track/interface/SimTrackContainer.h"
//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
//#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

// helper files
//#include <CLHEP/Vector/LorentzVector.h>
//#include "DataFormats/Math/interface/LorentzVector.h"
//#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>

#include "TString.h"

class GlobalHitsTester : public DQMEDAnalyzer
{
  
 public:

  explicit GlobalHitsTester(const edm::ParameterSet&);
  virtual ~GlobalHitsTester();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &,
    edm::Run const &, edm::EventSetup const &) override;
  
private:
  std::string fName;
  int verbosity;
  int frequency;
  int vtxunit;
  std::string label;
  bool getAllProvenances;
  bool printProvenanceInfo;

  std::string outputfile;
  bool doOutput;

  MonitorElement *meTestString;
  MonitorElement *meTestInt;
  MonitorElement *meTestFloat;
  MonitorElement *meTestTH1F;
  MonitorElement *meTestTH2F;
  MonitorElement *meTestTH3F;
  MonitorElement *meTestProfile1;
  MonitorElement *meTestProfile2;

  TRandom *Random;
  double RandomVal1;
  double RandomVal2;
  double RandomVal3;

 // private statistics information
  unsigned int count;
};

#endif
