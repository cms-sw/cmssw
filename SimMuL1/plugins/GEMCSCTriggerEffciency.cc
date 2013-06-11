#include <memory>
#include <fstream>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

// root include files
#include "TTree.h"
#include "TFile.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include <DataFormats/MuonDetId/interface/GEMDetId.h>

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
 
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include "CommonTools/UtilAlgos/interface/TFileService.h"


class GEMCSCTriggerEfficiency : public edm::EDAnalyzer 
{
public:
  /// constructor
  explicit GEMCSCTriggerEfficiency(const edm::ParameterSet&);
  /// destructor
  ~GEMCSCTriggerEfficiency();

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);

  virtual void beginJob() ;

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void endJob() ;
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
 
};

//
// constructors and destructor
//
GEMCSCTriggerEfficiency::GEMCSCTriggerEfficiency(const edm::ParameterSet& iConfig)
{
}

GEMCSCTriggerEfficiency::~GEMCSCTriggerEfficiency()
{
}

void GEMCSCTriggerEfficiency::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
}


void GEMCSCTriggerEfficiency::beginJob()
{
}


void GEMCSCTriggerEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


void GEMCSCTriggerEfficiency::endJob() 
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GEMCSCTriggerEfficiency::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plugin
DEFINE_FWK_MODULE(GEMCSCTriggerEfficiency);
