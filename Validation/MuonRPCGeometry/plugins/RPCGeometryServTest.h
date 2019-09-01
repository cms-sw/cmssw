/**
 *
 *  \author M. Maggi - INFN Bari
 */

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include <string>
#include <cmath>
#include <vector>
#include <map>
#include <iomanip>
#include <set>

class RPCGeometryServTest : public edm::EDAnalyzer {
public:
  RPCGeometryServTest(const edm::ParameterSet& pset);

  ~RPCGeometryServTest() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const std::string& myName() { return myName_; }

private:
  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
  std::map<int, std::pair<double, double> > barzranges;
  std::map<int, std::pair<double, double> > forRranges;
  std::map<int, std::pair<double, double> > bacRranges;
};
