/**
 *
 *  \author M. Maggi - INFN Bari
 */

#include <memory>
#include <fstream>
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Validation/MuonRPCGeometry/plugins/RPCGeometryServTest.h"

#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

using namespace std;

RPCGeometryServTest::RPCGeometryServTest(const edm::ParameterSet& iConfig)
    : dashedLineWidth_(104), dashedLine_(std::string(dashedLineWidth_, '-')), myName_("RPCGeometryServTest") {
  std::cout << "======================== Opening output file" << std::endl;
  rpcGeomToken_ = esConsumes();
}

RPCGeometryServTest::~RPCGeometryServTest() {}

void RPCGeometryServTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& pDD = iSetup.getHandle(rpcGeomToken_);

  std::cout << myName() << ": Analyzer..." << std::endl;
  std::cout << "start " << dashedLine_ << std::endl;

  std::cout << " Geometry node for RPCGeom is  " << &(*pDD) << std::endl;
  cout << " I have " << pDD->detTypes().size() << " detTypes" << endl;
  cout << " I have " << pDD->detUnits().size() << " detUnits" << endl;
  cout << " I have " << pDD->dets().size() << " dets" << endl;
  cout << " I have " << pDD->rolls().size() << " rolls" << endl;
  cout << " I have " << pDD->chambers().size() << " chambers" << endl;

  std::cout << myName() << ": Begin iteration over geometry..." << std::endl;
  std::cout << "iter " << dashedLine_ << std::endl;

  LocalPoint a(0., 0., 0.);
  for (TrackingGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) {
    //----------------------- RPCCHAMBER TEST ---------------------------

    if (dynamic_cast<const RPCChamber*>(*it) != nullptr) {
      const RPCChamber* ch = dynamic_cast<const RPCChamber*>(*it);

      std::vector<const RPCRoll*> rollsRaf = (ch->rolls());
      for (std::vector<const RPCRoll*>::iterator r = rollsRaf.begin(); r != rollsRaf.end(); ++r) {
        std::cout << dashedLine_ << " NEW ROLL" << std::endl;
        std::cout << "Region = " << (*r)->id().region() << "  Ring = " << (*r)->id().ring()
                  << "  Station = " << (*r)->id().station() << "  Sector = " << (*r)->id().sector()
                  << "  Layer = " << (*r)->id().layer() << "  Subsector = " << (*r)->id().subsector()
                  << "  Roll = " << (*r)->id().roll() << std::endl;
        RPCGeomServ s((*r)->id());
        GlobalPoint g = (*r)->toGlobal(a);
        std::cout << s.name() << " eta partition " << s.eta_partition() << " nroll=" << ch->nrolls() << " z=" << g.z()
                  << " phi=" << g.phi() << " R=" << g.perp() << std::endl;

        if ((*r)->id().region() == 0) {
          if (barzranges.find(s.eta_partition()) != barzranges.end()) {
            std::pair<double, double> cic = barzranges.find(s.eta_partition())->second;
            double cmin = cic.first;
            double cmax = cic.second;
            if (g.z() < cmin)
              cmin = g.z();
            if (g.z() > cmax)
              cmax = g.z();
            std::pair<double, double> cic2(cmin, cmax);
            barzranges[s.eta_partition()] = cic2;
          } else {
            std::pair<double, double> cic(g.z(), g.z());
            barzranges[s.eta_partition()] = cic;
          }
        } else if ((*r)->id().region() == +1) {
          if (forRranges.find(s.eta_partition()) != forRranges.end()) {
            std::pair<double, double> cic = forRranges.find(s.eta_partition())->second;
            double cmin = cic.first;
            double cmax = cic.second;
            if (g.perp() < cmin)
              cmin = g.perp();
            if (g.perp() > cmax)
              cmax = g.perp();
            std::pair<double, double> cic2(cmin, cmax);
            forRranges[s.eta_partition()] = cic2;
          } else {
            std::pair<double, double> cic(g.perp(), g.perp());
            forRranges[s.eta_partition()] = cic;
          }
        } else if ((*r)->id().region() == -1) {
          if (bacRranges.find(s.eta_partition()) != bacRranges.end()) {
            std::pair<double, double> cic = bacRranges.find(s.eta_partition())->second;
            double cmin = cic.first;
            double cmax = cic.second;
            if (g.perp() < cmin)
              cmin = g.perp();
            if (g.perp() > cmax)
              cmax = g.perp();
            std::pair<double, double> cic2(cmin, cmax);
            bacRranges[s.eta_partition()] = cic2;
          } else {
            std::pair<double, double> cic(g.perp(), g.perp());
            bacRranges[s.eta_partition()] = cic;
          }
        }
      }
    }
  }

  std::cout << std::endl;
  std::map<int, std::pair<double, double> >::iterator ieta;

  for (ieta = bacRranges.begin(); ieta != bacRranges.end(); ieta++) {
    std::cout << " Eta " << ieta->first << " Radii = ( " << ieta->second.first << ", " << ieta->second.second << ")"
              << std::endl;
  }

  for (ieta = barzranges.begin(); ieta != barzranges.end(); ieta++) {
    std::cout << " Eta " << ieta->first << " Z = ( " << ieta->second.first << ", " << ieta->second.second << ")"
              << std::endl;
  }

  for (ieta = forRranges.begin(); ieta != forRranges.end(); ieta++) {
    std::cout << " Eta " << ieta->first << " Radii = ( " << ieta->second.first << ", " << ieta->second.second << ")"
              << std::endl;
  }

  std::cout << dashedLine_ << " end" << std::endl;
}
