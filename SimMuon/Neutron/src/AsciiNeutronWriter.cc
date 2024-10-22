#include "SimMuon/Neutron/src/AsciiNeutronWriter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>
#include <fstream>

using namespace std;

AsciiNeutronWriter::AsciiNeutronWriter(string fileNameBase) : theFileNameBase(fileNameBase) {}

AsciiNeutronWriter::~AsciiNeutronWriter() {}

void AsciiNeutronWriter::writeCluster(int chamberType, const edm::PSimHitContainer& hits) {
  // open the correct file
  stringstream s;
  s << theFileNameBase << chamberType;
  LogDebug("NeutronWriter") << "opening " << s.str();
  ofstream os;
  os.open(s.str().c_str(), ofstream::app);
  os << hits.size() << endl;
  for (size_t i = 0; i < hits.size(); ++i) {
    const PSimHit& h = hits[i];
    os << h.entryPoint().x() << " " << h.entryPoint().y() << " " << h.entryPoint().z() << " " << h.exitPoint().x()
       << " " << h.exitPoint().y() << " " << h.exitPoint().z() << " " << h.pabs() << " "
       << " " << h.timeOfFlight() << " " << h.energyLoss() << " " << h.particleType() << " " << h.detUnitId() << " "
       << h.trackId() << " " << h.momentumAtEntry().theta() << " " << h.momentumAtEntry().phi() << endl;
  }
}
