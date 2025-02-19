#include "SimMuon/Neutron/interface/SubsystemNeutronReader.h"
#include "TRandom.h"


class TestNeutronReader : public SubsystemNeutronReader
{
public:
  TestNeutronReader(const edm::ParameterSet & p) : SubsystemNeutronReader(p)
  {
  }

  virtual int detId(int chamberIndex, int localDetId ) {return 0;}
};


int main() {
  edm::ParameterSet p;
  p.addParameter("luminosity", 1.);
  p.addParameter("startTime", -50.);
  p.addParameter("endTime", 150.);
  p.addParameter("eventOccupancy", std::vector<double>(11, 0.01));
  p.addParameter("reader", std::string("ROOT"));
  p.addParameter("output" , std::string("CSCNeutronHits.root"));
  p.addParameter("nChamberTypes", 10);
std::cout << "BUILD " << std::endl;
try {
  TestNeutronReader reader(p);
std::cout << "BUILT " << std::endl;
  for(int ev = 0; ev < 10; ++ev) {
    std::cout << "NEW EVENT " << std::endl;
    reader.clear();
    for(int ch = 1; ch <= 10; ++ch) {
      edm::PSimHitContainer cont;
      reader.generateChamberNoise(ch, ch, cont);
      std::cout << "This event has " << cont.size() << " SimHits " << std::endl;
    }
  }
}
catch (cms::Exception& x)
{
      std::cerr << "cms Exception caught, message follows\n"
                << x.what();
}

}

