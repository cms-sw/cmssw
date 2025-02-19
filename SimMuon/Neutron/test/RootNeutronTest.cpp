#include "SimMuon/Neutron/src/RootNeutronWriter.h"
#include "SimMuon/Neutron/src/RootNeutronReader.h"


/*
 *  should write:
 * one event of one hit
 * one event of two hits
 *  Then read out three events, so it loops back.
 */

using namespace std;

void testWriting()
{
  RootNeutronWriter writer("NeutronsME.root");

  int nevent = 5;
  int hitCount = 0;
  for(int event = 1; event <= nevent; ++event) {
    int detType = (rand() % 10) + 1;
    vector<PSimHit> hits;
    int nhits = event;
    for(int i = 0; i < nhits; ++i) {
      int layer   = i%6 + 1;
      ++hitCount;
      PSimHit pSimHit(LocalPoint(i, hitCount,-0.25), LocalPoint(i, hitCount,0.25),
              100., 300., 0.0001, 11, layer, 1, 0., 0.);
      hits.push_back(pSimHit);
    }
    writer.writeCluster(detType, hits);
  }

}


void testReading()
{
  RootNeutronReader reader("NeutronsME.root");
  for(int i = 0; i < 30; ++i) {
    for(int detType = 1; detType <= 10; ++detType) 
    {
      vector<PSimHit> hits;
      reader.readNextEvent(detType, hits);
      for(unsigned int i = 0; i != hits.size(); ++i) {
        cout <<  "OVAL DET " << detType << " " << hits[i] << endl;
      }
    }
  }

}


int main() 
{
  testWriting();
  testReading();
}
  

