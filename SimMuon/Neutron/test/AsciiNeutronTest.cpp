#include "Utilities/Configuration/interface/Architecture.h"
#include "Muon/MuonNeutrons/interface/AsciiNeutronHitWriter.h"
#include "Muon/MuonNeutrons/interface/AsciiNeutronHitReader.h"
#include "Profound/PersistentTrackingHits/interface/PSimHit.h"
#include "Muon/MuonNeutrons/test/DummySimHit.h"


/*
 *  should write:
 * one event of one hit
 * one event of two hits
 *  Then read out three events, so it loops back.
 */

int main() {
  AsciiNeutronHitWriter writer("NeutronsME");
  int detType = 10;
  int nevent = 2;
  int hitCount = 0;
  for(int event = 1; event <= nevent; ++event) {
    vector<PSimHit> hits;
    int nhits = event;
    for(int i = 0; i < nhits; ++i) {
      int layer   = i%6 + 1;
      ++hitCount;
      // leaks like crazy, but it's just a test
      PSimHit * pSimHit = new PSimHit(LocalPoint(i, hitCount,-0.25), LocalPoint(i, hitCount,0.25),
              100., 300., 0.0001, 11, layer, 1, 0., 0.);
      DummySimHit * simHit = new DummySimHit(pSimHit, 0., 1);
      hits.push_back(writer.makePSimHit(simHit, layer));
    }
    writer.writeCluster(detType, hits);
  }

  AsciiNeutronHitReader reader("NeutronsME", 11);
  for(int i = 0; i < 3; ++i) {
    vector<PSimHit> hits = reader.readNextEvent(detType);
    for(int i = 0; i != hits.size(); ++i) {
      cout <<  "OVAL " << hits[i] << endl;
    }
  }
}

  

