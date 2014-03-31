#include "Validation/MuonGEMHits/interface/DigiMatcher.h"
#include "Validation/MuonGEMHits/interface/SimHitMatcher.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include <cmath>

using namespace std;
using namespace matching;


namespace {

// translate half-strip number [1..nstrip*2] into fractional strip number [0..nstrip)
float halfstripToStrip(int hs)
{
  //return (hs + 1)/2;
  return 0.5 * hs - 0.25;
}


bool is_gem(unsigned int detid)
{
  DetId id;
  if (id.subdetId() == MuonSubdetId::GEM) return true;
  return false;
}


}


DigiMatcher::DigiMatcher(SimHitMatcher& sh)
: BaseMatcher(sh.trk(), sh.vtx(), sh.conf(), sh.event(), sh.eventSetup())
, simhit_matcher_(&sh)
{

}


DigiMatcher::~DigiMatcher() {}


GlobalPoint
DigiMatcher::digiPosition(const Digi& digi) const
{
  unsigned int id = digi_id(digi);
  int strip = digi_channel(digi);
  DigiType t = digi_type(digi);

  GlobalPoint gp;
  if ( t == GEM_STRIP )
  {
    GEMDetId idd(id);
    LocalPoint lp = gem_geo_->etaPartition(idd)->centreOfStrip(strip);
    gp = gem_geo_->idToDet(id)->surface().toGlobal(lp);
  }
  else if ( t == GEM_PAD )
  {
    GEMDetId idd(id);
    LocalPoint lp = gem_geo_->etaPartition(idd)->centreOfPad(strip);
    gp = gem_geo_->idToDet(id)->surface().toGlobal(lp);
  }
  else if ( t == GEM_COPAD)
  {
    GEMDetId id1(id);
    LocalPoint lp1 = gem_geo_->etaPartition(id1)->centreOfPad(strip);
    GlobalPoint gp1 = gem_geo_->idToDet(id)->surface().toGlobal(lp1);

    GEMDetId id2(id1.region(), id1.ring(), id1.station(), 2, id1.chamber(), id1.roll());
    LocalPoint lp2 = gem_geo_->etaPartition(id2)->centreOfPad(strip);
    GlobalPoint gp2 = gem_geo_->idToDet(id2())->surface().toGlobal(lp2);

    gp = GlobalPoint( (gp1.x()+gp2.x())/2., (gp1.y()+gp2.y())/2., (gp1.z()+gp2.z())/2.);
  }
  return gp;
}


GlobalPoint
DigiMatcher::digisMeanPosition(const DigiMatcher::DigiContainer& digis) const
{
  GlobalPoint point_zero;
  if (digis.empty()) return point_zero; // point "zero"

  float sumx, sumy, sumz;
  sumx = sumy = sumz = 0.f;
  size_t n = 0;
  for (auto& d: digis)
  {
    GlobalPoint gp = digiPosition(d);
    if (gp == point_zero) continue;

    sumx += gp.x();
    sumy += gp.y();
    sumz += gp.z();
    ++n;
  }
  if (n == 0) return GlobalPoint();
  return GlobalPoint(sumx/n, sumy/n, sumz/n);
}
