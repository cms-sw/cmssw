#ifndef RecoTracker_PixelLowPtUtilities_ClusterShapeHitFilter_h
#define RecoTracker_PixelLowPtUtilities_ClusterShapeHitFilter_h

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "RecoTracker/PixelLowPtUtilities/interface/ClusterData.h"

#include <utility>
#include <unordered_map>
#include <cstring>

/*****************************************************************************/
class PixelKeys {
public:
  PixelKeys(int part, int dx, int dy) : key((part == 0) ? barrelPacking(dx, dy) : endcapPacking(dx, dy)) {}

  operator unsigned int() const { return key; }

  static unsigned char endcapPacking(int dx, int dy) {
    if (dx < 0 || dy < 0)
      return N;
    if (dx > 10 || dy > 4)
      return N;
    return N_barrel + dx * offset_endcap_dy + dy;  // max 11*5 = 55
  }

  static unsigned char barrelPacking(int dx, int dy) {
    if (dx < 0 || dy < 0)
      return N;
    if (dx > 10 || dy > 15)
      return N;
    if (dx < 8)
      return dx * 16 + dy;  // max 8*16=128
    if (dy > 2)
      return N;
    return 128 + (dx - 8) * 3 + dy;  // max = 128+9 = 137
  }

  bool isValid() const { return key < N; }

  static const int offset_endcap_dy = 5;
  static const int offset_endcap_dx = 10;
  static const int N_endcap = 55;
  static const int N_barrel = 137;
  static const int N = N_barrel + N_endcap;

  bool operator<(const PixelKeys& right) const { return key < right.key; }

private:
  unsigned char key;
};

class StripKeys {
public:
  static const int N = 40;

  StripKeys(int width) : key(width > 0 ? width - 1 : N) {}

  operator unsigned int() const { return key; }

  bool isValid() const { return key < N; }

  bool operator<(const StripKeys& right) const { return key < right.key; }

private:
  unsigned char key;  // max 40;
};

struct PixelLimits {
  PixelLimits() {
    // init to make sure inside is true;
    auto limit = data[0];
    limit[0][0] = -10e10;
    limit[0][1] = 10e10;
    limit[1][0] = -10e10;
    limit[1][1] = 10e10;
    limit = data[1];
    limit[0][0] = -10e10;
    limit[0][1] = 10e10;
    limit[1][0] = -10e10;
    limit[1][1] = 10e10;
  }

  float data[2][2][2];

  bool isInside(const std::pair<float, float>& pred) const {
    auto limit = data[0];
    bool one = (pred.first > limit[0][0]) && (pred.first < limit[0][1]) && (pred.second > limit[1][0]) &&
               (pred.second < limit[1][1]);

    limit = data[1];
    bool two = (pred.first > limit[0][0]) && (pred.first < limit[0][1]) && (pred.second > limit[1][0]) &&
               (pred.second < limit[1][1]);

    return one || two;
  }
};

struct StripLimits {
  StripLimits() {
    data[0][0] = -10e10;
    data[0][1] = 10e10;
    data[1][0] = -10e10;
    data[1][1] = 10e10;
  }

  float data[2][2];

  bool isInside(float pred) const {
    float const* limit = data[0];
    bool one = pred > limit[0] && pred < limit[1];
    limit = data[1];
    bool two = pred > limit[0] && pred < limit[1];

    return one || two;
  }
};

/*****************************************************************************/
namespace edm {
  class EventSetup;
}

class TrackerGeometry;
class TrackerTopology;
class MagneticField;
class SiPixelLorentzAngle;
class SiStripLorentzAngle;
class PixelGeomDetUnit;
class StripGeomDetUnit;
class StripTopology;

// Function for testing ClusterShapeHitFilter
namespace test {
  namespace ClusterShapeHitFilterTest {
    int test();
  }
}  // namespace test

class ClusterShapeHitFilter {
  // For tests
  friend int test::ClusterShapeHitFilterTest::test();

public:
  struct PixelData {
    const PixelGeomDetUnit* det;
    unsigned short part;
    unsigned short layer;
    std::pair<float, float> drift;
    std::pair<float, float> cotangent;
  };

  struct StripData {
    const StripGeomDetUnit* det;
    StripTopology const* topology;
    float drift;
    float thickness;
    int nstrips;
  };

  typedef TrajectoryFilter::Record Record;

  ClusterShapeHitFilter(const TrackerGeometry* theTracker_,
                        const TrackerTopology* theTkTopol_,
                        const MagneticField* theMagneticField_,
                        const SiPixelLorentzAngle* theSiPixelLorentzAngle_,
                        const SiStripLorentzAngle* theSiStripLorentzAngle_,
                        const std::string& pixelShapeFile_,
                        const std::string& pixelShapeFileL1_);

  ~ClusterShapeHitFilter();

  void setShapeCuts(bool cutOnPixelShape, bool cutOnStripShape) {
    cutOnPixelShape_ = cutOnPixelShape;
    cutOnStripShape_ = cutOnStripShape;
  }

  void setChargeCuts(bool cutOnPixelCharge, float minGoodPixelCharge, bool cutOnStripCharge, float minGoodStripCharge) {
    cutOnPixelCharge_ = cutOnPixelCharge;
    minGoodPixelCharge_ = minGoodPixelCharge;
    cutOnStripCharge_ = cutOnStripCharge;
    minGoodStripCharge_ = minGoodStripCharge;
  }

  bool getSizes(const SiPixelRecHit& recHit,
                const LocalVector& ldir,
                const SiPixelClusterShapeCache& clusterShapeCache,
                int& part,
                ClusterData::ArrayType& meas,
                std::pair<float, float>& predr,
                PixelData const* pd = nullptr) const;
  bool isCompatible(const SiPixelRecHit& recHit,
                    const LocalVector& ldir,
                    const SiPixelClusterShapeCache& clusterShapeCache,
                    PixelData const* pd = nullptr) const;
  bool isCompatible(const SiPixelRecHit& recHit,
                    const GlobalVector& gdir,
                    const SiPixelClusterShapeCache& clusterShapeCache,
                    PixelData const* pd = nullptr) const;

  bool getSizes(DetId detId,
                const SiStripCluster& cluster,
                const LocalPoint& lpos,
                const LocalVector& ldir,
                int& meas,
                float& pred) const;
  bool getSizes(
      const SiStripRecHit2D& recHit, const LocalPoint& lpos, const LocalVector& ldir, int& meas, float& pred) const {
    return getSizes(recHit.geographicalId(), recHit.stripCluster(), lpos, ldir, meas, pred);
  }
  bool isCompatible(DetId detId, const SiStripCluster& cluster, const LocalPoint& lpos, const LocalVector& ldir) const;
  bool isCompatible(DetId detId, const SiStripCluster& cluster, const LocalVector& ldir) const {
    return isCompatible(detId, cluster, LocalPoint(0, 0, 0), ldir);
  }

  bool isCompatible(DetId detId, const SiStripCluster& cluster, const GlobalPoint& gpos, const GlobalVector& gdir) const;
  bool isCompatible(DetId detId, const SiStripCluster& cluster, const GlobalVector& gdir) const;

  bool isCompatible(const SiStripRecHit2D& recHit, const LocalPoint& lpos, const LocalVector& ldir) const {
    return isCompatible(recHit.geographicalId(), recHit.stripCluster(), lpos, ldir);
  }
  bool isCompatible(const SiStripRecHit2D& recHit, const LocalVector& ldir) const {
    return isCompatible(recHit.geographicalId(), recHit.stripCluster(), ldir);
  }
  bool isCompatible(const SiStripRecHit2D& recHit, const GlobalPoint& gpos, const GlobalVector& gdir) const {
    return isCompatible(recHit.geographicalId(), recHit.stripCluster(), gpos, gdir);
  }
  bool isCompatible(const SiStripRecHit2D& recHit, const GlobalVector& gdir) const {
    return isCompatible(recHit.geographicalId(), recHit.stripCluster(), gdir);
  }

private:
  // for testing purposes only
  ClusterShapeHitFilter(std::string const& f1, std::string const& f2) {
    loadPixelLimits(f1, pixelLimits);
    loadPixelLimits(f2, pixelLimitsL1);
    loadStripLimits();
  }

  const PixelData& getpd(const SiPixelRecHit& recHit, PixelData const* pd = nullptr) const {
    if (pd)
      return *pd;
    // Get detector
    DetId id = recHit.geographicalId();
    auto p = pixelData.find(id);
    return (*p).second;
  }

  const StripData& getsd(DetId id) const { return stripData.find(id)->second; }

  void loadPixelLimits(std::string const& file, PixelLimits* plim);
  void loadStripLimits();
  void fillPixelData();
  void fillStripData();

  std::pair<float, float> getCotangent(const PixelGeomDetUnit* pixelDet) const;
  float getCotangent(const ClusterShapeHitFilter::StripData& sd, const LocalPoint& p) const;

  std::pair<float, float> getDrift(const PixelGeomDetUnit* pixelDet) const;
  float getDrift(const StripGeomDetUnit* stripDet) const;

  bool isNormalOriented(const GeomDetUnit* geomDet) const;

  const TrackerGeometry* theTracker;
  const TrackerTopology* theTkTopol;
  const MagneticField* theMagneticField;

  const SiPixelLorentzAngle* theSiPixelLorentzAngle;
  const SiStripLorentzAngle* theSiStripLorentzAngle;

  std::unordered_map<unsigned int, PixelData> pixelData;
  std::unordered_map<unsigned int, StripData> stripData;

  PixelLimits pixelLimits[PixelKeys::N + 1];    // [2][2][2]
  PixelLimits pixelLimitsL1[PixelKeys::N + 1];  // for BPIX1

  StripLimits stripLimits[StripKeys::N + 1];  // [2][2]

  float theAngle[6];
  bool cutOnPixelCharge_, cutOnStripCharge_;
  float minGoodPixelCharge_, minGoodStripCharge_;
  bool cutOnPixelShape_, cutOnStripShape_;
  bool checkClusterCharge(DetId detId, const SiStripCluster& cluster, const LocalVector& ldir) const;
  bool checkClusterCharge(DetId detId, const SiPixelCluster& cluster, const LocalVector& ldir) const;
};

#endif
