#ifndef RecoTracker_TkTrackingRegions_AreaSeededTrackingRegionsBuilder_h
#define RecoTracker_TkTrackingRegions_AreaSeededTrackingRegionsBuilder_h

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/VecArray.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingSeedCandidates.h"
class AreaSeededTrackingRegionsBuilder {
public:
  using Origin = std::pair<GlobalPoint, float>;  // (origin, half-length in z)
  using Origins = std::vector<Origin>;

  class Area {
  public:
    Area() {}

    // phimin and phimax, and hence xmin+xmax and ymin+ymax are
    // ordered by which way one goes around the unit circle, so it may
    // happen that actually phimax < phimin
    Area(float rmin, float rmax, float phimin, float phimax, float zmin, float zmax) : m_zmin(zmin), m_zmax(zmax) {
      auto cosphimin = std::cos(phimin);
      auto sinphimin = std::sin(phimin);
      auto cosphimax = std::cos(phimax);
      auto sinphimax = std::sin(phimax);

      m_x_rmin_phimin = rmin * cosphimin;
      m_x_rmin_phimax = rmin * cosphimax;
      m_x_rmax_phimin = rmax * cosphimin;
      m_x_rmax_phimax = rmax * cosphimax;

      m_y_rmin_phimin = rmin * sinphimin;
      m_y_rmin_phimax = rmin * sinphimax;
      m_y_rmax_phimin = rmax * sinphimin;
      m_y_rmax_phimax = rmax * sinphimax;
    }

    float x_rmin_phimin() const { return m_x_rmin_phimin; }
    float x_rmin_phimax() const { return m_x_rmin_phimax; }
    float x_rmax_phimin() const { return m_x_rmax_phimin; }
    float x_rmax_phimax() const { return m_x_rmax_phimax; }
    float y_rmin_phimin() const { return m_y_rmin_phimin; }
    float y_rmin_phimax() const { return m_y_rmin_phimax; }
    float y_rmax_phimin() const { return m_y_rmax_phimin; }
    float y_rmax_phimax() const { return m_y_rmax_phimax; }

    float zmin() const { return m_zmin; }
    float zmax() const { return m_zmax; }

  private:
    // all of these are in global coordinates
    float m_x_rmin_phimin = 0;
    float m_x_rmin_phimax = 0;
    float m_x_rmax_phimin = 0;
    float m_x_rmax_phimax = 0;

    float m_y_rmin_phimin = 0;
    float m_y_rmin_phimax = 0;
    float m_y_rmax_phimin = 0;
    float m_y_rmax_phimax = 0;

    float m_zmin = 0;
    float m_zmax = 0;
  };

  class Builder {
  public:
    explicit Builder(const AreaSeededTrackingRegionsBuilder* conf) : m_conf(conf) {}
    ~Builder() = default;

    void setMeasurementTracker(const MeasurementTrackerEvent* mte) { m_measurementTracker = mte; }
    void setCandidates(const TrackingSeedCandidates::Objects cands) { candidates = cands; }

    std::vector<std::unique_ptr<TrackingRegion> > regions(const Origins& origins, const std::vector<Area>& areas) const;
    std::unique_ptr<TrackingRegion> region(const Origin& origin, const std::vector<Area>& areas) const;
    std::unique_ptr<TrackingRegion> region(const Origin& origin, const edm::VecArray<Area, 2>& areas) const;

  private:
    template <typename T>
    std::unique_ptr<TrackingRegion> regionImpl(const Origin& origin, const T& areas) const;

    const AreaSeededTrackingRegionsBuilder* m_conf = nullptr;
    const MeasurementTrackerEvent* m_measurementTracker = nullptr;
    TrackingSeedCandidates::Objects candidates;
  };

  AreaSeededTrackingRegionsBuilder(const edm::ParameterSet& regPSet, edm::ConsumesCollector&& iC)
      : AreaSeededTrackingRegionsBuilder(regPSet, iC) {}
  AreaSeededTrackingRegionsBuilder(const edm::ParameterSet& regPSet, edm::ConsumesCollector& iC);
  ~AreaSeededTrackingRegionsBuilder() = default;

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  Builder beginEvent(const edm::Event& e) const;

private:
  std::vector<Area> m_areas;
  TrackingSeedCandidates candidates_;
  float m_extraPhi;
  float m_extraEta;
  float m_ptMin;
  float m_originRadius;
  bool m_precise;
  edm::EDGetTokenT<MeasurementTrackerEvent> token_measurementTracker;
  RectangularEtaPhiTrackingRegion::UseMeasurementTracker m_whereToUseMeasurementTracker;
  bool m_searchOpt;
};

#endif
