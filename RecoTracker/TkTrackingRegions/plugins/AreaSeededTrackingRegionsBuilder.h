#ifndef RecoTracker_TkTrackingRegions_AreaSeededTrackingRegionsBuilder_h
#define RecoTracker_TkTrackingRegions_AreaSeededTrackingRegionsBuilder_h

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class AreaSeededTrackingRegionsBuilder {
public:
  using Origins = std::vector< std::pair< GlobalPoint, float > >; // (origin, half-length in z)

  class Area {
  public:
    Area(double r, double phimin, double phimax, double zmin, double zmax):
      m_xmin(r*std::cos(phimin)),
      m_xmax(r*std::cos(phimax)),
      m_ymin(r*std::sin(phimin)),
      m_ymax(r*std::sin(phimax)),
      m_zmin(zmin), m_zmax(zmax) {}

    float xmin() const { return m_xmin; }
    float xmax() const { return m_xmax; }
    float ymin() const { return m_ymin; }
    float ymax() const { return m_ymax; }
    float zmin() const { return m_zmin; }
    float zmax() const { return m_zmax; }

  private:
    const float m_xmin = 0;
    const float m_xmax = 0;
    const float m_ymin = 0;
    const float m_ymax = 0;
    const float m_zmin = 0;
    const float m_zmax = 0;
  };

  class Builder {
  public:
    explicit Builder(const AreaSeededTrackingRegionsBuilder *conf): m_conf(conf) {}
    ~Builder() = default;

    void setMeasurementTracker(const MeasurementTrackerEvent *mte) { m_measurementTracker = mte; }

    std::vector<std::unique_ptr<TrackingRegion> > regions(const Origins& origins, const std::vector<Area>& areas) const;

  private:
    const AreaSeededTrackingRegionsBuilder *m_conf = nullptr;
    const MeasurementTrackerEvent *m_measurementTracker = nullptr;
  };

  AreaSeededTrackingRegionsBuilder(const edm::ParameterSet& regPSet, edm::ConsumesCollector&& iC): AreaSeededTrackingRegionsBuilder(regPSet, iC) {}
  AreaSeededTrackingRegionsBuilder(const edm::ParameterSet& regPSet, edm::ConsumesCollector& iC);
  ~AreaSeededTrackingRegionsBuilder() = default;

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  Builder beginEvent(const edm::Event& e) const;

private:
  std::vector<Area> m_areas;

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
