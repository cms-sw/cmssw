#ifndef RecoTracker_TkTrackingRegions_AreaSeededTrackingRegionsProducer_h
#define RecoTracker_TkTrackingRegions_AreaSeededTrackingRegionsProducer_h


#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"

#include "AreaSeededTrackingRegionsBuilder.h"

#include <array>
#include <limits>

/** class AreaSeededTrackingRegionsProducer
 *
 * eta-phi TrackingRegions producer in directions defined by z-phi area-based objects of interest
 * from the "input" parameters.
 *
 * Four operational modes are supported ("mode" parameter):
 *
 *   BeamSpotFixed:
 *     origin is defined by the beam spot
 *     z-half-length is defined by a fixed zErrorBeamSpot parameter
 *   BeamSpotSigma:
 *     origin is defined by the beam spot
 *     z-half-length is defined by nSigmaZBeamSpot * beamSpot.sigmaZ
 *   VerticesFixed:
 *     origins are defined by vertices from VertexCollection (use maximum MaxNVertices of them)
 *     z-half-length is defined by a fixed zErrorVertex parameter
 *   VerticesSigma:
 *     origins are defined by vertices from VertexCollection (use maximum MaxNVertices of them)
 *     z-half-length is defined by nSigmaZVertex * vetex.zError
 *
 *   If, while using one of the "Vertices" modes, there's no vertices in an event, we fall back into
 *   either BeamSpotSigma or BeamSpotFixed mode, depending on the positiveness of nSigmaZBeamSpot.
 *
 */
class AreaSeededTrackingRegionsProducer {
public:

  typedef enum {BEAM_SPOT_FIXED, BEAM_SPOT_SIGMA, VERTICES_FIXED, VERTICES_SIGMA } Mode;

  AreaSeededTrackingRegionsProducer(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC):
    m_builder(conf.getParameter<edm::ParameterSet>("RegionPSet"), iC)
  {
    edm::ParameterSet regPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");
    for(const auto& area: regPSet.getParameter<std::vector<edm::ParameterSet> >("areas")) {
      m_areas.emplace_back(area.getParameter<double>("r"),
                           area.getParameter<double>("phimin"),
                           area.getParameter<double>("phimax"),
                           area.getParameter<double>("zmin"),
                           area.getParameter<double>("zmax"));
    }
    if(m_areas.empty())
      throw cms::Exception("Configuration") << "Empty 'areas' parameter.";
  }
  
  ~AreaSeededTrackingRegionsProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    edm::ParameterSetDescription descAreas;
    descAreas.add<double>("r", 0.0);
    descAreas.add<double>("zmin", 0.0);
    descAreas.add<double>("zmax", 0.0);
    descAreas.add<double>("phimin", 0.0);
    descAreas.add<double>("phimax", 0.0);
    std::vector<edm::ParameterSet> vDefaults;
    /*
    edm::ParameterSet vDefaults1;
    vDefaults1.addParameter<double>("r", 0.0);
    vDefaults1.addParameter<double>("zmin", 0.0);
    vDefaults1.addParameter<double>("zmax", 0.0);
    vDefaults1.addParameter<double>("phimin", 0.0);
    vDefaults1.addParameter<double>("phimax", 0.0);
    vDefaults.push_back(vDefaults1);
    */
    desc.addVPSet("areas", descAreas, vDefaults);

    AreaSeededTrackingRegionsBuilder::fillDescriptions(desc);

    // Only for backwards-compatibility
    edm::ParameterSetDescription descRegion;
    descRegion.add<edm::ParameterSetDescription>("RegionPSet", desc);

    descriptions.add("areaSeededTrackingRegion", descRegion);
  }

  std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event& e, const edm::EventSetup& es) const
  {
    auto builder = m_builder.beginEvent(e, es);
    return builder.regions(m_areas);
  }
  
private:
  AreaSeededTrackingRegionsBuilder m_builder;
  std::vector<AreaSeededTrackingRegionsBuilder::Area> m_areas;
};

#endif
