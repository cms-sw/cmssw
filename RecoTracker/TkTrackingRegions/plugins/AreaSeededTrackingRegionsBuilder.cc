#include "AreaSeededTrackingRegionsBuilder.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include <array>
#include <limits>

namespace {
  float perp2(const std::array<float, 2>& a) {
    return a[0]*a[0] + a[1]*a[1];
  }
}

AreaSeededTrackingRegionsBuilder::AreaSeededTrackingRegionsBuilder(const edm::ParameterSet& regPSet, edm::ConsumesCollector& iC) {
  // operation mode
  std::string modeString       = regPSet.getParameter<std::string>("mode");
  if      (modeString == "BeamSpotFixed") m_mode = BEAM_SPOT_FIXED;
  else if (modeString == "BeamSpotSigma") m_mode = BEAM_SPOT_SIGMA;
  else if (modeString == "VerticesFixed") m_mode = VERTICES_FIXED;
  else if (modeString == "VerticesSigma") m_mode = VERTICES_SIGMA;
  else  throw cms::Exception("Configuration") <<"Unknown mode string: "<<modeString;

  m_extraPhi = regPSet.getParameter<double>("extraPhi");
  m_extraEta = regPSet.getParameter<double>("extraEta");

  token_beamSpot     = iC.consumes<reco::BeamSpot>(regPSet.getParameter<edm::InputTag>("beamSpot"));
  m_maxNVertices     = 1;
  if (m_mode == VERTICES_FIXED || m_mode == VERTICES_SIGMA) {
    token_vertex       = iC.consumes<reco::VertexCollection>(regPSet.getParameter<edm::InputTag>("vertexCollection"));
    m_maxNVertices     = regPSet.getParameter<int>("maxNVertices");
  }

  // RectangularEtaPhiTrackingRegion parameters:
  m_ptMin            = regPSet.getParameter<double>("ptMin");
  m_originRadius     = regPSet.getParameter<double>("originRadius");
  m_zErrorBeamSpot   = regPSet.getParameter<double>("zErrorBeamSpot");
  m_precise          = regPSet.getParameter<bool>("precise");
  m_whereToUseMeasurementTracker = RectangularEtaPhiTrackingRegion::stringToUseMeasurementTracker(regPSet.getParameter<std::string>("whereToUseMeasurementTracker"));
  if(m_whereToUseMeasurementTracker != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever) {
    token_measurementTracker = iC.consumes<MeasurementTrackerEvent>(regPSet.getParameter<edm::InputTag>("measurementTrackerName"));
  }
  m_searchOpt = regPSet.getParameter<bool>("searchOpt");

  // mode-dependent z-halflength of tracking regions
  if (m_mode == VERTICES_SIGMA)  m_nSigmaZVertex   = regPSet.getParameter<double>("nSigmaZVertex");
  if (m_mode == VERTICES_FIXED)  m_zErrorVertex     = regPSet.getParameter<double>("zErrorVertex");
  m_nSigmaZBeamSpot = -1.;
  if (m_mode == BEAM_SPOT_SIGMA) {
    m_nSigmaZBeamSpot = regPSet.getParameter<double>("nSigmaZBeamSpot");
    if (m_nSigmaZBeamSpot < 0.)
      throw cms::Exception("Configuration") << "nSigmaZBeamSpot must be positive for BeamSpotSigma mode!";
  }
}

void AreaSeededTrackingRegionsBuilder::fillDescriptions(edm::ParameterSetDescription& desc) {
  desc.add<double>("extraPhi", 0.);
  desc.add<double>("extraEta", 0.);

  desc.add<std::string>("mode", "BeamSpotFixed");
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("vertexCollection", edm::InputTag("firstStepPrimaryVertices"));
  desc.add<int>("maxNVertices", -1);

  desc.add<double>("ptMin", 0.9);
  desc.add<double>("originRadius", 0.2);
  desc.add<double>("zErrorBeamSpot", 24.2);
  desc.add<bool>("precise", true);

  desc.add<double>("nSigmaZVertex", 3.);
  desc.add<double>("zErrorVertex", 0.2);
  desc.add<double>("nSigmaZBeamSpot", 4.);

  desc.add<std::string>("whereToUseMeasurementTracker", "Never");
  desc.add<edm::InputTag>("measurementTrackerName", edm::InputTag(""));

  desc.add<bool>("searchOpt", false);
}

AreaSeededTrackingRegionsBuilder::Builder AreaSeededTrackingRegionsBuilder::beginEvent(const edm::Event& e, const edm::EventSetup& es) const {
  auto builder = Builder(this);

  // always need the beam spot (as a fall back strategy for vertex modes)
  edm::Handle< reco::BeamSpot > bs;
  e.getByToken( token_beamSpot, bs );
  if( !bs.isValid() ) return builder;

  // this is a default origin for all modes
  GlobalPoint default_origin( bs->x0(), bs->y0(), bs->z0() );

  // vector of origin & halfLength pairs:
  std::vector< std::pair< GlobalPoint, float > > origins;

  // fill the origins and halfLengths depending on the mode
  if (m_mode == BEAM_SPOT_FIXED || m_mode == BEAM_SPOT_SIGMA) {
    builder.addOrigin(default_origin,
                      (m_mode == BEAM_SPOT_FIXED) ? m_zErrorBeamSpot : m_nSigmaZBeamSpot*bs->sigmaZ() );
  } else if (m_mode == VERTICES_FIXED || m_mode == VERTICES_SIGMA) {
    edm::Handle< reco::VertexCollection > vertices;
    e.getByToken( token_vertex, vertices );
    int n_vert = 0;
    for(const auto& v: *vertices) {
      if(v.isFake() || !v.isValid()) continue;

      builder.addOrigin( GlobalPoint( v.x(), v.y(), v.z() ),
                         (m_mode == VERTICES_FIXED) ? m_zErrorVertex : m_nSigmaZVertex*v.zError() );
      ++n_vert;
      if(m_maxNVertices >= 0 && n_vert >= m_maxNVertices) {
        break;
      }
    }
    // no-vertex fall-back case:
    if(builder.empty()) {
      builder.addOrigin( default_origin,
                         (m_nSigmaZBeamSpot > 0.) ? m_nSigmaZBeamSpot*bs->z0Error() : m_zErrorBeamSpot );
    }
  }

  if( !token_measurementTracker.isUninitialized() ) {
    edm::Handle<MeasurementTrackerEvent> hmte;
    e.getByToken(token_measurementTracker, hmte);
    builder.setMeasurementTracker(hmte.product());
  }

  return builder;
}


std::vector<std::unique_ptr<TrackingRegion> > AreaSeededTrackingRegionsBuilder::Builder::regions(const std::vector<Area>& areas) const {
  std::vector<std::unique_ptr<TrackingRegion> > result;

  // create tracking regions in directions of the points of interest
  int n_regions = 0;
  for(const auto& origin: m_origins) {
    float minEta=std::numeric_limits<float>::max(), maxEta=std::numeric_limits<float>::lowest();
    float minPhi=std::numeric_limits<float>::max(), maxPhi=std::numeric_limits<float>::lowest();

    const auto& orig = origin.first;

    LogDebug("AreaSeededTrackingRegionsProducer") << "Origin x,y,z " << orig.x() << "," << orig.y() << "," << orig.z();

    auto unitFromOrig = [&](std::array<float, 2>& vec2) {
      const auto invlen = 1.f/std::sqrt(vec2[0]*vec2[0] + vec2[1]*vec2[1]);
      vec2[0] = orig.x() - vec2[0]*invlen*m_conf->m_originRadius;
      vec2[1] = orig.y() - vec2[1]*invlen*m_conf->m_originRadius;
    };
    for(const auto& area: areas) {
      // straight line assumption is conservative, accounding for
      // low-pT bending would only tighten the eta-phi window

      // eta
      {
        // along orig->area.xymin, farthest point away from area
        std::array<float, 2> pmin = {{area.xmin() - orig.x(), area.ymin() - orig.y()}};
        std::array<float, 2> pmax = {{area.xmax() - orig.x(), area.ymax() - orig.y()}};
        unitFromOrig(pmin);
        unitFromOrig(pmax);
        // pick the one with largest redius to maximize the eta window
        const std::array<float, 2> p = perp2(pmin) > perp2(pmax) ? pmin : pmax;

        minEta = std::min(minEta, etaFromXYZ(area.xmin()-p[0], area.ymin()-p[1],
                                             area.zmin() - (orig.z()+origin.second) ));

        maxEta = std::max(maxEta, etaFromXYZ(area.xmax()-p[0], area.ymax()-p[1],
                                             area.zmax() - (orig.z()-origin.second) ));
      }

      // phi
      {
        // ortogonal to orig->area.xymin, direction for smallest phiMin
        std::array<float, 2> pmin = {{area.ymin() - orig.y(), orig.x() - area.xmin()}};
        unitFromOrig(pmin);

        // orthogonal to orig->area.xymax, direction for largest phiMax
        std::array<float, 2> pmax = {{orig.y() - area.ymax(), area.xmax() - orig.x()}};
        unitFromOrig(pmax);

        auto phimin = std::atan2(area.ymin()-pmin[1], area.xmin()-pmin[0]);
        auto phimax = std::atan2(area.ymax()-pmax[1], area.xmax()-pmax[0]);
        if(phimax < phimin) { // wrapped around, need to decide which one to wrap
          if(phimax < 0) phimax += 2*M_PI;
          else           phimin -= 2*M_PI;
        }

        minPhi = std::min(minPhi, phimin);
        maxPhi = std::max(maxPhi, phimax);
      }

      LogTrace("AreaSeededTrackingRegionsBuilder") << " area x " << area.xmin() << "," << area.ymin()
                                                   << " y " << area.ymin() << "," << area.ymax()
                                                   << " z " << area.zmin() << "," << area.zmax()
                                                   << " eta " << minEta << "," << maxEta
                                                   << " phi " << minPhi << "," << maxPhi;
    }

    const auto meanEta = (minEta+maxEta)/2.f;
    const auto meanPhi = (minPhi+maxPhi)/2.f;
    const auto deltaEta = maxEta-meanEta + m_conf->m_extraEta;
    const auto deltaPhi = maxPhi-meanPhi + m_conf->m_extraPhi;

    const auto x = std::cos(meanPhi);
    const auto y = std::sin(meanPhi);
    const auto z = (x*x+y*y)/std::tan(2.f*std::atan(std::exp(-meanEta))); // simplify?

    LogTrace("AreaSeededTrackingRegionsBuilder") << "Direction x,y,z " << x << "," << y << "," << z
                                                  << " eta,phi " << meanEta << "," << meanPhi
                                                  << " window eta " << (meanEta-deltaEta) << "," << (meanEta+deltaEta)
                                                  << " phi " << (meanPhi-deltaPhi) << "," << (meanPhi+deltaPhi);

    result.push_back( std::make_unique<RectangularEtaPhiTrackingRegion>(
        GlobalVector(x,y,z),
        origin.first, // GlobalPoint
        m_conf->m_ptMin,
        m_conf->m_originRadius,
        origin.second,
        deltaEta,
        deltaPhi,
        m_conf->m_whereToUseMeasurementTracker,
        m_conf->m_precise,
        m_measurementTracker,
        m_conf->m_searchOpt
      ));
    ++n_regions;
  }
  LogDebug("AreaSeededTrackingRegionsBuilder") << "produced "<<n_regions<<" regions";

  return result;
}
