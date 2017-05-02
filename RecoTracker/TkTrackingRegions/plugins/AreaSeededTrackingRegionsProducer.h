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

  AreaSeededTrackingRegionsProducer(const edm::ParameterSet& conf, edm::ConsumesCollector && iC) {
    edm::ParameterSet regPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");

    // operation mode
    std::string modeString       = regPSet.getParameter<std::string>("mode");
    if      (modeString == "BeamSpotFixed") m_mode = BEAM_SPOT_FIXED;
    else if (modeString == "BeamSpotSigma") m_mode = BEAM_SPOT_SIGMA;
    else if (modeString == "VerticesFixed") m_mode = VERTICES_FIXED;
    else if (modeString == "VerticesSigma") m_mode = VERTICES_SIGMA;
    else  throw cms::Exception("Configuration") <<"Unknown mode string: "<<modeString;

    // basic inputs
    for(const auto& area: regPSet.getParameter<std::vector<edm::ParameterSet> >("areas")) {
      m_areas.push_back(Area(area.getParameter<double>("r"),
                             area.getParameter<double>("phimin"),
                             area.getParameter<double>("phimax"),
                             area.getParameter<double>("zmin"),
                             area.getParameter<double>("zmax")));
    }
    if(m_areas.empty())
      throw cms::Exception("Configuration") << "Empty 'areas' parameter.";
    
    token_beamSpot     = iC.consumes<reco::BeamSpot>(regPSet.getParameter<edm::InputTag>("beamSpot"));
    m_maxNVertices     = 1;
    if (m_mode == VERTICES_FIXED || m_mode == VERTICES_SIGMA)
    {
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
    if (m_mode == BEAM_SPOT_SIGMA)
    {
      m_nSigmaZBeamSpot = regPSet.getParameter<double>("nSigmaZBeamSpot");
      if (m_nSigmaZBeamSpot < 0.)
        throw cms::Exception("Configuration") << "nSigmaZBeamSpot must be positive for BeamSpotSigma mode!";
    }
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

    // Only for backwards-compatibility
    edm::ParameterSetDescription descRegion;
    descRegion.add<edm::ParameterSetDescription>("RegionPSet", desc);

    descriptions.add("areaSeededTrackingRegion", descRegion);
  }

  std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event& e, const edm::EventSetup& es) const
  {
    std::vector<std::unique_ptr<TrackingRegion> > result;

    // always need the beam spot (as a fall back strategy for vertex modes)
    edm::Handle< reco::BeamSpot > bs;
    e.getByToken( token_beamSpot, bs );
    if( !bs.isValid() ) return result;

    // this is a default origin for all modes
    GlobalPoint default_origin( bs->x0(), bs->y0(), bs->z0() );

    // vector of origin & halfLength pairs:
    std::vector< std::pair< GlobalPoint, float > > origins;

    // fill the origins and halfLengths depending on the mode
    if (m_mode == BEAM_SPOT_FIXED || m_mode == BEAM_SPOT_SIGMA) {
      origins.push_back( std::make_pair( default_origin,
					 (m_mode == BEAM_SPOT_FIXED) ? m_zErrorBeamSpot : m_nSigmaZBeamSpot*bs->sigmaZ()
					 ));
    } else if (m_mode == VERTICES_FIXED || m_mode == VERTICES_SIGMA) {
      edm::Handle< reco::VertexCollection > vertices;
      e.getByToken( token_vertex, vertices );
      int n_vert = 0;
      for(const auto& v: *vertices) {
        if(v.isFake() || !v.isValid()) continue;
	
        origins.push_back( std::make_pair( GlobalPoint( v.x(), v.y(), v.z() ),
					   (m_mode == VERTICES_FIXED) ? m_zErrorVertex : m_nSigmaZVertex*v.zError()
					   ));
        ++n_vert;
        if(m_maxNVertices >= 0 && n_vert >= m_maxNVertices) {
          break;
        }
      }
      // no-vertex fall-back case:
      if(origins.empty()) {
        origins.push_back( std::make_pair( default_origin,
					   (m_nSigmaZBeamSpot > 0.) ? m_nSigmaZBeamSpot*bs->z0Error() : m_zErrorBeamSpot
					   ));
      }
    }
    
    const MeasurementTrackerEvent *measurementTracker = nullptr;
    if( !token_measurementTracker.isUninitialized() ) {
      edm::Handle<MeasurementTrackerEvent> hmte;
      e.getByToken(token_measurementTracker, hmte);
      measurementTracker = hmte.product();
    }

    // create tracking regions in directions of the points of interest
    int n_regions = 0;
    for(const auto& origin: origins) {
      float minEta=std::numeric_limits<float>::max(), maxEta=std::numeric_limits<float>::lowest();
      float minPhi=std::numeric_limits<float>::max(), maxPhi=std::numeric_limits<float>::lowest();

      const auto& orig = origin.first;

      LogDebug("AreaSeededTrackingRegionsProducer") << "Origin x,y,z " << orig.x() << "," << orig.y() << "," << orig.z();

      auto unitFromOrig = [&](std::array<float, 2>& vec2) {
        const auto invlen = 1.f/std::sqrt(vec2[0]*vec2[0] + vec2[1]*vec2[1]);
        vec2[0] = orig.x() - vec2[0]*invlen*m_originRadius;
        vec2[1] = orig.y() - vec2[1]*invlen*m_originRadius;
      };
      for(const auto& area: m_areas) {
        // straight line assumption is conservative, accounding for
        // low-pT bending would only tighten the eta-phi window

        // eta
        {
          // along orig->area.xymin, farthest point away from area
          std::array<float, 2> pmin = {{area.xmin - orig.x(), area.ymin - orig.y()}};
          std::array<float, 2> pmax = {{area.xmax - orig.x(), area.ymax - orig.y()}};
          unitFromOrig(pmin);
          unitFromOrig(pmax);
          // pick the one with largest redius to maximize the eta window
          const std::array<float, 2> p = perp2(pmin) > perp2(pmax) ? pmin : pmax;

          minEta = std::min(minEta, etaFromXYZ(area.xmin-p[0], area.ymin-p[1],
                                               area.zmin - (orig.z()+origin.second) ));

          maxEta = std::max(maxEta, etaFromXYZ(area.xmax-p[0], area.ymax-p[1],
                                               area.zmax - (orig.z()-origin.second) ));
        }

        // phi
        {
          // ortogonal to orig->area.xymin, direction for smallest phiMin
          std::array<float, 2> pmin = {{area.ymin - orig.y(), orig.x() - area.xmin}};
          unitFromOrig(pmin);

          // orthogonal to orig->area.xymax, direction for largest phiMax
          std::array<float, 2> pmax = {{orig.y() - area.ymax, area.xmax - orig.x()}};
          unitFromOrig(pmax);

          auto phimin = std::atan2(area.ymin-pmin[1], area.xmin-pmin[0]);
          auto phimax = std::atan2(area.ymax-pmax[1], area.xmax-pmax[0]);
          if(phimax < phimin) { // wrapped around, need to decide which one to wrap
            if(phimax < 0) phimax += 2*M_PI;
            else           phimin -= 2*M_PI;
          }

          minPhi = std::min(minPhi, phimin);
          maxPhi = std::max(maxPhi, phimax);
        }

        LogTrace("AreaSeededTrackingRegionsProducer") << " area x " << area.xmin << "," << area.ymin
                                                      << " y " << area.ymin << "," << area.ymax
                                                      << " z " << area.zmin << "," << area.zmax
                                                      << " eta " << minEta << "," << maxEta
                                                      << " phi " << minPhi << "," << maxPhi;
      }

      const auto meanEta = (minEta+maxEta)/2.f;
      const auto meanPhi = (minPhi+maxPhi)/2.f;
      const auto deltaEta = maxEta-meanEta;
      const auto deltaPhi = maxPhi-meanPhi;

      const auto x = std::cos(meanPhi);
      const auto y = std::sin(meanPhi);
      const auto z = (x*x+y*y)/std::tan(2.f*std::atan(std::exp(-meanEta))); // simplify?

      LogTrace("AreaSeededTrackingRegionsProducer") << "Direction x,y,z " << x << "," << y << "," << z
                                                    << " eta,phi " << meanEta << "," << meanPhi
                                                    << " window eta " << (meanEta-deltaEta) << "," << (meanEta+deltaEta)
                                                    << " phi " << (meanPhi-deltaPhi) << "," << (meanPhi+deltaPhi);

      result.push_back( std::make_unique<RectangularEtaPhiTrackingRegion>(
          GlobalVector(x,y,z),
          origin.first, // GlobalPoint
          m_ptMin,
          m_originRadius,
          origin.second,
          deltaEta,
          deltaPhi,
          m_whereToUseMeasurementTracker,
          m_precise,
          measurementTracker,
          m_searchOpt
        ));
      ++n_regions;
    }
    LogDebug("AreaSeededTrackingRegionsProducer") << "produced "<<n_regions<<" regions";
    
    return result;
  }
  
private:
  float perp2(const std::array<float, 2>& a) const {
    return a[0]*a[0] + a[1]*a[1];
  }

  Mode m_mode;

  edm::EDGetTokenT<reco::VertexCollection> token_vertex; 
  edm::EDGetTokenT<reco::BeamSpot> token_beamSpot; 
  int m_maxNVertices;

  struct Area {
    Area(double r, double phimin, double phimax, double zmin_, double zmax_):
      xmin(r*std::cos(phimin)),
      xmax(r*std::cos(phimax)),
      ymin(r*std::sin(phimin)),
      ymax(r*std::sin(phimax)),
      zmin(zmin_), zmax(zmax_) {}

    const float xmin = 0;
    const float xmax = 0;
    const float ymin = 0;
    const float ymax = 0;
    const float zmin = 0;
    const float zmax = 0;
  };
  std::vector<Area> m_areas;

  float m_ptMin;
  float m_originRadius;
  float m_zErrorBeamSpot;
  bool m_precise;
  edm::EDGetTokenT<MeasurementTrackerEvent> token_measurementTracker;
  RectangularEtaPhiTrackingRegion::UseMeasurementTracker m_whereToUseMeasurementTracker;
  bool m_searchOpt;

  float m_nSigmaZVertex;
  float m_zErrorVertex;
  float m_nSigmaZBeamSpot;
};

#endif
