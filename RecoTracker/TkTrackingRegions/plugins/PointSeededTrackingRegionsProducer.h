#ifndef PointSeededTrackingRegionsProducer_h
#define PointSeededTrackingRegionsProducer_h


#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <TMath.h>
#include <TLorentzVector.h>

/** class PointSeededTrackingRegionsProducer
 *
 * eta-phi TrackingRegions producer in directions defined by Point-based objects of interest
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
 *     z-half-length is defined by a fixed zErrorVetex parameter
 *   VerticesSigma:
 *     origins are defined by vertices from VertexCollection (use maximum MaxNVertices of them)
 *     z-half-length is defined by nSigmaZVertex * vetex.zError
 *
 *   If, while using one of the "Vertices" modes, there's no vertices in an event, we fall back into
 *   either BeamSpotSigma or BeamSpotFixed mode, depending on the positiveness of nSigmaZBeamSpot.
 *
 */
class  PointSeededTrackingRegionsProducer : public TrackingRegionProducer
{
public:

  typedef enum {BEAM_SPOT_FIXED, BEAM_SPOT_SIGMA, VERTICES_FIXED, VERTICES_SIGMA } Mode;

  explicit PointSeededTrackingRegionsProducer(const edm::ParameterSet& conf,
					      edm::ConsumesCollector && iC)
  {
    edm::ParameterSet regPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");

    // operation mode
    std::string modeString       = regPSet.getParameter<std::string>("mode");
    if      (modeString == "BeamSpotFixed") m_mode = BEAM_SPOT_FIXED;
    else if (modeString == "BeamSpotSigma") m_mode = BEAM_SPOT_SIGMA;
    else if (modeString == "VerticesFixed") m_mode = VERTICES_FIXED;
    else if (modeString == "VerticesSigma") m_mode = VERTICES_SIGMA;
    else  edm::LogError ("PointSeededTrackingRegionsProducer")<<"Unknown mode string: "<<modeString;

    // basic inputsi
    edm::ParameterSet points = regPSet.getParameter<edm::ParameterSet>("points");
    etaPoints = points.getParameter<std::vector<double>>("eta");
    phiPoints = points.getParameter<std::vector<double>>("phi");
    if (!(etaPoints.size() == phiPoints.size()))  throw edm::Exception(edm::errors::Configuration) << "The parameters 'eta' and 'phi' must have the same size";;
    m_maxNRegions      = regPSet.getParameter<int>("maxNRegions");
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
    m_deltaEta         = regPSet.getParameter<double>("deltaEta");
    m_deltaPhi         = regPSet.getParameter<double>("deltaPhi");
    m_precise          = regPSet.getParameter<bool>("precise");
    m_whereToUseMeasurementTracker = RectangularEtaPhiTrackingRegion::stringToUseMeasurementTracker(regPSet.getParameter<std::string>("whereToUseMeasurementTracker"));
    if(m_whereToUseMeasurementTracker != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever) {
      token_measurementTracker = iC.consumes<MeasurementTrackerEvent>(regPSet.getParameter<edm::InputTag>("measurementTrackerName"));
    }
    m_searchOpt = false;
    if (regPSet.exists("searchOpt")) m_searchOpt = regPSet.getParameter<bool>("searchOpt");

    // mode-dependent z-halflength of tracking regions
    if (m_mode == VERTICES_SIGMA)  m_nSigmaZVertex   = regPSet.getParameter<double>("nSigmaZVertex");
    if (m_mode == VERTICES_FIXED)  m_zErrorVetex     = regPSet.getParameter<double>("zErrorVetex");
    m_nSigmaZBeamSpot = -1.;
    if (m_mode == BEAM_SPOT_SIGMA)
    {
      m_nSigmaZBeamSpot = regPSet.getParameter<double>("nSigmaZBeamSpot");
      if (m_nSigmaZBeamSpot < 0.)
        edm::LogError ("PointSeededTrackingRegionsProducer")<<"nSigmaZBeamSpot must be positive for BeamSpotSigma mode!";
    }
  }
  
  virtual ~PointSeededTrackingRegionsProducer() {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    edm::ParameterSetDescription descPoints;
    descPoints.add<std::vector<double>> ("eta", {0.} ); 
    descPoints.add<std::vector<double>> ("phi", {0.} ); 
    desc.add<edm::ParameterSetDescription>("points", descPoints);
	
    desc.add<std::string>("mode", "BeamSpotFixed");
    desc.add<int>("maxNRegions", 10);
    desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
    desc.add<edm::InputTag>("vertexCollection", edm::InputTag("hltPixelVertices"));
    desc.add<int>("maxNVertices", 1);

    desc.add<double>("ptMin", 0.9);
    desc.add<double>("originRadius", 0.2);
    desc.add<double>("zErrorBeamSpot", 24.2);
    desc.add<double>("deltaEta", 0.5);
    desc.add<double>("deltaPhi", 0.5);
    desc.add<bool>("precise", true);

    desc.add<double>("nSigmaZVertex", 3.);
    desc.add<double>("zErrorVetex", 0.2);
    desc.add<double>("nSigmaZBeamSpot", 4.);

    desc.add<std::string>("whereToUseMeasurementTracker", "ForSiStrips");
    desc.add<edm::InputTag>("measurementTrackerName", edm::InputTag(""));
 
    desc.add<bool>("searchOpt", false); 

    // Only for backwards-compatibility
    edm::ParameterSetDescription descRegion;
    descRegion.add<edm::ParameterSetDescription>("RegionPSet", desc);
    //edm::ParameterSetDescription descPoint;
    //descPoint.add<edm::ParameterSetDescription>("point_input", desc);


    descriptions.add("pointSeededTrackingRegion", descRegion);
  }

    

  virtual std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event& e, const edm::EventSetup& es) const override
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
      for (reco::VertexCollection::const_iterator iv = vertices->begin(), ev = vertices->end();
	   iv != ev && n_vert < m_maxNVertices; ++iv) {
        if ( iv->isFake() || !iv->isValid() ) continue;
	
        origins.push_back( std::make_pair( GlobalPoint( iv->x(), iv->y(), iv->z() ),
					   (m_mode == VERTICES_FIXED) ? m_zErrorVetex : m_nSigmaZVertex*iv->zError()
					   ));
        ++n_vert;
      }
      // no-vertex fall-back case:
      if ( origins.empty() ) {
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

    // create tracking regions (maximum MaxNRegions of them) in directions of the
    // points of interest
    size_t n_points = etaPoints.size();
    int n_regions = 0;
    for(size_t i = 0; i < n_points && n_regions < m_maxNRegions; ++i ) {

      double x = std::cos(phiPoints[i]);
      double y = std::sin(phiPoints[i]);
      double theta = 2*std::atan(std::exp(-etaPoints[i]));
      double z = 1./std::tan(theta);

      GlobalVector direction( x,y,z );
	
      for (size_t  j=0; j<origins.size() && n_regions < m_maxNRegions; ++j) {

        result.push_back( std::make_unique<RectangularEtaPhiTrackingRegion>(
          direction, // GlobalVector
          origins[j].first, // GlobalPoint
          m_ptMin,
          m_originRadius,
          origins[j].second,
          m_deltaEta,
          m_deltaPhi,
          m_whereToUseMeasurementTracker,
          m_precise,
          measurementTracker,
          m_searchOpt
        ));
        ++n_regions;
      }
    }
    edm::LogInfo ("PointSeededTrackingRegionsProducer") << "produced "<<n_regions<<" regions";
    
    return result;
  }
  
private:

  Mode m_mode;

  int m_maxNRegions;
  edm::EDGetTokenT<reco::VertexCollection> token_vertex; 
  edm::EDGetTokenT<reco::BeamSpot> token_beamSpot; 
  int m_maxNVertices;

  std::vector<double> etaPoints;
  std::vector<double> phiPoints;

  float m_ptMin;
  float m_originRadius;
  float m_zErrorBeamSpot;
  float m_deltaEta;
  float m_deltaPhi;
  bool m_precise;
  edm::EDGetTokenT<MeasurementTrackerEvent> token_measurementTracker;
  RectangularEtaPhiTrackingRegion::UseMeasurementTracker m_whereToUseMeasurementTracker;
  bool m_searchOpt;

  float m_nSigmaZVertex;
  float m_zErrorVetex;
  float m_nSigmaZBeamSpot;
};

#endif
