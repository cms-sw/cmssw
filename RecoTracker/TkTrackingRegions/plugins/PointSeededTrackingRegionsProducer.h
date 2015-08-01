#ifndef PointSeededTrackingRegionsProducer_h
#define PointSeededTrackingRegionsProducer_h


#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
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
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

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

    // basic inputs
    edm::ParameterSet point_input = regPSet.getParameter<edm::ParameterSet>("point_input");
    eta_input          = point_input.getParameter<double>("eta");
    phi_input          = point_input.getParameter<double>("phi");
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
    m_whereToUseMeasurementTracker = RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kForSiStrips;
    if (regPSet.exists("measurementTrackerName"))
    {
      // FIXME: when next time altering the configuration of this
      // class, please change the types of the following parameters:
      // - whereToUseMeasurementTracker to at least int32 or to a string
      //   corresponding to the UseMeasurementTracker enumeration
      // - measurementTrackerName to InputTag
      if (regPSet.exists("whereToUseMeasurementTracker"))
        m_whereToUseMeasurementTracker = RectangularEtaPhiTrackingRegion::doubleToUseMeasurementTracker(regPSet.getParameter<double>("whereToUseMeasurementTracker"));
      if(m_whereToUseMeasurementTracker != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever)
        token_measurementTracker = iC.consumes<MeasurementTrackerEvent>(regPSet.getParameter<std::string>("measurementTrackerName"));
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
    

  virtual std::vector<TrackingRegion* > regions(const edm::Event& e, const edm::EventSetup& es) const
  {
    std::vector<TrackingRegion* > result;

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
    size_t n_points = 1;
    int n_regions = 0;
    for(size_t i = 0; i < n_points && n_regions < m_maxNRegions; ++i ) {

      double x = TMath::Cos(phi_input);
      double y = TMath::Sin(phi_input);
      double theta = 2*TMath::ATan(TMath::Exp(-eta_input));
      double z = (x*x+y*y)/TMath::Tan(theta);

      GlobalVector direction( x,y,z );
	
      for (size_t  j=0; j<origins.size() && n_regions < m_maxNRegions; ++j) {

        result.push_back( new RectangularEtaPhiTrackingRegion(
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
    //std::cout<<"n_seeded_regions = "<<n_regions<<std::endl;
    edm::LogInfo ("PointSeededTrackingRegionsProducer") << "produced "<<n_regions<<" regions";
    
    return result;
  }
  
private:

  Mode m_mode;

  int m_maxNRegions;
  edm::EDGetTokenT<reco::VertexCollection> token_vertex; 
  edm::EDGetTokenT<reco::BeamSpot> token_beamSpot; 
  double eta_input, phi_input;
  int m_maxNVertices;

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
