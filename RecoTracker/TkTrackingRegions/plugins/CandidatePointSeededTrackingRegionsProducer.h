#ifndef CandidatePointSeededTrackingRegionsProducer_h
#define CandidatePointSeededTrackingRegionsProducer_h


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
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/normalizedPhi.h"


/** class CandidatePointSeededTrackingRegionsProducer
 *
 * eta-phi TrackingRegions producer in directions defined by Candidate-based objects of interest
 * from a collection defined by the "input" parameter.
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
 * Three seeding modes are supported ("seedingMode" parameter):
 *
 *   Candidate-seeded:
 *      defines regions around candidates from the "input" collection
 *   Point-seeded:
 *      defines regions around fixed points in the detector (previously in PointSeededTrackingRegionsProducer)
 *   Candidate+point-seeded:
 *      defines regions as intersections of regions around candidates from the "input" collections and around fixed points in the detector
 *
 *   \authors Vadim Khotilovich + Thomas Strebler
 */
class  CandidatePointSeededTrackingRegionsProducer : public TrackingRegionProducer
{
public:

  enum class Mode {BEAM_SPOT_FIXED, BEAM_SPOT_SIGMA, VERTICES_FIXED, VERTICES_SIGMA };
  enum class SeedingMode {CANDIDATE_SEEDED, POINT_SEEDED, CANDIDATE_POINT_SEEDED};

  explicit CandidatePointSeededTrackingRegionsProducer(const edm::ParameterSet& conf,
	edm::ConsumesCollector && iC)
  {
    edm::ParameterSet regPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");

    // operation mode
    std::string modeString       = regPSet.getParameter<std::string>("mode");
    if      (modeString == "BeamSpotFixed") m_mode = Mode::BEAM_SPOT_FIXED;
    else if (modeString == "BeamSpotSigma") m_mode = Mode::BEAM_SPOT_SIGMA;
    else if (modeString == "VerticesFixed") m_mode = Mode::VERTICES_FIXED;
    else if (modeString == "VerticesSigma") m_mode = Mode::VERTICES_SIGMA;
    else throw edm::Exception(edm::errors::Configuration) << "Unknown mode string: "<<modeString;

    // seeding mode
    std::string seedingModeString = regPSet.getParameter<std::string>("seedingMode");
    if      (seedingModeString == "Candidate")      m_seedingMode = SeedingMode::CANDIDATE_SEEDED;
    else if (seedingModeString == "Point")          m_seedingMode = SeedingMode::POINT_SEEDED;
    else if (seedingModeString == "CandidatePoint") m_seedingMode = SeedingMode::CANDIDATE_POINT_SEEDED;
    else throw edm::Exception(edm::errors::Configuration) << "Unknown seeding mode string: "<<seedingModeString;

    // basic inputs
    if(m_seedingMode == SeedingMode::CANDIDATE_SEEDED || m_seedingMode == SeedingMode::CANDIDATE_POINT_SEEDED)
      token_input        = iC.consumes<reco::CandidateView>(regPSet.getParameter<edm::InputTag>("input"));

    // Specific points in the detector
    if(m_seedingMode == SeedingMode::POINT_SEEDED || m_seedingMode == SeedingMode::CANDIDATE_POINT_SEEDED){
      edm::ParameterSet points = regPSet.getParameter<edm::ParameterSet>("points");
      etaPoints = points.getParameter<std::vector<double>>("eta");
      phiPoints = points.getParameter<std::vector<double>>("phi");
      n_points = etaPoints.size();

      if (!(etaPoints.size() == phiPoints.size()))  throw edm::Exception(edm::errors::Configuration) << "The parameters 'eta' and 'phi' must have the same size";
      if (n_points == 0) throw edm::Exception(edm::errors::Configuration) << "At least one point should be defined for point or candidate+point seeding modes";

      for(size_t i = 0; i < n_points; ++i ){

      	double x = std::cos(phiPoints[i]);
	double y = std::sin(phiPoints[i]);
	double theta = 2*std::atan(std::exp(-etaPoints[i]));
	double z = 1./std::tan(theta);
	GlobalVector direction( x,y,z );
	directionPoints.push_back(direction);

      }

    }

    m_maxNRegions      = regPSet.getParameter<int>("maxNRegions");
    token_beamSpot     = iC.consumes<reco::BeamSpot>(regPSet.getParameter<edm::InputTag>("beamSpot"));
    m_maxNVertices     = 1;
    if (m_mode == Mode::VERTICES_FIXED || m_mode == Mode::VERTICES_SIGMA)
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

    if (m_seedingMode == SeedingMode::CANDIDATE_SEEDED){
      m_deltaEta_Cand = regPSet.getParameter<double>("deltaEta_Cand");
      if(m_deltaEta_Cand<0) m_deltaEta_Cand = m_deltaEta; //For backwards compatibility
      m_deltaPhi_Cand = regPSet.getParameter<double>("deltaPhi_Cand");
      if(m_deltaPhi_Cand<0) m_deltaPhi_Cand = m_deltaPhi; //For backwards compatibility
    }
    else if (m_seedingMode == SeedingMode::POINT_SEEDED){
      m_deltaEta_Point = regPSet.getParameter<double>("deltaEta_Point");
      if(m_deltaEta_Point<0) m_deltaEta_Point = m_deltaEta; //For backwards compatibility
      m_deltaPhi_Point = regPSet.getParameter<double>("deltaPhi_Point");
      if(m_deltaPhi_Point<0) m_deltaPhi_Point = m_deltaPhi; //For backwards compatibility
    }
    else if (m_seedingMode == SeedingMode::CANDIDATE_POINT_SEEDED){
      m_deltaEta_Cand = regPSet.getParameter<double>("deltaEta_Cand");
      m_deltaPhi_Cand = regPSet.getParameter<double>("deltaPhi_Cand");
      m_deltaEta_Point = regPSet.getParameter<double>("deltaEta_Point");
      m_deltaPhi_Point = regPSet.getParameter<double>("deltaPhi_Point");
      if (m_deltaEta_Cand<0 || m_deltaPhi_Cand<0 || m_deltaEta_Point<0 || m_deltaPhi_Point<0)  throw edm::Exception(edm::errors::Configuration) << "Delta eta and phi parameters must be set separately for candidates and points in candidate+point seeding mode";
    }   

    m_precise          = regPSet.getParameter<bool>("precise");
    m_whereToUseMeasurementTracker = RectangularEtaPhiTrackingRegion::stringToUseMeasurementTracker(regPSet.getParameter<std::string>("whereToUseMeasurementTracker"));
    if(m_whereToUseMeasurementTracker != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever) {
      token_measurementTracker = iC.consumes<MeasurementTrackerEvent>(regPSet.getParameter<edm::InputTag>("measurementTrackerName"));
    }
    m_searchOpt = false;
    if (regPSet.exists("searchOpt")) m_searchOpt = regPSet.getParameter<bool>("searchOpt");

    // mode-dependent z-halflength of tracking regions
    if (m_mode == Mode::VERTICES_SIGMA)  m_nSigmaZVertex   = regPSet.getParameter<double>("nSigmaZVertex");
    if (m_mode == Mode::VERTICES_FIXED)  m_zErrorVetex     = regPSet.getParameter<double>("zErrorVetex");
    m_nSigmaZBeamSpot = -1.;
    if (m_mode == Mode::BEAM_SPOT_SIGMA)
    {
      m_nSigmaZBeamSpot = regPSet.getParameter<double>("nSigmaZBeamSpot");
      if (m_nSigmaZBeamSpot < 0.)
	throw edm::Exception(edm::errors::Configuration) << "nSigmaZBeamSpot must be positive for BeamSpotSigma mode!";
    }
  }
  
  ~CandidatePointSeededTrackingRegionsProducer() override {}
    
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<std::string>("mode", "BeamSpotFixed");
    desc.add<std::string>("seedingMode", "Candidate");

    desc.add<edm::InputTag>("input", edm::InputTag(""));
    edm::ParameterSetDescription descPoints;
    descPoints.add<std::vector<double>> ("eta", {} ); 
    descPoints.add<std::vector<double>> ("phi", {} ); 
    desc.add<edm::ParameterSetDescription>("points", descPoints);

    desc.add<int>("maxNRegions", 10);
    desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
    desc.add<edm::InputTag>("vertexCollection", edm::InputTag("hltPixelVertices"));
    desc.add<int>("maxNVertices", 1);

    desc.add<double>("ptMin", 0.9);
    desc.add<double>("originRadius", 0.2);
    desc.add<double>("zErrorBeamSpot", 24.2);
    desc.add<double>("deltaEta", 0.5);
    desc.add<double>("deltaPhi", 0.5);
    desc.add<double>("deltaEta_Cand", -1.);
    desc.add<double>("deltaPhi_Cand", -1.);
    desc.add<double>("deltaEta_Point", -1.);
    desc.add<double>("deltaPhi_Point", -1.);
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

    descriptions.add("candidatePointSeededTrackingRegionsFromBeamSpot", descRegion);
  }


  std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event& e, const edm::EventSetup& es) const override
  {
    std::vector<std::unique_ptr<TrackingRegion> > result;

    // pick up the candidate objects of interest    
    edm::Handle< reco::CandidateView > objects;
    size_t n_objects = 0;

    if(m_seedingMode == SeedingMode::CANDIDATE_SEEDED || m_seedingMode == SeedingMode::CANDIDATE_POINT_SEEDED){
      e.getByToken( token_input, objects );
      n_objects = objects->size();
      if (n_objects == 0) return result;
    }

    const auto& objs = *objects;

    // always need the beam spot (as a fall back strategy for vertex modes)
    edm::Handle< reco::BeamSpot > bs;
    e.getByToken( token_beamSpot, bs );
    if( !bs.isValid() ) return result;

    // this is a default origin for all modes
    GlobalPoint default_origin( bs->x0(), bs->y0(), bs->z0() );

    // vector of origin & halfLength pairs:
    std::vector< std::pair< GlobalPoint, float > > origins;

    // fill the origins and halfLengths depending on the mode
    if (m_mode == Mode::BEAM_SPOT_FIXED || m_mode == Mode::BEAM_SPOT_SIGMA)
    {
      origins.emplace_back( default_origin,
			    (m_mode == Mode::BEAM_SPOT_FIXED) ? m_zErrorBeamSpot : m_nSigmaZBeamSpot*bs->sigmaZ()
			    );
    }
    else if (m_mode == Mode::VERTICES_FIXED || m_mode == Mode::VERTICES_SIGMA)
    {
      edm::Handle< reco::VertexCollection > vertices;
      e.getByToken( token_vertex, vertices );
      int n_vert = 0;
      for (reco::VertexCollection::const_iterator v = vertices->begin(); v != vertices->end() && n_vert < m_maxNVertices; ++v)
      {
        if ( v->isFake() || !v->isValid() ) continue;
	origins.emplace_back( GlobalPoint( v->x(), v->y(), v->z() ),
			      (m_mode == Mode::VERTICES_FIXED) ? m_zErrorVetex : m_nSigmaZVertex*v->zError()
			      );
        ++n_vert;
      }
      // no-vertex fall-back case:
      if (origins.empty())
      {
	origins.emplace_back( default_origin,
			      (m_nSigmaZBeamSpot > 0.) ? m_nSigmaZBeamSpot*bs->z0Error() : m_zErrorBeamSpot
			      );
      }
    }
    
    const MeasurementTrackerEvent *measurementTracker = nullptr;
    if(!token_measurementTracker.isUninitialized()) {
      edm::Handle<MeasurementTrackerEvent> hmte;
      e.getByToken(token_measurementTracker, hmte);
      measurementTracker = hmte.product();
    }

    // create tracking regions (maximum MaxNRegions of them) in directions of the
    // objects of interest (we expect that the collection was sorted in decreasing pt order)
    int n_regions = 0;

    if(m_seedingMode == SeedingMode::CANDIDATE_SEEDED) {

      for(size_t i = 0; i < n_objects && n_regions < m_maxNRegions; ++i ) {

	const reco::Candidate & object = objs[i];
	GlobalVector direction( object.momentum().x(), object.momentum().y(), object.momentum().z() );	

	for (size_t  j=0; j<origins.size() && n_regions < m_maxNRegions; ++j) {	

	  result.push_back(std::make_unique<RectangularEtaPhiTrackingRegion>(
		               direction,
			       origins[j].first,
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

    }
     

    else if(m_seedingMode == SeedingMode::POINT_SEEDED) {

      for(size_t i = 0; i < n_points && n_regions < m_maxNRegions; ++i ) {	

	for (size_t  j=0; j<origins.size() && n_regions < m_maxNRegions; ++j) {
	  	 
	  result.push_back( std::make_unique<RectangularEtaPhiTrackingRegion>(
			    directionPoints[i], // GlobalVector
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

    }


    else if(m_seedingMode == SeedingMode::CANDIDATE_POINT_SEEDED) {        

      for(size_t i = 0; i < n_objects && n_regions < m_maxNRegions; ++i ) {

	const reco::Candidate & object = objs[i];
	double eta_Cand = object.eta();
	double phi_Cand = object.phi();
	
	for(size_t j = 0 ; j < n_points && n_regions < m_maxNRegions; ++j) {

	  double eta_Point = etaPoints[j];
	  double phi_Point = phiPoints[j];
	  double dEta_Cand_Point = fabs(eta_Cand-eta_Point);
	  double dPhi_Cand_Point = fabs(deltaPhi(phi_Cand,phi_Point));

	  //Check if there is an overlap between Candidate- and Point-based regions of interest
	  if(dEta_Cand_Point > (m_deltaEta_Cand + m_deltaEta_Point) || dPhi_Cand_Point > (m_deltaPhi_Cand + m_deltaPhi_Point)) continue;

	  //Determines boundaries of intersection of RoIs
	  double etaMin_RoI = std::max(eta_Cand-m_deltaEta_Cand,eta_Point-m_deltaEta_Point);
	  double etaMax_RoI = std::min(eta_Cand+m_deltaEta_Cand,eta_Point+m_deltaEta_Point);

	  double phi_Cand_minus  = normalizedPhi(phi_Cand-m_deltaPhi_Cand);
	  double phi_Point_minus = normalizedPhi(phi_Point-m_deltaPhi_Point);
	  double phi_Cand_plus  = normalizedPhi(phi_Cand+m_deltaPhi_Cand);
	  double phi_Point_plus = normalizedPhi(phi_Point+m_deltaPhi_Point);

	  double phiMin_RoI = deltaPhi(phi_Cand_minus,phi_Point_minus)>0. ? phi_Cand_minus : phi_Point_minus ;	
	  double phiMax_RoI = deltaPhi(phi_Cand_plus,phi_Point_plus)<0. ? phi_Cand_plus : phi_Point_plus;

	  //Determines position and width of new RoI
	  double eta_RoI = 0.5*(etaMax_RoI+etaMin_RoI);
	  double deltaEta_RoI = etaMax_RoI - eta_RoI;

	  double phi_RoI = 0.5*(phiMax_RoI+phiMin_RoI);
	  if( phiMax_RoI < phiMin_RoI ) phi_RoI-=M_PI;
	  phi_RoI = normalizedPhi(phi_RoI);
	  double deltaPhi_RoI = deltaPhi(phiMax_RoI,phi_RoI);
	  
	  double x = std::cos(phi_RoI);
	  double y = std::sin(phi_RoI);
	  double theta = 2*std::atan(std::exp(-eta_RoI));
	  double z = 1./std::tan(theta);

	  GlobalVector direction( x,y,z );
      
	  for (size_t  k=0; k<origins.size() && n_regions < m_maxNRegions; ++k)
	    {
	      
	      result.push_back(std::make_unique<RectangularEtaPhiTrackingRegion>(
		direction,
		origins[k].first,
		m_ptMin,
		m_originRadius,
		origins[k].second,
		deltaEta_RoI,
		deltaPhi_RoI,
		m_whereToUseMeasurementTracker,
		m_precise,
		measurementTracker,
		m_searchOpt
	     ));
	    ++n_regions;

	    }

	}
	
      }

    }
    
    edm::LogInfo ("CandidatePointSeededTrackingRegionsProducer") << "produced "<<n_regions<<" regions";

    return result;
  }
  
private:

  Mode m_mode;
  SeedingMode m_seedingMode;

  int m_maxNRegions;
  edm::EDGetTokenT<reco::VertexCollection> token_vertex; 
  edm::EDGetTokenT<reco::BeamSpot> token_beamSpot; 
  edm::EDGetTokenT<reco::CandidateView> token_input; 
  int m_maxNVertices;

  size_t n_points;
  std::vector<double> etaPoints;
  std::vector<double> phiPoints;
  std::vector<GlobalVector> directionPoints;


  float m_ptMin;
  float m_originRadius;
  float m_zErrorBeamSpot;
  float m_deltaEta;
  float m_deltaPhi;
  float m_deltaEta_Cand;
  float m_deltaPhi_Cand;
  float m_deltaEta_Point;
  float m_deltaPhi_Point;
  bool m_precise;
  edm::EDGetTokenT<MeasurementTrackerEvent> token_measurementTracker;
  RectangularEtaPhiTrackingRegion::UseMeasurementTracker m_whereToUseMeasurementTracker;
  bool m_searchOpt;

  float m_nSigmaZVertex;
  float m_zErrorVetex;
  float m_nSigmaZBeamSpot;
};

#endif
