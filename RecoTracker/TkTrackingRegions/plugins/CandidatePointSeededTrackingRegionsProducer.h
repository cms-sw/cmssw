#ifndef CandidatePointSeededTrackingRegionsProducer_h
#define CandidatePointSeededTrackingRegionsProducer_h


#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

#include "VertexBeamspotOrigins.h"

/** class CandidatePointSeededTrackingRegionsProducer
 *
 * eta-phi TrackingRegions producer in directions defined by Candidate-based objects of interest
 * from a collection defined by the "input" parameter.
 *
 * Four operational modes are supported ("operationMode" parameter):
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

  enum class SeedingMode {CANDIDATE_SEEDED, POINT_SEEDED, CANDIDATE_POINT_SEEDED};

  explicit CandidatePointSeededTrackingRegionsProducer(const edm::ParameterSet& conf, edm::ConsumesCollector && iC):
    m_origins(conf.getParameter<edm::ParameterSet>("RegionPSet"), iC)
  {
    edm::ParameterSet regPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");

    // seeding mode
    std::string seedingModeString = regPSet.getParameter<std::string>("seedingMode");
    if      (seedingModeString == "Candidate")      m_seedingMode = SeedingMode::CANDIDATE_SEEDED;
    else if (seedingModeString == "Point")          m_seedingMode = SeedingMode::POINT_SEEDED;
    else if (seedingModeString == "CandidatePoint") m_seedingMode = SeedingMode::CANDIDATE_POINT_SEEDED;
    else throw edm::Exception(edm::errors::Configuration) << "Unknown seeding mode string: "<<seedingModeString;

    // basic inputs
    if(m_seedingMode == SeedingMode::CANDIDATE_SEEDED || m_seedingMode == SeedingMode::CANDIDATE_POINT_SEEDED)
      m_token_input        = iC.consumes<reco::CandidateView>(regPSet.getParameter<edm::InputTag>("input"));

    // Specific points in the detector
    if(m_seedingMode == SeedingMode::POINT_SEEDED || m_seedingMode == SeedingMode::CANDIDATE_POINT_SEEDED){
      edm::ParameterSet points = regPSet.getParameter<edm::ParameterSet>("points");
      std::vector<double> etaPoints = points.getParameter<std::vector<double>>("eta");
      std::vector<double> phiPoints = points.getParameter<std::vector<double>>("phi");

      if (!(etaPoints.size() == phiPoints.size()))  throw edm::Exception(edm::errors::Configuration) << "The parameters 'eta' and 'phi' must have the same size";
      if (etaPoints.empty()) throw edm::Exception(edm::errors::Configuration) << "At least one point should be defined for point or candidate+point seeding modes";

      for(size_t i = 0; i < etaPoints.size(); ++i ){

	m_etaPhiPoints.push_back(std::make_pair(etaPoints[i],phiPoints[i]));

      	double x = std::cos(phiPoints[i]);
	double y = std::sin(phiPoints[i]);
	double theta = 2*std::atan(std::exp(-etaPoints[i]));
	double z = 1./std::tan(theta);
	GlobalVector direction( x,y,z );
	m_directionPoints.push_back(direction);

      }

    }

    m_maxNRegions      = regPSet.getParameter<unsigned int>("maxNRegions");
    if(m_maxNRegions==0) throw edm::Exception(edm::errors::Configuration) << "maxNRegions should be greater than or equal to 1";

    // RectangularEtaPhiTrackingRegion parameters:
    m_ptMin            = regPSet.getParameter<double>("ptMin");
    m_originRadius     = regPSet.getParameter<double>("originRadius");

    if (m_seedingMode == SeedingMode::CANDIDATE_SEEDED){
      m_deltaEta_Cand = regPSet.getParameter<double>("deltaEta_Cand"); 
      m_deltaPhi_Cand = regPSet.getParameter<double>("deltaPhi_Cand");
      if (m_deltaEta_Cand<0 || m_deltaPhi_Cand<0)  throw edm::Exception(edm::errors::Configuration) << "Delta eta and phi parameters must be set for candidates in candidate seeding mode";
    }
    else if (m_seedingMode == SeedingMode::POINT_SEEDED){
      m_deltaEta_Point = regPSet.getParameter<double>("deltaEta_Point");
      m_deltaPhi_Point = regPSet.getParameter<double>("deltaPhi_Point");
      if (m_deltaEta_Point<0 || m_deltaPhi_Point<0)  throw edm::Exception(edm::errors::Configuration) << "Delta eta and phi parameters must be set for points in point seeding mode";
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
      m_token_measurementTracker = iC.consumes<MeasurementTrackerEvent>(regPSet.getParameter<edm::InputTag>("measurementTrackerName"));
    }
    m_searchOpt = false;
    if (regPSet.exists("searchOpt")) m_searchOpt = regPSet.getParameter<bool>("searchOpt");
  }
  
  ~CandidatePointSeededTrackingRegionsProducer() override {}
    
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<std::string>("seedingMode", "Candidate");

    desc.add<edm::InputTag>("input", edm::InputTag(""));
    edm::ParameterSetDescription descPoints;
    descPoints.add<std::vector<double>> ("eta", {} ); 
    descPoints.add<std::vector<double>> ("phi", {} ); 
    desc.add<edm::ParameterSetDescription>("points", descPoints);

    desc.add<unsigned int>("maxNRegions", 10);

    VertexBeamspotOrigins::fillDescriptions(desc, "hltOnlineBeamSpot", "hltPixelVertices", 1);

    desc.add<double>("ptMin", 0.9);
    desc.add<double>("originRadius", 0.2);
    desc.add<double>("deltaEta_Cand", -1.);
    desc.add<double>("deltaPhi_Cand", -1.);
    desc.add<double>("deltaEta_Point", -1.);
    desc.add<double>("deltaPhi_Point", -1.);
    desc.add<bool>("precise", true);

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
      e.getByToken( m_token_input, objects );
      n_objects = objects->size();
      if (n_objects == 0) return result;
    }

    const auto& objs = *objects;

    const auto& origins = m_origins.origins(e);
    if(origins.empty()) {
      return result;
    }

    const MeasurementTrackerEvent *measurementTracker = nullptr;
    if(!m_token_measurementTracker.isUninitialized()) {
      edm::Handle<MeasurementTrackerEvent> hmte;
      e.getByToken(m_token_measurementTracker, hmte);
      measurementTracker = hmte.product();
    }

    // create tracking regions (maximum MaxNRegions of them) in directions of the
    // objects of interest (we expect that the collection was sorted in decreasing pt order)
    int n_regions = 0;

    if(m_seedingMode == SeedingMode::CANDIDATE_SEEDED) {

      for(const auto& object : objs) {

	GlobalVector direction( object.momentum().x(), object.momentum().y(), object.momentum().z() );	

	for(const auto& origin : origins) {

	  result.push_back(std::make_unique<RectangularEtaPhiTrackingRegion>(
		               direction,			       
			       origin.first,
			       m_ptMin,
			       m_originRadius,
			       origin.second,
			       m_deltaEta_Cand,
			       m_deltaPhi_Cand,
			       m_whereToUseMeasurementTracker,
			       m_precise,
			       measurementTracker,
			       m_searchOpt
			   ));
	  ++n_regions;	  
	  if( n_regions >= m_maxNRegions ) break;

	}

	if( n_regions >= m_maxNRegions ) break;

      }

    }
     

    else if(m_seedingMode == SeedingMode::POINT_SEEDED) {

      for( const auto& direction : m_directionPoints ){

	for(const auto& origin : origins) {
	  	 
	  result.push_back( std::make_unique<RectangularEtaPhiTrackingRegion>(
			    direction, // GlobalVector
			    origin.first, // GlobalPoint
			    m_ptMin,
			    m_originRadius,
			    origin.second,
			    m_deltaEta_Point,
			    m_deltaPhi_Point,
			    m_whereToUseMeasurementTracker,
			    m_precise,
			    measurementTracker,
			    m_searchOpt
			   ));
	  ++n_regions;
	  if( n_regions >= m_maxNRegions ) break;

	}

	if( n_regions >= m_maxNRegions ) break;

      }

    }


    else if(m_seedingMode == SeedingMode::CANDIDATE_POINT_SEEDED) {        

      for(const auto& object : objs) {

	double eta_Cand = object.eta();
	double phi_Cand = object.phi();
	
	for (const auto& etaPhiPoint : m_etaPhiPoints ){

	  double eta_Point = etaPhiPoint.first;
	  double phi_Point = etaPhiPoint.second;
	  double dEta_Cand_Point = std::abs(eta_Cand-eta_Point);
	  double dPhi_Cand_Point = std::abs(deltaPhi(phi_Cand,phi_Point));

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
      
	  for(const auto& origin : origins) {
	      
	    result.push_back(std::make_unique<RectangularEtaPhiTrackingRegion>(
		direction,
		origin.first,
		m_ptMin,
		m_originRadius,
		origin.second,
		deltaEta_RoI,
		deltaPhi_RoI,
		m_whereToUseMeasurementTracker,
		m_precise,
		measurementTracker,
		m_searchOpt
	        ));	        
	    ++n_regions;
	    if( n_regions >= m_maxNRegions ) break;

	  }

	  if( n_regions >= m_maxNRegions ) break;

	}

	if( n_regions >= m_maxNRegions ) break;
	
      }

    }
    
    edm::LogInfo ("CandidatePointSeededTrackingRegionsProducer") << "produced "<<n_regions<<" regions";

    return result;
  }
  
private:

  VertexBeamspotOrigins m_origins;
  SeedingMode m_seedingMode;

  int m_maxNRegions;
  edm::EDGetTokenT<reco::CandidateView> m_token_input; 

  std::vector<std::pair<double,double> > m_etaPhiPoints;
  std::vector<GlobalVector> m_directionPoints;

  float m_ptMin;
  float m_originRadius;
  float m_deltaEta_Cand;
  float m_deltaPhi_Cand;
  float m_deltaEta_Point;
  float m_deltaPhi_Point;
  bool m_precise;
  edm::EDGetTokenT<MeasurementTrackerEvent> m_token_measurementTracker;
  RectangularEtaPhiTrackingRegion::UseMeasurementTracker m_whereToUseMeasurementTracker;
  bool m_searchOpt;
};

#endif
