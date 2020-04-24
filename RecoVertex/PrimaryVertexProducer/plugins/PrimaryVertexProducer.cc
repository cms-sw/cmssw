#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducer.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"

PrimaryVertexProducer::PrimaryVertexProducer(const edm::ParameterSet& conf)
  :theConfig(conf)
{

  fVerbose   = conf.getUntrackedParameter<bool>("verbose", false);

  trkToken = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("TrackLabel"));
  bsToken = consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpotLabel"));
  f4D = false;

  // select and configure the track selection
  std::string trackSelectionAlgorithm=conf.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<std::string>("algorithm");
  if(trackSelectionAlgorithm=="filter"){
    theTrackFilter= new TrackFilterForPVFinding( conf.getParameter<edm::ParameterSet>("TkFilterParameters") );
  }else if (trackSelectionAlgorithm=="filterWithThreshold"){
    theTrackFilter= new HITrackFilterForPVFinding(conf.getParameter<edm::ParameterSet>("TkFilterParameters"));
  }else{
    throw VertexException("PrimaryVertexProducerAlgorithm: unknown track selection algorithm: " + trackSelectionAlgorithm);  
  }


  // select and configure the track clusterizer
  std::string clusteringAlgorithm=conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<std::string>("algorithm");
  if (clusteringAlgorithm=="gap"){
    theTrackClusterizer = new GapClusterizerInZ(conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkGapClusParameters"));
  }else if(clusteringAlgorithm=="DA"){
    theTrackClusterizer = new DAClusterizerInZ(conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters"));
  } 
  // provide the vectorized version of the clusterizer, if supported by the build
   else if(clusteringAlgorithm == "DA_vect") {
    theTrackClusterizer = new DAClusterizerInZ_vect(conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters"));
  } else if( clusteringAlgorithm=="DA2D_vect" ) {
    theTrackClusterizer = new DAClusterizerInZT_vect(conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters"));
    f4D = true;
  }

  else{
    throw VertexException("PrimaryVertexProducerAlgorithm: unknown clustering algorithm: " + clusteringAlgorithm);  
  }

  if( f4D ) {
    trkTimesToken     = consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("TrackTimesLabel") );
    trkTimeResosToken = consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("TrackTimeResosLabel") );    
  }


  // select and configure the vertex fitters
  if (conf.exists("vertexCollections")){
    std::vector<edm::ParameterSet> vertexCollections =conf.getParameter< std::vector<edm::ParameterSet> >("vertexCollections");

    for( std::vector< edm::ParameterSet >::const_iterator algoconf = vertexCollections.begin(); algoconf != vertexCollections.end(); algoconf++){
      
      algo algorithm;
      std::string fitterAlgorithm = algoconf->getParameter<std::string>("algorithm");
      if (fitterAlgorithm=="KalmanVertexFitter") {
	algorithm.fitter= new KalmanVertexFitter();
      } else if( fitterAlgorithm=="AdaptiveVertexFitter") {
	algorithm.fitter= new AdaptiveVertexFitter( GeometricAnnealing( algoconf->getParameter<double>("chi2cutoff")));
      } else {
	throw VertexException("PrimaryVertexProducerAlgorithm: unknown algorithm: " + fitterAlgorithm);  
      }
      algorithm.label = algoconf->getParameter<std::string>("label");
      algorithm.minNdof = algoconf->getParameter<double>("minNdof");
      algorithm.useBeamConstraint=algoconf->getParameter<bool>("useBeamConstraint");
      algorithm.vertexSelector=new VertexCompatibleWithBeam(VertexDistanceXY(), algoconf->getParameter<double>("maxDistanceToBeam"));
      algorithms.push_back(algorithm);
      
      produces<reco::VertexCollection>(algorithm.label);
    }
  }else{
    edm::LogWarning("MisConfiguration")<<"this module's configuration has changed, please update to have a vertexCollections=cms.VPSet parameter.";

    algo algorithm;
    std::string fitterAlgorithm = conf.getParameter<std::string>("algorithm");
    if (fitterAlgorithm=="KalmanVertexFitter") {
      algorithm.fitter= new KalmanVertexFitter();
    } else if( fitterAlgorithm=="AdaptiveVertexFitter") {
      algorithm.fitter= new AdaptiveVertexFitter();
    } else {
      throw VertexException("PrimaryVertexProducerAlgorithm: unknown algorithm: " + fitterAlgorithm);  
    }
    algorithm.label = "";
    algorithm.minNdof = conf.getParameter<double>("minNdof");
    algorithm.useBeamConstraint=conf.getParameter<bool>("useBeamConstraint");
    
    algorithm.vertexSelector=new VertexCompatibleWithBeam(VertexDistanceXY(), conf.getParameter<edm::ParameterSet>("PVSelParameters").getParameter<double>("maxDistanceToBeam"));

    algorithms.push_back(algorithm);
    produces<reco::VertexCollection>(algorithm.label);
  }
 

}


PrimaryVertexProducer::~PrimaryVertexProducer() {
  if (theTrackFilter) delete theTrackFilter;
  if (theTrackClusterizer) delete theTrackClusterizer;
  for( std::vector <algo>::const_iterator algorithm=algorithms.begin(); algorithm!=algorithms.end(); algorithm++){
    if (algorithm->fitter) delete algorithm->fitter;
    if (algorithm->vertexSelector) delete algorithm->vertexSelector;
  }
}


void
PrimaryVertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get the BeamSpot, it will alwys be needed, even when not used as a constraint
  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(bsToken,recoBeamSpotHandle);
  if (recoBeamSpotHandle.isValid()){
    beamSpot = *recoBeamSpotHandle;
  }else{
    edm::LogError("UnusableBeamSpot") << "No beam spot available from EventSetup";
  }

  bool validBS = true;
  VertexState beamVertexState(beamSpot);
  if ( (beamVertexState.error().cxx() <= 0.) || 
       (beamVertexState.error().cyy() <= 0.) ||
       (beamVertexState.error().czz() <= 0.) ) {
    validBS = false;
    edm::LogError("UnusableBeamSpot") << "Beamspot with invalid errors "<<beamVertexState.error().matrix();
  }


  // get RECO tracks from the event
  // `tks` can be used as a ptr to a reco::TrackCollection
  edm::Handle<reco::TrackCollection> tks;
  iEvent.getByToken(trkToken, tks);


  // interface RECO tracks to vertex reconstruction
  edm::ESHandle<TransientTrackBuilder> theB;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
  std::vector<reco::TransientTrack> t_tks;

  if( f4D ) {
    edm::Handle<edm::ValueMap<float> > trackTimesH;
    edm::Handle<edm::ValueMap<float> > trackTimeResosH;    
    iEvent.getByToken(trkTimesToken, trackTimesH);
    iEvent.getByToken(trkTimeResosToken, trackTimeResosH);
    t_tks = (*theB).build(tks, beamSpot, *(trackTimesH.product()), *(trackTimeResosH.product()));
  } else {
    t_tks = (*theB).build(tks, beamSpot);
  }
  if(fVerbose) {std::cout << "RecoVertex/PrimaryVertexProducer"
		     << "Found: " << t_tks.size() << " reconstructed tracks" << "\n";
  }


  // select tracks
  std::vector<reco::TransientTrack> && seltks = theTrackFilter->select( t_tks );

  // clusterize tracks in Z
  std::vector< std::vector<reco::TransientTrack> > && clusters =  theTrackClusterizer->clusterize(seltks);

  if (fVerbose){std::cout <<  " clustering returned  "<< clusters.size() << " clusters  from " << seltks.size() << " selected tracks" <<std::endl;}


  // vertex fits
  for( std::vector <algo>::const_iterator algorithm=algorithms.begin(); algorithm!=algorithms.end(); algorithm++){


    auto result = std::make_unique<reco::VertexCollection>();
    reco::VertexCollection & vColl = (*result);


    std::vector<TransientVertex> pvs;
    for (std::vector< std::vector<reco::TransientTrack> >::const_iterator iclus
	   = clusters.begin(); iclus != clusters.end(); iclus++) {
      
      double meantime = 0.;
      double expv_x2 = 0.;
      double normw = 0.;  
      if( f4D ) {
        for( const auto& tk : *iclus ) {
          const double time = tk.timeExt();
          const double inverr = 1.0/tk.dtErrorExt();
          const double w = inverr*inverr;
          meantime += time*w;
          expv_x2  += time*time*w;
          normw    += w;
        }
        meantime = meantime/normw;
        expv_x2 = expv_x2/normw;
      }
      const double time_var = ( f4D ? expv_x2 - meantime*meantime : 0. ); 


      TransientVertex v; 
      if( algorithm->useBeamConstraint && validBS &&((*iclus).size()>1) ){
        
	v = algorithm->fitter->vertex(*iclus, beamSpot);
	
        if( f4D ) {
          if( v.isValid() ) {
            auto err = v.positionError().matrix4D();
            err(3,3) = time_var/(double)iclus->size();        
            v = TransientVertex(v.position(),meantime,err,v.originalTracks(),v.totalChiSquared());
          }
        }
	
      }else if( !(algorithm->useBeamConstraint) && ((*iclus).size()>1) ) {
              
	v = algorithm->fitter->vertex(*iclus);
        
        if( f4D ) {
          if( v.isValid() ) {
            auto err = v.positionError().matrix4D();
            err(3,3) = time_var/(double)iclus->size();          
            v = TransientVertex(v.position(),meantime,err,v.originalTracks(),v.totalChiSquared());
          }
        }
	
      }// else: no fit ==> v.isValid()=False


      if (fVerbose){
	if (v.isValid()) {
          std::cout << "x,y,z";
          if (f4D) std::cout << ",t";
          std::cout << "=" << v.position().x() <<" " << v.position().y() << " " <<  v.position().z();
          if (f4D) std::cout << " " << v.time();
          std::cout  << " cluster size = " << (*iclus).size() << std::endl;
        }
	else{
	  std::cout <<"Invalid fitted vertex,  cluster size=" << (*iclus).size() << std::endl;
	}
      }

      if ( v.isValid() 
           && (v.degreesOfFreedom()>=algorithm->minNdof) 
	   && (!validBS || (*(algorithm->vertexSelector))(v,beamVertexState))
           ) pvs.push_back(v);
    }// end of cluster loop

    if(fVerbose){
      std::cout << "PrimaryVertexProducerAlgorithm::vertices  candidates =" << pvs.size() << std::endl;
    }


    if (clusters.size()>2 && clusters.size() > 2*pvs.size()) 
      edm::LogWarning("PrimaryVertexProducer") << "more than half of candidate vertices lost " << pvs.size()  << ' ' << clusters.size();

    if (pvs.empty() && seltks.size()>5) 
       edm::LogWarning("PrimaryVertexProducer") << "no vertex found with " << seltks.size() << " tracks and " << clusters.size() <<" vertex-candidates";    

    // sort vertices by pt**2  vertex (aka signal vertex tagging)
    if(pvs.size()>1){
      sort(pvs.begin(), pvs.end(), VertexHigherPtSquared());
    }



    // convert transient vertices returned by the theAlgo to (reco) vertices
    for (std::vector<TransientVertex>::const_iterator iv = pvs.begin();
	 iv != pvs.end(); iv++) {
      reco::Vertex v = *iv;
      vColl.push_back(v);
    }

    if (vColl.empty()) {
      GlobalError bse(beamSpot.rotatedCovariance3D());
      if ( (bse.cxx() <= 0.) || 
	   (bse.cyy() <= 0.) ||
	   (bse.czz() <= 0.) ) {
	AlgebraicSymMatrix33 we;
	we(0,0)=10000; we(1,1)=10000; we(2,2)=10000;
	vColl.push_back(reco::Vertex(beamSpot.position(), we,0.,0.,0));
	if(fVerbose){
	  std::cout <<"RecoVertex/PrimaryVertexProducer: "
		    << "Beamspot with invalid errors "<<bse.matrix()<<std::endl;
	  std::cout << "Will put Vertex derived from dummy-fake BeamSpot into Event.\n";
	}
      } else {
	vColl.push_back(reco::Vertex(beamSpot.position(), 
				     beamSpot.rotatedCovariance3D(),0.,0.,0));
	if(fVerbose){
	  std::cout <<"RecoVertex/PrimaryVertexProducer: "
		    << " will put Vertex derived from BeamSpot into Event.\n";
	}
      }
    }

    if(fVerbose){
      int ivtx=0;
      for(reco::VertexCollection::const_iterator v=vColl.begin(); 
	  v!=vColl.end(); ++v){
	std::cout << "recvtx "<< ivtx++ 
		  << "#trk " << std::setw(3) << v->tracksSize()
		  << " chi2 " << std::setw(4) << v->chi2() 
		  << " ndof " << std::setw(3) << v->ndof() 
		  << " x "  << std::setw(6) << v->position().x() 
		  << " dx " << std::setw(6) << v->xError()
		  << " y "  << std::setw(6) << v->position().y() 
		  << " dy " << std::setw(6) << v->yError()
		  << " z "  << std::setw(6) << v->position().z() 
		  << " dz " << std::setw(6) << v->zError();
        if( f4D ) {
          std::cout << " t " << std::setw(6) << v->t()
                    << " dt " << std::setw(6) << v->tError();
        }
        std::cout << std::endl;
      }
    }

    iEvent.put(std::move(result), algorithm->label); 
  }
  
}


//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexProducer);
