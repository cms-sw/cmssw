#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "RecoVertex/AdaptiveVertexFinder/interface/TracksClusteringFromDisplacedSeed.h"

#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexSmoother.h"
#include "RecoVertex/MultiVertexFit/interface/MultiVertexFitter.h"

//#define VTXDEBUG 1

class InclusiveVertexFinder : public edm::stream::EDProducer<> {
    public:
	InclusiveVertexFinder(const edm::ParameterSet &params);

	virtual void produce(edm::Event &event, const edm::EventSetup &es) override;

    private:
	bool trackFilter(const reco::TrackRef &track) const;
        std::pair<std::vector<reco::TransientTrack>,GlobalPoint> nearTracks(const reco::TransientTrack &seed, const std::vector<reco::TransientTrack> & tracks, const reco::Vertex & primaryVertex) const;

	edm::EDGetTokenT<reco::BeamSpot> 	token_beamSpot; 
	edm::EDGetTokenT<reco::VertexCollection> token_primaryVertex;
	edm::EDGetTokenT<reco::TrackCollection>	token_tracks; 
	unsigned int				minHits;
	unsigned int				maxNTracks;
	double					maxLIP;
        double 					minPt;
        double 					vertexMinAngleCosine;
        double 					vertexMinDLen2DSig;
        double 					vertexMinDLenSig;
	double					fitterSigmacut;
	double  				fitterTini;
	double 					fitterRatio;
	bool 					useVertexFitter;
	bool 					useVertexReco;
	std::auto_ptr<VertexReconstructor>	vtxReco;
	std::auto_ptr<TracksClusteringFromDisplacedSeed>	clusterizer;

};

InclusiveVertexFinder::InclusiveVertexFinder(const edm::ParameterSet &params) :
	minHits(params.getParameter<unsigned int>("minHits")),
	maxNTracks(params.getParameter<unsigned int>("maxNTracks")),
       	maxLIP(params.getParameter<double>("maximumLongitudinalImpactParameter")),
 	minPt(params.getParameter<double>("minPt")), //0.8
        vertexMinAngleCosine(params.getParameter<double>("vertexMinAngleCosine")), //0.98
        vertexMinDLen2DSig(params.getParameter<double>("vertexMinDLen2DSig")), //2.5
        vertexMinDLenSig(params.getParameter<double>("vertexMinDLenSig")), //0.5
        fitterSigmacut(params.getParameter<double>("fitterSigmacut")),
        fitterTini(params.getParameter<double>("fitterTini")),
        fitterRatio(params.getParameter<double>("fitterRatio")),
	useVertexFitter(params.getParameter<bool>("useDirectVertexFitter")),
	useVertexReco(params.getParameter<bool>("useVertexReco")),
	vtxReco(new ConfigurableVertexReconstructor(params.getParameter<edm::ParameterSet>("vertexReco"))),
        clusterizer(new TracksClusteringFromDisplacedSeed(params.getParameter<edm::ParameterSet>("clusterizer")))

{
	token_beamSpot = consumes<reco::BeamSpot>(params.getParameter<edm::InputTag>("beamSpot"));
	token_primaryVertex = consumes<reco::VertexCollection>(params.getParameter<edm::InputTag>("primaryVertices"));
	token_tracks = consumes<reco::TrackCollection>(params.getParameter<edm::InputTag>("tracks"));
	produces<reco::VertexCollection>();
	//produces<reco::VertexCollection>("multi");
}

bool InclusiveVertexFinder::trackFilter(const reco::TrackRef &track) const
{
	if (track->hitPattern().numberOfValidHits() < (int)minHits)
//	if (track->hitPattern().trackerLayersWithMeasurement() < (int)minHits)
		return false;
	if (track->pt() < minPt )
		return false;
 
	return true;
}

void InclusiveVertexFinder::produce(edm::Event &event, const edm::EventSetup &es)
{
	using namespace reco;

  VertexDistance3D vdist;
  VertexDistanceXY vdist2d;
  MultiVertexFitter theMultiVertexFitter;
  AdaptiveVertexFitter theAdaptiveFitter(
                                            GeometricAnnealing(fitterSigmacut, fitterTini, fitterRatio),
                                            DefaultLinearizationPointFinder(),
                                            KalmanVertexUpdator<5>(),
                                            KalmanVertexTrackCompatibilityEstimator<5>(),
                                            KalmanVertexSmoother() );


	edm::Handle<BeamSpot> beamSpot;
	event.getByToken(token_beamSpot,beamSpot);

	edm::Handle<VertexCollection> primaryVertices;
	event.getByToken(token_primaryVertex, primaryVertices);

	edm::Handle<TrackCollection> tracks;
	event.getByToken(token_tracks, tracks);

	edm::ESHandle<TransientTrackBuilder> trackBuilder;
	es.get<TransientTrackRecord>().get("TransientTrackBuilder",
	                                   trackBuilder);


        std::auto_ptr<VertexCollection> recoVertices(new VertexCollection);
        if(primaryVertices->size()!=0) {
     
	const reco::Vertex &pv = (*primaryVertices)[0];
        
	std::vector<TransientTrack> tts;
        //Fill transient track vector 
	for(TrackCollection::const_iterator track = tracks->begin();
	    track != tracks->end(); ++track) {
		TrackRef ref(tracks, track - tracks->begin());
		if (!trackFilter(ref))
			continue;
                if( std::abs(ref->dz(pv.position())) > maxLIP)
			continue;
		TransientTrack tt = trackBuilder->build(ref);
		tt.setBeamSpot(*beamSpot);
		tts.push_back(tt);
	}
        std::vector<TracksClusteringFromDisplacedSeed::Cluster> clusters = clusterizer->clusters(pv,tts);

        //Create BS object from PV to feed in the AVR
	BeamSpot::CovarianceMatrix cov;
	for(unsigned int i = 0; i < 7; i++) {
		for(unsigned int j = 0; j < 7; j++) {
			if (i < 3 && j < 3)
				cov(i, j) = pv.covariance(i, j);
			else
				cov(i, j) = 0.0;
		}
	}
	BeamSpot bs(pv.position(), 0.0, 0.0, 0.0, 0.0, cov, BeamSpot::Unknown);


        int i=0;
#ifdef VTXDEBUG

	std::cout <<  "CLUSTERS " << clusters.size() << std::endl; 
#endif

	for(std::vector<TracksClusteringFromDisplacedSeed::Cluster>::iterator cluster = clusters.begin();
	    cluster != clusters.end(); ++cluster,++i)
        {
                if(cluster->tracks.size() == 0 || cluster->tracks.size() > maxNTracks ) 
		     continue;
        
 	        cluster->tracks.push_back(cluster->seedingTrack); //add the seed to the list of tracks to fit
	 	std::vector<TransientVertex> vertices;
		if(useVertexReco) {
			vertices = vtxReco->vertices(cluster->tracks, bs);  // attempt with config given reconstructor
		}
                TransientVertex singleFitVertex;
		if(useVertexFitter) {
			singleFitVertex = theAdaptiveFitter.vertex(cluster->tracks,cluster->seedPoint); //attempt with direct fitting
			if(singleFitVertex.isValid())
				vertices.push_back(singleFitVertex);
		}
		for(std::vector<TransientVertex>::const_iterator v = vertices.begin();
		    v != vertices.end(); ++v) {
//			if(v->degreesOfFreedom() > 0.2)
                        {
                         Measurement1D dlen= vdist.distance(pv,*v);
                         Measurement1D dlen2= vdist2d.distance(pv,*v);
			 reco::Vertex vv(*v);
#ifdef VTXDEBUG
                       std::cout << "V chi2/n: " << v->normalisedChiSquared() << " ndof: " <<v->degreesOfFreedom() ;
                         std::cout << " dlen: " << dlen.value() << " error: " << dlen.error() << " signif: " << dlen.significance();
                         std::cout << " dlen2: " << dlen2.value() << " error2: " << dlen2.error() << " signif2: " << dlen2.significance();
                         std::cout << " pos: " << vv.position() << " error: " <<vv.xError() << " " << vv.yError() << " " << vv.zError() << std::endl;
#endif
                         GlobalVector dir;  
			 std::vector<reco::TransientTrack> ts = v->originalTracks();
                        for(std::vector<reco::TransientTrack>::const_iterator i = ts.begin();
                            i != ts.end(); ++i) {
                                reco::TrackRef t = i->trackBaseRef().castTo<reco::TrackRef>();
                                float w = v->trackWeight(*i);
                                if (w > 0.5) dir+=i->impactPointState().globalDirection();
#ifdef VTXDEBUG
                                std::cout << "\t[" << (*t).pt() << ": "
                                          << (*t).eta() << ", "
                                          << (*t).phi() << "], "
                                          << w << std::endl;
#endif
                        }
		       GlobalPoint ppv(pv.position().x(),pv.position().y(),pv.position().z());
		       GlobalPoint sv((*v).position().x(),(*v).position().y(),(*v).position().z());
                       float vscal = dir.unit().dot((sv-ppv).unit()) ;
//                        std::cout << "Vscal: " <<  vscal << std::endl;
                       if(dlen.significance() > vertexMinDLenSig  && vscal > vertexMinAngleCosine &&  v->normalisedChiSquared() < 10 && dlen2.significance() > vertexMinDLen2DSig)
	            	  {	 
				recoVertices->push_back(*v);
#ifdef VTXDEBUG

	                        std::cout << "ADDED" << std::endl;
#endif

                         }
                      }
                   }
        }
#ifdef VTXDEBUG

        std::cout <<  "Final put  " << recoVertices->size() << std::endl;
#endif  
        }
 
	event.put(recoVertices);

}

DEFINE_FWK_MODULE(InclusiveVertexFinder);
