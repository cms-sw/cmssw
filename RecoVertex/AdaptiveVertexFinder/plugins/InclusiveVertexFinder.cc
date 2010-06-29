#include <memory>

#include "FWCore/Framework/interface/EDProducer.h"
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
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexSmoother.h"
#include "RecoVertex/MultiVertexFit/interface/MultiVertexFitter.h"

class InclusiveVertexFinder : public edm::EDProducer {
    public:
	InclusiveVertexFinder(const edm::ParameterSet &params);

	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	bool trackFilter(const reco::TrackRef &track) const;
        std::pair<std::vector<reco::TransientTrack>,GlobalPoint> nearTracks(const reco::TransientTrack &seed, const std::vector<reco::TransientTrack> & tracks, const reco::Vertex & primaryVertex) const;

	edm::InputTag				beamSpotCollection;
	edm::InputTag				primaryVertexCollection;
	edm::InputTag				trackCollection;
	unsigned int				minHits;
        double 					minPt;
        double 					min3DIPSignificance;
        double 					min3DIPValue;
        double 					clusterMaxDistance;
        double 					clusterMaxSignificance;
        double 					clusterScale;
        double 					clusterMinAngleCosine;
        double 					vertexMinAngleCosine;
        double 					vertexMinDLen2DSig;
        double 					vertexMinDLenSig;

	std::auto_ptr<VertexReconstructor>	vtxReco;

};

InclusiveVertexFinder::InclusiveVertexFinder(const edm::ParameterSet &params) :
	beamSpotCollection(params.getParameter<edm::InputTag>("beamSpot")),
	primaryVertexCollection(params.getParameter<edm::InputTag>("primaryVertices")),
	trackCollection(params.getParameter<edm::InputTag>("tracks")),
	minHits(params.getParameter<unsigned int>("minHits")),
        minPt(params.getParameter<double>("minPt")), //0.8
	min3DIPSignificance(params.getParameter<double>("seedMin3DIPSignificance")),
	min3DIPValue(params.getParameter<double>("seedMin3DIPValue")),
	clusterMaxDistance(params.getParameter<double>("clusterMaxDistance")),
        clusterMaxSignificance(params.getParameter<double>("clusterMaxSignificance")), //3
        clusterScale(params.getParameter<double>("clusterScale")),//10.
        clusterMinAngleCosine(params.getParameter<double>("clusterMinAngleCosine")), //0.0
        vertexMinAngleCosine(params.getParameter<double>("vertexMinAngleCosine")), //0.98
        vertexMinDLen2DSig(params.getParameter<double>("vertexMinDLen2DSig")), //2.5
        vertexMinDLenSig(params.getParameter<double>("vertexMinDLenSig")), //0.5
	vtxReco(new ConfigurableVertexReconstructor(params.getParameter<edm::ParameterSet>("vertexReco")))

{
	produces<reco::VertexCollection>();
	//produces<reco::VertexCollection>("multi");
}

bool InclusiveVertexFinder::trackFilter(const reco::TrackRef &track) const
{
	if (track->hitPattern().trackerLayersWithMeasurement() < (int)minHits)
		return false;
	if (track->pt() < minPt )
		return false;
        
	return true;
}

std::pair<std::vector<reco::TransientTrack>,GlobalPoint> InclusiveVertexFinder::nearTracks(const reco::TransientTrack &seed, const std::vector<reco::TransientTrack> & tracks, const  reco::Vertex & primaryVertex) const
{
      VertexDistance3D distanceComputer;
      GlobalPoint pv(primaryVertex.position().x(),primaryVertex.position().y(),primaryVertex.position().z());
      std::vector<reco::TransientTrack> result;
      TwoTrackMinimumDistance dist;
      GlobalPoint seedingPoint;
      float sumWeights=0;
      std::pair<bool,Measurement1D> ipSeed = IPTools::absoluteImpactParameter3D(seed,primaryVertex);
      float pvDistance = ipSeed.second.value();

      for(std::vector<reco::TransientTrack>::const_iterator tt = tracks.begin();tt!=tracks.end(); ++tt )   {

       if(*tt==seed) continue;

       std::pair<bool,Measurement1D> ip = IPTools::absoluteImpactParameter3D(*tt,primaryVertex);
       if(dist.calculate(tt->impactPointState(),seed.impactPointState()))
            {
		 GlobalPoint ttPoint          = dist.points().first;
		 GlobalError ttPointErr       = tt->impactPointState().cartesianError().position();
	         GlobalPoint seedPosition     = dist.points().second;
	         GlobalError seedPositionErr  = seed.impactPointState().cartesianError().position();
                 Measurement1D m = distanceComputer.distance(VertexState(seedPosition,seedPositionErr), VertexState(ttPoint, ttPointErr));
                 GlobalPoint cp(dist.crossingPoint()); 


                 float distanceFromPV =  (dist.points().second-pv).mag();
                 float distance = dist.distance();
		 GlobalVector trackDir2D(tt->impactPointState().globalDirection().x(),tt->impactPointState().globalDirection().y(),0.); 
		 GlobalVector seedDir2D(seed.impactPointState().globalDirection().x(),seed.impactPointState().globalDirection().y(),0.); 
                 float dotprodTrackSeed2D = trackDir2D.unit().dot(seedDir2D.unit());

                 float dotprodTrack = (dist.points().first-pv).unit().dot(tt->impactPointState().globalDirection().unit());
                 float dotprodSeed = (dist.points().second-pv).unit().dot(seed.impactPointState().globalDirection().unit());

                 float w = distanceFromPV*distanceFromPV/(pvDistance*distance);

          	 if(m.significance() < clusterMaxSignificance && 
                    dotprodSeed > clusterMinAngleCosine && //Angles between PV-PCAonSeed vectors and seed directions
                    dotprodTrack > clusterMinAngleCosine && //Angles between PV-PCAonTrack vectors and track directions
                    dotprodTrackSeed2D > clusterMinAngleCosine && //Angle between track and seed
                    distance*clusterScale < distanceFromPV*distanceFromPV/pvDistance && // cut scaling with track density
                    distance < clusterMaxDistance)  // absolute distance cut
                 {
                     result.push_back(*tt);
                     seedingPoint = GlobalPoint(cp.x()*w+seedingPoint.x(),cp.y()*w+seedingPoint.y(),cp.z()*w+seedingPoint.z());  
                     sumWeights+=w; 
                 }
            }
       }

seedingPoint = GlobalPoint(seedingPoint.x()/sumWeights,seedingPoint.y()/sumWeights,seedingPoint.z()/sumWeights);
return std::pair<std::vector<reco::TransientTrack>,GlobalPoint>(result,seedingPoint);

}


void InclusiveVertexFinder::produce(edm::Event &event, const edm::EventSetup &es)
{
	using namespace reco;

  double sigmacut = 3.0;
  double Tini = 256.;
  double ratio = 0.25;
  VertexDistance3D vdist;
  VertexDistanceXY vdist2d;
  MultiVertexFitter theMultiVertexFitter;
  AdaptiveVertexFitter theAdaptiveFitter(
                                            GeometricAnnealing(sigmacut, Tini, ratio),
                                            DefaultLinearizationPointFinder(),
                                            KalmanVertexUpdator<5>(),
                                            KalmanVertexTrackCompatibilityEstimator<5>(),
                                            KalmanVertexSmoother() );


	edm::Handle<BeamSpot> beamSpot;
	event.getByLabel(beamSpotCollection, beamSpot);

	edm::Handle<VertexCollection> primaryVertices;
	event.getByLabel(primaryVertexCollection, primaryVertices);

	edm::Handle<TrackCollection> tracks;
	event.getByLabel(trackCollection, tracks);

	edm::ESHandle<TransientTrackBuilder> trackBuilder;
	es.get<TransientTrackRecord>().get("TransientTrackBuilder",
	                                   trackBuilder);


	const reco::Vertex &pv = (*primaryVertices)[0];
        
	std::vector<TransientTrack> tts;
	std::vector<TransientTrack> seeds;
        //Fill transient track vector and find seeds
	for(TrackCollection::const_iterator track = tracks->begin();
	    track != tracks->end(); ++track) {
		TrackRef ref(tracks, track - tracks->begin());
		if (!trackFilter(ref))
			continue;

		TransientTrack tt = trackBuilder->build(ref);
		tt.setBeamSpot(*beamSpot);
		tts.push_back(tt);
                std::pair<bool,Measurement1D> ip = IPTools::absoluteImpactParameter3D(tt,pv);
//                std::cout << "track: " << ip.second.value() << " " << ip.second.significance() << " " << track->hitPattern().trackerLayersWithMeasurement() << " " << track->pt() << " " << track->eta() << std::endl;
                if(ip.first && ip.second.value() >= min3DIPValue && ip.second.significance() >= min3DIPSignificance)
                  { 
  //                  std::cout << "new seed " << ip.second.value() << " " << ip.second.significance() << " " << track->hitPattern().trackerLayersWithMeasurement() << " " << track->pt() << " " << track->eta() << std::endl;
                    seeds.push_back(tt);  
                  }
 
	}


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
	std::auto_ptr<VertexCollection> recoVertices(new VertexCollection);
        int i=0;
        std::vector< std::vector<TransientTrack> > clusters;
	for(std::vector<TransientTrack>::const_iterator s = seeds.begin();
	    s != seeds.end(); ++s,++i)
        {
                
//		std::cout << "Match pvd = "<< pvd[i] <<   std::endl;
        	std::pair<std::vector<reco::TransientTrack>,GlobalPoint>  ntracks = nearTracks(*s,tts,pv);
                if(ntracks.first.size() == 0 ) continue;
                ntracks.first.push_back(*s);
                clusters.push_back(ntracks.first); 
	 	std::vector<TransientVertex> vertices;
//		try {
			vertices = vtxReco->vertices(ntracks.first, bs);
                        TransientVertex singleFitVertex;
                        singleFitVertex = theAdaptiveFitter.vertex(ntracks.first,ntracks.second); //edPoint);
                        if(singleFitVertex.isValid())
                          vertices.push_back(singleFitVertex);
//		} catch(...) {
//			vertices.clear();
//		}

//		std::cout << "for this seed I found  " << vertices.size() << " vertices"<< std::endl;
		for(std::vector<TransientVertex>::const_iterator v = vertices.begin();
		    v != vertices.end(); ++v) {
//			if(v->degreesOfFreedom() > 0.2)
                        {
                         Measurement1D dlen= vdist.distance(pv,*v);
                         Measurement1D dlen2= vdist2d.distance(pv,*v);
			 reco::Vertex vv(*v);
  //                       std::cout << "V chi2/n: " << v->normalisedChiSquared() << " ndof: " <<v->degreesOfFreedom() ;
    //                     std::cout << " dlen: " << dlen.value() << " error: " << dlen.error() << " signif: " << dlen.significance();
      //                   std::cout << " dlen2: " << dlen2.value() << " error2: " << dlen2.error() << " signif2: " << dlen2.significance();
        //                 std::cout << " pos: " << vv.position() << " error: " <<vv.xError() << " " << vv.yError() << " " << vv.zError() << std::endl;
                         GlobalVector dir;  
			 std::vector<reco::TransientTrack> ts = v->originalTracks();
                        for(std::vector<reco::TransientTrack>::const_iterator i = ts.begin();
                            i != ts.end(); ++i) {
                                reco::TrackRef t = i->trackBaseRef().castTo<reco::TrackRef>();
                                float w = v->trackWeight(*i);
                                if (w > 0.5) dir+=i->impactPointState().globalDirection();
          //                      std::cout << "\t[" << (*t).pt() << ": "
            //                              << (*t).eta() << ", "
              //                            << (*t).phi() << "], "
                //                          << w << std::endl;
                        }
		       GlobalPoint ppv(pv.position().x(),pv.position().y(),pv.position().z());
		       GlobalPoint sv((*v).position().x(),(*v).position().y(),(*v).position().z());
                       float vscal = dir.unit().dot((sv-ppv).unit()) ;
//                        std::cout << "Vscal: " <<  vscal << std::endl;
                        if(dlen.significance() > vertexMinDLenSig  && vscal > vertexMinAngleCosine &&  v->normalisedChiSquared() < 10 && dlen2.significance() > vertexMinDLen2DSig)
       			 recoVertices->push_back(*v);


                        }
                   }
        }
	event.put(recoVertices);

}

DEFINE_FWK_MODULE(InclusiveVertexFinder);
