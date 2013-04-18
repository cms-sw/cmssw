#include <memory>
#include <set>


#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexSmoother.h"
#include "DataFormats/Math/interface/deltaR.h"


#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"



class TrackVertexArbitrator : public edm::EDProducer {
    public:
	TrackVertexArbitrator(const edm::ParameterSet &params);


	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	bool trackFilter(const reco::TrackRef &track) const;

	edm::InputTag				primaryVertexCollection;
	edm::InputTag				secondaryVertexCollection;
	edm::InputTag				trackCollection;
        edm::InputTag                           beamSpotCollection;
        double 					dRCut;
        double					distCut;
        double					sigCut;
	double					dLenFraction;
};

TrackVertexArbitrator::TrackVertexArbitrator(const edm::ParameterSet &params) :
	primaryVertexCollection(params.getParameter<edm::InputTag>("primaryVertices")),
	secondaryVertexCollection(params.getParameter<edm::InputTag>("secondaryVertices")),
	trackCollection(params.getParameter<edm::InputTag>("tracks")),
        beamSpotCollection(params.getParameter<edm::InputTag>("beamSpot")),
	dRCut(params.getParameter<double>("dRCut")),
	distCut(params.getParameter<double>("distCut")),
	sigCut(params.getParameter<double>("sigCut")),
	dLenFraction(params.getParameter<double>("dLenFraction"))
{
	produces<reco::VertexCollection>();
}

bool TrackVertexArbitrator::trackFilter(const reco::TrackRef &track) const
{
        if (track->hitPattern().trackerLayersWithMeasurement() < 4)
                return false;
        if (track->pt() < 0.4 )
                return false;

        return true;
}


void TrackVertexArbitrator::produce(edm::Event &event, const edm::EventSetup &es)
{
	using namespace reco;

	edm::Handle<VertexCollection> secondaryVertices;
	event.getByLabel(secondaryVertexCollection, secondaryVertices);

        edm::Handle<VertexCollection> primaryVertices;
        event.getByLabel(primaryVertexCollection, primaryVertices);

        edm::Handle<TrackCollection> tracks;
        event.getByLabel(trackCollection, tracks);

        edm::ESHandle<TransientTrackBuilder> trackBuilder;
        es.get<TransientTrackRecord>().get("TransientTrackBuilder",
                                           trackBuilder);

        edm::Handle<BeamSpot> beamSpot;
        event.getByLabel(beamSpotCollection, beamSpot);

        const reco::Vertex &pv = (*primaryVertices)[0];
//        std::cout << "PV: " << pv.position() << std::endl;
        VertexDistance3D dist;

  double sigmacut = 3.0;
  double Tini = 256.;
  double ratio = 0.25;

  AdaptiveVertexFitter theAdaptiveFitter(
                                            GeometricAnnealing(sigmacut, Tini, ratio),
                                            DefaultLinearizationPointFinder(),
                                            KalmanVertexUpdator<5>(),
                                            KalmanVertexTrackCompatibilityEstimator<5>(),
                                            KalmanVertexSmoother() );



	std::auto_ptr<VertexCollection> recoVertices(new VertexCollection);

  VertexDistance3D vdist;

for(std::vector<reco::Vertex>::const_iterator sv = secondaryVertices->begin();
	    sv != secondaryVertices->end(); ++sv) {
/*          recoVertices->push_back(*sv);
        

       for(std::vector<reco::Vertex>::iterator sv = recoVertices->begin();
	    sv != recoVertices->end(); ++sv) {
*/
	    GlobalPoint ppv(pv.position().x(),pv.position().y(),pv.position().z());
	    GlobalPoint ssv(sv->position().x(),sv->position().y(),sv->position().z());
            GlobalVector flightDir = ssv-ppv;
//            std::cout << "Vertex : " << sv-secondaryVertices->begin() << " " << sv->position() << std::endl;
            Measurement1D dlen= vdist.distance(pv,*sv);
            std::vector<reco::TransientTrack>  selTracks;

        for(TrackCollection::const_iterator track = tracks->begin();
            track != tracks->end(); ++track) {

                TrackRef ref(tracks, track - tracks->begin());
	        if (!trackFilter(ref))                         continue;

                TransientTrack tt = trackBuilder->build(ref);
                tt.setBeamSpot(*beamSpot);
	        float w = sv->trackWeight(ref);
                std::pair<bool,Measurement1D> ipv = IPTools::absoluteImpactParameter3D(tt,pv);
                std::pair<bool,Measurement1D> isv = IPTools::absoluteImpactParameter3D(tt,*sv);
                if( w > 0 || ( isv.second.significance() < sigCut && isv.second.value() < distCut && isv.second.value() < dlen.value()*dLenFraction ) )
                {
		  float dR = deltaR(flightDir.eta(), flightDir.phi(), tt.track().eta(), tt.track().phi());

                  if(isv.second.value() < ipv.second.value() && isv.second.value() < distCut && isv.second.value() < dlen.value()*dLenFraction 
                  && dR < dRCut ) 
                  {
//                     if(w > 0.5) std::cout << " = ";
  //                   else std::cout << " + ";
                     selTracks.push_back(tt);
                  } else
                  {
//                     if(w > 0.5 && isv.second.value() > ipv.second.value() ) std::cout << " - ";
  //                   else std::cout << "   ";
                     //add also the tracks used in previous fitting that are still closer to Sv than Pv 
                     if(w > 0.5 && isv.second.value() < ipv.second.value() && dR < dRCut) selTracks.push_back(tt);
                  }

    //              std::cout << "t : " << track-tracks->begin() <<  " w: " << w 
      //            << " svip: " << isv.second.significance() << " " << isv.second.value()  
        //          << " pvip: " << ipv.second.significance() << " " << ipv.second.value()  << " dr: "   << dR << std::endl;
 
                }
           }      

           if(selTracks.size() >= 2)
              { 
             	 TransientVertex singleFitVertex;
             	 singleFitVertex = theAdaptiveFitter.vertex(selTracks,ssv);
              	if(singleFitVertex.isValid())  recoVertices->push_back(singleFitVertex);
              } 
	}
	event.put(recoVertices);
}

DEFINE_FWK_MODULE(TrackVertexArbitrator);
