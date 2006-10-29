#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

using namespace edm;
using namespace reco;
using namespace std;

double TrackAssociatorByChi2::compareTracksParam ( TrackCollection::const_iterator rt, 
						   SimTrackContainer::const_iterator st, 
						   const HepLorentzVector vertexPosition, 
						   GlobalVector magField,
						   TrackBase::CovarianceMatrix  
						   invertedCovariance  ) {
  
  double thetares = st->momentum().theta();
  double phi0res = st->momentum().phi();
  double d0res = vertexPosition.perp();
  double dzres = vertexPosition.z();    
  double kres= -1 * rt->charge() * 2.99792458e-3 * magField.mag() / st->momentum().perp();
  
  TrackBase::ParameterVector sParameters;
  sParameters[0] = kres;
  sParameters[1] = thetares;
  sParameters[2] = phi0res;
  sParameters[3] = d0res;
  sParameters[4] = dzres;
  TrackBase::ParameterVector rParameters = rt->parameters();
  
  TrackBase::ParameterVector diffParameters = rParameters - sParameters;
  double chi2 = ROOT::Math::Dot(diffParameters * invertedCovariance, diffParameters);
//   std::cout << "FROM SIM TRACK - chi2: " << chi2 << std::endl;
  
  return chi2;
}


TrackAssociatorByChi2::RecoToSimPairAssociation 
TrackAssociatorByChi2::compareTracksParam(const TrackCollection& rtColl,
					  const SimTrackContainer& stColl,
					  const SimVertexContainer& svColl) {
  
  RecoToSimPairAssociation outputVec;

  for (TrackCollection::const_iterator track=rtColl.begin(); track!=rtColl.end(); track++){
     Chi2SimMap outMap;

    TrackBase::ParameterVector rParameters = track->parameters();
    TrackBase::CovarianceMatrix recoTrackCovMatrix = track->covariance();
    recoTrackCovMatrix.Invert();
    
    for (SimTrackContainer::const_iterator st=stColl.begin(); st!=stColl.end(); st++){
      const HepLorentzVector vertexPos = svColl[st->vertIndex()].position(); 
      GlobalVector magField = theMF->inTesla( GlobalPoint(vertexPos.x(),vertexPos.y(),vertexPos.z()));

      double thetares = st->momentum().theta();
      double phi0res = st->momentum().phi();
      double d0res = vertexPos.perp();
      double dzres = vertexPos.z();    
      double kres= -1 * track->charge() * 2.99792458e-3 * magField.mag() / st->momentum().perp();
      
      TrackBase::ParameterVector sParameters;
      sParameters[0] = kres;
      sParameters[1] = thetares;
      sParameters[2] = phi0res;
      sParameters[3] = d0res;
      sParameters[4] = dzres;
      
      TrackBase::ParameterVector diffParameters = rParameters - sParameters;
      double chi2 = ROOT::Math::Dot(diffParameters * recoTrackCovMatrix, diffParameters);
//       std::cout << "FROM SIM TRACK - chi2: " << chi2 << std::endl;

      chi2/=5;
      if (chi2<50) outMap[chi2]=*st;
    }
    outputVec.push_back(RecoToSimPair(*track,outMap));
  }
  return outputVec;
}


RecoToSimCollection TrackAssociatorByChi2::associateRecoToSim(edm::Handle<reco::TrackCollection>& tCH, 
							      edm::Handle<TrackingParticleCollection>& tPCH){

  RecoToSimCollection  outputCollection;
  double chi2;

  const TrackCollection tC = *(tCH.product());
  const TrackingParticleCollection tPC= *(tPCH.product());

  int tindex=0;
  for (TrackCollection::const_iterator rt=tC.begin(); rt!=tC.end(); rt++, tindex++){
 
    TrackBase::ParameterVector rParameters = rt->parameters();
    TrackBase::CovarianceMatrix recoTrackCovMatrix = rt->covariance();
    recoTrackCovMatrix.Invert();

//     cout << "Reco Track : " << rt->x() << " " << rt->y() << " " << rt->z() << " " << endl;

    int tpindex =0;
    for (TrackingParticleCollection::const_iterator tp=tPC.begin(); tp!=tPC.end(); tp++, ++tpindex){
      for (TrackingParticle::g4t_iterator t=tp->g4Track_begin(); t!=tp->g4Track_end(); ++t) {
	
// 	std::cout << "tindex=" << tindex << " - tpindex=" << tpindex << std::endl;
	
	double thetares = (*t)->momentum().theta();
	double phi0res = (*t)->momentum().phi();
	GlobalPoint vert;
	const TrackingVertex * tv = &(*(tp->parentVertex()));
	int vind=0;
	for (TrackingVertex::g4v_iterator v=tv->g4Vertices_begin(); v!=tv->g4Vertices_end(); v++){
// 	  cout << "Sim Track " << (*v)->position().x() << " " << (*v)->position().y() << " " <<(*v)->position().z() << " " << endl;
	  if (vind==(*t)->vertIndex()) vert=GlobalPoint((*v)->position().x(),(*v)->position().y(),(*v)->position().z());
	  vind++;
	}
	double d0res = vert.perp();
	double dzres = vert.z();
	GlobalVector magField = theMF->inTesla( vert);
	//should use *t->charge when implemented
	double kres= -1 * rt->charge() * 2.99792458e-3 * magField.mag() / (*t)->momentum().perp();
	
	TrackBase::ParameterVector sParameters;
	sParameters[0] = kres;
	sParameters[1] = thetares;
	sParameters[2] = phi0res;
	sParameters[3] = d0res;
	sParameters[4] = dzres;
	
	TrackBase::ParameterVector diffParameters = rParameters - sParameters;
	chi2 = ROOT::Math::Dot(diffParameters * recoTrackCovMatrix, diffParameters);
// 	std::cout << "FROM TRACKING PARTICLE - chi2: " << chi2 << std::endl;
	chi2 /= 5;

	if (chi2<50) {
	  outputCollection.insert(reco::TrackRef(tCH,tindex), 
				  edm::Ref<TrackingParticleCollection>(tPCH, tpindex));
	}
      }
    }
    
  }
  return outputCollection;

}

