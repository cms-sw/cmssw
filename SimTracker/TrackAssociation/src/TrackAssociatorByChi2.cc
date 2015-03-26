#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/GeometrySurface/interface/Line.h"
#include "TrackingTools/PatternTools/interface/trackingParametersAtClosestApproachToBeamSpot.h"
#include "getChi2.h"

using namespace edm;
using namespace reco;
using namespace std;

double TrackAssociatorByChi2::compareTracksParam ( TrackCollection::const_iterator rt, 
						   SimTrackContainer::const_iterator st, 
						   const math::XYZTLorentzVectorD& vertexPosition, 
						   const GlobalVector& magField,
						   const TrackBase::CovarianceMatrix& invertedCovariance,
						   const reco::BeamSpot& bs) const{
  
  Basic3DVector<double> momAtVtx(st->momentum().x(),st->momentum().y(),st->momentum().z());
  Basic3DVector<double> vert = (Basic3DVector<double>) vertexPosition;

  std::pair<bool,reco::TrackBase::ParameterVector> params = parametersAtClosestApproach(vert, momAtVtx, st->charge(), bs);
  if (params.first){
    TrackBase::ParameterVector sParameters = params.second;
    TrackBase::ParameterVector rParameters = rt->parameters();

    TrackBase::ParameterVector diffParameters = rParameters - sParameters;
    diffParameters[2] = reco::deltaPhi(diffParameters[2],0.f);
    double chi2 = ROOT::Math::Dot(diffParameters * invertedCovariance, diffParameters);
    
    return chi2;
  } else {
    return 10000000000.;
  }
}


TrackAssociatorByChi2::RecoToSimPairAssociation 
TrackAssociatorByChi2::compareTracksParam(const TrackCollection& rtColl,
					  const SimTrackContainer& stColl,
					  const SimVertexContainer& svColl,
					  const reco::BeamSpot& bs) const{
  
  RecoToSimPairAssociation outputVec;

  for (TrackCollection::const_iterator track=rtColl.begin(); track!=rtColl.end(); track++){
     Chi2SimMap outMap;

    TrackBase::ParameterVector rParameters = track->parameters();

    TrackBase::CovarianceMatrix recoTrackCovMatrix = track->covariance();
    if (onlyDiagonal){
      for (unsigned int i=0;i<5;i++){
	for (unsigned int j=0;j<5;j++){
	  if (i!=j) recoTrackCovMatrix(i,j)=0;
	}
      }
    }
    recoTrackCovMatrix.Invert();

    for (SimTrackContainer::const_iterator st=stColl.begin(); st!=stColl.end(); st++){

      Basic3DVector<double> momAtVtx(st->momentum().x(),st->momentum().y(),st->momentum().z());
      Basic3DVector<double> vert = (Basic3DVector<double>)  svColl[st->vertIndex()].position();

      std::pair<bool,reco::TrackBase::ParameterVector> params = parametersAtClosestApproach(vert, momAtVtx, st->charge(), bs);
      if (params.first){
	TrackBase::ParameterVector sParameters = params.second;
      
	TrackBase::ParameterVector diffParameters = rParameters - sParameters;
        diffParameters[2] = reco::deltaPhi(diffParameters[2],0.f);
	double chi2 = ROOT::Math::Dot(diffParameters * recoTrackCovMatrix, diffParameters);
	chi2/=5;
	if (chi2<chi2cut) outMap[chi2]=*st;
      }
    }
    outputVec.push_back(RecoToSimPair(*track,outMap));
  }
  return outputVec;
}

double TrackAssociatorByChi2::getChi2(const TrackBase::ParameterVector& rParameters,
				      const TrackBase::CovarianceMatrix& recoTrackCovMatrix,
				      const Basic3DVector<double>& momAtVtx,
				      const Basic3DVector<double>& vert,
				      int charge,
				      const reco::BeamSpot& bs) const{
  return track_associator::getChi2(rParameters, recoTrackCovMatrix,momAtVtx, vert, charge, *theMF, bs);
}


double TrackAssociatorByChi2::associateRecoToSim( TrackCollection::const_iterator rt, 
						  TrackingParticleCollection::const_iterator tp, 
						  const reco::BeamSpot& bs) const{  
  TrackBase::ParameterVector rParameters = rt->parameters();
  TrackBase::CovarianceMatrix recoTrackCovMatrix = rt->covariance();
  if (onlyDiagonal){
    for (unsigned int i=0;i<5;i++){
      for (unsigned int j=0;j<5;j++){
	if (i!=j) recoTrackCovMatrix(i,j)=0;
      }
    }
  } 
  
  recoTrackCovMatrix.Invert();
  Basic3DVector<double> momAtVtx(tp->momentum().x(),tp->momentum().y(),tp->momentum().z());
  Basic3DVector<double> vert(tp->vertex().x(),tp->vertex().y(),tp->vertex().z());
  int charge = tp->charge();
  return getChi2(rParameters,recoTrackCovMatrix,momAtVtx,vert,charge,bs);
}

pair<bool,TrackBase::ParameterVector> 
TrackAssociatorByChi2::parametersAtClosestApproach(const Basic3DVector<double>& vertex,
						   const Basic3DVector<double>& momAtVtx,
						   float charge,
						   const BeamSpot& bs) const{
  return reco::trackingParametersAtClosestApproachToBeamSpot(vertex,momAtVtx,charge, *theMF, bs);
}

RecoToSimCollection TrackAssociatorByChi2::associateRecoToSim(const edm::RefToBaseVector<reco::Track>& tC, 
							      const edm::RefVector<TrackingParticleCollection>& tPCH,
							      const edm::Event * e,
                                                              const edm::EventSetup *setup ) const{
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e->getByLabel(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot bs = *recoBeamSpotHandle;      

  RecoToSimCollection  outputCollection;

  //dereference the edm::Ref only once
  std::vector<const TrackingParticle*> tPC;
  tPC.reserve(tPCH.size());
  for(auto const& ref: tPCH) {
    tPC.push_back(&(*ref));
  }

  int tindex=0;
  for (RefToBaseVector<reco::Track>::const_iterator rt=tC.begin(); rt!=tC.end(); rt++, tindex++){

    LogDebug("TrackAssociator") << "=========LOOKING FOR ASSOCIATION===========" << "\n"
				<< "rec::Track #"<<tindex<<" with pt=" << (*rt)->pt() <<  "\n"
				<< "===========================================" << "\n";
 
    TrackBase::ParameterVector rParameters = (*rt)->parameters();

    TrackBase::CovarianceMatrix recoTrackCovMatrix = (*rt)->covariance();
    if (onlyDiagonal){
      for (unsigned int i=0;i<5;i++){
	for (unsigned int j=0;j<5;j++){
	  if (i!=j) recoTrackCovMatrix(i,j)=0;
	}
      }
    } 

    recoTrackCovMatrix.Invert();

    int tpindex =0;
    for (auto tp=tPC.begin(); tp!=tPC.end(); tp++, ++tpindex){
	
      //skip tps with a very small pt
      //if (sqrt((*tp)->momentum().perp2())<0.5) continue;
      int charge = (*tp)->charge();
      if (charge==0) continue;
      Basic3DVector<double> momAtVtx((*tp)->momentum().x(),(*tp)->momentum().y(),(*tp)->momentum().z());
      Basic3DVector<double> vert=(Basic3DVector<double>) (*tp)->vertex();

      double chi2 = getChi2(rParameters,recoTrackCovMatrix,momAtVtx,vert,charge,bs);
      
      if (chi2<chi2cut) {
        //NOTE: tPCH and tPC have the same index for the same object
	outputCollection.insert(tC[tindex], 
				std::make_pair(tPCH[tpindex],
					       -chi2));//-chi2 because the Association Map is ordered using std::greater
      }
    }
  }
  outputCollection.post_insert();
  return outputCollection;
}


SimToRecoCollection TrackAssociatorByChi2::associateSimToReco(const edm::RefToBaseVector<reco::Track>& tC, 
							      const edm::RefVector<TrackingParticleCollection>& tPCH,
							      const edm::Event * e,
                                                              const edm::EventSetup *setup ) const {
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e->getByLabel(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot bs = *recoBeamSpotHandle;      

  SimToRecoCollection  outputCollection;

  //dereference the edm::Ref only once
  std::vector<const TrackingParticle*> tPC;
  tPC.reserve(tPCH.size());
  for(auto const& ref: tPCH) {
    tPC.push_back(&(*ref));
  }

  int tpindex =0;
  for (auto tp=tPC.begin(); tp!=tPC.end(); tp++, ++tpindex){
    
    //skip tps with a very small pt
    //if (sqrt((*tp)->momentum().perp2())<0.5) continue;
    int charge = (*tp)->charge();
    if (charge==0) continue;
    
    LogDebug("TrackAssociator") << "=========LOOKING FOR ASSOCIATION===========" << "\n"
				<< "TrackingParticle #"<<tpindex<<" with pt=" << sqrt((*tp)->momentum().perp2()) << "\n"
				<< "===========================================" << "\n";
    
    Basic3DVector<double> momAtVtx((*tp)->momentum().x(),(*tp)->momentum().y(),(*tp)->momentum().z());
    Basic3DVector<double> vert((*tp)->vertex().x(),(*tp)->vertex().y(),(*tp)->vertex().z());
      
    int tindex=0;
    for (RefToBaseVector<reco::Track>::const_iterator rt=tC.begin(); rt!=tC.end(); rt++, tindex++){
      
      TrackBase::ParameterVector rParameters = (*rt)->parameters();      
      TrackBase::CovarianceMatrix recoTrackCovMatrix = (*rt)->covariance();
      if (onlyDiagonal) {
	for (unsigned int i=0;i<5;i++){
	  for (unsigned int j=0;j<5;j++){
	    if (i!=j) recoTrackCovMatrix(i,j)=0;
	  }
	}
      }
      recoTrackCovMatrix.Invert();
      
      double chi2 = getChi2(rParameters,recoTrackCovMatrix,momAtVtx,vert,charge,bs);
      
      if (chi2<chi2cut) {
        //NOTE: tPCH and tPC have the same index for the same object
	outputCollection.insert(tPCH[tpindex],
				std::make_pair(tC[tindex],
					       -chi2));//-chi2 because the Association Map is ordered using std::greater
      }
    }
  }
  outputCollection.post_insert();
  return outputCollection;
}




