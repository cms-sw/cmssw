#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/GeometrySurface/interface/Line.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

using namespace edm;
using namespace reco;
using namespace std;

double TrackAssociatorByChi2::compareTracksParam ( TrackCollection::const_iterator rt, 
						   SimTrackContainer::const_iterator st, 
						   const math::XYZTLorentzVectorD vertexPosition, 
						   GlobalVector magField,
						   TrackBase::CovarianceMatrix  
						   invertedCovariance  ) const{
  
  Basic3DVector<double> momAtVtx(st->momentum().x(),st->momentum().y(),st->momentum().z());
  Basic3DVector<double> vert = (Basic3DVector<double>) vertexPosition;

  std::pair<bool,reco::TrackBase::ParameterVector> params = parametersAtClosestApproach(vert, momAtVtx, st->charge());
  if (params.first){
    TrackBase::ParameterVector sParameters =params.second;
    TrackBase::ParameterVector rParameters = rt->parameters();
    
    TrackBase::ParameterVector diffParameters = rParameters - sParameters;
    double chi2 = ROOT::Math::Dot(diffParameters * invertedCovariance, diffParameters);
    
    return chi2;
  } else {
    return 10000000000.;
  }
}


TrackAssociatorByChi2::RecoToSimPairAssociation 
TrackAssociatorByChi2::compareTracksParam(const TrackCollection& rtColl,
					  const SimTrackContainer& stColl,
					  const SimVertexContainer& svColl) const{
  
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

      std::pair<bool,reco::TrackBase::ParameterVector> params = parametersAtClosestApproach(vert, momAtVtx, st->charge());
      if (params.first){
	TrackBase::ParameterVector sParameters = params.second;
      
	TrackBase::ParameterVector diffParameters = rParameters - sParameters;
	double chi2 = ROOT::Math::Dot(diffParameters * recoTrackCovMatrix, diffParameters);
	chi2/=5;
	if (chi2<chi2cut) outMap[chi2]=*st;
      }
    }
    outputVec.push_back(RecoToSimPair(*track,outMap));
  }
  return outputVec;
}


RecoToSimCollection TrackAssociatorByChi2::associateRecoToSim(edm::Handle<reco::TrackCollection>& tCH, 
							      edm::Handle<TrackingParticleCollection>& tPCH,
							      const edm::Event * e ) const{

  RecoToSimCollection  outputCollection;
  double chi2;

  const TrackCollection tC = *(tCH.product());
  const TrackingParticleCollection tPC= *(tPCH.product());

  int tindex=0;
  for (TrackCollection::const_iterator rt=tC.begin(); rt!=tC.end(); rt++, tindex++){

    LogDebug("TrackAssociator") << "=========LOOKING FOR ASSOCIATION===========" << "\n"
				<< "rec::Track #"<<tindex<<" with pt=" << rt->pt() <<  "\n"
				<< "===========================================" << "\n";
 
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

    int tpindex =0;
    for (TrackingParticleCollection::const_iterator tp=tPC.begin(); tp!=tPC.end(); tp++, ++tpindex){
	
      //skip tps with a very small pt
      if (sqrt(tp->momentum().perp2())<0.5) continue;
	
      Basic3DVector<double> momAtVtx(tp->momentum().x(),tp->momentum().y(),tp->momentum().z());
      Basic3DVector<double> vert=(Basic3DVector<double>) tp->vertex();

      std::pair<bool,reco::TrackBase::ParameterVector> params = parametersAtClosestApproach(vert, momAtVtx, tp->charge());
      if (params.first){
	TrackBase::ParameterVector sParameters=params.second;
	
	TrackBase::ParameterVector diffParameters = rParameters - sParameters;
	
	chi2 = ROOT::Math::Similarity(diffParameters, recoTrackCovMatrix);
	chi2 /= 5;
	
	LogDebug("TrackAssociator") << "====RECO TRACK WITH PT=" << rt->pt() << "====\n" 
				    << "qoverp simG: " << sParameters[0] << "\n" 
				    << "lambda simG: " << sParameters[1] << "\n" 
				    << "phi    simG: " << sParameters[2] << "\n" 
				    << "dxy    simG: " << sParameters[3] << "\n" 
				    << "dsz    simG: " << sParameters[4] << "\n" 
				    << ": " /*<< */ << "\n" 
				    << "qoverp rec: " << rt->qoverp()/*rParameters[0]*/ << "\n" 
				    << "lambda rec: " << rt->lambda()/*rParameters[1]*/ << "\n" 
				    << "phi    rec: " << rt->phi()/*rParameters[2]*/ << "\n" 
				    << "dxy    rec: " << rt->dxy()/*rParameters[3]*/ << "\n" 
				    << "dsz    rec: " << rt->dsz()/*rParameters[4]*/ << "\n" 
				    << ": " /*<< */ << "\n" 
				    << "chi2: " << chi2 << "\n";
	
	if (chi2<chi2cut) {
	  outputCollection.insert(reco::TrackRef(tCH,tindex), 
				  std::make_pair(edm::Ref<TrackingParticleCollection>(tPCH, tpindex),
						 -chi2));//-chi2 because the Association Map is ordered using std::greater
	}
      }
    }
  }
  outputCollection.post_insert();
  return outputCollection;
}



SimToRecoCollection TrackAssociatorByChi2::associateSimToReco(edm::Handle<reco::TrackCollection>& tCH, 
							      edm::Handle<TrackingParticleCollection>& tPCH,
							      const edm::Event * e ) const {
  SimToRecoCollection  outputCollection;
  double chi2;

  const TrackCollection tC = *(tCH.product());
  const TrackingParticleCollection tPC= *(tPCH.product());

  int tpindex =0;
  for (TrackingParticleCollection::const_iterator tp=tPC.begin(); tp!=tPC.end(); tp++, ++tpindex){
    
    //skip tps with a very small pt
    if (sqrt(tp->momentum().perp2())<0.5) continue;
    
    LogDebug("TrackAssociator") << "=========LOOKING FOR ASSOCIATION===========" << "\n"
				<< "TrackingParticle #"<<tpindex<<" with pt=" << sqrt(tp->momentum().perp2()) << "\n"
				<< "===========================================" << "\n";
    
    Basic3DVector<double> momAtVtx(tp->momentum().x(),tp->momentum().y(),tp->momentum().z());
    Basic3DVector<double> vert(tp->vertex().x(),tp->vertex().y(),tp->vertex().z());
    
    std::pair<bool,reco::TrackBase::ParameterVector> params = parametersAtClosestApproach(vert, momAtVtx, tp->charge());
    if (params.first){
      TrackBase::ParameterVector sParameters=params.second;
      
      int tindex=0;
      for (TrackCollection::const_iterator rt=tC.begin(); rt!=tC.end(); rt++, tindex++){
	
	TrackBase::ParameterVector rParameters = rt->parameters();
	TrackBase::CovarianceMatrix recoTrackCovMatrix = rt->covariance();
	if (onlyDiagonal) {
	  for (unsigned int i=0;i<5;i++){
	    for (unsigned int j=0;j<5;j++){
	      if (i!=j) recoTrackCovMatrix(i,j)=0;
	    }
	  }
	}
	
	recoTrackCovMatrix.Invert();
	
	TrackBase::ParameterVector diffParameters = rParameters - sParameters;
	
	chi2 = ROOT::Math::Similarity(recoTrackCovMatrix, diffParameters);
	chi2 /= 5;
	
	LogDebug("TrackAssociator") << "====RECO TRACK WITH PT=" << rt->pt() << "====\n" 
				    << "qoverp simG: " << sParameters[0] << "\n" 
				    << "lambda simG: " << sParameters[1] << "\n" 
				    << "phi    simG: " << sParameters[2] << "\n" 
				    << "dxy    simG: " << sParameters[3] << "\n" 
				    << "dsz    simG: " << sParameters[4] << "\n" 
				    << ": " /*<< */ << "\n" 
				    << "qoverp rec: " << rt->qoverp()/*rParameters[0]*/ << "\n" 
				    << "lambda rec: " << rt->lambda()/*rParameters[1]*/ << "\n" 
				    << "phi    rec: " << rt->phi()/*rParameters[2]*/ << "\n" 
				    << "dxy    rec: " << rt->dxy()/*rParameters[3]*/ << "\n" 
				    << "dsz    rec: " << rt->dsz()/*rParameters[4]*/ << "\n" 
				    << ": " /*<< */ << "\n" 
				    << "chi2: " << chi2 << "\n";
	
	if (chi2<chi2cut) {
	  outputCollection.insert(edm::Ref<TrackingParticleCollection>(tPCH, tpindex),
				  std::make_pair(reco::TrackRef(tCH,tindex),
						 -chi2));//-chi2 because the Association Map is ordered using std::greater
	}
      }
    }
  }
  outputCollection.post_insert();
  return outputCollection;
}

double TrackAssociatorByChi2::associateRecoToSim( TrackCollection::const_iterator rt, 
						  TrackingParticleCollection::const_iterator tp ) const{
  
  double chi2;
  
  TrackBase::ParameterVector rParameters = rt->parameters();
  TrackBase::CovarianceMatrix recoTrackCovMatrix = rt->covariance();
      if (onlyDiagonal) {
	for (unsigned int i=0;i<5;i++){
	  for (unsigned int j=0;j<5;j++){
	    if (i!=j) recoTrackCovMatrix(i,j)=0;
	  }
	}
      }
  recoTrackCovMatrix.Invert();
  
  Basic3DVector<double> momAtVtx(tp->momentum().x(),tp->momentum().y(),tp->momentum().z());
  Basic3DVector<double> vert(tp->vertex().x(),tp->vertex().y(),tp->vertex().z());

  std::pair<bool,reco::TrackBase::ParameterVector> params = parametersAtClosestApproach(vert, momAtVtx, tp->charge());
  if (params.first){
    TrackBase::ParameterVector sParameters=params.second;
    
    TrackBase::ParameterVector diffParameters = rParameters - sParameters;
    chi2 = ROOT::Math::Dot(diffParameters * recoTrackCovMatrix, diffParameters);
    chi2 /= 5;
    
    LogDebug("TrackAssociator") << "====NEW RECO TRACK WITH PT=" << rt->pt() << "====\n" 
				<< "qoverp simG: " << sParameters[0] << "\n" 
				<< "lambda simG: " << sParameters[1] << "\n" 
				<< "phi    simG: " << sParameters[2] << "\n" 
				<< "dxy    simG: " << sParameters[3] << "\n" 
				<< "dsz    simG: " << sParameters[4] << "\n" 
				<< ": " /*<< */ << "\n" 
				<< "qoverp rec: " << rt->qoverp()/*rParameters[0]*/ << "\n" 
				<< "lambda rec: " << rt->lambda()/*rParameters[1]*/ << "\n" 
				<< "phi    rec: " << rt->phi()/*rParameters[2]*/ << "\n" 
				<< "dxy    rec: " << rt->dxy()/*rParameters[3]*/ << "\n" 
				<< "dsz    rec: " << rt->dsz()/*rParameters[4]*/ << "\n" 
				<< ": " /*<< */ << "\n" 
				<< "chi2: " << chi2 << "\n";
    
    return chi2;  
  } else {
    return 10000000000.;
  }
}

pair<bool,TrackBase::ParameterVector> 
TrackAssociatorByChi2::parametersAtClosestApproach(Basic3DVector<double> vertex,
						   Basic3DVector<double> momAtVtx,
						   float charge) const{
  
  TrackBase::ParameterVector sParameters;
  try {
    FreeTrajectoryState ftsAtProduction(GlobalPoint(vertex.x(),vertex.y(),vertex.z()),
					GlobalVector(momAtVtx.x(),momAtVtx.y(),momAtVtx.z()),
					TrackCharge(charge),
					theMF.product());
    TSCPBuilderNoMaterial tscpBuilder;
    TrajectoryStateClosestToPoint tsAtClosestApproach 
      = tscpBuilder(ftsAtProduction,GlobalPoint(0,0,0));//as in TrackProducerAlgorithm
    GlobalPoint v = tsAtClosestApproach.theState().position();
    GlobalVector p = tsAtClosestApproach.theState().momentum();
    
    sParameters[0] = tsAtClosestApproach.charge()/p.mag();
    sParameters[1] = Geom::halfPi() - p.theta();
    sParameters[2] = p.phi();
    sParameters[3] = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
    sParameters[4] = v.z()*p.perp()/p.mag() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.mag();
    
    return make_pair<bool,TrackBase::ParameterVector>(true,sParameters);
  } catch ( ... ) {
    return make_pair<bool,TrackBase::ParameterVector>(false,sParameters);
  }
}

