#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/GeometrySurface/interface/Line.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace edm;
using namespace reco;
using namespace std;

double TrackAssociatorByChi2::compareTracksParam ( TrackCollection::const_iterator rt, 
						   SimTrackContainer::const_iterator st, 
						   const math::XYZTLorentzVectorD vertexPosition, 
						   GlobalVector magField,
						   TrackBase::CovarianceMatrix invertedCovariance,
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

double TrackAssociatorByChi2::getChi2(TrackBase::ParameterVector& rParameters,
				      TrackBase::CovarianceMatrix& recoTrackCovMatrix,
				      Basic3DVector<double>& momAtVtx,
				      Basic3DVector<double>& vert,
				      int& charge,
				      const reco::BeamSpot& bs) const{
  
  double chi2;
  
  std::pair<bool,reco::TrackBase::ParameterVector> params = parametersAtClosestApproach(vert, momAtVtx, charge, bs);
  if (params.first){
    TrackBase::ParameterVector sParameters=params.second;
    
    TrackBase::ParameterVector diffParameters = rParameters - sParameters;
    diffParameters[2] = reco::deltaPhi(diffParameters[2],0.f);
    chi2 = ROOT::Math::Dot(diffParameters * recoTrackCovMatrix, diffParameters);
    chi2 /= 5;
    
    LogDebug("TrackAssociator") << "====NEW RECO TRACK WITH PT=" << sin(rParameters[1])*float(charge)/rParameters[0] << "====\n" 
				<< "qoverp sim: " << sParameters[0] << "\n" 
				<< "lambda sim: " << sParameters[1] << "\n" 
				<< "phi    sim: " << sParameters[2] << "\n" 
				<< "dxy    sim: " << sParameters[3] << "\n" 
				<< "dsz    sim: " << sParameters[4] << "\n" 
				<< ": " /*<< */ << "\n" 
				<< "qoverp rec: " << rParameters[0] << "\n" 
				<< "lambda rec: " << rParameters[1] << "\n" 
				<< "phi    rec: " << rParameters[2] << "\n" 
				<< "dxy    rec: " << rParameters[3] << "\n" 
				<< "dsz    rec: " << rParameters[4] << "\n" 
				<< ": " /*<< */ << "\n" 
				<< "chi2: " << chi2 << "\n";
    
    return chi2;  
  } else {
    return 10000000000.;
  }
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
TrackAssociatorByChi2::parametersAtClosestApproach(Basic3DVector<double> vertex,
						   Basic3DVector<double> momAtVtx,
						   float charge,
						   const BeamSpot& bs) const{
  
  TrackBase::ParameterVector sParameters;
  try {
    FreeTrajectoryState ftsAtProduction(GlobalPoint(vertex.x(),vertex.y(),vertex.z()),
					GlobalVector(momAtVtx.x(),momAtVtx.y(),momAtVtx.z()),
					TrackCharge(charge),
					theMF.product());
    TSCBLBuilderNoMaterial tscblBuilder;
    TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,bs);//as in TrackProducerAlgorithm
    
    GlobalPoint v = tsAtClosestApproach.trackStateAtPCA().position();
    GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();
    sParameters[0] = tsAtClosestApproach.trackStateAtPCA().charge()/p.mag();
    sParameters[1] = Geom::halfPi() - p.theta();
    sParameters[2] = p.phi();
    sParameters[3] = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
    sParameters[4] = v.z()*p.perp()/p.mag() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.mag();
    
    return pair<bool,TrackBase::ParameterVector>(true,sParameters);
  } catch ( ... ) {
    return pair<bool,TrackBase::ParameterVector>(false,sParameters);
  }
}

RecoToSimCollection TrackAssociatorByChi2::associateRecoToSim(const edm::RefToBaseVector<reco::Track>& tC, 
							      const edm::RefVector<TrackingParticleCollection>& tPCH,
							      const edm::Event * e,
                                                              const edm::EventSetup *setup ) const{
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e->getByLabel(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot bs = *recoBeamSpotHandle;      

  RecoToSimCollection  outputCollection;

  TrackingParticleCollection tPC;
  if (tPCH.size()!=0)  tPC = *tPCH.product();

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
    for (TrackingParticleCollection::const_iterator tp=tPC.begin(); tp!=tPC.end(); tp++, ++tpindex){
	
      //skip tps with a very small pt
      //if (sqrt(tp->momentum().perp2())<0.5) continue;
      Basic3DVector<double> momAtVtx(tp->momentum().x(),tp->momentum().y(),tp->momentum().z());
      Basic3DVector<double> vert=(Basic3DVector<double>) tp->vertex();
      int charge = tp->charge();

      double chi2 = getChi2(rParameters,recoTrackCovMatrix,momAtVtx,vert,charge,bs);
      
      if (chi2<chi2cut) {
	outputCollection.insert(tC[tindex], 
				std::make_pair(edm::Ref<TrackingParticleCollection>(tPCH, tpindex),
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

  TrackingParticleCollection tPC;
  if (tPCH.size()!=0)  tPC = *tPCH.product();

  int tpindex =0;
  for (TrackingParticleCollection::const_iterator tp=tPC.begin(); tp!=tPC.end(); tp++, ++tpindex){
    
    //skip tps with a very small pt
    //if (sqrt(tp->momentum().perp2())<0.5) continue;
    
    LogDebug("TrackAssociator") << "=========LOOKING FOR ASSOCIATION===========" << "\n"
				<< "TrackingParticle #"<<tpindex<<" with pt=" << sqrt(tp->momentum().perp2()) << "\n"
				<< "===========================================" << "\n";
    
    Basic3DVector<double> momAtVtx(tp->momentum().x(),tp->momentum().y(),tp->momentum().z());
    Basic3DVector<double> vert(tp->vertex().x(),tp->vertex().y(),tp->vertex().z());
    int charge = tp->charge();
      
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
	outputCollection.insert(edm::Ref<TrackingParticleCollection>(tPCH, tpindex),
				std::make_pair(tC[tindex],
					       -chi2));//-chi2 because the Association Map is ordered using std::greater
      }
    }
  }
  outputCollection.post_insert();
  return outputCollection;
}




RecoToGenCollection TrackAssociatorByChi2::associateRecoToGen(const edm::RefToBaseVector<reco::Track>& tC, 
							      const edm::RefVector<GenParticleCollection>& tPCH,
							      const edm::Event * e,
                                                              const edm::EventSetup *setup ) const{
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e->getByLabel(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot bs = *recoBeamSpotHandle;      

  RecoToGenCollection  outputCollection;

  GenParticleCollection tPC;
  if (tPCH.size()!=0)  tPC = *tPCH.product();

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
    for (GenParticleCollection::const_iterator tp=tPC.begin(); tp!=tPC.end(); tp++, ++tpindex){
	
      //skip tps with a very small pt
      //if (sqrt(tp->momentum().perp2())<0.5) continue;
      Basic3DVector<double> momAtVtx(tp->momentum().x(),tp->momentum().y(),tp->momentum().z());
      Basic3DVector<double> vert=(Basic3DVector<double>) tp->vertex();
      int charge = tp->charge();

      double chi2 = getChi2(rParameters,recoTrackCovMatrix,momAtVtx,vert,charge,bs);
      
      if (chi2<chi2cut) {
	outputCollection.insert(tC[tindex], 
				std::make_pair(edm::Ref<GenParticleCollection>(tPCH, tpindex),
					       -chi2));//-chi2 because the Association Map is ordered using std::greater
      }
    }
  }
  outputCollection.post_insert();
  return outputCollection;
}


GenToRecoCollection TrackAssociatorByChi2::associateGenToReco(const edm::RefToBaseVector<reco::Track>& tC, 
							      const edm::RefVector<GenParticleCollection>& tPCH,
							      const edm::Event * e,
							      const edm::EventSetup *setup ) const {

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e->getByLabel(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot bs = *recoBeamSpotHandle;      

  GenToRecoCollection  outputCollection;

  GenParticleCollection tPC;
  if (tPCH.size()!=0)  tPC = *tPCH.product();

  int tpindex =0;
  for (GenParticleCollection::const_iterator tp=tPC.begin(); tp!=tPC.end(); tp++, ++tpindex){
    
    //skip tps with a very small pt
    //if (sqrt(tp->momentum().perp2())<0.5) continue;
    
    LogDebug("TrackAssociator") << "=========LOOKING FOR ASSOCIATION===========" << "\n"
				<< "TrackingParticle #"<<tpindex<<" with pt=" << sqrt(tp->momentum().perp2()) << "\n"
				<< "===========================================" << "\n";
    
    Basic3DVector<double> momAtVtx(tp->momentum().x(),tp->momentum().y(),tp->momentum().z());
    Basic3DVector<double> vert(tp->vertex().x(),tp->vertex().y(),tp->vertex().z());
    int charge = tp->charge();
      
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
	outputCollection.insert(edm::Ref<GenParticleCollection>(tPCH, tpindex),
				std::make_pair(tC[tindex],
					       -chi2));//-chi2 because the Association Map is ordered using std::greater
      }
    }
  }
  outputCollection.post_insert();
  return outputCollection;
}
