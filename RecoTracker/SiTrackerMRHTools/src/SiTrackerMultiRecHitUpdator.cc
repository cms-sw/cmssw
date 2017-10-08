#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "DataFormats/Math/interface/ProjectMatrix.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "RecoTracker/SiTrackerMRHTools/interface/GenericProjectedRecHit2D.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiTrackerMultiRecHitUpdator::SiTrackerMultiRecHitUpdator(const TransientTrackingRecHitBuilder* builder,
							 const TrackingRecHitPropagator* hitpropagator,
							 const float Chi2Cut1D,
							 const float Chi2Cut2D,
						         const std::vector<double>& anAnnealingProgram,
							 bool debug):
  theBuilder(builder),
  theHitPropagator(hitpropagator),
  theChi2Cut1D(Chi2Cut1D),
  theChi2Cut2D(Chi2Cut2D),
  theAnnealingProgram(anAnnealingProgram),
  debug_(debug){
    theHitCloner = static_cast<TkTransientTrackingRecHitBuilder const *>(builder)->cloner();
  }


TransientTrackingRecHit::RecHitPointer  SiTrackerMultiRecHitUpdator::buildMultiRecHit(const std::vector<const TrackingRecHit*>& rhv,
                                                                          	      const TrajectoryStateOnSurface& tsos,
 										      MeasurementDetWithData& measDet, float annealing) const{

  LogTrace("SiTrackerMultiRecHitUpdator") << "Calling SiTrackerMultiRecHitUpdator::buildMultiRecHit with AnnealingFactor: "  << annealing;

  TransientTrackingRecHit::ConstRecHitContainer tcomponents;	
  for (std::vector<const TrackingRecHit*>::const_iterator iter = rhv.begin(); iter != rhv.end(); iter++){

    TransientTrackingRecHit::RecHitPointer transient = theBuilder->build(*iter);
    if(transient->isValid()) tcomponents.push_back(transient);

  }
  return update(tcomponents, tsos, measDet, annealing); 
  
}

TransientTrackingRecHit::RecHitPointer SiTrackerMultiRecHitUpdator::update( TransientTrackingRecHit::ConstRecHitPointer original,
                                                                	    const TrajectoryStateOnSurface& tsos,                                                                            MeasurementDetWithData& measDet,
									    double annealing ) const{

  LogTrace("SiTrackerMultiRecHitUpdator") << "Calling SiTrackerMultiRecHitUpdator::update with AnnealingFactor: "  << annealing;

  if(!tsos.isValid()) {
    //return original->clone();
    throw cms::Exception("SiTrackerMultiRecHitUpdator") << "!!! MultiRecHitUpdator::update(..): tsos NOT valid!!! ";
  }	

  //check if to clone is the right thing
  if(original->isValid()){
    if (original->transientHits().empty()){
      return theHitCloner.makeShared(original,tsos);
    }
  } else {
    return theHitCloner.makeShared(original,tsos);
  }

  TransientTrackingRecHit::ConstRecHitContainer tcomponents = original->transientHits();	
  return update(tcomponents, tsos, measDet, annealing);
}

/*------------------------------------------------------------------------------------------------------------------------*/
TransientTrackingRecHit::RecHitPointer SiTrackerMultiRecHitUpdator::update( TransientTrackingRecHit::ConstRecHitContainer& tcomponents,
                                                                	    const TrajectoryStateOnSurface& tsos,
									    MeasurementDetWithData& measDet, double annealing) const{

  if (tcomponents.empty()){
    LogTrace("SiTrackerMultiRecHitUpdator") << "Empty components vector passed to SiTrackerMultiRecHitUpdator::update, returning an InvalidTransientRecHit ";
    return std::make_shared<InvalidTrackingRecHit>(measDet.mdet().geomDet(), TrackingRecHit::missing);
  }		
  
  if(!tsos.isValid()) {
    LogTrace("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::update: tsos NOT valid!!!, returning an InvalidTransientRecHit";
    return std::make_shared<InvalidTrackingRecHit>(measDet.mdet().geomDet(), TrackingRecHit::missing);
  }
  
  std::vector<TransientTrackingRecHit::RecHitPointer> updatedcomponents;
  const GeomDet* geomdet = nullptr;

  //running on all over the MRH components 
  for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator iter = tcomponents.begin(); iter != tcomponents.end(); iter++){

    //the first rechit must belong to the same surface of TSOS
    if (iter == tcomponents.begin()) {

      if (&((*iter)->det()->surface())!=&(tsos.surface())){
	throw cms::Exception("SiTrackerMultiRecHitUpdator") << "the Trajectory state and the first rechit "
	"passed to the SiTrackerMultiRecHitUpdator lay on different surfaces!: state lays on surface " 
        << tsos.surface().position() << " hit with detid " << (*iter)->det()->geographicalId().rawId() 
        << " lays on surface " << (*iter)->det()->surface().position();
      }

      geomdet = (*iter)->det();

    }

    //if the rechit does not belong to the surface of the tsos
    //GenericProjectedRecHit2D is used to prepagate
    if (&((*iter)->det()->surface())!=&(tsos.surface())){

      TransientTrackingRecHit::RecHitPointer cloned = theHitPropagator->project<GenericProjectedRecHit2D>(*iter,
										 *geomdet, tsos, theBuilder);
      //if it is used a sensor by sensor grouping this should not appear
      if (cloned->isValid()) updatedcomponents.push_back(cloned);

    } else {
      TransientTrackingRecHit::RecHitPointer cloned = theHitCloner.makeShared(*iter,tsos);
      if (cloned->isValid()){
        updatedcomponents.push_back(cloned);
      }
    }
  }	

  std::vector<std::pair<const TrackingRecHit*, float> > mymap;
  std::vector<std::pair<const TrackingRecHit*, float> > normmap;
  
  double a_sum=0, c_sum=0;


  for(std::vector<TransientTrackingRecHit::RecHitPointer>::iterator ihit = updatedcomponents.begin(); 
	ihit != updatedcomponents.end(); ihit++) {

    double a_i = ComputeWeight(tsos, *(*ihit), false, annealing); //exp(-0.5*Chi2)
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t a_i:" << a_i ;
    //double c_i = ComputeWeight(tsos, *(*ihit), true, annealing);  //exp(-0.5*theChi2Cut/annealing)/(2.*M_PI*sqrt(det));
    //LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t c_i:" << c_i ;
    mymap.push_back(std::pair<const TrackingRecHit*, float>((*ihit)->hit(), a_i));

    a_sum += a_i;
    //with the new definition, the cut weight is computed only once
    if( ihit == updatedcomponents.begin() ) c_sum = ComputeWeight(tsos, *(*ihit), true, annealing);   //exp(-0.5*theChi2Cut/annealing)
  }
  double total_sum = a_sum + c_sum;    
  LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t c_sum:" << c_sum ;
  LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t total sum:" << total_sum ;

  unsigned int counter = 0;
  bool invalid = true;
  for(std::vector<TransientTrackingRecHit::RecHitPointer>::iterator ihit = updatedcomponents.begin(); 
	ihit != updatedcomponents.end(); ihit++) {

    double p = ((mymap[counter].second)/total_sum > 1.e-12 ? (mymap[counter].second)/total_sum : 1.e-12);
    //ORCA: float p = ((mymap[counter].second)/total_sum > 0.01 ? (mymap[counter].second)/total_sum : 1.e-6);


    LogTrace("SiTrackerMultiRecHitUpdator")<< "  Component hit type " << typeid(*mymap[counter].first).name()
                                           << " and dim:" << mymap[counter].first->dimension()
                                           << " position (PRECISE!!!)" << mymap[counter].first->localPosition()
                                           << " error " << mymap[counter].first->localPositionError()
                                           << " with weight " << p ;

    if( p > 10e-6 ){
      invalid = false;
      normmap.push_back(std::pair<const TrackingRecHit*,float>(mymap[counter].first, p));
    }

    counter++;
  }
 
  if(!invalid){

    SiTrackerMultiRecHitUpdator::LocalParameters param = calcParameters(tsos, normmap);
    SiTrackerMultiRecHit updated(param.first, param.second, *normmap.front().first->det(), normmap, annealing);
    LogTrace("SiTrackerMultiRecHitUpdator") << " Updated Hit position " << updated.localPosition() 
     					    << " updated error " << updated.localPositionError() << std::endl;

    return std::make_shared<SiTrackerMultiRecHit>(param.first, param.second, *normmap.front().first->det(), normmap, annealing);

  } else {
    LogTrace("SiTrackerMultiRecHitUpdator") << " No hits with weight (> 10e-6) have been found for this MRH." << std::endl;
    return std::make_shared<InvalidTrackingRecHit>(measDet.mdet().geomDet(), TrackingRecHit::missing);
  }
}


//---------------------------------------------------------------------------------------------------------------
double SiTrackerMultiRecHitUpdator::ComputeWeight(const TrajectoryStateOnSurface& tsos, 
							const TransientTrackingRecHit& aRecHit, bool CutWeight, double annealing) const{
     switch (aRecHit.dimension()) {
         case 1: return ComputeWeight<1>(tsos,aRecHit,CutWeight,annealing);
         case 2: return ComputeWeight<2>(tsos,aRecHit,CutWeight,annealing);
         case 3: return ComputeWeight<3>(tsos,aRecHit,CutWeight,annealing);
         case 4: return ComputeWeight<4>(tsos,aRecHit,CutWeight,annealing);
         case 5: return ComputeWeight<5>(tsos,aRecHit,CutWeight,annealing);
     }
     throw cms::Exception("Rec hit of invalid dimension (not 1,2,3,4,5)") <<
          "The value was " << aRecHit.dimension() << 
         ", type is " << typeid(aRecHit).name() << "\n";
}
 
//---------------------------------------------------------------------------------------------------------------
template <unsigned int N>
double SiTrackerMultiRecHitUpdator::ComputeWeight(const TrajectoryStateOnSurface& tsos,
                                                const TransientTrackingRecHit& aRecHit, bool CutWeight, double annealing) const {

//  typedef typename AlgebraicROOTObject<N,5>::Matrix MatN5;
//  typedef typename AlgebraicROOTObject<5,N>::Matrix Mat5N;
//  typedef typename AlgebraicROOTObject<N,N>::SymMatrix SMatNN;
//  typedef typename AlgebraicROOTObject<N>::Vector VecN;

  // define variables that will be used to setup the KfComponentsHolder
  ProjectMatrix<double,5,N>  pf;
  typename AlgebraicROOTObject<N>::Vector r, rMeas;
  typename AlgebraicROOTObject<N,N>::SymMatrix R, RMeas, W;
  AlgebraicVector5 x = tsos.localParameters().vector();
  const AlgebraicSymMatrix55 &C = (tsos.localError().matrix());

  // setup the holder with the correct dimensions and get the values
  KfComponentsHolder holder;
  holder.template setup<N>(&r, &R,  &pf, &rMeas, &RMeas, x, C);
  aRecHit.getKfComponents(holder);

  typename AlgebraicROOTObject<N>::Vector diff = r - rMeas;

  if(!CutWeight){
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t r:" << r ;
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t tsospos:" << rMeas ;
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t diff:" << diff ;
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t R:" << R ;
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t RMeas:" << RMeas ;
  }

  R += RMeas;						//assume that TSOS is predicted || comb one
  if(!CutWeight)  LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t R+RMeas:" << R ;


  //computing chi2 with the smoothTsos
 // SMatNN R_smooth = R - RMeas;
 // if(!CutWeight)  LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t R-=Rmeas:" << R_smooth ;
 // bool ierr2_bis = invertPosDefMatrix(R_smooth);
 // double Chi2_smooth = ROOT::Math::Similarity(diff, R_smooth); 

  //computing chi2 with the smoothTsos
  //SMatNN R_pred   = R + RMeas;
  //if(!CutWeight)  LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t R+=Rmeas:" << R_pred   ;
  //bool ierr2_bis = invertPosDefMatrix(R_pred  );
  //double Chi2_pred   = ROOT::Math::Similarity(diff, R_pred  ); 

  //Det2 method will preserve the content of the Matrix 
  //and return true when the calculation is successfull
  double det;
  bool ierr = R.Det2(det);

  bool ierr2 = invertPosDefMatrix(R);			//ierr will be set to true when inversion is successfull
  double Chi2 = ROOT::Math::Similarity(diff, R);

  if( !ierr || !ierr2 ) {
    LogTrace("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::ComputeWeight: W not valid!"<<std::endl;
    LogTrace("SiTrackerMultiRecHitUpdator")<<"V: "<<R<<" AnnealingFactor: "<<annealing<<std::endl;
  }


  if(!CutWeight){
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t det:" << det;
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t Chi2:" << Chi2;
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t Chi2/ann:" << Chi2/annealing;
  }

  double temp_weight = 0.0;
  if( CutWeight && N == 1 ){
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t Chi2Cut1D:" << theChi2Cut1D;
    temp_weight = exp(-0.5*theChi2Cut1D/annealing);
  } else if( CutWeight && N == 2 ) {
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t Chi2Cut2D:" << theChi2Cut2D;
    temp_weight = exp(-0.5*theChi2Cut2D/annealing);
  }

  if(!CutWeight) {
    temp_weight = exp(-0.5*Chi2/annealing); 
  }

  return temp_weight;

}

//-----------------------------------------------------------------------------------------------------------
SiTrackerMultiRecHitUpdator::LocalParameters SiTrackerMultiRecHitUpdator::calcParameters(const TrajectoryStateOnSurface& tsos, std::vector<std::pair<const TrackingRecHit*, float> >& aHitMap) const{

  //supposing all the hits inside of a MRH have the same dimension
  LogTrace("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::LocalParameters: dim first recHit: " << aHitMap[0].first->dimension() <<std::endl;
  switch (aHitMap[0].first->dimension()) {
    case 1: return calcParameters<1>(tsos,aHitMap);
    case 2: return calcParameters<2>(tsos,aHitMap);
  }
  throw cms::Exception("Rec hit of invalid dimension for computing MRH (not 1,2)") <<
        "The value was " << aHitMap[0].first->dimension() <<
        ", type is " << typeid(aHitMap[0].first).name() << "\n";

}
  
//-----------------------------------------------------------------------------------------------------------
template <unsigned int N>
SiTrackerMultiRecHitUpdator::LocalParameters SiTrackerMultiRecHitUpdator::calcParameters(const TrajectoryStateOnSurface& tsos, std::vector<std::pair<const TrackingRecHit*, float> >& aHitMap) const{

  typedef typename AlgebraicROOTObject<N,N>::SymMatrix SMatNN;
  typedef typename AlgebraicROOTObject<N>::Vector VecN;

  VecN m_sum;
  SMatNN W_sum;
  LocalPoint position;
  LocalError error;

  //for TID and TEC the correlation is really high -> need to be scorrelated and then correlated again
  float s = 0.1;

  for( std::vector<std::pair<const TrackingRecHit*, float> >::const_iterator ihit = aHitMap.begin(); 
	ihit != aHitMap.end(); ihit++ ){

    // define variables that will be used to setup the KfComponentsHolder
    ProjectMatrix<double,5,N>  pf;
    typename AlgebraicROOTObject<N>::Vector r, rMeas;
    typename AlgebraicROOTObject<N,N>::SymMatrix V, VMeas, Wtemp;
    AlgebraicVector5 x = tsos.localParameters().vector();
    const AlgebraicSymMatrix55 &C = (tsos.localError().matrix());

    // setup the holder with the correct dimensions and get the values
    KfComponentsHolder holder;
    holder.template setup<N>(&r, &V, &pf, &rMeas, &VMeas, x, C);
    (ihit->first)->getKfComponents(holder);

    LogTrace("SiTrackerMultiRecHitUpdator") << "\t position: " << r;  
    LogTrace("SiTrackerMultiRecHitUpdator") << "\t error: " << V;  

    //scorrelation  in TID and TEC
    if( N==2 && TIDorTEChit(ihit->first) ) { 
      V(0,1) = V(1,0) = V(0,1)*s; 
//      V(1,0) = V(1,0)*s; 
      LogTrace("SiTrackerMultiRecHitUpdator") << "\t error scorr: " << V;
    }

    bool ierr = invertPosDefMatrix(V);
    if( !ierr ) {
      edm::LogError("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::calcParameters: V not valid!";
    }
    LogTrace("SiTrackerMultiRecHitUpdator") << "\t inverse error: " << V;  

    //compute m_sum and W_sum
    m_sum += (ihit->second*V*r);
    W_sum += (ihit->second*V);
  
  }

  bool ierr_sum = invertPosDefMatrix(W_sum);
  if( !ierr_sum ) {
    edm::LogError("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::calcParameters: W_sum not valid!";
  }
 
  LogTrace("SiTrackerMultiRecHitUpdator") << "\t inverse total error: " << W_sum; 
  typename AlgebraicROOTObject<N>::Vector parameters = W_sum*m_sum;
  if( N == 1 ){
    position = LocalPoint(parameters(0),0.f);
    error = LocalError(W_sum(0,0),0.f,std::numeric_limits<float>::max());
  }
  else if( N == 2 ){
    position = LocalPoint(parameters(0), parameters(1));
    //ri-correlation  in TID and TEC
    if( TIDorTEChit(aHitMap.at(0).first) )	error = LocalError(W_sum(0,0), W_sum(0,1)/s, W_sum(1,1));
    else 					error = LocalError(W_sum(0,0), W_sum(0,1), W_sum(1,1));
  }


  return std::make_pair(position,error);

}

bool SiTrackerMultiRecHitUpdator::TIDorTEChit(const TrackingRecHit* const& hit) const{

  DetId hitId = hit->geographicalId();

  if( hitId.det() == DetId::Tracker && 
	  ( hitId.subdetId() == StripSubdetector::TEC || hitId.subdetId() == StripSubdetector::TID) ) {
    return true;
  }    

  return false;
}
