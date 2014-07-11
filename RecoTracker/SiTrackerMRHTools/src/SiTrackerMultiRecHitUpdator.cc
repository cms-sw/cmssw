#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "DataFormats/Math/interface/ProjectMatrix.h"

#include "RecoTracker/SiTrackerMRHTools/interface/GenericProjectedRecHit2D.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiTrackerMultiRecHitUpdator::SiTrackerMultiRecHitUpdator(const TransientTrackingRecHitBuilder* builder,
							 const TrackingRecHitPropagator* hitpropagator,
							 const float Chi2Cut,
						         const std::vector<double>& anAnnealingProgram,
							 bool debug):
  theBuilder(builder),
  theHitPropagator(hitpropagator),
  theChi2Cut(Chi2Cut),
  theAnnealingProgram(anAnnealingProgram),
  debug_(debug){
    theHitCloner = static_cast<TkTransientTrackingRecHitBuilder const *>(builder)->cloner();
  }


TransientTrackingRecHit::RecHitPointer  SiTrackerMultiRecHitUpdator::buildMultiRecHit(const std::vector<const TrackingRecHit*>& rhv,
                                                                          	      const TrajectoryStateOnSurface& tsos,
 										      float annealing) const{

  LogTrace("SiTrackerMultiRecHitUpdator") << "Calling SiTrackerMultiRecHitUpdator::buildMultiRecHit with AnnealingFactor: "  << annealing;

  TransientTrackingRecHit::ConstRecHitContainer tcomponents;	
  for (std::vector<const TrackingRecHit*>::const_iterator iter = rhv.begin(); iter != rhv.end(); iter++){

    TransientTrackingRecHit::RecHitPointer transient = theBuilder->build(*iter);
    if(transient->isValid()) tcomponents.push_back(transient);

  }
  return update(tcomponents, tsos, annealing); 
  
}

TransientTrackingRecHit::RecHitPointer SiTrackerMultiRecHitUpdator::update( TransientTrackingRecHit::ConstRecHitPointer original,
                                                                	    const TrajectoryStateOnSurface& tsos,
									    double annealing) const{

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
  return update(tcomponents, tsos, annealing);
}

/*------------------------------------------------------------------------------------------------------------------------*/
TransientTrackingRecHit::RecHitPointer SiTrackerMultiRecHitUpdator::update( TransientTrackingRecHit::ConstRecHitContainer& tcomponents,
                                                                	    const TrajectoryStateOnSurface& tsos,
									    double annealing) const{

  if (tcomponents.empty()){
    LogTrace("SiTrackerMultiRecHitUpdator") << "Empty components vector passed to SiTrackerMultiRecHitUpdator::update, returning an InvalidTransientRecHit ";
    return std::make_shared<InvalidTrackingRecHitNoDet>(); 
  }		
  
  if(!tsos.isValid()) {
    LogTrace("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::update: tsos NOT valid!!!, returning an InvalidTransientRecHit";
    return std::make_shared<InvalidTrackingRecHitNoDet>();
  }
  
  std::vector<TransientTrackingRecHit::RecHitPointer> updatedcomponents;
  const GeomDet* geomdet = 0;

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

    double a_i = ComputeWeight(tsos, *(*ihit), false, annealing); //exp(-0.5*Chi2)/(2.*M_PI*sqrt(det));
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t a_i:" << a_i ;
    double c_i = ComputeWeight(tsos, *(*ihit), true, annealing);  //exp(-0.5*theChi2Cut/annealing)/(2.*M_PI*sqrt(det));
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t c_i:" << c_i ;
    mymap.push_back(std::pair<const TrackingRecHit*, float>((*ihit)->hit(), a_i));

    a_sum += a_i;
    c_sum += c_i;   
  }
  double total_sum = a_sum + c_sum;    
  LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t total sum:" << total_sum ;

  unsigned int counter = 0;
  for(std::vector<TransientTrackingRecHit::RecHitPointer>::iterator ihit = updatedcomponents.begin(); 
	ihit != updatedcomponents.end(); ihit++) {

    double p = ((mymap[counter].second)/total_sum > 1.e-6 ? (mymap[counter].second)/total_sum : 1.e-6);
    //ORCA: float p = ((mymap[counter].second)/total_sum > 0.01 ? (mymap[counter].second)/total_sum : 1.e-6);
    normmap.push_back(std::pair<const TrackingRecHit*,float>(mymap[counter].first, p));

    LogTrace("SiTrackerMultiRecHitUpdator")<< "  Component hit type " << typeid(*mymap[counter].first).name()
					   << " and dim:" << mymap[counter].first->dimension() 
					   << " position " << mymap[counter].first->localPosition() 
					   << " error " << mymap[counter].first->localPositionError()
					   << " with weight " << p ;
    counter++;
  }
 
  SiTrackerMultiRecHitUpdator::LocalParameters param = calcParameters(tsos, normmap);

  SiTrackerMultiRecHit updated(param.first, param.second, *normmap.front().first->det(), normmap, annealing);
  LogTrace("SiTrackerMultiRecHitUpdator") << " Updated Hit position " << updated.localPosition() 
   					  << " updated error " << updated.localPositionError() << std::endl;

  return std::make_shared<SiTrackerMultiRecHit>(param.first, param.second, *normmap.front().first->det(), normmap, annealing);
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

  typedef typename AlgebraicROOTObject<N,5>::Matrix MatN5;
  typedef typename AlgebraicROOTObject<5,N>::Matrix Mat5N;
  typedef typename AlgebraicROOTObject<N,N>::SymMatrix SMatNN;
  typedef typename AlgebraicROOTObject<N>::Vector VecN;

  VecN r, rMeas; 
  SMatNN R, RMeas;
  MatN5 dummyProjMatrix;
  auto && v = tsos.localParameters().vector();
  auto && m = tsos.localError().matrix();

  // setup the holder with the correct dimensions and get the values
  KfComponentsHolder holder;
  holder.template setup<N>(&r, &R, &dummyProjMatrix, &rMeas, &RMeas, v, m);
  aRecHit.getKfComponents(holder);

  VecN diff = r - rMeas;
  //R += RMeas;						//assume that TSOS is predicted one
  if(!CutWeight)  LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t R:" << R ;
  R *= annealing;					//assume that TSOS is smoothed one
  if(!CutWeight)  LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t R*= ann:" << R ;

  if(!CutWeight){
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t r:" << r ;
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t tsospos:" << rMeas ;
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t diff:" << diff ;
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t RMeas:" << RMeas ;
  }

  //Det2 method will preserve the content of the Matrix 
  //and return true when the calculation is successfull
  double det;
  bool ierr = R.Det2(det);

  bool ierr2 = invertPosDefMatrix(R);			//ierr will be set to true when inversion is successfull
  double Chi2 = ROOT::Math::Similarity(diff, R);

  if( !ierr || !ierr2) {
    LogTrace("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::ComputeWeight: W not valid!"<<std::endl;
    LogTrace("SiTrackerMultiRecHitUpdator")<<"V: "<<R<<" AnnealingFactor: "<<annealing<<std::endl;
  }

  if(!CutWeight){
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t det:" << det;
    LogTrace("SiTrackerMultiRecHitUpdator")<< "\t\t Chi2:" << Chi2;
  }

  double temp_weight;
  if( CutWeight ) 	temp_weight = exp(-0.5*theChi2Cut/annealing)/(2.*M_PI*sqrt(det));
  else 			temp_weight = exp(-0.5*Chi2)/(2.*M_PI*sqrt(det)); 

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

  for( std::vector<std::pair<const TrackingRecHit*, float> >::const_iterator ihit = aHitMap.begin(); 
	ihit != aHitMap.end(); ihit++ ){

    // define variables that will be used to setup the KfComponentsHolder
    ProjectMatrix<double,5,N>  pf;
    typename AlgebraicROOTObject<N,5>::Matrix H;
    AlgebraicVector5 x = tsos.localParameters().vector();
    const AlgebraicSymMatrix55 &C = (tsos.localError().matrix());

    VecN r, rMeas;
    SMatNN V, VMeas, Wtemp;

    // setup the holder with the correct dimensions and get the values
    KfComponentsHolder holder;
    holder.template setup<N>(&r, &V, &H, &pf, &rMeas, &VMeas, x, C);
    (ihit->first)->getKfComponents(holder);

    LogTrace("SiTrackerMultiRecHitUpdator") << "\t position: " << r;  
    LogTrace("SiTrackerMultiRecHitUpdator") << "\t error: " << V;  
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
  VecN parameters = W_sum*m_sum;
  if( N == 1 ){
    position = LocalPoint(parameters(0),0.f);
    error = LocalError(W_sum(0,0),0.f,std::numeric_limits<float>::max());
  }
  else if( N == 2 ){
    position = LocalPoint(parameters(0), parameters(1));
    error = LocalError(W_sum(0,0), W_sum(0,1), W_sum(1,1));
  }


  return std::make_pair(position,error);

}


