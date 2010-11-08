#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerMultiRecHit.h"
#include "RecoTracker/SiTrackerMRHTools/interface/GenericProjectedRecHit2D.h"
//#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


SiTrackerMultiRecHitUpdator::SiTrackerMultiRecHitUpdator(const TransientTrackingRecHitBuilder* builder,
							 const TrackingRecHitPropagator* hitpropagator,
							 const float Chi2Cut,
						         const std::vector<double>& anAnnealingProgram):
  theBuilder(builder),
  theHitPropagator(hitpropagator),
  theChi2Cut(Chi2Cut),
  theAnnealingProgram(anAnnealingProgram){}
//theAnnealingStep(0),
//theIsUpdating(true){}


TransientTrackingRecHit::RecHitPointer  SiTrackerMultiRecHitUpdator::buildMultiRecHit(const std::vector<const TrackingRecHit*>& rhv,
                                                                          	      TrajectoryStateOnSurface tsos,
 										      float annealing) const{
  TransientTrackingRecHit::ConstRecHitContainer tcomponents;	
  for (std::vector<const TrackingRecHit*>::const_iterator iter = rhv.begin(); iter != rhv.end(); iter++){
    TransientTrackingRecHit::RecHitPointer transient = theBuilder->build(*iter);
    if (transient->isValid()) tcomponents.push_back(transient);
  }
  
  return update(tcomponents, tsos, annealing); 
  
}

TransientTrackingRecHit::RecHitPointer  SiTrackerMultiRecHitUpdator::update( TransientTrackingRecHit::ConstRecHitPointer original,
                                                                	     TrajectoryStateOnSurface tsos,
									     double annealing) const{
  LogTrace("SiTrackerMultiRecHitUpdator") << "Calling SiTrackerMultiRecHitUpdator::update with AnnealingFactor: "  << annealing;
  if (original->isValid())
    LogTrace("SiTrackerMultiRecHitUpdator") << "Original Hit position " << original->localPosition() << " original error " 
					    << original->parametersError();
  else LogTrace("SiTrackerMultiRecHitUpdator") << "Invalid hit";	
  
  if(!tsos.isValid()) {
    //return original->clone();
    throw cms::Exception("SiTrackerMultiRecHitUpdator") << "!!! MultiRecHitUpdator::update(..): tsos NOT valid!!! ";
  }	
  
  //check if to clone is the right thing
  if (original->transientHits().empty()) return original->clone(tsos);
  
  TransientTrackingRecHit::ConstRecHitContainer tcomponents = original->transientHits();	
  return update(tcomponents, tsos, annealing);
}

TransientTrackingRecHit::RecHitPointer  SiTrackerMultiRecHitUpdator::update( TransientTrackingRecHit::ConstRecHitContainer& tcomponents,
                                                                	     TrajectoryStateOnSurface tsos,
									     double annealing) const{
  
  if (tcomponents.empty()){
    LogTrace("SiTrackerMultiRecHitUpdator") << "Empty components vector passed to SiTrackerMultiRecHitUpdator::update, returning an InvalidTransientRecHit ";
    return InvalidTransientRecHit::build(0); 
  }		
  
  if(!tsos.isValid()) {
    LogTrace("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::update: tsos NOT valid!!!, returning an InvalidTransientRecHit";
    return InvalidTransientRecHit::build(0);
  }
  
  std::vector<TransientTrackingRecHit::RecHitPointer> updatedcomponents;
  const GeomDet* geomdet = 0;
  for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator iter = tcomponents.begin(); iter != tcomponents.end(); iter++){
    if (iter == tcomponents.begin()) {
      if (&((*iter)->det()->surface())!=&(tsos.surface())){
	throw cms::Exception("SiTrackerMultiRecHitUpdator") << "the Trajectory state and the first rechit passed to the SiTrackerMultiRecHitUpdator lay on different surfaces!: state lays on surface " << tsos.surface().position() << " hit with detid " << (*iter)->det()->geographicalId().rawId() << " lays on surface " << (*iter)->det()->surface().position();
      }
      geomdet = (*iter)->det();
      LogTrace("SiTrackerMultiRecHitUpdator") << "Current reference surface located at " << geomdet->surface().position();
      //  LogTrace("SiTrackerMultiRecHitUpdator")<<  "TSOS position " << tsos.localPosition(); 
    }
    if (&((*iter)->det()->surface())!=&(tsos.surface())){
      TransientTrackingRecHit::RecHitPointer cloned = theHitPropagator->project<GenericProjectedRecHit2D>(*iter, *geomdet, tsos);
      //      LogTrace("SiTrackerMultiRecHitUpdator") << "hit propagated";

      if (cloned->isValid()) updatedcomponents.push_back(cloned);
    } else {
      TransientTrackingRecHit::RecHitPointer cloned = (*iter)->clone(tsos);
      if (cloned->isValid()) updatedcomponents.push_back(cloned);
    }
  }	
  //  LogTrace("SiTrackerMultiRecHitUpdator") << "hit cloned";
  int ierr;
  std::vector<std::pair<const TrackingRecHit*, float> > mymap;
  std::vector<std::pair<const TrackingRecHit*, float> > normmap;
  
  double a_sum=0, c_sum=0;
  
  AlgebraicVector2 tsospos;
  tsospos[0]=tsos.localPosition().x();
  tsospos[1]=tsos.localPosition().y();
  LogTrace("SiTrackerMultiRecHitUpdator")<<  "TSOS position " << tsos.localPosition(); 
  for(std::vector<TransientTrackingRecHit::RecHitPointer>::iterator ihit = updatedcomponents.begin(); ihit != updatedcomponents.end(); ihit++) {
    AlgebraicVector2 r(asSVector<2>((*ihit)->parameters()) - tsospos);
    AlgebraicSymMatrix22 V(asSMatrix<2>((*ihit)->parametersError()));
    V *= annealing;//assume that TSOS is smoothed one
    //V += me.measuredError(*ihit);// result = b*V + H*C*H.T()
    AlgebraicSymMatrix22 W(V.Inverse(ierr));
    double det;
    bool ierr2=!(V.Det2(det));

    if(ierr != 0|| ierr2) {
      LogTrace("SiTrackerMultiRecHitUpdator")<<"MultiRecHitUpdator::update: W not valid!"<<std::endl;
      LogTrace("SiTrackerMultiRecHitUpdator")<<"V: "<<V<<" AnnealingFactor: "<<annealing<<std::endl;
    }
    double Chi2 =  ROOT::Math::Similarity(r,W);// Chi2 = r.T()*W*r
    double a_i = exp(-0.5*Chi2)/(2.*M_PI*sqrt(det));
    mymap.push_back(std::pair<const TrackingRecHit*, float>((*ihit)->hit(), a_i));
    double c_i = exp(-0.5*theChi2Cut/annealing)/(2.*M_PI*sqrt(det));
    a_sum += a_i;
    c_sum += c_i;   
  }
  double total_sum = a_sum + c_sum;    
  
  unsigned int counter = 0;
  TransientTrackingRecHit::ConstRecHitContainer finalcomponents;
  for(std::vector<TransientTrackingRecHit::RecHitPointer>::iterator ihit = updatedcomponents.begin(); ihit != updatedcomponents.end(); ihit++) {
    //uncomment lines below to have like ORCA
    double p = ((mymap[counter].second)/total_sum > 1.e-6 ? (mymap[counter].second)/total_sum : 1.e-6);
    //float p = ((mymap[counter].second)/total_sum > 0.01 ? (mymap[counter].second)/total_sum : 1.e-6);
    normmap.push_back(std::pair<const TrackingRecHit*, float>(mymap[counter].first, p));
    //let's store the weight in the component TransientTrackingRecHit too
    (*ihit)->setWeight(p);
    (*ihit)->setAnnealingFactor(annealing);
    finalcomponents.push_back(*ihit);	
    LogTrace("SiTrackerMultiRecHitUpdator")<< "Component hit type " << typeid(*mymap[counter].first).name() 
					   << " position " << mymap[counter].first->localPosition() 
					   << " error " << mymap[counter].first->localPositionError()
					   << " with weight " << p;
    counter++;
  }
  
  mymap = normmap;
  //  LocalError er = calcParametersError(finalcomponents);
  //  LocalPoint p  = calcParameters(finalcomponents, er);
  SiTrackerMultiRecHitUpdator::LocalParameters param=calcParameters(finalcomponents);
  SiTrackerMultiRecHit updated(param.first, param.second, normmap.front().first->geographicalId(), normmap);
  LogTrace("SiTrackerMultiRecHitUpdator") << "Updated Hit position " << updated.localPosition() << " updated error " << updated.parametersError() << std::endl;
  //return new SiTrackerMultiRecHit(normmap); 	
  return TSiTrackerMultiRecHit::build(geomdet, &updated, finalcomponents, annealing);
}


SiTrackerMultiRecHitUpdator::LocalParameters SiTrackerMultiRecHitUpdator::calcParameters(TransientTrackingRecHit::ConstRecHitContainer& map)const{
  AlgebraicSymMatrix22 W_sum;
  AlgebraicVector2 m_sum;
  int ierr;
  for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator ihit = map.begin(); ihit != map.end(); ihit ++) {
    AlgebraicVector2 m(asSVector<2>((*ihit)->parameters()));
    AlgebraicSymMatrix22 V(asSMatrix<2>((*ihit)->parametersError()));
    AlgebraicSymMatrix22 W(V.Inverse(ierr));
    
    if(ierr != 0) {
      edm::LogError("SiTrackerMultiRecHitUpdator")<<"MultiRecHit::checkParameters: W not valid!"<<std::endl;
    }
    
    else {
      W_sum += ((*ihit)->weight()*W);
      m_sum += ((*ihit)->weight()*(W*m));
    }
  }
  AlgebraicSymMatrix22  V_sum= W_sum.Inverse(ierr);
  AlgebraicVector2 parameters = V_sum*m_sum;
  LocalError error=LocalError(V_sum(0,0), V_sum(0,1), V_sum(1,1));
  LocalPoint position=LocalPoint(parameters(0), parameters(1));
  return std::make_pair(position,error);
}

LocalError SiTrackerMultiRecHitUpdator::calcParametersError(TransientTrackingRecHit::ConstRecHitContainer& map) const {
  AlgebraicSymMatrix22 W_sum;
  int ierr;
  for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator ihit = map.begin(); ihit != map.end(); ihit ++) {
    AlgebraicSymMatrix22 V(asSMatrix<2>((*ihit)->parametersError()));
    AlgebraicSymMatrix22 W(V.Inverse(ierr));
    
    if(ierr != 0) {
      edm::LogError("SiTrackerMultiRecHitUpdator")<<"MultiRecHit::checkParametersError: W not valid!"<<std::endl;
    }
    
    else W_sum += ((*ihit)->weight()*W);
  }
  AlgebraicSymMatrix22 parametersError = W_sum.Inverse(ierr);
  return LocalError(parametersError(0,0), parametersError(0,1), parametersError(1,1));
} 

LocalPoint SiTrackerMultiRecHitUpdator::calcParameters(TransientTrackingRecHit::ConstRecHitContainer& map, const LocalError& er) const {
  AlgebraicVector2 m_sum;
  int ierr;
  for( TransientTrackingRecHit::ConstRecHitContainer::const_iterator ihit = map.begin(); ihit != map.end(); ihit ++) {
    AlgebraicVector2 m(asSVector<2>((*ihit)->parameters()));
    AlgebraicSymMatrix22 V(asSMatrix<2>((*ihit)->parametersError()));
    AlgebraicSymMatrix22 W(V.Inverse(ierr));
    
    if(ierr != 0) {
      edm::LogError("SiTrackerMultiRecHitUpdator")<<"MultiRecHit::checkParameters: W not valid!"<<std::endl;
    }
    
    //m_sum += ihit->weight()*(W*m);      
    else m_sum += ((*ihit)->weight()*(W*m));
  }
  AlgebraicSymMatrix22 V_sum;
	
  V_sum(0,0) = er.xx();
  V_sum(0,1) = er.xy();
  V_sum(1,1) = er.yy();
  //AlgebraicSymMatrix V_sum(parametersError());
  AlgebraicVector2 parameters = V_sum*m_sum;
  return LocalPoint(parameters(0), parameters(1));
}
                            
