#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdatorMTF.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerMultiRecHit.h"
#include "RecoTracker/SiTrackerMRHTools/interface/GenericProjectedRecHit2D.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiTrajectoryMeasurement.h"
//#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


SiTrackerMultiRecHitUpdatorMTF::SiTrackerMultiRecHitUpdatorMTF(const TransientTrackingRecHitBuilder* builder,
							       const TrackingRecHitPropagator* hitpropagator,
							       const float Chi2Cut,
							       const std::vector<double>& anAnnealingProgram):
  theBuilder(builder),
  theHitPropagator(hitpropagator),
  theChi2Cut(Chi2Cut),
  theAnnealingProgram(anAnnealingProgram){}
//theAnnealingStep(0),
//theIsUpdating(true){}



//modified respect to the DAF, to get an mtm, and from this one the vector of tsos to put the trajectories in competition for the hit
TransientTrackingRecHit::RecHitPointer SiTrackerMultiRecHitUpdatorMTF::buildMultiRecHit(TrajectoryStateOnSurface& state, 
											TransientTrackingRecHit::ConstRecHitContainer& rhv,
											MultiTrajectoryMeasurement* mtm,
											float annealing) const{
  
  //LogDebug("SiTrackerMultiRechitUpdatorMTF") << "Vector pushed back " << td::endl;
  
	
	//get the other variables needed to do the update
	std::vector<TransientTrackingRecHit::RecHitPointer> updatedcomponents = updatecomponents(rhv, state, annealing);
	

	//	LogDebug("SiTrackerMultiRechitUpdatorMTF") << "built a vector (updated) of size: "<<updatedcomponents.size() 
	//				   << std::endl;
	  
	double rowsum = updaterow(rhv, state, updatedcomponents, annealing);
	//if (rowsum==-1)return an invalid hit?
	//float columnsum = updatecolumn(hit, mtm, annealing);
	double c = calculatecut(rhv, state, updatedcomponents, annealing);
	//LogDebug("SiTrackerMultiRechitUpdatorMTF") <<" rowsum = " << rowsum <<std::endl
	//					   <<" columnsum = "<<columnsum<<std::endl
	//					   <<" c = "<<c<<std::endl;
	
	std::vector<std::pair<TransientTrackingRecHit::RecHitPointer, double> > themap = mapmaker(rhv, state, annealing);
	//LogDebug("SiTrackerMultiRechitUpdatorMTF") << " made a map of "<<themap.size()<<"components" 
	//					   << std::endl;

	//LogDebug("SiTrackerMultiRechitUpdatorMTF") << " local position "<<themap.front().first->localPosition() 
	//					   << std::endl;
	
	//returns a vector updated given 6 variables
	return update(rowsum,mtm,c,themap,updatedcomponents,annealing); 
	
}



double SiTrackerMultiRecHitUpdatorMTF::update(TransientTrackingRecHit::ConstRecHitPointer original,
					     TrajectoryStateOnSurface tsos,
					     float annealing) const{
  
  //  LogTrace("SiTrackerMultiRecHitUpdator") << "Calling SiTrackerMultiRecHitUpdator::update with AnnealingFactor: "  << annealing;
   if (original->isValid())
    LogTrace("SiTrackerMultiRecHitUpdator") << "Original Hit position " << original->localPosition() << " original error " 
					    << original->parametersError();
  else LogTrace("SiTrackerMultiRecHitUpdator") << "Invalid hit";	
  
  if(!tsos.isValid()) {
    //return original->clone();
    throw cms::Exception("SiTrackerMultiRecHitUpdator") << "!!! MultiRecHitUpdator::update(..): tsos NOT valid!!! ";
  }	
  
  //check if to clone is the right thing
  if (original->transientHits().empty()) return 0;
    //return original->clone(tsos);
  //
  TransientTrackingRecHit::ConstRecHitContainer tcomponents = original->transientHits();

  std::vector<TransientTrackingRecHit::RecHitPointer> updatedcomponents = updatecomponents(tcomponents, tsos, annealing);	
  
  return updaterow(tcomponents, tsos, updatedcomponents, annealing);
}




//this method calculate the DAF weight. It is only a part of the MTF weight (we call it the updaterow method) when the tsos is fix and the weight of each
//hit respect to the tsos is computed
double SiTrackerMultiRecHitUpdatorMTF::updaterow(TransientTrackingRecHit::ConstRecHitContainer& tcomponents,
						TrajectoryStateOnSurface& tsos,
						std::vector<TransientTrackingRecHit::RecHitPointer>& updatedcomponents,
						float annealing) const{
  
  if (tcomponents.empty()){
    LogTrace("SiTrackerMultiRecHitUpdator") << "Empty components vector passed to SiTrackerMultiRecHitUpdator::updateraw";
    return -1; 
  }		
  
  if(!tsos.isValid()) {
    LogTrace("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::updateraw: tsos NOT valid!!!";
    return -2;
  }
  
  
  
  //a loop to propagate all the hits on the same surface...
  //std::vector<TransientTrackingRecHit::RecHitPointer> updatedcomponents = updatecomponents(tcomponents,tsos,annealing);
  
  int ierr;
  
  double a_sum=0;
  
  AlgebraicVector2 tsospos;
  tsospos[0]=tsos.localPosition().x();
  tsospos[1]=tsos.localPosition().y();
  //  LogDebug("SiTrackerMultiRecHitUpdatorMTF")<<  "TSOS position " << tsos.localPosition(); 
  
  for(std::vector<TransientTrackingRecHit::RecHitPointer>::iterator ihit = updatedcomponents.begin(); ihit != updatedcomponents.end(); ihit++) {
    
    //    LogDebug("SiTrackerMultiRecHitUpdatorMTF")<<  "hit position " << (*ihit)->localPosition();
    AlgebraicVector2 r(asSVector<2>((*ihit)->parameters()) - tsospos);
    AlgebraicSymMatrix22 V(asSMatrix<2>((*ihit)->parametersError()));
    V *= annealing;//assume that TSOS is smoothed one
    //V += me.measuredError(*ihit);// result = b*V + H*C*H.T()
    AlgebraicSymMatrix22 W(V.Inverse(ierr));
    double det;
    bool ierr2=!(V.Det2(det));
    
    if(ierr != 0|| ierr2) {
      LogTrace("SiTrackerMultiRecHitUpdatorMTF")<<"MultiRecHitUpdator::update: W not valid!"<<std::endl;
      LogTrace("SiTrackerMultiRecHitUpdatorMTF")<<"V: "<<V<<" AnnealingFactor: "<<annealing<<std::endl;
    }
    else{
      double Chi2 = ROOT::Math::Similarity(r,W);// Chi2 = r.T()*W*r
      //LogDebug("SiTrackerMultiRecHitUpdatorMTF")<<"Chi2 (updaterow method)= "<<Chi2<<std::endl;
      //this is the probability of the hit respect to the track
      double a_i = exp(-0.5*Chi2)/(2.*M_PI*sqrt(det));
      a_sum += a_i;
    }
  }
  
  return a_sum;
}

//can make a new nethod on the base of the first one, which updates the sum over the columns(it gets a mtm and returns a double)
double SiTrackerMultiRecHitUpdatorMTF::updatecolumn(TransientTrackingRecHit::ConstRecHitPointer trechit, 
						   MultiTrajectoryMeasurement* mtm, 
						   float annealing) const{
  
  
  if(!trechit->isValid()) {
    LogTrace("SiTrackerMultiRecHitUpdatorMTF")<<"SiTrackerMultiRecHitUpdatorMTF::updatecolumn: rechit NOT valid!!!, returning an InvalidTransientRecHit";
    //return InvalidTransientRecHit::build(0);
  }
  
  int ierr;
  
  //const GeomDet* geomdet = 0;
  
  double a_sum=0;
  
  
  AlgebraicVector2 hitpos(asSVector<2>(trechit->parameters()));
  //hitpos[0]=trechit->localPosition().x();
  //hitpos[1]=trechit->localPosition().y();
  //  LogDebug("SiTrackerMultiRecHitUpdatorMTF")<<  "Hit position " << trechit->localPosition(); 

  //calculate parameters outside for cycle
  AlgebraicSymMatrix22 V( asSMatrix<2>(trechit->parametersError()));  
  V *= annealing;//assume that TSOS is smoothed one
  double det=0;
  bool ierr2=!(V.Det2(det));
  double denom= (2.*M_PI*sqrt(det));
  AlgebraicSymMatrix22 W(V.Inverse(ierr));
  
  if(ierr != 0 ||ierr2) {
    edm::LogWarning("SiTrackerMultiRecHitUpdatorMTF")<<"MultiRecHitUpdator::update: W not valid!"<<std::endl;
    LogTrace("SiTrackerMultiRecHitUpdatorMTF")<<"V: "<<V<<" AnnealingFactor: "<<annealing<<std::endl;
  }
  else{
    AlgebraicVector2 tsospos;
    //substitute the loop over the hits with a loop over the tsos
    for(std::map<int,TSOS>::const_iterator imtm=mtm->filteredStates().begin(); imtm!=mtm->filteredStates().end(); imtm++) {
    

    //LogDebug("SiTrackerMultiRecHitUpdatorMTF") << "entered in the for cicle";
    //std::cout << "debug message for cicle";
    //every step a different tsos
    //here I call a constructor
    //TSOS tsos( (*(imtm->second.freeState())), imtm->second.surface(), imtm->second.surfaceSide() );
    
    //then I fill the vector with the positions
  
    tsospos[0]= imtm->second.localPosition().x();
    tsospos[1]= imtm->second.localPosition().y();
    
    //LogDebug("SiTrackerMultiRecHitUpdatorMTF")<<  "TSOS position " << imtm->second.localPosition() ; 
    
    AlgebraicVector2 r( hitpos - tsospos);

    

    //V += me.measuredError(*ihit);// result = b*V + H*C*H.T()
    
    
    double Chi2 = ROOT::Math::Similarity(r,W);// Chi2 = r.T()*W*r
    //LogDebug("SiTrackerMultiRecHitUpdatorMTF")<<"Chi2 (updatecolumn method)= "<<Chi2<<std::endl;
    
    //this is the probability of the hit respect to the track
    //float a_i = exp(-0.5*Chi2)/(2.*M_PI*sqrt(V.determinant()));
    double a_i = exp(-0.5*Chi2)/denom;
    a_sum += a_i;
    
    //    LogDebug("SiTrackerMultiRecHitUpdatorMTF")<<  "Value of a " << a_i ; 
    }
  }
  
  return a_sum;
  
}



double SiTrackerMultiRecHitUpdatorMTF::calculatecut(TransientTrackingRecHit::ConstRecHitContainer& tcomponents, 
						   TrajectoryStateOnSurface& tsos, 
						   std::vector<TransientTrackingRecHit::RecHitPointer>& updatedcomponents,
						   float annealing) const{
  
  if (tcomponents.empty()){
    LogTrace("SiTrackerMultiRecHitUpdator") 
      << "Empty components rechit vector passed to SiTrackerMultiRecHitUpdator::calculatecut";
    return -1; 
  }		
  
  if(!tsos.isValid()) {
    LogTrace("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::calculatecut: tsos NOT valid!!!";
    return -2;
  }
  
  //a loop to propagate all the hits on the same surface...
  //std::vector<TransientTrackingRecHit::RecHitPointer> updatedcomponents = updatecomponents(tcomponents,tsos,annealing);
  
  double c_sum=0;
  int ierr;
  AlgebraicVector2 tsospos;
  tsospos[0]=tsos.localPosition().x();
  tsospos[1]=tsos.localPosition().y();
  //LogTrace("SiTrackerMultiRecHitUpdator")<<  "Hit position " << tcomponents->localPosition(); 
  
  for(std::vector<TransientTrackingRecHit::RecHitPointer>::iterator ihit = updatedcomponents.begin(); ihit != updatedcomponents.end(); ihit++) {
    //    AlgebraicVector2 r(asSVector<2>((*ihit)->parameters()) - tsospos);
    AlgebraicSymMatrix22 V(asSMatrix<2>((*ihit)->parametersError()));
    V *= annealing;//assume that TSOS is smoothed one
    //V += me.measuredError(*ihit);// result = b*V + H*C*H.T()
    AlgebraicSymMatrix22 W(V.Inverse(ierr));
    double det=0;
    bool ierr2=!(V.Det2(det));
    
    if(ierr != 0||ierr2) {
      LogTrace("SiTrackerMultiRecHitUpdator")<<"MultiRecHitUpdator::calculatecut: W not valid!"<<std::endl;
      LogTrace("SiTrackerMultiRecHitUpdator")<<"V: "<<V<<" AnnealingFactor: "<<annealing<<std::endl;
    }
    else{
    //float Chi2 = W.similarity(r);// Chi2 = r.T()*W*r 
    //this is a normalization factor, to define a cut c, for dropping the probability to 0 
  LogTrace("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::calculatecut: det= "<<det<<std::endl;
    double c_i = exp(-0.5*theChi2Cut/annealing)/(2.*M_PI*sqrt(det));
    c_sum += c_i;   
    }
  }
  LogTrace("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::calculatecut: csum= "<<c_sum<<std::endl;
  return c_sum;
  
	
}

std::vector< std::pair<TransientTrackingRecHit::RecHitPointer, double> > 
SiTrackerMultiRecHitUpdatorMTF::mapmaker(TransientTrackingRecHit::ConstRecHitContainer& tcomponents,
					 TrajectoryStateOnSurface& tsos,
					 float annealing) const{
  if (tcomponents.empty()){
    LogTrace("SiTrackerMultiRecHitUpdator") 
      << "Empty components vector passed to SiTrackerMultiRecHitUpdator::update, returning an InvalidTransientRecHit ";
    // return InvalidTransientRecHit::build(0); 
  }		
  
  if(!tsos.isValid()) {
    LogTrace("SiTrackerMultiRecHitUpdator")<<"SiTrackerMultiRecHitUpdator::update: tsos NOT valid!!!, returning an InvalidTransientRecHit";
    //return InvalidTransientRecHit::build(0);
  }
  
  std::vector<TransientTrackingRecHit::RecHitPointer> updatedcomponents = updatecomponents(tcomponents, tsos,annealing);
  	
  int ierr;
  std::vector<std::pair<TransientTrackingRecHit::RecHitPointer, double> > mymap;
  
  
  AlgebraicVector2 tsospos;
  tsospos[0]=tsos.localPosition().x();
  tsospos[1]=tsos.localPosition().y();
  LogTrace("SiTrackerMultiRecHitUpdatorMTF")<<  "TSOS position " << tsos.localPosition(); 
  for(std::vector<TransientTrackingRecHit::RecHitPointer>::iterator ihit = updatedcomponents.begin(); ihit != updatedcomponents.end(); ihit++) {
    AlgebraicVector2 r(asSVector<2>((*ihit)->parameters()) - tsospos);
    AlgebraicSymMatrix22 V(asSMatrix<2>((*ihit)->parametersError()));
    V *= annealing;//assume that TSOS is smoothed one
    //V += me.measuredError(*ihit);// result = b*V + H*C*H.T()
    AlgebraicSymMatrix22 W(V.Inverse(ierr));
    double det=0;
    bool ierr2=!(V.Det2(det));
    double a_i=0;
    if(ierr != 0||ierr2) {
      LogTrace("SiTrackerMultiRecHitUpdatorMTF")<<"MultiRecHitUpdator::update: W not valid!"<<std::endl;
      LogTrace("SiTrackerMultiRecHitUpdatorMTF")<<"V: "<<V<<" AnnealingFactor: "<<annealing<<std::endl;
      
    }
    else{
      double Chi2 =ROOT::Math::Similarity(r,W);// Chi2 = r.T()*W*r
      //this is the probability of the hit respect to the track
      a_i=exp(-0.5*Chi2)/(2.*M_PI*sqrt(det));
    }
    mymap.push_back(std::pair<TransientTrackingRecHit::RecHitPointer, float>((*ihit), a_i));
    //this is a normalization factor, to define a cut c, for dropping the probability to 0 
    //a_sum += a_i;	
  }
  
  LogDebug("SiTrackerMultiRecHitUpdatorMTF") << "Map mymap size: " << mymap.size() << std::endl
					     << "Position 1st element: " << mymap.front().first->localPosition()<<std::endl;
  
  
  return mymap;
}

std::vector<TransientTrackingRecHit::RecHitPointer> 
SiTrackerMultiRecHitUpdatorMTF::updatecomponents(TransientTrackingRecHit::ConstRecHitContainer& tcomponents,
						 TrajectoryStateOnSurface& tsos,
						 float annealing) const{
  
  if (tcomponents.empty()){
    LogTrace("SiTrackerMultiRecHitUpdator") 
      << "Empty components vector passed to SiTrackerMultiRecHitUpdatorMTF::updatecomponents, returning an InvalidTransientRecHit ";
    //return InvalidTransientRecHit::build(0); 
  }		
  
  if(!tsos.isValid()) 
    {
      LogTrace("SiTrackerMultiRecHitUpdatorMTF")
	<<"SiTrackerMultiRecHitUpdatorMTF::updatecomponents: tsos NOT valid!!!, returning an InvalidTransientRecHit";
      //return InvalidTransientRecHit::build(0);
    }
  
  std::vector<TransientTrackingRecHit::RecHitPointer> updatedcomponents;
  const GeomDet* geomdet = 0;
  for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator iter = tcomponents.begin(); iter != tcomponents.end(); iter++){
    if (iter == tcomponents.begin()) {
      //to be fixed, to take into account hits coming from different modules
      if (&((*iter)->det()->surface())!=&(tsos.surface())){
	throw cms::Exception("SiTrackerMultiRecHitUpdatorMTF") 
	  << "the Trajectory state and the first rechit passed to the SiTrackerMultiRecHitUpdatorMTF lay on different surfaces!: state lays on surface " 
	  << tsos.surface().position() 
	  << "hit with detid " 
	  << (*iter)->det()->geographicalId().rawId() 
	  << "lays on surface " 
	  << (*iter)->det()->surface().position();
      }
      geomdet = (*iter)->det();
      //      LogTrace("SiTrackerMultiRecHitUpdatorMTF") << "Current reference surface located at " << geomdet->surface().position();
    }
    if (&((*iter)->det()->surface())!=&(tsos.surface())){
    TransientTrackingRecHit::RecHitPointer cloned = theHitPropagator->project<GenericProjectedRecHit2D>(*iter, *geomdet, tsos);
    LogTrace("SiTrackerMultiRecHitUpdatorMTF") << "Projecting a hit from surface " << (*iter)->det()->surface().position() 
					       << " to surface " << tsos.surface().position()  << " original global position " 
					       << (*iter)->globalPosition() << " projected position " << cloned->globalPosition();
    if (cloned->isValid()) updatedcomponents.push_back(cloned);
      } else {
    
      
      //changed to limit the use of clone method
      //if ( (*iter)->isValid() ) updatedcomponents.push_back(*iter); 
	LogTrace("SiTrackerMultiRecHitUpdatorMTF") << "about to clone the tracking rechit with the method clone " << std::endl;
	TransientTrackingRecHit::RecHitPointer cloned = (*iter)->clone(tsos);
	LogTrace("SiTrackerMultiRecHitUpdatorMTF") << "cloned";
	LogTrace("SiTrackerMultiRecHitUpdatorMTF") <<"Original position: "<<(*iter)->localPosition()<<std::endl;
	LogTrace("SiTrackerMultiRecHitUpdatorMTF") <<"New position: "<<cloned->localPosition()<<std::endl;
	
	if (cloned->isValid()) updatedcomponents.push_back(cloned);
      }
  }
  
  return updatedcomponents;
}


TransientTrackingRecHit::RecHitPointer 
SiTrackerMultiRecHitUpdatorMTF::update(double rowsum, 
				       MultiTrajectoryMeasurement* mtm, 
				       double c, 
				       std::vector<std::pair<TransientTrackingRecHit::RecHitPointer,double> >& mymap,
				       std::vector<TransientTrackingRecHit::RecHitPointer>& updatedcomponents, 
				       float annealing) const{
  
  //LogDebug("SiTrackerMultiRecHitUpdatorMTF") << "check if the map is good" << std::endl
  //				     << "size: " << mymap.size()<< std::endl;
    
   
  unsigned int counter = 0;
  TransientTrackingRecHit::ConstRecHitContainer finalcomponents;
  //create a map normalized
  std::vector<std::pair<TransientTrackingRecHit::RecHitPointer, double> > normmap;
  const GeomDet* geomdet = updatedcomponents.front()->det(); 
  LogTrace("SiTrackerMultiRecHitUpdatorMTF")<<"Multirechit size"<<updatedcomponents.size()<<std::endl;
    for(std::vector<TransientTrackingRecHit::RecHitPointer>::iterator ihit = updatedcomponents.begin(); ihit != updatedcomponents.end(); ihit++) {
     
      // LogDebug("SiTrackerMultiRechitUpdatorMTF") <<" the value of phi is  "<< mymap[counter].second << std::endl
      //					 <<" the value of totalsum is "<< total_sum << std::endl
      //					 <<" the value of the cut is "<< c << std::endl
      //					 <<" the size of the map is " << mymap.size() << std::endl
      //					 <<" printing the first element of the map" << mymap[counter].first << std::endl;
      // <<" the local position of the hit is: "
      
      double colsum = updatecolumn(*ihit, mtm, annealing);
      double total_sum = rowsum + colsum; 
      LogTrace("SiTrackerMultiRecHitUpdatorMTF")<<"rowsum= "<<rowsum<<", colsum= "<<colsum<<std::endl;
      LogTrace("SiTrackerMultiRecHitUpdatorMTF")<<"phi_i_j= "<<mymap[counter].second<<", c="<<c<<std::endl;
      //float p = ( (mymap[counter].second)/(total_sum - mymap[counter].second ) > 1.e-6 ? (mymap[counter].second)/(total_sum- mymap[counter].second) : 1.e-6);
            double p = ( (mymap[counter].second)/(total_sum - mymap[counter].second + c) > 1.e-6 ? (mymap[counter].second)/(total_sum- mymap[counter].second +c) : 1.e-6);
      
      //    LogDebug("SiTrackerMultiRechitUpdatorMTF") << " the probability of this hit is "
      //						 << p << std::endl;
      
      //float p = ((mymap[counter].second)/total_sum > 0.01 ? (mymap[counter].second)/total_sum : 1.e-6);
      normmap.push_back(std::pair<TransientTrackingRecHit::RecHitPointer, double>(mymap[counter].first, p) );
      //LogDebug("SiTrackerMultiRechitUpdatorMTF") <<"stored the pair <hit,p> in the map "
      //					 <<std::endl;
      
      
      
      //let's store the weight in the component TransientTrackingRecHit too
      (*ihit)->setWeight(p);
      //      LogDebug("SiTrackerMultiRechitUpdatorMTF")<<"Weight set"<<std::endl;
      
      (*ihit)->setAnnealingFactor(annealing);
      // LogDebug("SiTrackerMultiRechitUpdatorMTF")<<"Annealing factor set"<<std::endl;
      
      finalcomponents.push_back(*ihit);	
      //LogDebug("SiTrackerMultiRechitUpdatorMTF")<<"Component pushed back"<<std::endl;
      
      //LogDebug("SiTrackerMultiRecHitUpdatorMTF")<<"FinalComponents size: "<<finalcomponents.size()<<std::endl;
      
       LogDebug("SiTrackerMultiRecHitUpdatorMTF")<<"Component hit type "<< typeid(*(normmap[counter].first)).name() <<std::endl
	
      << "position " << mymap[counter].first->localPosition()<<std::endl
      << "error " << mymap[counter].first->localPositionError()<<std::endl
      << "with weight " <<p<<std::endl;
      
      counter++;
    
    }
  
    
    
    mymap = normmap;
    SiTrackerMultiRecHitUpdatorMTF::LocalParameters param=calcParameters(finalcomponents);

    //    LocalError er = calcParametersError(finalcomponents);
    
    //LocalPoint pos  = calcParameters(finalcomponents, er);
    
    std::vector<std::pair<const TrackingRecHit*,float> > newmap;


    for(std::vector<std::pair<TransientTrackingRecHit::RecHitPointer,double> >::iterator imap=mymap.begin(); imap!=mymap.end(); imap++)
      {
	
	newmap.push_back(std::pair<const TrackingRecHit*, float>(imap->first->hit(),imap->second));
      
      }
    
    
    

    
    SiTrackerMultiRecHit updatedmrh(param.first,param.second, newmap.front().first->geographicalId(), newmap);
    
    LogDebug("SiTrackerMultiRecHitUpdatorMTF") << "SiTrackerMultiRecHit built " << std::endl;
    
    LogDebug("SiTrackerMultiRecHitUpdatorMTF") << "Updated Hit position " 
					       << updatedmrh.localPosition() << " updated error " 
					       << updatedmrh.parametersError() << std::endl;
    
    //return new SiTrackerMultiRecHit(normmap); 	
    
    return TSiTrackerMultiRecHit::build(geomdet, &updatedmrh, finalcomponents, annealing);
    
}

SiTrackerMultiRecHitUpdatorMTF::LocalParameters SiTrackerMultiRecHitUpdatorMTF::calcParameters(TransientTrackingRecHit::ConstRecHitContainer& map)const{
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


LocalError SiTrackerMultiRecHitUpdatorMTF::calcParametersError(TransientTrackingRecHit::ConstRecHitContainer& map) const {
  AlgebraicSymMatrix22 W_sum;
        int ierr;
        for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator ihit = map.begin(); ihit != map.end(); ihit ++) {
	  AlgebraicSymMatrix22 V(asSMatrix<2>((*ihit)->parametersError()));
	  AlgebraicSymMatrix22 W(V.Inverse(ierr));

	  if(ierr != 0) {
	    std::cout<<"MultiRecHit::checkParametersError: W not valid!"<<std::endl;
	  }

	  else W_sum += ((*ihit)->weight()*W);
        }
        AlgebraicSymMatrix22 parametersError = W_sum.Inverse(ierr);
        return LocalError(parametersError(0,0), parametersError(0,1), parametersError(1,1));
} 

LocalPoint SiTrackerMultiRecHitUpdatorMTF::calcParameters(TransientTrackingRecHit::ConstRecHitContainer& map, const LocalError& er) const {
  AlgebraicVector2 m_sum;
  int ierr;
  for( TransientTrackingRecHit::ConstRecHitContainer::const_iterator ihit = map.begin(); ihit != map.end(); ihit ++) {
    AlgebraicVector2 m(asSVector<2>((*ihit)->parameters()));
    AlgebraicSymMatrix22 V(asSMatrix<2>((*ihit)->parametersError()));
    AlgebraicSymMatrix22 W(V.Inverse(ierr));
    
    if(ierr != 0) {
      std::cout<<"MultiRecHit::checkParameters: W not valid!"<<std::endl;
    }
    
    //m_sum += ihit->weight()*(W*m);      
    else    m_sum += ((*ihit)->weight()*(W*m));
  }
  AlgebraicSymMatrix22 V_sum;
	
  V_sum(0,0) = er.xx();
  V_sum(0,1) = er.xy();
  V_sum(1,1) = er.yy();
  //AlgebraicSymMatrix V_sum(parametersError());
  AlgebraicVector2 parameters = V_sum*m_sum;
  return LocalPoint(parameters(0), parameters(1));
}

