
#include "DataFormats/GeometrySurface/interface/Line.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/TertiaryTracksVertexFinder/interface/GetLineCovMatrix.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalTrajectoryExtrapolatorToLine.h"

#include "RecoVertex/TertiaryTracksVertexFinder/interface/AddTvTrack.h"

using namespace std;
using namespace reco;

AddTvTrack::AddTvTrack( vector<TransientVertex> *PrimaryVertices, vector<TransientVertex> *SecondaryVertices, double maxSigOnDistTrackToB) 
{
  thePrimaryVertices    = PrimaryVertices;
  theSecondaryVertices  = SecondaryVertices;

  MaxSigOnDistTrackToB  = maxSigOnDistTrackToB; 

 // why is the argument passed in the ctor overwritten here ???
  MaxSigOnDistTrackToB = 10.0;

  // min. transverse impact parameter significance
  theIPSig = 1.5; // should this be 2 or 1.5 ???

}

//-----------------------------------------------------------------------------

vector<TransientVertex> AddTvTrack::getSecondaryVertices(const vector<TransientTrack> & unusedTracks ) {

  VertexDistance3D theVertexDistance3D; 

  theTrackInfoVector.clear();

  // this was commented out in ORCA implementation
  // Measurement1D PvSvMeasurement = theVertexDistance3D.distance( (*thePrimaryVertices)[0],  (*theSecondaryVertices)[0] );
  // if( PvSvMeasurement.value() < 0.05 ) return *theSecondaryVertices;

  // first, filter tracks on min. IP significance -----------------------------

  vector<TransientTrack> unusedTrackswithIPSig; // filtered Tracks on IPsig
  for( vector<TransientTrack>::const_iterator itT = unusedTracks.begin() ; 
    itT != unusedTracks.end() ; itT++ ) { 

    GlobalPoint vtxPoint((*thePrimaryVertices)[0].position());

    TrajectoryStateClosestToPoint tscp=(*itT).trajectoryStateClosestToPoint(vtxPoint);
    double val=tscp.perigeeParameters().transverseImpactParameter();
    double error=sqrt((tscp.perigeeError().covarianceMatrix_old())[3][3]);

    if (debug) cout <<"[AddTvTrack] tip val,err"<<val<<","<<error<<endl;

    //double val= 0.; //(*itT).transverseImpactParameter().value();
    //double error= 9999.; //(*itT).transverseImpactParameter().error();

    double err=sqrt(pow(error,2)+pow(0.0015,2));
    if( abs(val/err)>theIPSig ) unusedTrackswithIPSig.push_back(*itT);  
  }

  if (debug) cout<<"[AddTvTrack] tracks surviving IPSig cut: "
                 <<unusedTrackswithIPSig.size()<<endl;

  // if no tracks survived, we are done already
  if( unusedTrackswithIPSig.empty() )  return *theSecondaryVertices;  

  // construct b-flight line (from PV to SV)
  GlobalPoint  PVposition((*thePrimaryVertices)[0].position());
  double px = (*theSecondaryVertices)[0].position().x() 
              - (*thePrimaryVertices)[0].position().x();
  double py = (*theSecondaryVertices)[0].position().y()
              - (*thePrimaryVertices)[0].position().y();
  double pz = (*theSecondaryVertices)[0].position().z()
              - (*thePrimaryVertices)[0].position().z();
  GlobalVector bVector(px, py, pz); 
  Line bFlightLine( PVposition , bVector);           

  // and the corresponding cov matrix                  
  GetLineCovMatrix MyLineCovMatrix( PVposition, 
    (*theSecondaryVertices)[0].position(), 
    (*thePrimaryVertices)[0].positionError(), 
    (*theSecondaryVertices)[0].positionError() );  

  // the tracks to be added to the SV
  vector<TransientTrack> TracksToAdd;

  // main loop over tracks 
  for( vector<TransientTrack>::const_iterator itT = 
    unusedTrackswithIPSig.begin() ; itT != unusedTrackswithIPSig.end() ; 
    itT++ ) {  
    try {  

      // point on track closest to b flight line 

      // ORCA implementation

      // get closest Point to bFlightLine on TransientTrack   
      // TrajectoryStateOnSurface MyTransientTrackTrajectory = 
      //   (*itT).stateAtLine(bFlightLine); 
      // Point on  TransientTrack closest to bFlightLine 
      // GlobalPoint GlobalPoint2=MyTransientTrackTrajectory.globalPosition(); 

      // begin lenghy orca translation - - - - - - - - - - - - - - - - - - - - 

      TransientTrack track = *itT;
      AnalyticalTrajectoryExtrapolatorToLine TETL(track.field());
      TrajectoryStateOnSurface MyTransientTrackTrajectory;

      // approach from impact point first
      // create a fts without errors for faster propagation
      FreeTrajectoryState fastFts(
        track.impactPointState().freeTrajectoryState()->parameters());
      TrajectoryStateOnSurface cptl = TETL.extrapolate(fastFts, bFlightLine);
      // extrapolate from closest measurement
      if (cptl.isValid()) {
        // FreeTrajectoryState* fts = 
        //   theTrack.closestState(cptl.globalPosition()).freeState();
        // transienttrack::closestState() not provided in CMSSW, do by hand
        FreeTrajectoryState fts;
        GlobalVector d1 = track.innermostMeasurementState().globalPosition()
                         -cptl.globalPosition();
        GlobalVector d2 = track.outermostMeasurementState().globalPosition()
                         -cptl.globalPosition();
        if (d1.mag()<d2.mag()) 
          fts=*(track.innermostMeasurementState().freeState());
	else                     
	  fts=*(track.outermostMeasurementState().freeState());
        TrajectoryStateOnSurface cptl2 = TETL.extrapolate(fts, bFlightLine);
        MyTransientTrackTrajectory=cptl2;
      } 
      else { 
        MyTransientTrackTrajectory=cptl;
      } 

      // Point on  TransientTrack closest to bFlightLine  
      GlobalPoint GlobalPoint2 = MyTransientTrackTrajectory.globalPosition();  

      // end lengthy orca translation - - - - - - - - - - - - - - - - - - - - -

      GlobalVector VecDistTrackbFlightLine=bFlightLine.distance(GlobalPoint2);
      double X = GlobalPoint2.x() + VecDistTrackbFlightLine.x();
      double Y = GlobalPoint2.y() + VecDistTrackbFlightLine.y();
      double Z = GlobalPoint2.z() + VecDistTrackbFlightLine.z();
      // Point on  bFlightLine closest to TransientTrack   
      GlobalPoint GlobalPoint1( X , Y , Z );  

      pair<GlobalPoint,GlobalPoint> TheTwoPoints(GlobalPoint1, GlobalPoint2);
      // TheTwoPoints.first : Point on b-flight-trajectory
      // TheTwoPoints.second: Point on TransientTrack = pseudo tertiary vertex
    
      // Get Error of  b-flight-trajectory at TheTwoPoints.first
      GlobalError ErrOnBFlightTraj = 
        MyLineCovMatrix.GetMatrix( TheTwoPoints.first );  

      // used to build pseudo vtx      
      vector<TransientTrack> emptyVectorTracks;  

      VertexState theState1 (TheTwoPoints.first , ErrOnBFlightTraj );  
      TransientVertex V1(theState1 , emptyVectorTracks , 0.0 ) ;
      VertexState theState2 ( TheTwoPoints.second, 
        MyTransientTrackTrajectory.cartesianError().position() ); 
      TransientVertex V2(theState2 , emptyVectorTracks, 0.0 ) ;
      
      //VertexDistance3D theVertexDistance3D;   // Distance of the point on the b-flight-trajectory and the point on the TransientTrack                 changed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

      Measurement1D TheMeasurement = theVertexDistance3D.distance( V1, V2 );
      double Sig = TheMeasurement.significance(); 

      // check if point on b-flight-trajectory is in the correct hemisphere
      bool HemisphereOk = false;  
      if ( abs( (*thePrimaryVertices)[0].position().x() 
		- (*theSecondaryVertices)[0].position().x() ) >  
           abs( (*thePrimaryVertices)[0].position().y() 
		- (*theSecondaryVertices)[0].position().y() )   ) { 
	  if ( (*thePrimaryVertices)[0].position().x() 
	       < (*theSecondaryVertices)[0].position().x()  &&  
	       (*thePrimaryVertices)[0].position().x() 
	       < TheTwoPoints.first.x() )  HemisphereOk = true;
	  if ( (*thePrimaryVertices)[0].position().x() 
	       > (*theSecondaryVertices)[0].position().x()  &&  
	       (*thePrimaryVertices)[0].position().x() 
	       > TheTwoPoints.first.x() )  HemisphereOk = true;
	}
      else {
	  if ( (*thePrimaryVertices)[0].position().y() 
	       < (*theSecondaryVertices)[0].position().y()  &&  
	       (*thePrimaryVertices)[0].position().y() 
	       < TheTwoPoints.first.y() )  HemisphereOk = true;
	  if ( (*thePrimaryVertices)[0].position().y() 
	       > (*theSecondaryVertices)[0].position().y()  &&  
	       (*thePrimaryVertices)[0].position().y() 
	       > TheTwoPoints.first.y() )  HemisphereOk = true;
	}

      double HemisOK = 0.0;
      if( HemisphereOk ) HemisOK = 1.0; 
 
      // start Max Dist Cut
      //double RDistTheTwoPointsFirst = sqrt( TheTwoPoints.second.x() * TheTwoPoints.second.x() + TheTwoPoints.second.y() * TheTwoPoints.second.y() );   // changed first to second !!!!
      //if( RDistTheTwoPointsFirst >= 2.0 ) HemisphereOk = false; 
      // end Max Dist Cut

      // distance to PV      
      double Dist_X=TheTwoPoints.second.x()
	-(*thePrimaryVertices)[0].position().x();
      double Dist_Y=TheTwoPoints.second.y()
	-(*thePrimaryVertices)[0].position().y();
      double Dist_Z=TheTwoPoints.second.z()
	-(*thePrimaryVertices)[0].position().z();
      double Dist_to_PV = sqrt(Dist_X*Dist_X + Dist_Y*Dist_Y + Dist_Z*Dist_Z);
                                                                       
      //if( Dist_to_PV < 0.2 ) HemisphereOk = false;   !! changed
      //if( Dist_to_PV < 0.2  &&   Dist_to_PV >= 2.0 ) HemisphereOk = false; 
      //if( Dist_to_PV < 0.01 ) HemisphereOk = false;    

      // distance to beam      
      Dist_X = TheTwoPoints.second.x();  
      Dist_Y = TheTwoPoints.second.y();
      double Dist_to_Beam = sqrt(Dist_X*Dist_X + Dist_Y*Dist_Y); 
      if( Dist_to_Beam < 0.01  ||  Dist_to_Beam > 2.5 ) HemisphereOk = false; 
      // !!!!!!!!!!!!!!!!!!!!! changed !!   0.2 -> 0.01

      // add track      
      if(Sig<MaxSigOnDistTrackToB && HemisphereOk) TracksToAdd.push_back(*itT);

      // start TDR INFO -------------------------------------------------------

      //double IP= 0.; //(*itT).transverseImpactParameter().value();
      //double error= 999.; //(*itT).transverseImpactParameter().error();

      GlobalPoint vtxPoint((*thePrimaryVertices)[0].position());
      TrajectoryStateClosestToPoint tscp=(*itT).trajectoryStateClosestToPoint(vtxPoint);
      double IP=tscp.perigeeParameters().transverseImpactParameter();
      double error=sqrt((tscp.perigeeError().covarianceMatrix_old())[3][3]);
      double err=sqrt(pow(error,2)+pow(0.0015,2));
      double IPSig = IP/err;

      double InfArray[7];
      InfArray[0] = Sig;
      InfArray[1] = Dist_to_Beam;
      InfArray[2] = Dist_to_PV;
      InfArray[3] = HemisOK;
      InfArray[4] = IP;
      InfArray[5] = IPSig;
      if (Sig > 10.0) Sig = 10.0 ;
      if ( HemisphereOk ) InfArray[6]=Sig;
      else InfArray[6]=-Sig;

      theTrackInfoVector.push_back(TrackInfo(&*itT,InfArray));

      //pair<TransientTrack,double* > TrackInfoToPush2( (*itT) , InfArray ); 
      //TrackInfo2.push_back( TrackInfoToPush2 );
      
      //if (Sig > 10.0) Sig = 10.0 ;
      
      //if ( HemisphereOk ) {
      //	pair<TransientTrack,double> TrackInfoToPush( (*itT) , Sig );
      //	TrackInfo.push_back( TrackInfoToPush );
      //}
      //else {
      //	pair<TransientTrack,double> TrackInfoToPush( (*itT) , -Sig );
      //TrackInfo.push_back( TrackInfoToPush );
      //}


      // end TDR INFO ---------------------------------------------------------
   
    } // end try block 
    catch(...) {
      cout << " [AddTvTrack]::getSecondaryVertices throws exception " << endl;
    }

  } // end  main loop over tracks

  //TracksToAdd.clear();  // for using as PVR  

  if (debug) cout <<"[AddTvTrack] tracks to add: "<<TracksToAdd.size()<<endl;

  if( ! TracksToAdd.empty() ) {

    // build new Vertex, position and CovMatrix are not changed
    vector<TransientVertex> NewSecondaryVertices = *theSecondaryVertices;
    vector<TransientTrack>  VertexTracks = 
      NewSecondaryVertices[0].originalTracks(); 

    VertexState theState ( NewSecondaryVertices[0].position(), 
			   NewSecondaryVertices[0].positionError() );  

    // map<TransientTrack, float, ltrt> TrackWeightMap;
    TransientTrackToFloatMap TrackWeightMap;

    // extend old TrackWeightMap
    if( NewSecondaryVertices[0].hasTrackWeight() ) {  
      TrackWeightMap = NewSecondaryVertices[0].weightMap() ;  
      //map<TransientTrack,float,ltrt>::iterator itMapEnd=TrackWeightMap.end();
      TransientTrackToFloatMap::iterator itMapEnd = TrackWeightMap.end();
      for(vector<TransientTrack>::const_iterator itT = TracksToAdd.begin();
	  itT != TracksToAdd.end(); itT++) {
	// insert weight 0.0 for new Tracks because they are not used 
	// for refitting the TransientVertex
	pair< TransientTrack , float > TrackMapPair( (*itT) , 0.0);  
	itMapEnd = TrackWeightMap.insert( itMapEnd, TrackMapPair ); 
      }
    }
    // build  build TrackWeightMap
    else { 
      //map<TransientTrack,float,ltrt>::iterator itMapEnd=TrackWeightMap.end();
      TransientTrackToFloatMap::iterator itMapEnd = TrackWeightMap.end();
      for(vector<TransientTrack>::const_iterator itT = VertexTracks.begin(); 
	  itT != VertexTracks.end(); itT++) {
        // insert weight 1.0 for original Tracks because they are used
	// for fitting the TransientVertex
	pair< TransientTrack , float > TrackMapPair( (*itT) , 1.0);   
	itMapEnd = TrackWeightMap.insert( itMapEnd, TrackMapPair );       
      }
      for(vector<TransientTrack>::const_iterator itT = TracksToAdd.begin(); 
	  itT != TracksToAdd.end(); itT++) {
	// insert weight 0.0 for new Tracks because they are not used 
	// for refitting the TransientVertex
	pair< TransientTrack , float > TrackMapPair( (*itT) , 0.0);   
	itMapEnd = TrackWeightMap.insert( itMapEnd, TrackMapPair );       
      }
    }

    for(vector<TransientTrack>::const_iterator itT = TracksToAdd.begin(); 
	itT!=TracksToAdd.end(); itT++)
      VertexTracks.push_back(*itT);     

    //TransientVertex NewVertex(theState , VertexTracks, 
    //  NewSecondaryVertices[0].totalChiSquared(), 
    //  NewSecondaryVertices[0].degreesOfFreedom(), TrackWeightMap) ;

    TransientVertex NewVertex(theState , VertexTracks, 
      NewSecondaryVertices[0].totalChiSquared(), 
      NewSecondaryVertices[0].degreesOfFreedom()); 
    NewVertex.weightMap(TrackWeightMap);

    // end build new Vertex

    NewSecondaryVertices[0] = NewVertex;      
    return NewSecondaryVertices;
  }
  
  return *theSecondaryVertices;
}
