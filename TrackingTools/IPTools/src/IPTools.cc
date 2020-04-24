#include "DataFormats/GeometrySurface/interface/Line.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalTrajectoryExtrapolatorToLine.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Matrix/Vector.h"
#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"


namespace IPTools
{
  using namespace std;
  using namespace reco;


    std::pair<bool,Measurement1D> absoluteImpactParameter(const TrajectoryStateOnSurface & tsos  , const  reco::Vertex & vertex, VertexDistance & distanceComputer) {
        if(!tsos.isValid()) {
         return pair<bool,Measurement1D>(false,Measurement1D(0.,0.)) ;
        }
        GlobalPoint refPoint          = tsos.globalPosition();
        GlobalError refPointErr       = tsos.cartesianError().position();
        GlobalPoint vertexPosition    = RecoVertex::convertPos(vertex.position());
        GlobalError vertexPositionErr = RecoVertex::convertError(vertex.error());
        return pair<bool,Measurement1D>(true,distanceComputer.distance(VertexState(vertexPosition,vertexPositionErr), VertexState(refPoint, refPointErr)));
   }

    std::pair<bool,Measurement1D> absoluteImpactParameter3D(const reco::TransientTrack & transientTrack, const  reco::Vertex & vertex)
    {
      AnalyticalImpactPointExtrapolator extrapolator(transientTrack.field());
      VertexDistance3D dist;
      return absoluteImpactParameter(extrapolator.extrapolate(transientTrack.impactPointState(), RecoVertex::convertPos(vertex.position())), vertex, dist);
    }
    std::pair<bool,Measurement1D> absoluteTransverseImpactParameter(const reco::TransientTrack & transientTrack, const  reco::Vertex & vertex)
    {
      TransverseImpactPointExtrapolator extrapolator(transientTrack.field());
      VertexDistanceXY dist;
      return absoluteImpactParameter(extrapolator.extrapolate(transientTrack.impactPointState(), RecoVertex::convertPos(vertex.position())), vertex,  dist);
    }
    
  pair<bool,Measurement1D> signedTransverseImpactParameter(const TransientTrack & track,
                                                           const GlobalVector & direction, const  Vertex & vertex){
    //Extrapolate to closest point on transverse plane
    TransverseImpactPointExtrapolator extrapolator(track.field());
    TrajectoryStateOnSurface closestOnTransversePlaneState = extrapolator.extrapolate(track.impactPointState(),RecoVertex::convertPos(vertex.position()));
    
    //Compute absolute value
    VertexDistanceXY dist;
    pair<bool,Measurement1D> result = absoluteImpactParameter(closestOnTransversePlaneState, vertex,  dist);
    if(!result.first) return result;

    //Compute Sign
    GlobalPoint impactPoint    = closestOnTransversePlaneState.globalPosition();
    GlobalVector IPVec(impactPoint.x()-vertex.x(),impactPoint.y()-vertex.y(),0.);
    double prod = IPVec.dot(direction);
    double sign = (prod>=0) ? 1. : -1.;
    
    //Apply sign to the result
    return pair<bool,Measurement1D>(result.first,Measurement1D(sign*result.second.value(), result.second.error()));
  }

  pair<bool,Measurement1D> signedImpactParameter3D(const TransientTrack & track,
                                                           const GlobalVector & direction, const  Vertex & vertex){
    //Extrapolate to closest point on transverse plane
    AnalyticalImpactPointExtrapolator extrapolator(track.field());
    TrajectoryStateOnSurface closestIn3DSpaceState = extrapolator.extrapolate(track.impactPointState(),RecoVertex::convertPos(vertex.position()));

    //Compute absolute value
    VertexDistance3D dist;
    pair<bool,Measurement1D> result = absoluteImpactParameter(closestIn3DSpaceState, vertex,  dist);
    if(!result.first) return result;

    //Compute Sign
    GlobalPoint impactPoint = closestIn3DSpaceState.globalPosition();
    GlobalVector IPVec(impactPoint.x()-vertex.x(),impactPoint.y()-vertex.y(),impactPoint.z()-vertex.z());
    double prod = IPVec.dot(direction);
    double sign = (prod>=0) ? 1. : -1.;

    //Apply sign to the result
    return pair<bool,Measurement1D>(result.first,Measurement1D(sign*result.second.value(), result.second.error()));
  }



  pair<bool,Measurement1D> signedDecayLength3D(const   TrajectoryStateOnSurface & closestToJetState,
					       const GlobalVector & direction, const  Vertex & vertex)  {

    //Check if extrapolation has been successfull
    if(!closestToJetState.isValid()) {
      return pair<bool,Measurement1D>(false,Measurement1D(0.,0.));
    }

    GlobalVector jetDirection = direction.unit();
    GlobalPoint vertexPosition(vertex.x(),vertex.y(),vertex.z());
  
    double decayLen = jetDirection.dot(closestToJetState.globalPosition()-vertexPosition);

    //error calculation

    AlgebraicVector3 j;
    j[0] = jetDirection.x();
    j[1] = jetDirection.y();
    j[2] = jetDirection.z();
    AlgebraicVector6 jj;
    jj[0] = jetDirection.x();
    jj[1] = jetDirection.y();
    jj[2] = jetDirection.z();
    jj[3] =0.;
    jj[4] =0.;
    jj[5] =0.;///chech it!!!!!!!!!!!!!!!!!!!!!!!
     
    //TODO: FIXME: the extrapolation uncertainty is very relevant here should be taken into account!!
    double trackError2 = ROOT::Math::Similarity(jj,closestToJetState.cartesianError().matrix());
    double vertexError2 = ROOT::Math::Similarity(j,vertex.covariance());

    double decayLenError = sqrt(trackError2+vertexError2);
    
    return pair<bool,Measurement1D>(true,Measurement1D(decayLen,decayLenError));

  }


    
  pair<bool,Measurement1D> linearizedSignedImpactParameter3D(const   TrajectoryStateOnSurface & closestToJetState ,
						   const GlobalVector & direction, const  Vertex & vertex)
  {
    //Check if extrapolation has been successfull
    if(!closestToJetState.isValid()) {
      return pair<bool,Measurement1D>(false,Measurement1D(0.,0.));
    }

    GlobalPoint vertexPosition(vertex.x(),vertex.y(),vertex.z());
    GlobalVector impactParameter = linearImpactParameter(closestToJetState, vertexPosition);
    GlobalVector jetDir = direction.unit();
    GlobalVector flightDistance(closestToJetState.globalPosition()-vertexPosition);
    double theDistanceAlongJetAxis = jetDir.dot(flightDistance);
    double signedIP = impactParameter.mag()*((theDistanceAlongJetAxis!=0)?theDistanceAlongJetAxis/abs(theDistanceAlongJetAxis):1.);


    GlobalVector ipDirection = impactParameter.unit();
    //GlobalPoint closestPoint = closestToJetState.globalPosition();
    GlobalVector momentumAtClosestPoint = closestToJetState.globalMomentum(); 
    GlobalVector momentumDir = momentumAtClosestPoint.unit();
    
    AlgebraicVector3 deriv_v;
    deriv_v[0] = - ipDirection.x();
    deriv_v[1] = - ipDirection.y();
    deriv_v[2] = - ipDirection.z();

 
    AlgebraicVector6 deriv;
    deriv[0] = ipDirection.x();
    deriv[1] = ipDirection.y();
    deriv[2] = ipDirection.z();
    deriv[3] = -  (momentumDir.dot(flightDistance)*ipDirection.x())/momentumAtClosestPoint.mag();
    deriv[4] = -  (momentumDir.dot(flightDistance)*ipDirection.y())/momentumAtClosestPoint.mag();
    deriv[5] = -  (momentumDir.dot(flightDistance)*ipDirection.z())/momentumAtClosestPoint.mag();

    double trackError2 = ROOT::Math::Similarity(deriv , closestToJetState.cartesianError().matrix());
    double vertexError2 = ROOT::Math::Similarity(deriv_v , vertex.covariance());
    double ipError = sqrt(trackError2+vertexError2);

    return pair<bool,Measurement1D>(true,Measurement1D(signedIP,ipError));
  }



  TrajectoryStateOnSurface closestApproachToJet(const TrajectoryStateOnSurface & state,const Vertex & vertex, const GlobalVector& direction,const MagneticField * field) {
  
    Line::PositionType pos(GlobalPoint(vertex.x(),vertex.y(),vertex.z()));
    Line::DirectionType dir(direction.unit());
    Line jetLine(pos,dir);
  
    AnalyticalTrajectoryExtrapolatorToLine extrapolator(field);

    return extrapolator.extrapolate(state, jetLine);
  }

  /**
   * Compute the impact parameter of a track, linearized from the given state, with respect to a given point 
   */
  GlobalVector linearImpactParameter(const TrajectoryStateOnSurface & state, const GlobalPoint & point)  {

    Line::PositionType pos(state.globalPosition());
    Line::DirectionType dir((state.globalMomentum()).unit());
    Line trackLine(pos,dir);
    const GlobalPoint&  tmp=point; 
    return  trackLine.distance(tmp);
  }

  pair<double,Measurement1D> jetTrackDistance(const TransientTrack & track, const GlobalVector & direction, const Vertex & vertex) {
    double  theLDist_err(0.);
  
    //FIXME
    float weight=0.;//vertex.trackWeight(track);

    TrajectoryStateOnSurface stateAtOrigin = track.impactPointState(); 
    if(!stateAtOrigin.isValid())
      {
	//TODO: throw instead?
	return pair<bool,Measurement1D>(false,Measurement1D(0.,0.));
      }
   
    //get the Track line at origin
    Line::PositionType posTrack(stateAtOrigin.globalPosition());
    Line::DirectionType dirTrack((stateAtOrigin.globalMomentum()).unit());
    Line trackLine(posTrack,dirTrack);
    // get the Jet  line 
    // Vertex vertex(vertex);
    GlobalVector jetVector = direction.unit();    
    Line::PositionType posJet(GlobalPoint(vertex.x(),vertex.y(),vertex.z()));
    Line::DirectionType dirJet(jetVector);
    Line jetLine(posJet,dirJet);
  
    // now compute the distance between the two lines
    // If the track has been used to refit the Primary vertex then sign it positively, otherwise negative
    double theDistanceToJetAxis = (jetLine.distance(trackLine)).mag();
    if (weight<1) theDistanceToJetAxis= -theDistanceToJetAxis;

    // ... and the flight distance along the Jet axis.
    GlobalPoint  V = jetLine.position();    
    GlobalVector Q = dirTrack - jetVector.dot(dirTrack) * jetVector;
    GlobalVector P = jetVector - jetVector.dot(dirTrack) * dirTrack;
    double theDistanceAlongJetAxis = P.dot(V-posTrack)/Q.dot(dirTrack);

    //
    // get the covariance matrix of the vertex and compute the error on theDistanceToJetAxis
    //
    
    ////AlgebraicSymMatrix vertexError = vertex.positionError().matrix();

    // build the vector of closest approach between lines


    //FIXME: error not computed.
    GlobalVector H((jetVector.cross(dirTrack).unit()));
    CLHEP::HepVector Hh(3);
    Hh[0] = H.x();
    Hh[1] = H.y();
    Hh[2] = H.z();
    
    //  theLDist_err = sqrt(vertexError.similarity(Hh));

    //    cout << "distance to jet axis : "<< theDistanceToJetAxis <<" and error : "<< theLDist_err<<endl;
    // Now the impact parameter ...

    /*    GlobalPoint T0 = track.position();
	  GlobalVector D = (T0-V)- (T0-V).dot(dir) * dir;
	  double IP = D.mag();    
	  GlobalVector Dold = distance(aTSOS, aJet.vertex(), jetDirection);
	  double IPold = Dold.mag();
    */



  
    Measurement1D DTJA(theDistanceToJetAxis,theLDist_err);
  
    return pair<double,Measurement1D> (theDistanceAlongJetAxis,DTJA);
  }

}
