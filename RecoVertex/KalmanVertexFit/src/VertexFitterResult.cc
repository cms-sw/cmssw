#include "RecoVertex/KalmanVertexFit/interface/VertexFitterResult.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/PatternTools/interface/trackingParametersAtClosestApproachToBeamSpot.h"

using namespace reco;
using namespace std;

VertexFitterResult::VertexFitterResult(const int maxTracks, const MagneticField* magField)
  : theMagField(magField)
{
  theMaxTracks = maxTracks;
  if (theMagField==nullptr) theMaxTracks=0;
  for ( int i=0; i<5; i++ ) {
    if ( maxTracks>0 ) {
      simPars[i] = new float[maxTracks];
      recPars[i] = new float[maxTracks];
      refPars[i] = new float[maxTracks];
      recErrs[i] = new float[maxTracks];
      refErrs[i] = new float[maxTracks];
    }
    else {
      simPars[i] = 0;
      recPars[i] = 0;
      refPars[i] = 0;
      recErrs[i] = 0;
      refErrs[i] = 0;
    }
  }
  trackWeight = new float[maxTracks];
  simIndex = new int[maxTracks];
  recIndex = new int[maxTracks];
  numberOfRecTracks=theMaxTracks;
  numberOfSimTracks=theMaxTracks;
  reset();
}

VertexFitterResult::~VertexFitterResult()
{
    //
    // delete arrays
    //
    for ( int i=0; i<5; i++ ) {
      delete [] simPars[i];
      delete [] recPars[i];
      delete [] refPars[i];
      delete [] recErrs[i];
      delete [] refErrs[i];
    }
    delete trackWeight;
    delete simIndex;
    delete recIndex;
}

void VertexFitterResult::fill(const TransientVertex & recVertex,
	const TrackingVertex * simv, reco::RecoToSimCollection *recSimColl,
	const float &time) 
{
  TTrackCont recTrackV;
  if (recVertex.isValid()) recTrackV = recVertex.originalTracks();
  fill(recVertex, recTrackV, simv, recSimColl, time);
}

void VertexFitterResult::fill(const TransientVertex & recVertex, 
	const TTrackCont & recTrackV, const TrackingVertex * simv,
	reco::RecoToSimCollection *recSimColl, const float &time)
{
  TrackingParticleRefVector simTrackV;

  Basic3DVector<double> vert;
  if (recVertex.isValid()) {
    recPos[0] = recVertex.position().x();
    recPos[1] = recVertex.position().y();
    recPos[2] = recVertex.position().z();

    recErr[0] = sqrt(recVertex.positionError().cxx());
    recErr[1] = sqrt(recVertex.positionError().cyy());
    recErr[2] = sqrt(recVertex.positionError().czz());
    vert = (Basic3DVector<double>) recVertex.position();

    chi[0] = recVertex.totalChiSquared();
    chi[1] = recVertex.degreesOfFreedom();
    chi[2] = ChiSquaredProbability(recVertex.totalChiSquared(), 
					   recVertex.degreesOfFreedom());
    vertex = 2;
    fitTime = time;
    tracks[1] = recVertex.originalTracks().size();
  }

  if (simv!=0) {
    simPos[0] = simv->position().x();
    simPos[1] = simv->position().y();
    simPos[2] = simv->position().z();

    simTrackV = simv->daughterTracks();
    vertex += 1;
    for(TrackingVertex::tp_iterator simTrack = simv->daughterTracks_begin();
                 (simTrack != simv->daughterTracks_end() && (numberOfSimTracks<theMaxTracks));
		 simTrack++) {
      
      Basic3DVector<double> momAtVtx((**simTrack).momentum());

      std::pair<bool, reco::TrackBase::ParameterVector> paramPair =
	reco::trackingParametersAtClosestApproachToBeamSpot(vert, momAtVtx, 
                                                            (float) (**simTrack).charge(), 
                                                            *theMagField, 
                                                            recTrackV.front().stateAtBeamLine().beamSpot());
        if (paramPair.first) {
	  fillParameters(paramPair.second, simPars, numberOfSimTracks);
	  simIndex[numberOfSimTracks] = -1;
	  ++numberOfSimTracks;
        }
    }
    tracks[0] = numberOfSimTracks;
  }


  // now store all the recTrack...  

  for(TTrackCont::const_iterator recTrack =recTrackV.begin();
               (recTrack != recTrackV.end() 
		&& (numberOfRecTracks<theMaxTracks));
	       recTrack++) {
    //    std::cout << "Input; 1/Pt " << 1./(*recTrack).momentumAtVertex().transverse() << std::endl;

    //looking for sim tracks corresponding to our reconstructed tracks:  
    simIndex[numberOfRecTracks] = -1;

    std::vector<std::pair<TrackingParticleRef, double> > simFound;
    try {
      const TrackTransientTrack* ttt = dynamic_cast<const TrackTransientTrack*>(recTrack->basicTransientTrack());
      if ((ttt!=0) && (recSimColl!=0)) simFound = (*recSimColl)[ttt->trackBaseRef()];
//       if (recSimColl!=0) simFound = (*recSimColl)[recTrack->persistentTrackRef()];
//      if (recSimColl!=0) simFound = (*recSimColl)[recTrack];

    } catch (cms::Exception e) {
//       LogDebug("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt() 
// 				 << " NOT associated to any TrackingParticle" << "\n";
//       edm::LogError("TrackValidator") << e.what() << "\n";
    }

    if(simFound.size() != 0) {
      //OK, it was associated, so get the state on the same surface as the 'SimState'
      TrackingParticleRefVector::const_iterator simTrackI = 
	find(simTrackV.begin(), simTrackV.end(), simFound[0].first);
      if (simTrackI!=simTrackV.end()) ++tracks[2];
      int simTrackIndex = simTrackI-simTrackV.begin();
      if (simTrackIndex<numberOfSimTracks) {
        simIndex[numberOfRecTracks] = simTrackIndex;
        recIndex[simTrackIndex] = numberOfRecTracks;
	//	cout << "Assoc; 1/Pt " << 1./(*recTrack).momentumAtVertex().transverse() << std::endl;
      }
    }

    TrajectoryStateClosestToPoint tscp = recTrack->trajectoryStateClosestToPoint(recVertex.position());
    fillParameters(recTrack->track().parameters(), recPars, numberOfRecTracks);
    fillErrors(tscp.perigeeError(), recErrs, numberOfRecTracks);
//     trackWeight[numberOfRecTracks] = recVertex.trackWeight(*recTrack);
// 
//     if ((recVertex.isValid())&&(recVertex.hasRefittedTracks())) {
//       //looking for corresponding refitted tracks:
//       TrajectoryStateOnSurface refip;
//       RecTrackCont::iterator refTrackI = 
//       		find_if(refTracks.begin(), refTracks.end(), RecTrackMatch(*recTrack));
//       if (refTrackI!=refTracks.end()) {
//         // If it was not found, it would mean that it was not used in the fit,
// 	// or with a low weight such that the track was then discarded.
// 	if(simFound.size() != 0) {
// 	  refip = refTrackI->stateOnSurface(simFound[0]->impactPointStateOnSurface().surface());
// 	} else {
//           refip = refTrackI->innermostState();
// 	}
// 
// 	fillParameters(refip, refPars, numberOfRecTracks);
// 	fillErrors(refip, refErrs, numberOfRecTracks);
//       }
//     }
// 
    ++numberOfRecTracks;
  }
  
}

void VertexFitterResult::fillParameters (const reco::TrackBase::ParameterVector& perigee,
	float* params[5], int trackNumber)
{
  params[0][trackNumber] = perigee[0];
  params[1][trackNumber] = perigee[1];
  params[2][trackNumber] = perigee[2];
  params[3][trackNumber] = perigee[3];
  params[4][trackNumber] = perigee[4];
}

void VertexFitterResult::fillParameters (const PerigeeTrajectoryParameters & ptp,
	float* params[5], int trackNumber)
{
  const AlgebraicVector5 & perigee = ptp.vector();
  params[0][trackNumber] = perigee[0];
  params[1][trackNumber] = perigee[1];
  params[2][trackNumber] = perigee[2];
  params[3][trackNumber] = perigee[3];
  params[4][trackNumber] = perigee[4];
}

void VertexFitterResult::fillErrors (const PerigeeTrajectoryError & pte,
	float* errors[5], int trackNumber)
{
  errors[0][trackNumber] = pte.transverseCurvatureError(); 
  errors[1][trackNumber] = pte.thetaError(); 
  errors[2][trackNumber] = pte.phiError(); 
  errors[3][trackNumber] = pte.transverseImpactParameterError(); 
  errors[4][trackNumber] = pte.longitudinalImpactParameterError(); 
}

void VertexFitterResult::reset()
{
  for ( int i=0; i<3; ++i ) {
    simPos[i] = 0.;
    recPos[i] = 0.;
    recErr[i] = 0.;
    chi[i] = 0.;
    tracks[i] = 0;
  }
  vertex =0;
  fitTime = 0;

  for ( int j=0; j<numberOfRecTracks; ++j ) {
    for ( int i=0; i<5; ++i ) {
       recPars[i][j] = 0;
       refPars[i][j] = 0;
       recErrs[i][j] = 0;
       refErrs[i][j] = 0;
    }
    trackWeight[j] = 0;
    simIndex[j] = -1;
  }
  for ( int j=0; j<numberOfSimTracks; ++j ) {
    for ( int i=0; i<5; ++i ) {
       simPars[i][j] = 0;
    }
    recIndex[j] = -1;
  }

  numberOfRecTracks=0;
  numberOfSimTracks=0;
}
