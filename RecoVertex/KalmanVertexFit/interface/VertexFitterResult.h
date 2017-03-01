#ifndef VertexFitterResult_H
#define VertexFitterResult_H

#include <string>
// #include "Vertex/VertexRecoAnalysis/interface/NumberOfSharedTracks.h"
// #include "CommonReco/PatternTools/interface/RefittedRecTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

#include <vector>

/**
 * Very basic class containing only the positions of the simulated and reconstructed 
 * vertices, total chi**2, chi**2 probability and 
 * number of degrees of freedom.
 * The only thing to be done is to call the method fill for each vertex.
 */


class VertexFitterResult {
public:

//   typedef std::vector<const TkSimTrack*>	SimTrackCont;
  typedef std::vector<reco::TransientTrack>		TTrackCont;

  VertexFitterResult(const int maxTracks = 100, const MagneticField* = 0);
  ~VertexFitterResult();

  void fill(const TransientVertex & recv, const TrackingVertex * simv = 0, 
  	    reco::RecoToSimCollection *recSimColl = 0,
  	    const float &time = 0);

  void fill(const TransientVertex & recVertex, const TTrackCont & recTrackV,
	    const TrackingVertex * simv = 0, 
      	    reco::RecoToSimCollection *recSimColl = 0, const float &time = 0);


  const float* simVertexPos() const {return simPos;}
  const float* recVertexPos() const {return recPos;}
  const float* recVertexErr() const {return recErr;}
  const int* trackInformation() const {return tracks;}
  const float* chi2Information() const {return chi;}
  const int* vertexPresent() const {return &vertex;}
  const float* time() const {return &fitTime;}
  void reset();
  const int* numberSimTracks() {return &numberOfSimTracks;}
  const int* numberRecTracks() {return &numberOfRecTracks;}
// This array contains, for each SimTrack, the index of the associated RecTrack
  const int* simTrack_recIndex() {return recIndex;}
// This array contains, for each RecTrack, the index of the associated simTrack
  const int* recTrack_simIndex() {return simIndex;}
  const float* recTrackWeight() {return trackWeight;}
  const float* recParameters (const int i) const
  {
    if ( i<0 || i>=5 )  return 0;
    return recPars[i];
  }
  const float* refParameters (const int i) const
  {
    if ( i<0 || i>=5 )  return 0;
    return refPars[i];
  }
  const float* simParameters (const int i) const
  {
    if ( i<0 || i>=5 )  return 0;
    return simPars[i];
  }
  const float* recErrors (const int i) const
  {
    if ( i<0 || i>=5 )  return 0;
    return recErrs[i];
  }
  const float* refErrors (const int i) const
  {
    if ( i<0 || i>=5 )  return 0;
    return refErrs[i];
  }

private:

//   typedef std::vector<const TkSimTrack*> SimTrkCont;
// 
  void fillParameters (const reco::TrackBase::ParameterVector& perigee, float* params[5],
  			int trackNumber);
//   void fillErrors (const reco::TrackBase::CovarianceMatrix& perigeeCov, float* errors[5],
//   			int trackNumber);
  void fillParameters (const PerigeeTrajectoryParameters & ptp, float* params[5],
  			int trackNumber);
  void fillErrors (const PerigeeTrajectoryError & pte, float* errors[5],
  			int trackNumber);

private:
//   class RecTrackMatch{
//     public:
//       RecTrackMatch(const RecTrack & aRecTrack):theRecTrack(aRecTrack){}
//       ~RecTrackMatch(){}
//       bool operator() (const RecTrack & theRefTrack) {
//         const RefittedRecTrack* refTa = dynamic_cast <const RefittedRecTrack*>(theRefTrack.tTrack());
// 	return ((refTa!=0) && theRecTrack.sameAddress(refTa->originalTrack()));
// //       return theRecTrack.sameAddress(theRefTrack);
// //   TrackAssociator * recTa = const_cast < TrackAssociator *> (vtxAssocFact.trackAssociator());
// //        return theRecTrack.sameAddress(theRefTrack.originalTrack());
//       }
//     private:
//       const RecTrack & theRecTrack;
//   };

  // NumberOfSharedTracks numberOfSharedTracks;

  const MagneticField * theMagField;

  float simPos[3];
  float recPos[3];
  float recErr[3];
  float chi[3];
  int tracks[3];
  int vertex; // 0x1 is Sim, 0x10 is Rec
  float fitTime;

  int theMaxTracks;
  float* simPars[5];
  float* recPars[5];
  float* refPars[5];
  float* recErrs[5];
  float* refErrs[5];
  int numberOfRecTracks, numberOfSimTracks;
  float* trackWeight;
  int *simIndex, *recIndex;

};
#endif

