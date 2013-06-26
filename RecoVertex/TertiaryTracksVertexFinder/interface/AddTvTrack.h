#ifndef ADDTVTRACK_H
#define ADDTVTRACK_H

// Class for Secondary Vertex Finding in Jets
// It adds Tracks of a not reconstructed Tertiary Vertex to a Secondary Vertex
// New Tracks are not used for refitting the Vertex but for the kinematical 
// Variables for the b-tagging
// It uses the first PV and SV in the vectors

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class AddTvTrack  {

 public:

  // constructor
  AddTvTrack( std::vector<TransientVertex>*, std::vector<TransientVertex>*,
             double) ;

  // destructor   
  ~AddTvTrack() {} 

  // does the work
  std::vector<TransientVertex> getSecondaryVertices(
    const std::vector<reco::TransientTrack> & );

  // Access to parameters
  inline std::vector<TransientVertex>* getPrimaryVertices() const { 
    return thePrimaryVertices; 
  }
  inline std::vector<TransientVertex>* getSecondaryVertices() const { 
    return theSecondaryVertices; 
  }
  inline double getMaxSigOnDistTrackToB() const { 
    return MaxSigOnDistTrackToB; 
  }  


  struct TrackInfo {
    TrackInfo(const reco::TransientTrack* ptrack_, double* param_) {
      ptrack=ptrack_; 
      for(int i=0;i<7;i++) param[i]=param_[i];
    };
    const reco::TransientTrack* ptrack;
    double param[7];
  };
  typedef std::vector<TrackInfo> TrackInfoVector;

  TrackInfoVector getTrackInfo() { return theTrackInfoVector; }

  //std::vector<pair<reco::TransientTrack,double> > getTrackInfo() {
  //  return TrackInfo;
  //}
  //std::vector<pair<reco::TransientTrack,double*> > getTrackInfo2() {
  //  return TrackInfo2;
  //}


  // Set parameters
  inline void setPrimaryVertices(std::vector<TransientVertex> & 
				 ThePrimaryVertices) { 
    thePrimaryVertices = thePrimaryVertices; // TYPO ?!?!?!?!?!?!?!
  }
  inline void setSecondaryVertices(std::vector<TransientVertex> & 
				   TheSecondaryVertices) { 
    theSecondaryVertices = theSecondaryVertices; // TYPO ?!?!?!?!?!?
  }
  inline void setMaxSigOnDistTrackToB(double maxSigOnDistTrackToB) { 
    MaxSigOnDistTrackToB  = maxSigOnDistTrackToB; 
  }


 private:    

  typedef std::map<reco::TransientTrack, float> TransientTrackToFloatMap;

  std::vector<TransientVertex> *thePrimaryVertices;
  std::vector<TransientVertex> *theSecondaryVertices;
  double MaxSigOnDistTrackToB;
  double theIPSig;

  // TDR Studies
  //static std::vector<pair<reco::TransientTrack,double> > TrackInfo;
  //static std::vector<pair<reco::TransientTrack,double* > > TrackInfo2;
  //std::vector<pair<reco::TransientTrack,double> > TrackInfo;
  //std::vector<pair<reco::TransientTrack,double* > > TrackInfo2;

  TrackInfoVector theTrackInfoVector;

  static const bool debug = false;

};

#endif
