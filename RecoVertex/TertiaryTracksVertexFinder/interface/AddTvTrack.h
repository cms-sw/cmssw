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
  AddTvTrack( std::vector<TransientVertex>*, std::vector<TransientVertex>*, double) ;
   
  ~AddTvTrack() {}  // nothing special to be done here

  std::vector<TransientVertex> getSecondaryVertices(const std::vector<reco::TransientTrack> & );


  // Access to parameters
  inline std::vector<TransientVertex>* getPrimaryVertices() const { return thePrimaryVertices; }
  inline std::vector<TransientVertex>* getSecondaryVertices() const { return theSecondaryVertices; }
  inline double getMaxSigOnDistTrackToB() const { return MaxSigOnDistTrackToB; }  
  
  // Set parameters
  inline void setPrimaryVertices(std::vector<TransientVertex> & ThePrimaryVertices) { thePrimaryVertices = thePrimaryVertices; }
  inline void setSecondaryVertices(std::vector<TransientVertex> & TheSecondaryVertices) { theSecondaryVertices = theSecondaryVertices; }
  inline void setMaxSigOnDistTrackToB(double maxSigOnDistTrackToB) { MaxSigOnDistTrackToB  = maxSigOnDistTrackToB; }
  
  // TDR Studies
  //static std::vector<pair<reco::TransientTrack,double> > TrackInfo;
  //static std::vector<pair<reco::TransientTrack,double* > > TrackInfo2;
  std::vector<pair<reco::TransientTrack,double> > TrackInfo;
  std::vector<pair<reco::TransientTrack,double* > > TrackInfo2;

 private:    

  typedef std::map<reco::TransientTrack, float> TransientTrackToFloatMap;

  std::vector<TransientVertex> *thePrimaryVertices;
  std::vector<TransientVertex> *theSecondaryVertices;
  double MaxSigOnDistTrackToB;

  static const bool debug = false;

};

#endif
