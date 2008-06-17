#ifndef Application_SimTrackManager_h
#define Application_SimTrackManager_h
// -*- C++ -*-
//
// Package:     Application
// Class  :     SimTrackManager
// 
/**\class SimTrackManager SimTrackManager.h SimG4Core/Application/interface/SimTrackManager.h

 Description: Holds tracking information used by the sensitive detectors

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Fri Nov 25 17:36:41 EST 2005
// $Id: SimTrackManager.h,v 1.8 2008/01/09 10:42:16 fambrogl Exp $
//

// system include files
#include <map>
#include <vector>

// user include files
#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackContainer.h" 

// forward declarations

class SimTrackManager
{
  
public:
  
  class StrictWeakOrdering{
  public:
    bool operator() ( TrackWithHistory * & p,const unsigned int& i) const {return p->trackID() < i;}
  };
  //      enum SpecialNumbers {InvalidID = 65535};
  /// this map contains association between vertex number and position
  typedef std::pair<int,math::XYZVectorD> MapVertexPosition;
  typedef std::vector<std::pair<int,math::XYZVectorD> > MapVertexPositionVector;
  typedef std::map<int,MapVertexPositionVector> MotherParticleToVertexMap;
  typedef MotherParticleToVertexMap VertexMap;
  
  SimTrackManager(bool iCollapsePrimaryVertices =false);
  virtual ~SimTrackManager();
  
  // ---------- const member functions ---------------------
  const TrackContainer * trackContainer() const { 
    return m_trksForThisEvent; 
  }
  
  
  // ---------- member functions ---------------------------
  void storeTracks(G4SimEvent * simEvent);
  
  void reset();
  void deleteTracks();
  void cleanTkCaloStateInfoMap();
  
  void addTrack(TrackWithHistory* iTrack, bool inHistory) {
    if(idsave.size()<=iTrack->trackID()){
      int newsize = iTrack->trackID()*2;
      idsave.resize(newsize);
    }
    idsave[iTrack->trackID()] = iTrack->parentID();
    if (inHistory) m_trksForThisEvent->push_back(iTrack);
  }
  
  void addTkCaloStateInfo(uint32_t t,std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> p){
    std::map<uint32_t,std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> >::const_iterator it = 
      mapTkCaloStateInfo.find(t);
    
    if (it ==  mapTkCaloStateInfo.end())
      mapTkCaloStateInfo.insert(std::pair<uint32_t,std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> >(t,p));
    
  }
  void setCollapsePrimaryVertices(bool iSet) {
    m_collapsePrimaryVertices=iSet;
  }
  int idSavedTrack (int) const;
private:
  SimTrackManager(const SimTrackManager&); // stop default
  
  const SimTrackManager& operator=(const SimTrackManager&); // stop default
  
  void saveTrackAndItsBranch(TrackWithHistory *);
  int getOrCreateVertex(TrackWithHistory *,int,G4SimEvent * simEvent);
  void cleanVertexMap();
  void reallyStoreTracks(G4SimEvent * simEvent);
  
  // ---------- member data --------------------------------
  TrackContainer * m_trksForThisEvent;
  bool m_SaveSimTracks;
  MotherParticleToVertexMap m_vertexMap;
  int m_nVertices;
  bool m_collapsePrimaryVertices;
  std::map<uint32_t,std::pair<math::XYZVectorD,math::XYZTLorentzVectorD > > mapTkCaloStateInfo;
  std::vector<int> idsave;

  int niteration;
  int avgsizeidsave;
  int sizeidsave;
  
  int avgcalomapsize;
  int calomapsize;

  int avgvtxmapsize;
  int vtxmapsize;

};


class trkIDLess
{
public:
    bool operator()(TrackWithHistory * trk1, TrackWithHistory * trk2) const
    { return (trk1->trackID() < trk2->trackID()); }
};

#endif
