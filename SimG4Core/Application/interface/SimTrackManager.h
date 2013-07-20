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
// $Id: SimTrackManager.h,v 1.14 2013/05/30 21:14:57 gartung Exp $
//

// system include files
#include <map>
#include <vector>

// user include files
#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackContainer.h" 

#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

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
  
  void addTrack(TrackWithHistory* iTrack, bool inHistory, bool withAncestor) {
    std::pair<int, int> thePair(iTrack->trackID(),iTrack->parentID());
    idsave.push_back(thePair);
    if (inHistory) m_trksForThisEvent->push_back(iTrack);
    if (withAncestor) { std::pair<int,int> thisPair(iTrack->trackID(),0); ancestorList.push_back(thisPair); }
  }
  
  void addTkCaloStateInfo(uint32_t t,const std::pair<math::XYZVectorD,math::XYZTLorentzVectorD>& p){
    std::map<uint32_t,std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> >::const_iterator it = 
      mapTkCaloStateInfo.find(t);
    
    if (it ==  mapTkCaloStateInfo.end())
      mapTkCaloStateInfo.insert(std::pair<uint32_t,std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> >(t,p));
    
  }
  void setCollapsePrimaryVertices(bool iSet) {
    m_collapsePrimaryVertices=iSet;
  }
  int giveMotherNeeded(int i) const { 
    int theResult = 0;
    for (unsigned int itr=0; itr<idsave.size(); itr++) { if ((idsave[itr]).first == i) { theResult = (idsave[itr]).second; break; } }
    return theResult ; 
  }
  bool trackExists(unsigned int i) const {
    bool flag = false;
    for (unsigned int itr=0; itr<(*m_trksForThisEvent).size(); ++itr) {
      if ((*m_trksForThisEvent)[itr]->trackID() == i) {
	flag = true; break;
      }
    }
    return flag;
  }
  void cleanTracksWithHistory();
  void setLHCTransportLink( const edm::LHCTransportLinkContainer * thisLHCTlink ) { theLHCTlink = thisLHCTlink; }

private:
  SimTrackManager(const SimTrackManager&); // stop default
  
  const SimTrackManager& operator=(const SimTrackManager&); // stop default
  
  void saveTrackAndItsBranch(TrackWithHistory *);
  int getOrCreateVertex(TrackWithHistory *,int,G4SimEvent * simEvent);
  void cleanVertexMap();
  void reallyStoreTracks(G4SimEvent * simEvent);
  void fillMotherList();
  int idSavedTrack (int) const;

  // to restore the pre-LHCTransport GenParticle id link to a SimTrack
  void resetGenID();

  // ---------- member data --------------------------------
  TrackContainer * m_trksForThisEvent;
  bool m_SaveSimTracks;
  MotherParticleToVertexMap m_vertexMap;
  int m_nVertices;
  bool m_collapsePrimaryVertices;
  std::map<uint32_t,std::pair<math::XYZVectorD,math::XYZTLorentzVectorD > > mapTkCaloStateInfo;
  std::vector< std::pair<int, int> > idsave;

  std::vector<std::pair<int, int> > ancestorList; 

  unsigned int lastTrack;
  unsigned int lastHist;

  const edm::LHCTransportLinkContainer * theLHCTlink;

};


class trkIDLess
{
public:
    bool operator()(TrackWithHistory * trk1, TrackWithHistory * trk2) const
    { return (trk1->trackID() < trk2->trackID()); }
};

#endif
