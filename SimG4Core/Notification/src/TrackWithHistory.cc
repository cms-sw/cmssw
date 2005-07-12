#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Notification/interface/TrackInformationExtractor.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/GenParticleInfoExtractor.h"

#include "G4VProcess.hh"

#include <iostream>

using std::cout;
using std::endl;

//#define DEBUG

G4TrackToParticleID * TrackWithHistory::theG4TrackToParticleID(0);

TrackWithHistory::TrackWithHistory(const G4Track * g4trk) 
  : trackID_(0),particleID_(0),parentID_(0),momentum_(0),totalEnergy_(0),
    vertexPosition_(0),globalTime_(0),localTime_(0),properTime_(0),
    creatorProcess_(0),weight_(0),storeTrack_(false),saved_(false)
{
    if (theG4TrackToParticleID == 0) theG4TrackToParticleID = new G4TrackToParticleID;  
    if (g4trk!=0) 
    {
	TrackInformationExtractor extractor;
	trackID_ = g4trk->GetTrackID();
	particleID_ = theG4TrackToParticleID->particleID(g4trk);
	parentID_ = g4trk->GetParentID();
	momentum_ = g4trk->GetMomentum();
	totalEnergy_ = g4trk->GetTotalEnergy();
	vertexPosition_  = g4trk->GetPosition();
	globalTime_  = g4trk->GetGlobalTime();
	localTime_  = g4trk->GetLocalTime();
	properTime_  = g4trk->GetProperTime();
	creatorProcess_ = g4trk->GetCreatorProcess();
	weight_ = g4trk->GetWeight();
	storeTrack_ = extractor(g4trk).storeTrack();
	saved_ = false;
	genParticleID_ = extractGenID( g4trk);
#ifdef DEBUG	
	cout << " TrackWithHistory : created history for " << trackID_
	     << " with mother " << parentID_ << endl;
#endif
    }
}

void TrackWithHistory::checkAtEnd(const G4Track * gt)
{
    bool ok = true;
    double epsilon = 1.e-6;
    double eps2 = epsilon*epsilon;
    if ((vertexPosition_-gt->GetVertexPosition()).mag2() > eps2) 
    {
	cout << "TrackWithHistory vertex position check failed" << endl;
	cout << "At construction: " << vertexPosition_ << endl;
	cout << "At end:          " << gt->GetVertexPosition() << endl;
	ok = false;
    }
    Hep3Vector dirDiff = momentum_.unit() - gt->GetVertexMomentumDirection();
    if (dirDiff.mag2() > eps2) 
    {
	cout << "TrackWithHistory momentum direction check failed" << endl;
	cout << "At construction: " << momentum_.unit() << endl;
	cout << "At end:          " << gt->GetVertexMomentumDirection() << endl;
	ok = false;
    }
    if (!ok) throw SimG4Exception("TrackWithHistory::checkAtEnd failed");
}

int TrackWithHistory::extractGenID(const G4Track* gt) const
{
    void * vgprimary = gt->GetDynamicParticle()->GetPrimaryParticle();
    if (vgprimary == 0) return -1;
    // replace old-style cast with appropriate new-style cast...
    G4PrimaryParticle* gprimary = (G4PrimaryParticle*) vgprimary;
    GenParticleInfoExtractor ext;
    return ext(gprimary).id();
}
