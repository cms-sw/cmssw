#ifndef SimG4Core_TrackInformation_H
#define SimG4Core_TrackInformation_H 

#include "G4VUserTrackInformation.hh"

#include "G4Allocator.hh"

class TrackInformation : public G4VUserTrackInformation
{
public:
    virtual ~TrackInformation() {}
    inline void * operator new(size_t);
    inline void   operator delete(void * TrackInformation);

    bool storeTrack() const 	{ return storeTrack_; }
    /// can only be set to true, cannot be reset to false!
    void storeTrack(bool v)    
    { if (v) storeTrack_ = v; if (v == true) putInHistory(); }

    bool isPrimary() const 	{ return isPrimary_; }
    void isPrimary(bool v) 	{ isPrimary_ = v; }

    bool hasHits() const 	{ return hasHits_; }
    void hasHits(bool v) 	{ hasHits_ = v; }

    bool isGeneratedSecondary() const { return isGeneratedSecondary_; }
    void isGeneratedSecondary(bool v) { isGeneratedSecondary_ = v; }

    bool isInHistory() const { return isInHistory_; }
    void putInHistory()      { isInHistory_= true; }

    // Calo section
    int getIDonCaloSurface() const 	{ return idOnCaloSurface_; }
    void setIDonCaloSurface(int id, int ical, int last)
    { idOnCaloSurface_ = id; idCaloVolume_ = ical; idLastVolume_ = last; }
    int getIDCaloVolume() const         { return idCaloVolume_; }
    int getIDLastVolume() const         { return idLastVolume_; }
    bool caloIDChecked() const 		{ return caloIDChecked_; }
    void setCaloIDChecked(bool f) 	{ caloIDChecked_ = f; }

    // Generator information
    int    genParticlePID() const       { return genParticlePID_; }
    void   setGenParticlePID(int id)    { genParticlePID_ = id; }
    double genParticleP() const         { return genParticleP_; }
    void   setGenParticleP(double p)    { genParticleP_ = p; }

    virtual void Print() const;
private:
    bool storeTrack_;    
    bool isPrimary_;
    bool hasHits_;
    bool isGeneratedSecondary_;
    bool isInHistory_;
    int  idOnCaloSurface_;
    int  idCaloVolume_;
    int  idLastVolume_;
    bool caloIDChecked_;
    int    genParticlePID_;
    double genParticleP_;

    // Restrict construction to friends
    TrackInformation() :G4VUserTrackInformation(),storeTrack_(false),isPrimary_(false),
			hasHits_(false),isGeneratedSecondary_(false),isInHistory_(false),
                        idOnCaloSurface_(0),idCaloVolume_(-1),idLastVolume_(-1),
                        caloIDChecked_(false), genParticlePID_(-1),
                        genParticleP_(0) {}
    friend class NewTrackAction;
};

extern G4Allocator<TrackInformation> TrackInformationAllocator;

inline void * TrackInformation::operator new(size_t)
{
    void * trkInfo;
    trkInfo = (void *) TrackInformationAllocator.MallocSingle();
    return trkInfo;
}

inline void TrackInformation::operator delete(void * trkInfo)
{  TrackInformationAllocator.FreeSingle((TrackInformation*) trkInfo); }

#endif
