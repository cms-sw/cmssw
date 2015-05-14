#ifndef SimDataFormats_TrackingParticle_h
#define SimDataFormats_TrackingParticle_h

#include <vector>
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

//
// Forward declarations
//
class TrackingVertex;
class SimTrack;
class EncodedEventId;

/** @brief Monte Carlo truth information used for tracking validation.
 *
 * Object with references to the original SimTrack and parent and daughter TrackingVertices.
 * Simulation with high (~100) pileup was taking too much memory so the class was slimmed down
 * and copies of the SimHits were removed.
 *
 * @author original author unknown, re-engineering and slimming by Subir Sarkar (subir.sarkar@cern.ch),
 * some tweaking and documentation by Mark Grimes (mark.grimes@bristol.ac.uk).
 * @date original date unknown, re-engineering Jan-May 2013
 */
class TrackingParticle
{
    friend std::ostream& operator<< (std::ostream& s, TrackingParticle const& tp);
public:
    typedef int Charge; ///< electric charge type
    typedef math::XYZTLorentzVectorD LorentzVector; ///< Lorentz vector
    typedef math::PtEtaPhiMLorentzVector PolarLorentzVector; ///< Lorentz vector
    typedef math::XYZPointD Point; ///< point in the space
    typedef math::XYZVectorD Vector; ///< point in the space

    /// reference to reco::GenParticle
    typedef reco::GenParticleRefVector::iterator   genp_iterator;
    typedef std::vector<SimTrack>::const_iterator  g4t_iterator;

    /** @brief Default constructor. Note that the object will be useless until it is provided
     * with a SimTrack and parent TrackingVertex.
     *
     * Most of the methods assume there is a SimTrack and parent TrackingVertex set, so will either
     * crash or give undefined results if this isn't true. This constructor should only be used to
     * create a placeholder until setParentVertex() and addG4Track() can be called.
     */
    TrackingParticle();

    TrackingParticle( const SimTrack& simtrk, const TrackingVertexRef& parentVertex );

    // destructor
    ~TrackingParticle();

    /** @brief PDG ID.
     *
     * Returns the PDG ID of the first associated gen particle. If there are no gen particles associated
     * then it returns type() from the first SimTrack. */
    int pdgId() const;
    /** @brief Signal source, crossing number.
     *
     * Note this is taken from the first SimTrack only, but there shouldn't be any SimTracks from different
     * crossings in the TrackingParticle. */
    EncodedEventId eventId() const;

    // Setters for G4 and reco::GenParticle
    void addGenParticle( const reco::GenParticleRef& ref);
    void addG4Track( const SimTrack& t);
    /// iterators
    genp_iterator genParticle_begin() const;
    genp_iterator genParticle_end() const;
    g4t_iterator g4Track_begin() const;
    g4t_iterator g4Track_end() const;
    void setParentVertex(const TrackingVertexRef& ref);
    void addDecayVertex(const TrackingVertexRef& ref);
    void clearParentVertex();
    void clearDecayVertices();
    // Getters for Embd and Sim Tracks
    const reco::GenParticleRefVector& genParticles() const;
    const std::vector<SimTrack>& g4Tracks() const;
    const TrackingVertexRef& parentVertex() const;

    // Accessors for vector of decay vertices
    const TrackingVertexRefVector& decayVertices() const;
    tv_iterator decayVertices_begin() const;
    tv_iterator decayVertices_end() const;


    /// @brief Electric charge. Note this is taken from the first SimTrack only.
    float charge() const { return g4Tracks_[0].charge(); }
    /// Gives charge in unit of quark charge (should be 3 times "charge()")
    int threeCharge() const { return lrintf(3.f*charge()); }

    const LorentzVector& p4() const; ///< @brief Four-momentum Lorentz vector. Note this is taken from the first SimTrack only.

    Vector momentum() const; ///< spatial momentum vector

    Vector boostToCM() const; ///< @brief Vector to boost to the particle centre of mass frame.

    double p() const; ///< @brief Magnitude of momentum vector. Note this is taken from the first SimTrack only.
    double energy() const; ///< @brief Energy. Note this is taken from the first SimTrack only.
    double et() const; ///< @brief Transverse energy. Note this is taken from the first SimTrack only.
    double mass() const; ///< @brief Mass. Note this is taken from the first SimTrack only.
    double massSqr() const; ///< @brief Mass squared. Note this is taken from the first SimTrack only.
    double mt() const; ///< @brief Transverse mass. Note this is taken from the first SimTrack only.
    double mtSqr() const; ///< @brief Transverse mass squared. Note this is taken from the first SimTrack only.
    double px() const; ///< @brief x coordinate of momentum vector. Note this is taken from the first SimTrack only.
    double py() const; ///< @brief y coordinate of momentum vector. Note this is taken from the first SimTrack only.
    double pz() const; ///< @brief z coordinate of momentum vector. Note this is taken from the first SimTrack only.
    double pt() const; ///< @brief Transverse momentum. Note this is taken from the first SimTrack only.
    double phi() const; ///< @brief Momentum azimuthal angle. Note this is taken from the first SimTrack only.
    double theta() const; ///< @brief Momentum polar angle. Note this is taken from the first SimTrack only.
    double eta() const; ///< @brief Momentum pseudorapidity. Note this is taken from the first SimTrack only.
    double rapidity() const; ///< @brief Rapidity. Note this is taken from the first SimTrack only.
    double y() const; ///< @brief Same as rapidity().

    /// @brief Parent vertex position
    Point vertex() const {
       const TrackingVertex::LorentzVector & p = (*parentVertex_).position();
       return Point(p.x(),p.y(),p.z());
    }  

    double vx() const; ///< @brief x coordinate of parent vertex position
    double vy() const; ///< @brief y coordinate of parent vertex position
    double vz() const; ///< @brief z coordinate of parent vertex position
    /** @brief Status word.
     *
     * Returns status() from the first gen particle, or -99 if there are no gen particles attached. */
    int status() const {
      return genParticles_.empty() ? -99 : (*genParticles_[0]).status();
    }

    static const unsigned int longLivedTag; ///< long lived flag

    /// is long lived?
    bool longLived() const { return status()&longLivedTag;}

   /** @brief Gives the total number of hits, including muon hits. Hits on overlaps in the same layer count separately.
    *
    * Equivalent to trackPSimHit().size() in the old TrackingParticle implementation. */
   int numberOfHits() const {return numberOfHits_;}

   /** @brief The number of hits in the tracker. Hits on overlaps in the same layer count separately.
    *
    * Equivalent to trackPSimHit(DetId::Tracker).size() in the old TrackingParticle implementation. */
   int numberOfTrackerHits() const {return numberOfTrackerHits_;}

   /** @deprecated The number of hits in the tracker but taking account of overlaps.
    * Deprecated in favour of the more aptly named numberOfTrackerLayers(). */
   int matchedHit() const;
   /** @brief The number of tracker layers with a hit.
    *
    * Different from numberOfTrackerHits because this method counts multiple hits on overlaps in the layer as one hit. */
   int numberOfTrackerLayers() const {return numberOfTrackerLayers_;}

   void setNumberOfHits( int numberOfHits );
   void setNumberOfTrackerHits( int numberOfTrackerHits );
   void setNumberOfTrackerLayers( const int numberOfTrackerLayers );
private:
    int numberOfHits_; ///< @brief The total number of hits
    int numberOfTrackerHits_; ///< @brief The number of tracker only hits
    int numberOfTrackerLayers_; ///< @brief The number of tracker layers with hits. Equivalent to the old matchedHit.

    /// references to G4 and reco::GenParticle tracks
    std::vector<SimTrack> g4Tracks_;
    reco::GenParticleRefVector genParticles_;

    // Source and decay vertices
    TrackingVertexRef parentVertex_;
    TrackingVertexRefVector decayVertices_;
};

#endif // SimDataFormats_TrackingParticle_H
