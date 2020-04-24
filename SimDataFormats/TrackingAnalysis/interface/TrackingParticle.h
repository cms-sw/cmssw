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
    int pdgId() const {
      if( genParticles_.empty() ) return g4Tracks_[0].type();
      else return (*genParticles_.begin())->pdgId();
    }

    /** @brief Signal source, crossing number.
     *
     * Note this is taken from the first SimTrack only, but there shouldn't be any SimTracks from different
     * crossings in the TrackingParticle. */
    EncodedEventId eventId() const {
      return g4Tracks_[0].eventId();
    }

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
    const reco::GenParticleRefVector& genParticles() const { return genParticles_; }
    const std::vector<SimTrack>& g4Tracks() const { return g4Tracks_; }
    const TrackingVertexRef& parentVertex() const { return parentVertex_; }

    // Accessors for vector of decay vertices
    const TrackingVertexRefVector& decayVertices() const { return decayVertices_; }
    tv_iterator decayVertices_begin() const { return decayVertices_.begin(); }
    tv_iterator decayVertices_end() const { return decayVertices_.end(); }


    /// @brief Electric charge. Note this is taken from the first SimTrack only.
    float charge() const { return g4Tracks_[0].charge(); }
    /// Gives charge in unit of quark charge (should be 3 times "charge()")
    int threeCharge() const { return lrintf(3.f*charge()); }

    /// @brief Four-momentum Lorentz vector. Note this is taken from the first SimTrack only.
    const LorentzVector& p4() const {
      return g4Tracks_[0].momentum();
    }

    /// @brief spatial momentum vector
    Vector momentum() const {
      return p4().Vect();
    }

    /// @brief Vector to boost to the particle centre of mass frame.
    Vector boostToCM() const {
      return p4().BoostToCM();
    }

    /// @brief Magnitude of momentum vector. Note this is taken from the first SimTrack only.
    double p() const {
      return p4().P();
    }

    /// @brief Energy. Note this is taken from the first SimTrack only.
    double energy() const {
      return p4().E();
    }

    /// @brief Transverse energy. Note this is taken from the first SimTrack only.
    double et() const {
      return p4().Et();
    }

    /// @brief Mass. Note this is taken from the first SimTrack only.
    double mass() const {
      return p4().M();
    }

    /// @brief Mass squared. Note this is taken from the first SimTrack only.
    double massSqr() const {
      return pow( mass(), 2 );
    }

    /// @brief Transverse mass. Note this is taken from the first SimTrack only.
    double mt() const {
      return p4().Mt();
    }

    /// @brief Transverse mass squared. Note this is taken from the first SimTrack only.
    double mtSqr() const {
      return p4().Mt2();
    }

    /// @brief x coordinate of momentum vector. Note this is taken from the first SimTrack only.
    double px() const {
      return p4().Px();
    }

    /// @brief y coordinate of momentum vector. Note this is taken from the first SimTrack only.
    double py() const {
      return p4().Py();
    }

    /// @brief z coordinate of momentum vector. Note this is taken from the first SimTrack only.
    double pz() const {
      return p4().Pz();
    }

    /// @brief Transverse momentum. Note this is taken from the first SimTrack only.
    double pt() const {
      return p4().Pt();
    }

    /// @brief Momentum azimuthal angle. Note this is taken from the first SimTrack only.
    double phi() const {
      return p4().Phi();
    }

    /// @brief Momentum polar angle. Note this is taken from the first SimTrack only.
    double theta() const {
      return p4().Theta();
    }

    /// @brief Momentum pseudorapidity. Note this is taken from the first SimTrack only.
    double eta() const {
      return p4().Eta();
    }

    /// @brief Rapidity. Note this is taken from the first SimTrack only.
    double rapidity() const {
      return p4().Rapidity();
    }

    /// @brief Same as rapidity().
    double y() const {
      return rapidity();
    }

    /// @brief Parent vertex position
    Point vertex() const {
       const TrackingVertex::LorentzVector & p = (*parentVertex_).position();
       return Point(p.x(),p.y(),p.z());
    }  

    /// @brief x coordinate of parent vertex position
    double vx() const {
      const TrackingVertex& r=( *parentVertex_);
      return r.position().X();
    }

    /// @brief y coordinate of parent vertex position
    double vy() const {
      const TrackingVertex& r=( *parentVertex_);
      return r.position().Y();
    }
    // @brief z coordinate of parent vertex position
    double vz() const {
      const TrackingVertex& r=( *parentVertex_);
      return r.position().Z();
    }

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
