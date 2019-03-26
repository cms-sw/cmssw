//==========================================================================
//  AIDA Detector description implementation 
//--------------------------------------------------------------------------
// Copyright (C) Organisation europeenne pour la Recherche nucleaire (CERN)
// All rights reserved.
//
// For the licensing terms see $DD4hepINSTALL/LICENSE.
// For the list of contributors see $DD4hepINSTALL/doc/CREDITS.
//
// Author     : M.Frank
//
//==========================================================================

#ifndef DD4HEP_GEANT4PARTICLE_H
#define DD4HEP_GEANT4PARTICLE_H

// Framework include files
#include "DD4hep/Memory.h"

// ROOT includes
#include "Math/Vector4D.h"

// Geant4 forward declarations
class G4ParticleDefinition;
class G4VProcess;

// C/C++ include files
#include <set>
#include <map>
#include <vector>

/// Namespace for the AIDA detector description toolkit
namespace dd4hep {

  /// Namespace for the Geant4 based simulation part of the AIDA detector description toolkit
  namespace sim {

    // Forward declarations
    class Geant4Particle;

    /// Base class to extend the basic particle class used by DDG4 with user information
    /**
     *  \author  M.Frank
     *  \version 1.0
     *  \ingroup DD4HEP_SIMULATION
     */
    class ParticleExtension  {
    public:
      /// Default constructor
      ParticleExtension() {}
      /// Default destructor
      virtual ~ParticleExtension();
    };

    /// Track properties
    enum Geant4ParticleProperties {
      G4PARTICLE_CREATED_HIT = 1<<1,
      G4PARTICLE_PRIMARY = 1<<2,
      G4PARTICLE_HAS_SECONDARIES = 1<<3,
      G4PARTICLE_ABOVE_ENERGY_THRESHOLD = 1<<4,
      G4PARTICLE_KEEP_PROCESS = 1<<5,
      G4PARTICLE_KEEP_PARENT = 1<<6,
      G4PARTICLE_CREATED_CALORIMETER_HIT = 1<<7,
      G4PARTICLE_CREATED_TRACKER_HIT = 1<<8,
      G4PARTICLE_KEEP_USER = 1<<9,
      G4PARTICLE_KEEP_ALWAYS = 1<<10,
      G4PARTICLE_FORCE_KILL = 1<<11,

      // Generator status for a given particles: bit 0...4, agreed by many formats (HepMC, LCIO, ....):
      G4PARTICLE_GEN_EMPTY           = 1<<0,  // Empty line
      G4PARTICLE_GEN_STABLE          = 1<<1,  // undecayed particle, stable in the generator
      G4PARTICLE_GEN_DECAYED         = 1<<2,  // particle decayed in the generator
      G4PARTICLE_GEN_DOCUMENTATION   = 1<<3,  // documentation line
      G4PARTICLE_GEN_BEAM            = 1<<4,  // beam particle

      G4PARTICLE_GEN_OTHER           = 1<<9,  // any other generator status

      G4PARTICLE_GEN_GENERATOR       =        // Particle comes from generator
      (  G4PARTICLE_GEN_EMPTY+G4PARTICLE_GEN_STABLE+
         G4PARTICLE_GEN_DECAYED+G4PARTICLE_GEN_DOCUMENTATION+
         G4PARTICLE_GEN_BEAM+G4PARTICLE_GEN_OTHER),
      G4PARTICLE_GEN_STATUS          = 0x3FF, // Mask for generator status (bit 0...9)
      G4PARTICLE_GEN_STATUS_MASK     = 0xFFFF,// Mask for the raw generator status (max 65k values)
      // Simulation status of a given particle
      G4PARTICLE_SIM_CREATED         = 1<<10, // True if the particle has been created by the simulation program (rather than the generator)
      G4PARTICLE_SIM_BACKSCATTER     = 1<<11, // True if the particle is the result of a backscatter from a calorimeter shower.
      G4PARTICLE_SIM_DECAY_CALO      = 1<<12, // True if the particle has interacted in a calorimeter region.
      G4PARTICLE_SIM_DECAY_TRACKER   = 1<<13, // True if the particle has interacted in a tracking region.
      G4PARTICLE_SIM_STOPPED         = 1<<14, // True if the particle has been stopped by the simulation program.
      G4PARTICLE_SIM_LEFT_DETECTOR   = 1<<15, // True if the particle has left the world volume undecayed.
      G4PARTICLE_SIM_PARENT_RADIATED = 1<<16, // True if the particle's vertex is not the endpoint of the  parent particle.
      G4PARTICLE_SIM_OVERLAY         = 1<<17, // True if the particle has been overlayed by the simulation (or digitization)  program.

      G4PARTICLE_LAST_NOTHING = 1<<31
    };



    /// Data structure to store the MC particle information
    /**
     *  \author  M.Frank
     *  \version 1.0
     *  \ingroup DD4HEP_SIMULATION
     */
    class Geant4Particle {
    public:
      typedef std::set<int> Particles;
      /// Reference counter
      int ref = 0;           //! not persistent
      int id  = 0;
      int originalG4ID = 0;  //! not persistent
      int g4Parent = 0, reason = 0, mask = 0;
      int steps  = 0, secondaries = 0, pdgID = 0;
      int status = 0, colorFlow[2] {0,0};
      unsigned short genStatus= 0;
      char  charge = 0;
      char  _spare[1] {0};
      float spin[3] {0E0,0E0,0E0};
      // 12 ints + 4 bytes + 3 floats should be aligned to 8 bytes....
      double vsx  = 0E0, vsy  = 0E0, vsz = 0E0;
      double vex  = 0E0, vey  = 0E0, vez = 0E0;
      double psx  = 0E0, psy  = 0E0, psz = 0E0;
      double pex  = 0E0, pey  = 0E0, pez = 0E0;
      double mass = 0E0, time = 0E0, properTime = 0E0;
      /// The list of daughters of this MC particle
      Particles parents;
      Particles daughters;

      /// User data extension if required
#ifdef DD4HEP_DD4HEP_PTR_AUTO
      dd4hep_ptr<ParticleExtension> extension;
#else
      dd4hep_ptr<ParticleExtension> extension;   //! not persisten. ROOT cannot handle
#endif
      const G4VProcess *process = 0;             //! not persistent
      /// Default constructor
      Geant4Particle();
      /// Constructor with ID initialization
      Geant4Particle(int part_id);
      /// NO copy constructor
      Geant4Particle(const Geant4Particle& copy) = delete;
      /// Default destructor
      virtual ~Geant4Particle();
      /// NO assignment operation
      Geant4Particle& operator=(const Geant4Particle& copy) = delete;
      /// Increase reference count
      Geant4Particle* addRef()  {
        ++ref;
        return this;
      }
      /// Decrease reference count. Deletes object if NULL
      void release();
      /// Assignment operator
      Geant4Particle& get_data(Geant4Particle& c);
      /// Remove daughter from set
      void removeDaughter(int id_daughter);
      /// Charge accessor (for python etc.)
      int charge3() const  {  return charge; }
    };

#ifndef __DDG4_STANDALONE_DICTIONARIES__

    /// Data structure to access derived MC particle information
    /**
     *  \author  M.Frank
     *  \version 1.0
     *  \ingroup DD4HEP_SIMULATION
     */
    class Geant4ParticleHandle {
    public:
      typedef ROOT::Math::PxPyPzM4D<double> FourVector;
      typedef ROOT::Math::Cartesian3D<double> ThreeVector;
    protected:
      /// Particle pointer
      Geant4Particle* particle;
    public:
      /// Default constructor
      Geant4ParticleHandle(Geant4Particle* part);
      /// Initializing constructor
      Geant4ParticleHandle(const Geant4ParticleHandle& h);
      /// Assignment operator
      Geant4ParticleHandle& operator=(Geant4Particle* part);
      /// Overloaded -> operator to access particle details
      Geant4Particle* operator->() const;
      /// Conversion  operator to pointer
      operator Geant4Particle*() const;
      /// Accessor to the number of particle parents
      size_t numParent() const;
      /// Accessor to the number of particle daughters
      size_t numDaughter() const;
      /// Scalar particle momentum squared
      double momentum2() const;
      /// Scalar particle energy
      double energy() const;
      /// Scalar particle momentum
      double momentum() const   {  return std::sqrt(momentum2());  }
      /// Geant4 charge of the particle
      double charge()  const    { return double(particle->charge); }
      /// Geant4 mass of the particle
      double mass() const       { return particle->mass;           }
      /// Geant4 time of the particle
      double time() const       { return particle->time;           }
      /// Access to the Geant4 particle name
      std::string particleName() const;
      /// Access to the Geant4 particle type
      std::string particleType() const;
      /// Access to the creator process name
      std::string processName() const;
      /// Access to the creator process type name
      std::string processTypeName() const;
      /// Access patricle momentum, energy as 4 vector
      FourVector pxPyPzM() const;
      /// Access patricle momentum, energy as 4 vector
      ThreeVector startVertex() const;
      /// Access patricle momentum, energy as 4 vector
      ThreeVector endVertex()  const;
      /// Access the Geant4 particle definition object (expensive!)
      const G4ParticleDefinition *definition() const;

      /// Various output formats:

      /// Output type 1:+++ "tag"   10 def:0xde4eaa8 [gamma     ,   gamma] reason:      20 E:+1.017927e+03  \#Par:  1/4    \#Dau:  2
      void dump1(int level, const std::string& src, const char* tag) const;
      /// Output type 2:+++ "tag"   20 G4:   7 def:0xde4eaa8 [gamma     ,   gamma] reason:      20 E:+3.304035e+01 in record:YES  \#Par:  1/18   \#Dau:  0
      void dump2(int level, const std::string& src, const char* tag, int g4id, bool inrec) const;
      /// Output type 3:+++ "tag" ID:  0 e-           status:00000014 type:       11 Vertex:(+0.00e+00,+0.00e+00,+0.00e+00) [mm] time: +0.00e+00 [ns] \#Par:  0 \#Dau:  4
      void dumpWithVertex(int level, const std::string& src, const char* tag) const;
      void dumpWithMomentum(int level, const std::string& src, const char* tag) const;
      void dumpWithMomentumAndVertex(int level, const std::string& src, const char* tag) const;
      static void header4(int level, const std::string& src, const char* tag);
      void dump4(int level, const std::string& src, const char* tag) const;

      /// Handlers

      /// Offset all particle identifiers (id, parents, daughters) by a constant number
      void offset(int off)  const;

      /// Access Geant4 particle definitions by regular expression
      static std::vector<G4ParticleDefinition*> g4DefinitionsRegEx(const std::string& expression);
      /// Access Geant4 particle definitions by exact match
      static G4ParticleDefinition* g4DefinitionsExact(const std::string& expression);
    };

    inline Geant4ParticleHandle::Geant4ParticleHandle(const Geant4ParticleHandle& c)
      : particle(c.particle) {
    }

    /// Initializing constructor
    inline Geant4ParticleHandle::Geant4ParticleHandle(Geant4Particle* p)
      : particle(p)  {
    }

    /// Overloaded -> operator to access particle details
    inline Geant4Particle* Geant4ParticleHandle::operator->() const  {
      return particle;
    }
    /// Assignment operator
    inline Geant4ParticleHandle& Geant4ParticleHandle::operator=(Geant4Particle* part) {
      particle = part;
      return *this;
    }
    /// Conversion  operator to pointer
    inline Geant4ParticleHandle::operator Geant4Particle*() const  {
      return particle;
    }
    /// Accessor to the number of particle parents
    inline size_t Geant4ParticleHandle::numParent() const   {
      return particle->parents.size();
    }
    /// Accessor to the number of particle daughters
    inline size_t Geant4ParticleHandle::numDaughter() const   {
      return particle->daughters.size();
    }
    /// Access patricle momentum, energy as 4 vector
    inline ROOT::Math::PxPyPzM4D<double> Geant4ParticleHandle::pxPyPzM() const {
      const Geant4Particle* p = particle;
      return ROOT::Math::PxPyPzM4D<double>(p->psx,p->psy,p->psz,p->mass);
    }

    /// Access patricle momentum, energy as 4 vector
    inline ROOT::Math::Cartesian3D<double> Geant4ParticleHandle::startVertex() const {
      const Geant4Particle* p = particle;
      return ROOT::Math::Cartesian3D<double>(p->vsx,p->vsy,p->vsz);
    }

    /// Access patricle momentum, energy as 4 vector
    inline ROOT::Math::Cartesian3D<double> Geant4ParticleHandle::endVertex()  const {
      const Geant4Particle* p = particle;
      return ROOT::Math::Cartesian3D<double>(p->vex,p->vey,p->vez);
    }

    /// Scalar particle energy
    inline double Geant4ParticleHandle::energy() const {
      const Geant4Particle* p = particle;
      ROOT::Math::PxPyPzM4D<double> v(p->psx,p->psy,p->psz,p->mass);
      return v.E();
    }

    /// Scalar particle momentum squared
    inline double Geant4ParticleHandle::momentum2() const  {
      const Geant4Particle* p = particle;
      return (p->psx*p->psx + p->psy*p->psy + p->psz*p->psz);
    }

    /// Data structure to map particles produced during the generation and the simulation
    /**
     *  This data structure is added to the Geant4Event data extensions
     *  by the Geant4GenerationInit action.
     *  Particles are added:
     *  - During the generation if the required modules are activated
     *  - At the end of the handling of the MC truth are particles to be kept
     *    are inserted if the required modules are activated such as the
     *    Geant4ParticleHandler.
     *
     *  Note: This object takes OWNERSHIP of the inserted particles!
     *        beware of double deletion of objects!
     *
     *  \author  M.Frank
     *  \version 1.0
     *  \ingroup DD4HEP_SIMULATION
     */
    class Geant4ParticleMap  {
    public:
      typedef Geant4Particle          Particle;
      typedef std::map<int,Particle*> ParticleMap;
      typedef std::map<int,int>       TrackEquivalents;
      /// Mapping of particles of this event
      ParticleMap particleMap; //! not persistent
      /// Map associating the G4Track identifiers with identifiers of existing MCParticles
      TrackEquivalents equivalentTracks;

      /// Default constructor
      Geant4ParticleMap() {}
      /// Default destructor
      virtual ~Geant4ParticleMap();
      /// Check if the particle map was ever filled (ie. some particle handler was present)
      bool isValid() const;
      /// Dump content
      void dump()  const;
      /// Clear particle maps
      void clear();
      /// Adopt particle maps
      void adopt(ParticleMap& pm, TrackEquivalents& equiv);
      /// Access the particle map
      const ParticleMap& particles() const  {  return particleMap; }
      /// Access the map of track equivalents
      const TrackEquivalents& equivalents() const  {  return equivalentTracks;  }
      /// Access the equivalent track id (shortcut to the usage of TrackEquivalents)
      int particleID(int track, bool throw_if_not_found=true) const;
    };
#endif

  }    // End namespace sim
}      // End namespace dd4hep
#endif // DD4HEP_GEANT4PARTICLE_H
