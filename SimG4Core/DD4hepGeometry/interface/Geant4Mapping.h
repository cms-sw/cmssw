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
#ifndef DD4HEP_DDG4_GEANT4MAPPING_H
#define DD4HEP_DDG4_GEANT4MAPPING_H

// Framework include files
#include "DD4hep/Detector.h"
#include "DD4hep/Volumes.h"
#include "DD4hep/GeoHandler.h"
//FIXME: #include "DDG4/Geant4GeometryInfo.h"
#include "SimG4Core/DD4hepGeometry/interface/Geant4GeometryInfo.h"
//FIXME: #include "DDG4/Geant4VolumeManager.h"

/// Namespace for the AIDA detector description toolkit
namespace dd4hep {

  /// Namespace for the Geant4 based simulation part of the AIDA detector description toolkit
  namespace sim {

    /// Geometry mapping from dd4hep to Geant 4.
    /**
     *  \author  M.Frank
     *  \version 1.0
     *  \ingroup DD4HEP_SIMULATION
     */
    class Geant4Mapping: public detail::GeoHandlerTypes {
    protected:
      const Detector& m_detDesc;
      Geant4GeometryInfo* m_dataPtr;

      /// When resolving pointers, we must check for the validity of the data block
      void checkValidity() const;
    public:
      /// Initializing Constructor
      Geant4Mapping(const Detector& description);

      /// Standard destructor
      virtual ~Geant4Mapping();

      /// Possibility to define a singleton instance
      static Geant4Mapping& instance();

      /// Accesor to the Detector instance
      const Detector& detectorDescription() const {
        return m_detDesc;
      }

      /// Access to the data pointer
      Geant4GeometryInfo& data() const {
        return *m_dataPtr;
      }

      /// Access to the data pointer
      Geant4GeometryInfo* ptr() const {
        return m_dataPtr;
      }

      /// Create and attach new data block. Delete old data block if present.
      Geant4GeometryInfo& init();

      /// Release data and pass over the ownership
      Geant4GeometryInfo* detach();

      /// Set a new data block
      void attach(Geant4GeometryInfo* data);

      /// Access the volume manager
      //Geant4VolumeManager volumeManager() const;

      /// Accessor to resolve geometry placements
      PlacedVolume placement(const G4VPhysicalVolume* node) const;
    };
  }    // End namespace sim
}      // End namespace dd4hep

#endif // DD4HEP_DDG4_GEANT4MAPPING_H
