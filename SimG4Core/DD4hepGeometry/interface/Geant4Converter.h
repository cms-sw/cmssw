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
#ifndef DD4HEP_GEANT4CONVERTER_H
#define DD4HEP_GEANT4CONVERTER_H

// Framework include files
#include "DD4hep/Printout.h"
//FIXME: #include "DDG4/Geant4Mapping.h"
#include "SimG4Core/DD4hepGeometry/interface/Geant4Mapping.h"

/// Namespace for the AIDA detector description toolkit
namespace dd4hep {

  /// Namespace for the Geant4 based simulation part of the AIDA detector description toolkit
  namespace sim {

    /// Geometry converter from dd4hep to Geant 4.
    /**
     *  \author  M.Frank
     *  \version 1.0
     *  \ingroup DD4HEP_SIMULATION
     */
    class Geant4Converter : public detail::GeoHandler, public Geant4Mapping {
    public:
      /// Property: Flag to debug materials during conversion mechanism
      bool debugMaterials  = false;
      /// Property: Flag to debug elements during conversion mechanism
      bool debugElements   = false;
      /// Property: Flag to debug shapes during conversion mechanism
      bool debugShapes     = false;
      /// Property: Flag to debug volumes during conversion mechanism
      bool debugVolumes    = false;
      /// Property: Flag to debug placements during conversion mechanism
      bool debugPlacements = false;
      /// Property: Flag to debug regions during conversion mechanism
      bool debugRegions    = false;
      /// Property: Flag to debug surfaces during conversion mechanism
      bool debugSurfaces   = false;

      /// Property: Flag to dump all placements after the conversion procedure
      bool printPlacements = false;
      /// Property: Flag to dump all sensitives after the conversion procedure
      bool printSensitives = false;

      /// Property: Check geometrical overlaps for volume placements and G4 imprints 
      bool       checkOverlaps;
      /// Property: Output level for debug printing
      PrintLevel outputLevel;

      /// Initializing Constructor
      Geant4Converter(const Detector& description);

      /// Initializing Constructor
      Geant4Converter(Detector& description, PrintLevel level);

      /// Standard destructor
      virtual ~Geant4Converter();

      /// Create geometry conversion
      Geant4Converter& create(DetElement top);

#if ROOT_VERSION_CODE >= ROOT_VERSION(6,17,0)
      /// Convert the geometry type material into the corresponding Geant4 object(s).
      virtual void* handleMaterialProperties(TObject* matrix) const;

      /// Convert the optical surface to Geant4
      void* handleOpticalSurface(TObject* surface) const;

      /// Convert the skin surface to Geant4
      void* handleSkinSurface(TObject* surface) const;

      /// Convert the border surface to Geant4
      void* handleBorderSurface(TObject* surface) const;
#endif
      /// Convert the geometry type material into the corresponding Geant4 object(s).
      virtual void* handleMaterial(const std::string& name, Material medium) const;

      /// Convert the geometry type element into the corresponding Geant4 object(s).
      virtual void* handleElement(const std::string& name, Atom element) const;

      /// Convert the geometry type solid into the corresponding Geant4 object(s).
      virtual void* handleSolid(const std::string& name, const TGeoShape* volume) const;

      /// Convert the geometry type logical volume into the corresponding Geant4 object(s).
      virtual void* handleVolume(const std::string& name, const TGeoVolume* volume) const;
      virtual void* collectVolume(const std::string& name, const TGeoVolume* volume) const;

      /// Convert the geometry type volume placement into the corresponding Geant4 object(s).
      virtual void* handlePlacement(const std::string& name, const TGeoNode* node) const;
      virtual void* handleAssembly(const std::string& name, const TGeoNode* node) const;

      /// Convert the geometry type field into the corresponding Geant4 object(s).
      ///virtual void* handleField(const std::string& name, Ref_t field) const;

      /// Convert the geometry type region into the corresponding Geant4 object(s).
      virtual void* handleRegion(Region region, const std::set<const TGeoVolume*>& volumes) const;

      /// Convert the geometry visualisation attributes to the corresponding Geant4 object(s).
      virtual void* handleVis(const std::string& name, VisAttr vis) const;

      /// Convert the geometry type LimitSet into the corresponding Geant4 object(s).
      virtual void* handleLimitSet(LimitSet limitset, const std::set<const TGeoVolume*>& volumes) const;

      /// Handle the geant 4 specific properties
      void handleProperties(Detector::Properties& prp) const;

      /// Print the geometry type SensitiveDetector
      virtual void printSensitive(SensitiveDetector sens_det, const std::set<const TGeoVolume*>& volumes) const;

      /// Print Geant4 placement
      virtual void* printPlacement(const std::string& name, const TGeoNode* node) const;
    };
  }    // End namespace sim
}      // End namespace dd4hep

#endif // DD4HEP_GEANT4CONVERTER_H
