#ifndef SimG4Core_CMSGDMLWriteStructure_H
#define SimG4Core_CMSGDMLWriteStructure_H

#include "G4GDMLWriteStructure.hh"
#include <xercesc/dom/DOM.hpp>

class G4LogicalVolume;
class G4VRangeToEnergyConverter;

class CMSGDMLWriteStructure : public G4GDMLWriteStructure
{
public:

  CMSGDMLWriteStructure();
  
  ~CMSGDMLWriteStructure() override;

  void AddExtension(xercesc::DOMElement* volumeElement,
			    const G4LogicalVolume* const glv) override;


 private:

  G4VRangeToEnergyConverter *converter[4];
};

#endif
