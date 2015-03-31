#ifndef SimG4Core_CMSGDMLWriteStructure_H
#define SimG4Core_CMSGDMLWriteStructure_H

#include "G4GDMLWriteStructure.hh"
#include <xercesc/dom/DOM.hpp>

class G4LogicalVolume;

class CMSGDMLWriteStructure : public G4GDMLWriteStructure
{
public:

  CMSGDMLWriteStructure();
  
  virtual ~CMSGDMLWriteStructure();

  virtual void AddExtension(xercesc::DOMElement* volumeElement,
			    const G4LogicalVolume* const glv);

};

#endif
