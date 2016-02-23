#include "SimG4Core/Application/interface/CMSGDMLWriteStructure.h"

#include "G4LogicalVolume.hh"
#include "G4ProductionCuts.hh"
#include "G4GDMLParser.hh"
#include "G4VRangeToEnergyConverter.hh"
#include "G4RToEConvForGamma.hh"
#include "G4RToEConvForElectron.hh"
#include "G4RToEConvForPositron.hh"
#include "G4RToEConvForProton.hh"
#include "G4SystemOfUnits.hh"

CMSGDMLWriteStructure::CMSGDMLWriteStructure()
{
  converter[0] = new G4RToEConvForGamma();
  converter[1] = new G4RToEConvForElectron();
  converter[2] = new G4RToEConvForPositron();
  converter[3] = new G4RToEConvForProton();
}
  
CMSGDMLWriteStructure::~CMSGDMLWriteStructure()
{}

void 
CMSGDMLWriteStructure::AddExtension(xercesc::DOMElement* volumeElement,
				     const G4LogicalVolume* const glv)
{
  xercesc::DOMElement* auxiliaryElement = 0;
  std::stringstream ss;
  const char* cutnames[4] = {"pcutg","pcutem","pcutep","pcutp"};
 
  auxiliaryElement = NewElement("auxiliary");
  auxiliaryElement->setAttributeNode(NewAttribute("auxtype","G4Region"));
  auxiliaryElement->setAttributeNode(NewAttribute("auxvalue",glv->GetRegion()->GetName()));
  volumeElement->appendChild(auxiliaryElement);

  auxiliaryElement = NewElement("auxiliary");
  auxiliaryElement->setAttributeNode(NewAttribute("auxtype","pcutunit"));
  auxiliaryElement->setAttributeNode(NewAttribute("auxvalue","GeV"));
  volumeElement->appendChild(auxiliaryElement);

  //     G4cout << "I have been called " << glv->GetName() << " in region " 
  // << glv->GetRegion()->GetName() << G4endl;
  G4ProductionCuts *cuts = glv->GetRegion()->GetProductionCuts();

  for(G4int ic=0; ic<4; ++ic) {
    G4cout << ic << ". " << cutnames[ic] << " converter: " << converter[ic] 
	   << " cuts: " << cuts << " glv: " << glv << G4endl;
    ss.clear(); ss.str("");
    ss << converter[ic]->Convert(cuts->GetProductionCut(ic),glv->GetMaterial())/CLHEP::GeV;
    //	 G4cout << cutnames[ic] << " = " << ss.str() << G4endl;
    auxiliaryElement = NewElement("auxiliary");
    auxiliaryElement->setAttributeNode(NewAttribute("auxtype",cutnames[ic]));
    auxiliaryElement->setAttributeNode(NewAttribute("auxvalue",ss.str()));
    volumeElement->appendChild(auxiliaryElement);
  }
}
