#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/FieldFactory.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/LocalFieldManager.h"

#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"

FieldFactory::FieldFactory(seal::Context * ic, const std::string & iname) 
    : super(ic), theLocalFM(0)
{ 
    initialize(); 
    std::cout << " FieldFactory initialized " << std::endl; 
    theFieldBuilder = FieldBuilder::instance();
}

FieldFactory::~FieldFactory() 
{ 
    if (theFieldBuilder != 0) delete theFieldBuilder;
    if (theLocalFM != 0) delete theLocalFM;
}

void FieldFactory::update(const DDDWorld * d)
{
    bool useMagneticField = true;
    bool useLocalManagers = false;
    
    if (!useMagneticField) return;
    static bool fieldIsInitialized = false;
    if(!fieldIsInitialized)
    {
	theFieldBuilder->setField();    
        G4TransportationManager * tM = G4TransportationManager::GetTransportationManager(); 
	theFieldBuilder->configure("MagneticFieldType",
				   tM->GetFieldManager(),tM->GetPropagatorInField());
// 	if (useLocalManagers)
// 	    theLocalFM = (*LocalFieldManagerFactory::instance())();
        fieldIsInitialized = true;
    }
}
