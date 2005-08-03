#ifndef SimG4Core_LocalFieldManager_H
#define SimG4Core_LocalFieldManager_H

#include "G4FieldManager.hh"

class LocalFieldManager : public G4FieldManager
{
public:
    LocalFieldManager() : G4FieldManager() {}
    virtual ~LocalFieldManager() {}
};

#endif 
