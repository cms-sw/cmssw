#ifndef SimG4Core_GFlash_GflashMediaMap_H
#define SimG4Core_GFlash_GflashMediaMap_H

#include <map>
#include "G4Region.hh"
#include "G4FastTrack.hh"

#include "SimG4Core/GFlash/interface/GflashCalorimeterNumber.h"

typedef std::map<G4String, GflashCalorimeterNumber> GflashIndexMap;
typedef std::map<GflashCalorimeterNumber, G4Material*> GflashMaterialMap;

class GflashMediaMap 
{
public:
    GflashMediaMap(G4Region* envelope);
    ~GflashMediaMap();

    void buildGflashMediaMap(G4Region* envelope);
    void printGflashMediaMap();

    void printName() { G4cout << " MediaMap Name " << theName << G4endl; }  
    void SetName(G4String name) { theName = name ; };

    GflashCalorimeterNumber  getIndex(G4String name);
    G4Material* getMaterial(GflashCalorimeterNumber);
    GflashCalorimeterNumber getCalorimeterNumber(const G4FastTrack& fastTrack);

private:
    G4String theName;
    GflashIndexMap    gIndexMap;
    GflashMaterialMap gMaterialMap;
};

#endif
