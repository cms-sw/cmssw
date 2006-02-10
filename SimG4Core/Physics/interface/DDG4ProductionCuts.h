#ifndef SimG4Core_DDG4ProductionCuts_H
#define SimG4Core_DDG4ProductionCuts_H

#include <string>
#include <vector>

class DDLogicalPart;
class G4Region;
class G4LogicalVolume;
class G4ProductionCuts;

class DDG4ProductionCuts 
{
public:
    DDG4ProductionCuts();
    ~DDG4ProductionCuts();
    void update();
    void SetVerbosity( int verb ) { m_Verbosity = verb; return ; }
private:
    void SetProdCuts(const DDLogicalPart lpart, G4LogicalVolume * lvolume);
    G4Region * GetRegion(const std::string & region);
    G4ProductionCuts * GetProductionCuts(G4Region * region);

    std::string m_KeywordRegion;    
    int         m_Verbosity ;
    
};

#endif
