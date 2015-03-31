#ifndef CastorShowerEvent_h
#define CastorShowerEvent_h

#include "TROOT.h"
#include "Rtypes.h"
#include "TObject.h"
#include "TClass.h"
#include "TDictionary.h"

#include "DataFormats/Math/interface/Point3D.h"
#include <vector>
#include <string>

  // class CastorShowerEvent {
  class CastorShowerEvent : public TObject {
  
  public:
  
    /// point in the space
    typedef math::XYZPoint Point;

    CastorShowerEvent();
    ~CastorShowerEvent();
    
    void Clear();
    
//  private:
  
    // Data members
    unsigned int              nhit;
    std::vector<unsigned int> detID;
    std::vector<Point>        hitPosition;
    std::vector<float>        nphotons;
    std::vector<float>        time;
    float                     primaryEnergy;
    float                     primEta , primPhi;
    float                     primX , primY , primZ;
    
    // Setters
    void setNhit(unsigned int i)   { nhit = i; };
    void setDetID(unsigned int id) { detID.push_back(id); };
    void setHitPosition(const Point& p)   { hitPosition.push_back(p); };
    void setNphotons(float np)     { nphotons.push_back(np); };
    void setTime(float t)          { time.push_back(t); };
    void setPrimE(float e)         { primaryEnergy = e; };
    void setPrimEta(float eta)     { primEta = eta; };
    void setPrimPhi(float phi)     { primPhi = phi; };
    void setPrimX(float x)         { primX = x; };
    void setPrimY(float y)         { primY = y; };
    void setPrimZ(float z)         { primZ = z; };
    
    // Accessors
    unsigned int getNhit()       { return nhit; };
    unsigned int getDetID(int i) { return detID[i]; };
    Point getHitPosition(int i)  { return hitPosition[i]; };
    float getNphotons(int i)     { return nphotons[i]; };
    float getTime(int i)         { return time[i]; };
    float getPrimE()             { return primaryEnergy; };
    float getPrimEta()           { return primEta; };
    float getPrimPhi() const     { return primPhi; };
    float getPrimX()             { return primX; };
    float getPrimY()             { return primY; };
    float getPrimZ()             { return primZ; };
    
    ClassDef(CastorShowerEvent,2)
    
  };

#endif
