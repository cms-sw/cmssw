#ifndef SimG4CMS_TkSimHitPrinter_H
#define SimG4CMS_TkSimHitPrinter_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include <string>
#include <fstream>

class G4Track;

class TkSimHitPrinter 
{
public:
    TkSimHitPrinter(std::string);
    ~TkSimHitPrinter();
    void startNewSimHit(std::string,std::string,int,int,int);
    void printLocal(Local3DPoint,Local3DPoint) const;
    void printGlobal(Local3DPoint,Local3DPoint) const;
    void printHitData(float, float) const;
    void printMomentumOfTrack(float, std::string, int sign) const;
    int getPropagationSign(Local3DPoint ,Local3DPoint);
    void printGlobalMomentum(float,float,float)const ;
private:
    int eventno;
    static std::ofstream * theFile;
};

#endif

