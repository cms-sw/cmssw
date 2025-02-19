#include "SimG4CMS/Tracker/interface/TkSimHitPrinter.h"

#include <iomanip>
#include <fstream>

#include "G4Track.hh"

std::ofstream * TkSimHitPrinter::theFile(0);

TkSimHitPrinter::TkSimHitPrinter(std::string filename){
  if (theFile) return;
  const char* theName = filename.c_str();
  theFile = new std::ofstream(theName, std::ios::out);
}

TkSimHitPrinter::~TkSimHitPrinter(){
  //  theFile->close();
}

void TkSimHitPrinter::startNewSimHit(std::string s, std::string d, int i,int j,int eve){
  (*theFile) <<"Event: "<<eve<<" "<<s<<" "<<d<<" "<<i<<" Track "<<j;
}

void TkSimHitPrinter::printLocal(Local3DPoint  p,Local3DPoint e ) const{
  (*theFile) << " Local: "<<p.x()<<" "<<p.y()<<" "<<p.z()
	     <<" ; "<<e.x()<<" "<<e.y()<<" "<<e.z();
}

void TkSimHitPrinter::printGlobal(Local3DPoint  p ,Local3DPoint  e ) const {
  (*theFile) << " Global: "<<p.x()<<" "<<p.y()<<" "<<p.z()
	     <<" ; "<<e.x()<<" "<<e.y()<<" "<<e.z();
}

void TkSimHitPrinter::printHitData(float energy, float tof) const {
  (*theFile)<< " Energy: "<<energy<<" ToF: "<<tof;
}
void TkSimHitPrinter::printMomentumOfTrack(float m, std::string s, int sign)const {
  (*theFile)<<" Momentum "<<m*sign<<" Particle "<<s;
}
void TkSimHitPrinter::printGlobalMomentum(float px, float py,float pz)const {
  (*theFile)<<" Momentum "<<px<<" "<<py<<" "<<pz<<std::endl;
}
int TkSimHitPrinter::getPropagationSign(Local3DPoint  p ,Local3DPoint  e ){
  float dr;
  dr = p.x()*(e-p).x()+ p.y()*(e-p).y()+p.z()*(e-p).z();
  if (dr>0) return +1;
  return -1;
}
