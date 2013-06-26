#include "SimG4CMS/Muon/interface/SimHitPrinter.h"

#include<iomanip>
#include<iostream>

std::ofstream * SimHitPrinter::theFile(0);

SimHitPrinter::SimHitPrinter(std::string filename){
  if (theFile) return;
  const char* theName = filename.c_str();
  theFile = new std::ofstream(theName, std::ios::out);
}

SimHitPrinter::~SimHitPrinter(){
  //  theFile->close();
}

void SimHitPrinter::startNewSimHit(std::string s){
  std::cout.width(10);
  std::cout.setf(std::ios::right,std::ios::adjustfield);
  std::cout.setf(std::ios::scientific,std::ios::floatfield);
  std::cout.precision(6);
  std::cout << "SimHit in "<<s<<std::endl;
  (*theFile).width(10);
  (*theFile).setf(std::ios::right,std::ios::adjustfield);
  (*theFile).setf(std::ios::scientific|std::ios::uppercase|std::ios::showpos,std::ios::floatfield);
  (*theFile).precision(5);
  (*theFile) << "SimHit in "<<s;
}

void SimHitPrinter::startNewEvent(int num){
  (*theFile) << "Event "<<num<<std::endl;
}

void SimHitPrinter::printId(int id) const{
  std::cout << " Id: "<<id<<std::endl;
  (*theFile) << " id ";
  (*theFile).width(10);
  (*theFile).setf(std::ios::right,std::ios::adjustfield);
  (*theFile) <<id;
}

void SimHitPrinter::printTrack(int id) const{
  std::cout << " Track: "<<id<<std::endl;
  (*theFile) << " trk ";
  (*theFile).width(10);
  (*theFile).setf(std::ios::right,std::ios::adjustfield);
  (*theFile) << id;
}

void SimHitPrinter::printPabs(float pabs) const{
  std::cout << " Pabs: "<<pabs<<std::endl;
  (*theFile) << " p "<<pabs;
}

void SimHitPrinter::printEloss(float eloss) const{
  std::cout << " Eloss: "<<eloss<<std::endl;
  (*theFile) << " e "<<eloss;
}

void SimHitPrinter::printLocal(LocalPoint localen,LocalPoint localex) const{
  (*theFile).width(10);
  (*theFile).setf(std::ios::right,std::ios::adjustfield);
  (*theFile).setf(std::ios::floatfield);
  (*theFile).precision(6);
  std::cout << " Local(en/ex): "<<localen.x()<<" "<< localen.y()<<" "
       <<localen.z()<<" / "<<localex.x()<<" "<< localex.y()<<" "
       <<localex.z()<<std::endl;
  (*theFile) << " en/ex "<<localen.x()<<" "<< localen.y()<<" "
       <<localen.z()<<" / "<<localex.x()<<" "<< localex.y()<<" "
       <<localex.z();
}

void SimHitPrinter::printGlobal(GlobalPoint global) const {
  (*theFile).width(10);
  (*theFile).setf(std::ios::right,std::ios::adjustfield);
  (*theFile).setf(std::ios::floatfield);
  (*theFile).precision(6);
  std::cout << " Global(en): "<<global.x()<<" "<< global.y()<<" "
	     <<global.z()<<std::endl;
  (*theFile) << " gl "<<global.x()<<" "<< global.y()<<" "
	     <<global.z()<<std::endl;
}
