// This class holds a list of stubs for an input link.
//This modules 'owns' the pointers to the stubs. All
//subsequent modules that handles stubs uses a pointer
//to the original stored here

 
#ifndef FPGAINPUTLINK_H
#define FPGAINPUTLINK_H

#include "L1TStub.hh"
#include "FPGAStub.hh"
#include "FPGAMemoryBase.hh"
#include <sstream>

using namespace std;

class FPGAInputLink:public FPGAMemoryBase{

public:

  FPGAInputLink(string name, unsigned int iSector, 
		double phimin, double phimax):
    FPGAMemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addStub(L1TStub al1stub) {
    L1TStub* l1stub=new L1TStub(al1stub);
    FPGAStub* stub=new FPGAStub(*l1stub,phimin_,phimax_);
    std::pair<FPGAStub*,L1TStub*> tmp(stub,l1stub);
    int dctregion=stub->fedregion();
    int ilink=stub->ilink();
    bool add=false;
    //cout << "Stub candidate in "<<getName()<<" "<<al1stub.phi()<<" "
    //	 <<al1stub.z()<<" "<<ilink<<" "<<dctregion<<endl;
    if (ilink==1&&dctregion==1&&getName()=="IL1_D1") add=true;
    if (ilink==2&&dctregion==1&&getName()=="IL2_D1") add=true;
    if (ilink==3&&dctregion==1&&getName()=="IL3_D1") add=true;
    if (ilink==1&&dctregion==2&&getName()=="IL1_D2") add=true;
    if (ilink==2&&dctregion==2&&getName()=="IL2_D2") add=true;
    if (ilink==3&&dctregion==2&&getName()=="IL3_D2") add=true;
    if (ilink==1&&dctregion==3&&getName()=="IL1_D3") add=true;
    if (ilink==2&&dctregion==3&&getName()=="IL2_D3") add=true;
    if (ilink==3&&dctregion==3&&getName()=="IL3_D3") add=true;
    if (ilink==1&&dctregion==4&&getName()=="IL1_D4") add=true;
    if (ilink==2&&dctregion==4&&getName()=="IL2_D4") add=true;
    if (ilink==3&&dctregion==4&&getName()=="IL3_D4") add=true;
    if (ilink==1&&dctregion==5&&getName()=="IL1_D5") add=true;
    if (ilink==2&&dctregion==5&&getName()=="IL2_D5") add=true;
    if (ilink==3&&dctregion==5&&getName()=="IL3_D5") add=true;
    if (ilink==1&&dctregion==6&&getName()=="IL1_D6") add=true;
    if (ilink==2&&dctregion==6&&getName()=="IL2_D6") add=true;
    if (ilink==3&&dctregion==6&&getName()=="IL3_D6") add=true;
    if (ilink==1&&dctregion==7&&getName()=="IL1_D7") add=true;
    if (ilink==2&&dctregion==7&&getName()=="IL2_D7") add=true;
    if (ilink==3&&dctregion==7&&getName()=="IL3_D7") add=true;
    if (ilink==1&&dctregion==8&&getName()=="IL1_D8") add=true;
    if (ilink==2&&dctregion==8&&getName()=="IL2_D8") add=true;
    if (ilink==3&&dctregion==8&&getName()=="IL3_D8") add=true;

    if (!add) {
      delete l1stub;
      delete stub;
      return;
    }
    //cout << "Will add stub in "<<getName()<<endl;
    if (stubs_.size()<MAXSTUBSLINK) {
      stubs_.push_back(tmp);
    }
  }

  unsigned int nStubs() const {return stubs_.size();}

  FPGAStub* getFPGAStub(unsigned int i) const {return stubs_[i].first;}
  L1TStub* getL1TStub(unsigned int i) const {return stubs_[i].second;}
  std::pair<FPGAStub*,L1TStub*> getStub(unsigned int i) const {return stubs_[i];}

  void writeStubs(bool first, bool w2) {

    //Barrel
    std::string fname="InputStubs_";
    fname+=getName();
    fname+="_";
    ostringstream oss;
    oss << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=oss.str();
    if(w2)
      fname+="_in2.dat";
    else
      fname+=".dat";

    if (first) {
      bx_=0;
      event_=1;
      out_.open(fname.c_str());
    }
    else 
      out_.open(fname.c_str(),std::ofstream::app);

    int nlay[6];
    for (unsigned int i=0;i<6;i++) {
      nlay[i]=0;
    }

    for (unsigned int i=0;i<stubs_.size();i++){
      if(stubs_[i].first->isBarrel()){
	int lay=stubs_[i].first->layer().value();
	assert(lay>=0);
	assert(lay<6);
	nlay[lay]++;
      }
    }
    unsigned long int nlay2[6];
    nlay2[0] = nlay[0];
    for(unsigned int i=1; i<6; ++i)
      nlay2[i] = nlay2[i-1]+nlay[i];

    string marker="";

    if(!w2) {
      //Write out the header
      out_ << "11111111111"<<marker;  
      out_ << (bitset<3>)bx_; //Dummy BX
      for (unsigned int i=0;i<3;i++) {
	out_ <<marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      out_ << "00000000000"<<marker<<(bitset<3>)bx_;  
      for (unsigned int i=3;i<6;i++) {
	out_ << marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      
      //now write the stubs in layer order
      string leftover="";
      
      for (unsigned int i=0;i<6;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isBarrel()){
	    unsigned int lay=stubs_[j].first->layer().value();
	    if (lay==i) {
	      string tmp=stubs_[j].first->strbare();
	      string stub=leftover+tmp.substr(0,32-leftover.size());
	      leftover=tmp.substr(32-leftover.size(),leftover.size()+4);
	      out_ << stub << endl;
	      if (leftover.size()==32){
		out_ << leftover << endl;
		leftover="";
	      }
	    }
	  }
	}
      }
      if (leftover.size()!=0) {
	string tmp="00000000000000000000000000000000";
	out_ << leftover << tmp.substr(0,32-leftover.size())<<endl;
      }
    }
    else{ 
      //format in2 for IPBUS
      unsigned int xline = 10;
      unsigned long int xword;
      //Header
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25)|(0x1ffffff);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //Stub counts
      xline = (xline+1)&15;
      xword = (nlay2[5])|(nlay2[4]<<6)|(nlay2[3]<<12)|(nlay2[2]<<18)
	|(nlay2[1]<<24)|(nlay2[0]<<30); 
      out_<< std::hex << xline<<"000";
      out_.fill('0');
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //the stubs      
      for (unsigned int i=0;i<6;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isBarrel()){
	    unsigned int lay=stubs_[j].first->layer().value();
	    if (lay==i) {
	      string tmp=stubs_[j].first->strbareUNFLIPPED();
	      bitset<36> btmp(tmp);
	      xword = btmp.to_ulong();
	      xline = (xline+1)&15;
	      out_<< std::hex << xline<<"000";
	      out_.width(4);
	      out_<<(xword>>20);
	      out_<< " "<< xline<<"00";
	      out_.width(5);
	      out_<<(xword & 0xfffff);
	      out_<<" 51000003 51000007\n";
	    }
	  }
	}
      }

      //Trailer
      xline = (xline+1)&15;
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
    }    
  
    out_.close();
      
    //Now the disks

    //forward

    fname="InputStubsDiskF_";
    fname+=getName();
    fname+="_";
    ostringstream ossF;
    ossF << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=ossF.str();
    if(w2)
      fname+="_in2.dat";
    else
      fname+=".dat";

    if (first) {
      bx_=0;
      event_=1;
      out_.open(fname.c_str());
    }
    else 
      out_.open(fname.c_str(),std::ofstream::app);

    for (unsigned int i=0;i<5;i++) {
      nlay[i]=0;
      nlay2[i]=0;
    }

    for (unsigned int i=0;i<stubs_.size();i++){
      if(stubs_[i].first->isDisk()&&stubs_[i].first->disk().value()>0){
	int lay=stubs_[i].first->disk().value()-1;
	assert(lay>=0);
	assert(lay<5);
	nlay[lay]++;
      }
    }
    nlay[5] = 0;
    nlay2[5] = 0;
    nlay2[0] = nlay[0];
    for(unsigned int i=1; i<5; ++i)
      nlay2[i] = nlay2[i-1]+nlay[i];

    if(!w2) {
      //Write out the header
      out_ << "11111111111"<<marker;  
      out_ << (bitset<3>)bx_; //Dummy BX
      for (unsigned int i=0;i<3;i++) {
	out_ <<marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      out_ << "00000000000"<<marker<<(bitset<3>)bx_;  
      for (unsigned int i=3;i<6;i++) {
	out_ << marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      
      //now write the stubs in disk order
      string leftover="";
      
      for (unsigned int i=0;i<5;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isDisk()&&stubs_[j].first->disk().value()>0){
	    unsigned int lay=stubs_[j].first->disk().value()-1;
	    if (lay==i) {
	      string tmp=stubs_[j].first->strbare();
	      string stub=leftover+tmp.substr(0,32-leftover.size());
	      leftover=tmp.substr(32-leftover.size(),leftover.size()+4);
	      out_ << stub << endl;
	      if (leftover.size()==32){
		out_ << leftover << endl;
		leftover="";
	      }
	    }
	  }
	}
      }
      if (leftover.size()!=0) {
	string tmp="00000000000000000000000000000000";
	out_ << leftover << tmp.substr(0,32-leftover.size())<<endl;
      }
    }
    else{ 
      //format in2 for IPBUS
      unsigned int xline = 10;
      unsigned long int xword;
      //Header
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25)|(0x1ffffff);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //Stub counts
      xline = (xline+1)&15;
      xword = (nlay2[5])|(nlay2[4]<<6)|(nlay2[3]<<12)|(nlay2[2]<<18)
	|(nlay2[1]<<24)|(nlay2[0]<<30); 
      out_<< std::hex << xline<<"000";
      out_.fill('0');
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //the stubs      
      for (unsigned int i=0;i<5;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isDisk()&&stubs_[j].first->disk().value()>0){
	    unsigned int lay=stubs_[j].first->disk().value()-1;
	    if (lay==i) {
	      string tmp=stubs_[j].first->strbareUNFLIPPED();
	      bitset<36> btmp(tmp);
	      xword = btmp.to_ulong();
	      xline = (xline+1)&15;
	      out_<< std::hex << xline<<"000";
	      out_.width(4);
	      out_<<(xword>>20);
	      out_<< " "<< xline<<"00";
	      out_.width(5);
	      out_<<(xword & 0xfffff);
	      out_<<" 51000003 51000007\n";
	    }
	  }
	}
      }

      //Trailer
      xline = (xline+1)&15;
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";

    }    
  
    out_.close();

    //back

    fname="InputStubsDiskB_";
    fname+=getName();
    fname+="_";
    ostringstream ossB;
    ossB << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=ossB.str();
    if(w2)
      fname+="_in2.dat";
    else
      fname+=".dat";

    if (first) {
      bx_=0;
      event_=1;
      out_.open(fname.c_str());
    }
    else 
      out_.open(fname.c_str(),std::ofstream::app);

    for (unsigned int i=0;i<5;i++) {
      nlay[i]=0;
      nlay2[i]=0;
    }

    for (unsigned int i=0;i<stubs_.size();i++){
      if(stubs_[i].first->isDisk()&&stubs_[i].first->disk().value()<0){
	int lay=-stubs_[i].first->disk().value()-1;
	assert(lay>=0);
	assert(lay<5);
	nlay[lay]++;
      }
    }
    nlay[5] = 0;
    nlay2[5] = 0;
    nlay2[0] = nlay[0];
    for(unsigned int i=1; i<5; ++i)
      nlay2[i] = nlay2[i-1]+nlay[i];

    if(!w2) {
      //Write out the header
      out_ << "11111111111"<<marker;  
      out_ << (bitset<3>)bx_; //Dummy BX
      for (unsigned int i=0;i<3;i++) {
	out_ <<marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      out_ << "00000000000"<<marker<<(bitset<3>)bx_;  
      for (unsigned int i=3;i<6;i++) {
	out_ << marker<<(bitset<6>)nlay[i];
      }
      out_ << endl;
      
      //now write the stubs in disk order
      string leftover="";
      
      for (unsigned int i=0;i<5;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isDisk()&&stubs_[j].first->disk().value()<0){
	    unsigned int lay=-stubs_[j].first->disk().value()-1;
	    if (lay==i) {
	      string tmp=stubs_[j].first->strbare();
	      string stub=leftover+tmp.substr(0,32-leftover.size());
	      leftover=tmp.substr(32-leftover.size(),leftover.size()+4);
	      out_ << stub << endl;
	      if (leftover.size()==32){
		out_ << leftover << endl;
		leftover="";
	      }
	    }
	  }
	}
      }
      if (leftover.size()!=0) {
	string tmp="00000000000000000000000000000000";
	out_ << leftover << tmp.substr(0,32-leftover.size())<<endl;
      }
    }
    else{ 
      //format in2 for IPBUS
      unsigned int xline = 10;
      unsigned long int xword;
      //Header
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25)|(0x1ffffff);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //Stub counts
      xline = (xline+1)&15;
      xword = (nlay2[5])|(nlay2[4]<<6)|(nlay2[3]<<12)|(nlay2[2]<<18)
	|(nlay2[1]<<24)|(nlay2[0]<<30); 
      out_<< std::hex << xline<<"000";
      out_.fill('0');
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";
      //the stubs      
      for (unsigned int i=0;i<5;i++) {
	for (unsigned int j=0;j<stubs_.size();j++){
	  if(stubs_[j].first->isDisk()&&stubs_[j].first->disk().value()<0){
	    unsigned int lay=-stubs_[j].first->disk().value()-1;
	    if (lay==i) {
	      string tmp=stubs_[j].first->strbareUNFLIPPED();
	      bitset<36> btmp(tmp);
	      xword = btmp.to_ulong();
	      xline = (xline+1)&15;
	      out_<< std::hex << xline<<"000";
	      out_.width(4);
	      out_<<(xword>>20);
	      out_<< " "<< xline<<"00";
	      out_.width(5);
	      out_<<(xword & 0xfffff);
	      out_<<" 51000003 51000007\n";
	    }
	  }
	}
      }

      //Trailer
      xline = (xline+1)&15;
      xword = ((unsigned long int)7<<33)|(((unsigned long int) bx_)<<25);
      out_<< std::hex << xline<<"000";
      out_.width(4);
      out_<<(xword>>20);
      out_<< " "<< xline<<"00";
      out_.width(5);
      out_<<(xword & 0xfffff);
      out_<<" 51000003 51000007\n";

    }    
  
    out_.close();

    //increment BX
    bx_++;
    if (bx_>7) bx_=0;

  }
  
  

  void clean() {
    for(unsigned int i=0;i<stubs_.size();i++){
      delete stubs_[i].first;
      delete stubs_[i].second;
    }
    stubs_.clear();
  }


private:

  double phimin_;
  double phimax_;
  std::vector<std::pair<FPGAStub*,L1TStub*> > stubs_;

};

#endif
