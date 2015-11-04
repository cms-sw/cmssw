//This class implementes the projection tranceiver
#ifndef FPGAPROJECTIONTRANSCEIVER_H
#define FPGAPROJECTIONTRANSCEIVER_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAProjectionTransceiver:public FPGAProcessBase{

public:

  FPGAProjectionTransceiver(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="projout"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojections_.push_back(tmp);
      return;
    }
    /*
    if (output=="output_L1L2_2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojectionsL1L2_2_=tmp;
      return;
    }
    if (output=="output_L1L2_3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojectionsL1L2_3_=tmp;
      return;
    }
    if (output=="output_L1L2_4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojectionsL1L2_4_=tmp;
      return;
    }
    if (output=="output_L3L4_1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojectionsL3L4_1_=tmp;
      return;
    }
    if (output=="output_L3L4_2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojectionsL3L4_2_=tmp;
      return;
    }
    if (output=="output_L3L4_3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojectionsL3L4_3_=tmp;
      return;
    }
    if (output=="output_L3L4_4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojectionsL3L4_4_=tmp;
      return;
    }
    if (output=="output_L5L6_1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojectionsL5L6_1_=tmp;
      return;
    }
    if (output=="output_L5L6_2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojectionsL5L6_2_=tmp;
      return;
    }
    if (output=="output_L5L6_3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojectionsL5L6_3_=tmp;
      return;
    }
    if (output=="output_L5L6_4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      outputprojectionsL5L6_4_=tmp;
      return;
    }
    */
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }

    /*
    if (input=="input_L1L2_1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL1L2_1_=tmp;
      return;
    }
    if (input=="input_L1L2_2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL1L2_2_=tmp;
      return;
    }
    if (input=="input_L1L2_3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL1L2_3_=tmp;
      return;
    }
    if (input=="input_L1L2_4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL1L2_4_=tmp;
      return;
    }
    if (input=="input_L3L4_1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL3L4_1_=tmp;
      return;
    }
    if (input=="input_L3L4_2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL3L4_2_=tmp;
      return;
    }
    if (input=="input_L3L4_3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL3L4_3_=tmp;
      return;
    }
    if (input=="input_L3L4_4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL3L4_4_=tmp;
      return;
    }
    if (input=="input_L5L6_1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL5L6_1_=tmp;
      return;
    }
    if (input=="input_L5L6_2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL5L6_2_=tmp;
      return;
    }
    if (input=="input_L5L6_3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL5L6_3_=tmp;
      return;
    }
    if (input=="input_L5L6_4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojectionsL5L6_4_=tmp;
      return;
    }
    */
    if (input=="projin"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputprojections_.push_back(tmp);
      return;
    }

    assert(0);
  }

  //Copy otherSector->inputprojections_ to this->outputprojections_ 
  void execute(FPGAProjectionTransceiver* otherSector){
    int count=0;
    //cout << "in FPGAProjectionTransceiver "<<otherSector->inputprojections_.size()<<endl;
    for(unsigned int i=0;i<otherSector->inputprojections_.size();i++){
      FPGATrackletProjections* otherProj=otherSector->inputprojections_[i];
      if (otherProj->nTracklets()==0) continue;
      string regionname=otherProj->getName().substr(otherProj->getName().size()-2,2);      
      //cout << "Non empty projection : "<<otherProj->getName()<<" "<<regionname<<endl;     
      bool wrote=false;
      for(unsigned int j=0;j<outputprojections_.size();j++){
	FPGATrackletProjections* thisProj=outputprojections_[j];
	std::size_t found = thisProj->getName().find(regionname);
	if (found!=std::string::npos) {
	  for (unsigned int l=0;l<otherProj->nTracklets();l++){
	    int layer=0;
	    int disk=0;
	    if (regionname=="L1") layer=1;
	    if (regionname=="L2") layer=2;
	    if (regionname=="L3") layer=3;
	    if (regionname=="L4") layer=4;
	    if (regionname=="L5") layer=5;
	    if (regionname=="L6") layer=6;
	    if (regionname=="F1") disk=1;
	    if (regionname=="F2") disk=2;
	    if (regionname=="F3") disk=3;
	    if (regionname=="F4") disk=4;
	    if (regionname=="F5") disk=5;
	    if (regionname=="B1") disk=-1;
	    if (regionname=="B2") disk=-2;
	    if (regionname=="B3") disk=-3;
	    if (regionname=="B4") disk=-4;
	    if (regionname=="B5") disk=-5;

	    assert((layer!=0)||(disk!=0));

	    if (layer!=0) {

	      FPGAWord fpgaz=otherProj->getFPGATracklet(l)->fpgazproj(layer);

	      assert(!fpgaz.atExtreme());

	      int iz=4+(fpgaz.value()>>(fpgaz.nbits()-3));
	      iz=iz/2+1;
	    
	      assert(iz>0);
	      assert(iz<=4);

	      string dctregion=thisProj->getName().substr(thisProj->getName().size()-2,2);      

	      int idct=0;
	      if (dctregion=="D1") idct=1;
	      if (dctregion=="D2") idct=2;
	      if (dctregion=="D3") idct=3;
	      if (dctregion=="D4") idct=4;
	      assert(idct!=0);

	      //cout << " Adding to target: "<<thisProj->getName()
	      //	   <<" "<<regionname<<" "<<iz<<" "<<dctregion<<" "<<idct<<endl;
	      if (iz!=idct) continue;
	      wrote=true;
	      count++;
	      thisProj->addTracklet(otherProj->getFPGATracklet(l));
	    }


	    if (disk!=0) {

	      FPGAWord fpgar=otherProj->getFPGATracklet(l)->fpgarprojdisk(disk);

	      assert(!fpgar.atExtreme());

	      int ir=2*(fpgar.value()*krprojshiftdisk-rmindisk)/(rmaxdisk-rmindisk)+1;

	      assert(ir>0);
	      assert(ir<=2);

	      string dctregion=thisProj->getName().substr(thisProj->getName().size()-2,2);      

	      int idct=0;
	      if (dctregion=="D5") idct=1;
	      if (dctregion=="D6") idct=2;
	      if (dctregion=="D7") idct=1;
	      if (dctregion=="D8") idct=2;
	      assert(idct!=0);

	      //cout << " Adding to target: "<<thisProj->getName()
	      //     <<" "<<regionname<<" "<<ir<<" "<<dctregion<<" "<<idct<<endl;
	      if (ir!=idct) continue;
	      wrote=true;
	      count++;
	      thisProj->addTracklet(otherProj->getFPGATracklet(l));
	    }


	  }
	} 
      }
      assert(wrote!=false);
    }

    if (writeProjectionTransceiver) {
      static ofstream out("projectiontransceiver.txt");
      out << getName() << " " 
	  << count << endl;
    }

  }


private:

  vector<FPGATrackletProjections*> inputprojections_;

  vector<FPGATrackletProjections*> outputprojections_;


  /*
  FPGATrackletProjections* inputprojectionsL1L2_1_;
  FPGATrackletProjections* outputprojectionsL1L2_1_;

  FPGATrackletProjections* inputprojectionsL1L2_2_;
  FPGATrackletProjections* outputprojectionsL1L2_2_;

  FPGATrackletProjections* inputprojectionsL1L2_3_;
  FPGATrackletProjections* outputprojectionsL1L2_3_;

  FPGATrackletProjections* inputprojectionsL1L2_4_;
  FPGATrackletProjections* outputprojectionsL1L2_4_;

  FPGATrackletProjections* inputprojectionsL3L4_1_;
  FPGATrackletProjections* outputprojectionsL3L4_1_;

  FPGATrackletProjections* inputprojectionsL3L4_2_;
  FPGATrackletProjections* outputprojectionsL3L4_2_;

  FPGATrackletProjections* inputprojectionsL3L4_3_;
  FPGATrackletProjections* outputprojectionsL3L4_3_;

  FPGATrackletProjections* inputprojectionsL3L4_4_;
  FPGATrackletProjections* outputprojectionsL3L4_4_;

  FPGATrackletProjections* inputprojectionsL5L6_1_;
  FPGATrackletProjections* outputprojectionsL5L6_1_;

  FPGATrackletProjections* inputprojectionsL5L6_2_;
  FPGATrackletProjections* outputprojectionsL5L6_2_;

  FPGATrackletProjections* inputprojectionsL5L6_3_;
  FPGATrackletProjections* outputprojectionsL5L6_3_;

  FPGATrackletProjections* inputprojectionsL5L6_4_;
  FPGATrackletProjections* outputprojectionsL5L6_4_;
  */

};

#endif
