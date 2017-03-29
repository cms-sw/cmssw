//This class implementes the projection router
#ifndef FPGAPROJECTIONROUTER_H
#define FPGAPROJECTIONROUTER_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAProjectionRouter:public FPGAProcessBase{

public:

  FPGAProjectionRouter(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    string subname=name.substr(0,5);
    //cout << "subname : "<<subname<<endl;
    layer_=0;
    disk_=0;
    vmprojPHI1R1_=0;
    vmprojPHI1R2_=0;
    vmprojPHI2R1_=0;
    vmprojPHI2R2_=0;
    vmprojPHI3R1_=0;
    vmprojPHI3R2_=0;
    vmprojPHI4R1_=0;
    vmprojPHI4R2_=0;

    if (subname=="PR_L1") layer_=1;
    if (subname=="PR_L2") layer_=2;
    if (subname=="PR_L3") layer_=3;
    if (subname=="PR_L4") layer_=4;
    if (subname=="PR_L5") layer_=5;
    if (subname=="PR_L6") layer_=6;
    if (subname=="PR_F1") disk_=1;
    if (subname=="PR_F2") disk_=2;
    if (subname=="PR_F3") disk_=3;
    if (subname=="PR_F4") disk_=4;
    if (subname=="PR_F5") disk_=5;
    if (subname=="PR_B1") disk_=-1;
    if (subname=="PR_B2") disk_=-2;
    if (subname=="PR_B3") disk_=-3;
    if (subname=="PR_B4") disk_=-4;
    if (subname=="PR_B5") disk_=-5;
    allproj_=0;
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="allprojout"){
      FPGAAllProjections* tmp=dynamic_cast<FPGAAllProjections*>(memory);
      assert(tmp!=0);
      allproj_=tmp;
      return;
    }
    if (output=="vmprojoutPHI1Z1"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI1Z1_=tmp;
      return;
    }
    if (output=="vmprojoutPHI1Z2"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI1Z2_=tmp;
      return;
    }
    if (output=="vmprojoutPHI2Z1"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI2Z1_=tmp;
      return;
    }
    if (output=="vmprojoutPHI2Z2"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI2Z2_=tmp;
      return;
    }
    if (output=="vmprojoutPHI3Z1"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI3Z1_=tmp;
      return;
    }
    if (output=="vmprojoutPHI3Z2"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI3Z2_=tmp;
      return;
    }
    if (output=="vmprojoutPHI4Z1"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI4Z1_=tmp;
      return;
    }
    if (output=="vmprojoutPHI4Z2"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI4Z2_=tmp;
      return;
    }
    if (output=="vmprojoutPHI1R1"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI1R1_=tmp;
      return;
    }
    if (output=="vmprojoutPHI1R2"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI1R2_=tmp;
      return;
    }
    if (output=="vmprojoutPHI2R1"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI2R1_=tmp;
      return;
    }
    if (output=="vmprojoutPHI2R2"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI2R2_=tmp;
      return;
    }
    if (output=="vmprojoutPHI3R1"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI3R1_=tmp;
      return;
    }
    if (output=="vmprojoutPHI3R2"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI3R2_=tmp;
      return;
    }
    if (output=="vmprojoutPHI4R1"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI4R1_=tmp;
      return;
    }
    if (output=="vmprojoutPHI4R2"){
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojPHI4R2_=tmp;
      return;
    }
    cout << "Did not find output : "<<output<<endl;
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="proj1in"||input=="proj2in"||
	input=="proj3in"||input=="proj4in"||
	input=="proj5in"||input=="proj6in"||
	input=="proj7in"||input=="proj8in"||
	input=="proj9in"||input=="proj10in"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputproj_.push_back(tmp);
      return;
    }
    if (input=="projplusin"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputplusproj_=tmp;
      return;
    }
    if (input=="projminusin"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      inputminusproj_=tmp;
      return;
    }
    cout << "Could not find input : "<<input<<endl;
    assert(0);
  }

  void execute() {

    if (layer_!=0) {
      for (unsigned int j=0;j<inputproj_.size();j++){
	//cout << "Inputproj : "<<inputproj_[j]->getName()<<" "
	//     <<inputproj_[j]->nTracklets()<<endl;
	for (unsigned int i=0;i<inputproj_[j]->nTracklets();i++){
	  //cout << "Doing projection"<<endl;
	  
	  FPGAWord fpgaphi=inputproj_[j]->getFPGATracklet(i)->fpgaphiproj(layer_);
	  FPGAWord fpgaz=inputproj_[j]->getFPGATracklet(i)->fpgazproj(layer_);

	  //if (inputproj_[j]->getFPGATracklet(i)->plusNeighbor(layer_)) {
	  //  cout << "Found plus neighbor in : "<<inputproj_[j]->getName()<<endl;
	  //} 
	  
	  //skip if projection is out of range!
	  if (fpgaz.atExtreme()) continue;
	  if (fpgaphi.atExtreme()) continue;
	
	  int iz=4+(fpgaz.value()>>(fpgaz.nbits()-3));
	  int iphitmp=fpgaphi.value();
	  if ((layer_%2)==1) iphitmp-=(1<<(fpgaphi.nbits()-3));  
	  int iphi=iphitmp>>(fpgaphi.nbits()-2);
	  
	  //cout << "iphi iz : "<<iphi<<" "<<iz<<endl;
	  
	  iz=iz%2;
	  
	  //if (layer_==3) {
	  //  cout << "Will add to allproj_ in "<<getName()
	  //	 <<" z = "<<fpgaz.value()*kzproj
	  //		 <<" from :"<<inputproj_[j]->getName()
	  //		 <<endl;
	  //}

	  assert(allproj_!=0);

	  allproj_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  
	  if (iphi==0&&iz==0) {
	    vmprojPHI1Z1_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  if (iphi==0&&iz==1) {
	    vmprojPHI1Z2_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  
	  if (iphi==1&&iz==0) {
	    vmprojPHI2Z1_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  if (iphi==1&&iz==1) {
	    vmprojPHI2Z2_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  
	  if (iphi==2&&iz==0) {
	    vmprojPHI3Z1_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  if (iphi==2&&iz==1) {
	    vmprojPHI3Z2_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	
	  if (iphi==3&&iz==0) {
	    vmprojPHI4Z1_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  if (iphi==3&&iz==1) {
	    vmprojPHI4Z2_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	}
      }
    } else {
      for (unsigned int j=0;j<inputproj_.size();j++){
	for (unsigned int i=0;i<inputproj_[j]-> nTracklets();i++){
	  //cout << "Doing disk projection disk="<<disk_<<" in "<<getName()<<endl;
	  //assert(inputproj_[j]->getFPGATracklet(i)->isDisk());
	  FPGAWord fpgaphi=inputproj_[j]->getFPGATracklet(i)->fpgaphiprojdisk(disk_);
	  FPGAWord fpgar=inputproj_[j]->getFPGATracklet(i)->fpgarprojdisk(disk_);
	  
	  //skip if projection is out of range!
	  if (fpgar.atExtreme()) continue;
	  if (fpgaphi.atExtreme()) continue;
	
	  int ir=(1<<2)*(fpgar.value()*krprojshiftdisk-rmindisk)/(rmaxdisk-rmindisk);

	  int iphitmp=fpgaphi.value();
	  if ((disk_%2)==0) iphitmp-=(1<<(fpgaphi.nbits()-3));  
	  int iphi=iphitmp>>(fpgaphi.nbits()-2);
	  
	  //cout << "iphi ir : "<<iphi<<" "<<ir<<endl;
	  
	  ir=ir%2;
	  
	  //cout << "Will add to allproj_ in "<<getName()<<endl;
	  assert(allproj_!=0);
	  allproj_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  
	  if (iphi==0&&ir==0) {
	    vmprojPHI1R1_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  if (iphi==0&&ir==1) {
	    vmprojPHI1R2_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  
	  if (iphi==1&&ir==0) {
	    vmprojPHI2R1_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  if (iphi==1&&ir==1) {
	    vmprojPHI2R2_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  
	  if (iphi==2&&ir==0) {
	    vmprojPHI3R1_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  if (iphi==2&&ir==1) {
	    vmprojPHI3R2_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	
	  if (iphi==3&&ir==0&&vmprojPHI4R1_!=0) {
	    vmprojPHI4R1_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	  if (iphi==3&&ir==1&&vmprojPHI4R2_!=0) {
	    vmprojPHI4R2_->addTracklet(inputproj_[j]->getFPGATracklet(i));
	  }
	}
      }
    }


    if (layer_!=0) {
      if (writeAllProjections) {
	static ofstream out("allprojections.txt"); 
	out << getName() << " " << allproj_->nTracklets() << endl;
      } 
    }

    if (layer_!=0) {
      if (writeVMProjections) {
	static ofstream out("vmprojections.txt"); 
	out << vmprojPHI1Z1_->getName() << " " << vmprojPHI1Z1_->nTracklets() << endl; 
	out << vmprojPHI1Z2_->getName() << " " << vmprojPHI1Z2_->nTracklets() << endl; 
	out << vmprojPHI2Z1_->getName() << " " << vmprojPHI2Z1_->nTracklets() << endl; 
	out << vmprojPHI2Z2_->getName() << " " << vmprojPHI2Z2_->nTracklets() << endl; 
	out << vmprojPHI3Z1_->getName() << " " << vmprojPHI3Z1_->nTracklets() << endl; 
	out << vmprojPHI3Z2_->getName() << " " << vmprojPHI3Z2_->nTracklets() << endl; 
	if (layer_%2==0) {
	  out << vmprojPHI4Z1_->getName() << " " << vmprojPHI4Z1_->nTracklets() << endl; 
	  out << vmprojPHI4Z2_->getName() << " " << vmprojPHI4Z2_->nTracklets() << endl;
	} 
      }
    }
  }
  

private:

  int layer_; 
  int disk_; 

  vector<FPGATrackletProjections*> inputproj_;
  FPGATrackletProjections* inputplusproj_;
  FPGATrackletProjections* inputminusproj_;

  FPGAAllProjections* allproj_;
  FPGAVMProjections* vmprojPHI1Z1_;
  FPGAVMProjections* vmprojPHI1Z2_;
  FPGAVMProjections* vmprojPHI2Z1_;
  FPGAVMProjections* vmprojPHI2Z2_;
  FPGAVMProjections* vmprojPHI3Z1_;
  FPGAVMProjections* vmprojPHI3Z2_;
  FPGAVMProjections* vmprojPHI4Z1_;
  FPGAVMProjections* vmprojPHI4Z2_;

  FPGAVMProjections* vmprojPHI1R1_;
  FPGAVMProjections* vmprojPHI1R2_;
  FPGAVMProjections* vmprojPHI2R1_;
  FPGAVMProjections* vmprojPHI2R2_;
  FPGAVMProjections* vmprojPHI3R1_;
  FPGAVMProjections* vmprojPHI3R2_;
  FPGAVMProjections* vmprojPHI4R1_;
  FPGAVMProjections* vmprojPHI4R2_;


};

#endif
