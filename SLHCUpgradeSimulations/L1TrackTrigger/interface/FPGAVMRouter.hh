//This class implementes the VM router
#ifndef FPGAVMROUTER_H
#define FPGAVMROUTER_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAVMRouter:public FPGAProcessBase{

public:

  FPGAVMRouter(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="allstubout"||
	output=="allstuboutn1"||
	output=="allstuboutn2"||
	output=="allstuboutn3"||
	output=="allstuboutn4"||
	output=="allstuboutn5"||
	output=="allstuboutn6"
	){
      FPGAAllStubs* tmp=dynamic_cast<FPGAAllStubs*>(memory);
      assert(tmp!=0);
      allstubs_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI1Z1n1"||
	output=="vmstuboutPHI1Z1n2"||
	output=="vmstuboutPHI1Z1n3"||
	output=="vmstuboutPHI1Z1n4"||
	output=="vmstuboutPHI1Z1n5"||
	output=="vmstuboutPHI1Z1n6"||
	output=="vmstuboutPHI1Z1n7"||
	output=="vmstuboutPHI1Z1n8"||
	output=="vmstuboutPHI1Z1n9"||
	output=="vmstuboutPHI1Z1n10"||
	output=="vmstuboutPHI1Z1n11"||
	output=="vmstuboutPHI1Z1n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI1Z1_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI1Z2n1"||
	output=="vmstuboutPHI1Z2n2"||
	output=="vmstuboutPHI1Z2n3"||
	output=="vmstuboutPHI1Z2n4"||
	output=="vmstuboutPHI1Z2n5"||
	output=="vmstuboutPHI1Z2n6"||
	output=="vmstuboutPHI1Z2n7"||
	output=="vmstuboutPHI1Z2n8"||
	output=="vmstuboutPHI1Z2n9"||
	output=="vmstuboutPHI1Z2n10"||
	output=="vmstuboutPHI1Z2n11"||
	output=="vmstuboutPHI1Z2n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI1Z2_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI2Z1n1"||
	output=="vmstuboutPHI2Z1n2"||
	output=="vmstuboutPHI2Z1n3"||
	output=="vmstuboutPHI2Z1n4"||
	output=="vmstuboutPHI2Z1n5"||
	output=="vmstuboutPHI2Z1n6"||
	output=="vmstuboutPHI2Z1n7"||
	output=="vmstuboutPHI2Z1n8"||
	output=="vmstuboutPHI2Z1n9"||
	output=="vmstuboutPHI2Z1n10"||
	output=="vmstuboutPHI2Z1n11"||
	output=="vmstuboutPHI2Z1n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI2Z1_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI2Z2n1"||
	output=="vmstuboutPHI2Z2n2"||
	output=="vmstuboutPHI2Z2n3"||
	output=="vmstuboutPHI2Z2n4"||
	output=="vmstuboutPHI2Z2n5"||
	output=="vmstuboutPHI2Z2n6"||
	output=="vmstuboutPHI2Z2n7"||
	output=="vmstuboutPHI2Z2n8"||
	output=="vmstuboutPHI2Z2n9"||
	output=="vmstuboutPHI2Z2n10"||
	output=="vmstuboutPHI2Z2n11"||
	output=="vmstuboutPHI2Z2n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI2Z2_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI3Z1n1"||
	output=="vmstuboutPHI3Z1n2"||
	output=="vmstuboutPHI3Z1n3"||
	output=="vmstuboutPHI3Z1n4"||
	output=="vmstuboutPHI3Z1n5"||
	output=="vmstuboutPHI3Z1n6"||
	output=="vmstuboutPHI3Z1n7"||
	output=="vmstuboutPHI3Z1n8"||
	output=="vmstuboutPHI3Z1n9"||
	output=="vmstuboutPHI3Z1n10"||
	output=="vmstuboutPHI3Z1n11"||
	output=="vmstuboutPHI3Z1n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI3Z1_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI3Z2n1"||
	output=="vmstuboutPHI3Z2n2"||
	output=="vmstuboutPHI3Z2n3"||
	output=="vmstuboutPHI3Z2n4"||
	output=="vmstuboutPHI3Z2n5"||
	output=="vmstuboutPHI3Z2n6"||
	output=="vmstuboutPHI3Z2n7"||
	output=="vmstuboutPHI3Z2n8"||
	output=="vmstuboutPHI3Z2n9"||
	output=="vmstuboutPHI3Z2n10"||
	output=="vmstuboutPHI3Z2n11"||
	output=="vmstuboutPHI3Z2n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI3Z2_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI4Z1n1"||
	output=="vmstuboutPHI4Z1n2"||
	output=="vmstuboutPHI4Z1n3"||
	output=="vmstuboutPHI4Z1n4"||
	output=="vmstuboutPHI4Z1n5"||
	output=="vmstuboutPHI4Z1n6"||
	output=="vmstuboutPHI4Z1n7"||
	output=="vmstuboutPHI4Z1n8"||
	output=="vmstuboutPHI4Z1n9"||
	output=="vmstuboutPHI4Z1n10"||
	output=="vmstuboutPHI4Z1n11"||
	output=="vmstuboutPHI4Z1n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI4Z1_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI4Z2n1"||
	output=="vmstuboutPHI4Z2n2"||
	output=="vmstuboutPHI4Z2n3"||
	output=="vmstuboutPHI4Z2n4"||
	output=="vmstuboutPHI4Z2n5"||
	output=="vmstuboutPHI4Z2n6"||
	output=="vmstuboutPHI4Z2n7"||
	output=="vmstuboutPHI4Z2n8"||
	output=="vmstuboutPHI4Z2n9"||
	output=="vmstuboutPHI4Z2n10"||
	output=="vmstuboutPHI4Z2n11"||
	output=="vmstuboutPHI4Z2n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI4Z2_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI1R1"||
	output=="vmstuboutPHI1R1n1"||
	output=="vmstuboutPHI1R1n2"||
	output=="vmstuboutPHI1R1n3"||
	output=="vmstuboutPHI1R1n4"||
	output=="vmstuboutPHI1R1n5"||
	output=="vmstuboutPHI1R1n6"||
	output=="vmstuboutPHI1R1n7"||
	output=="vmstuboutPHI1R1n8"||
	output=="vmstuboutPHI1R1n9"||
	output=="vmstuboutPHI1R1n10"||
	output=="vmstuboutPHI1R1n11"||
	output=="vmstuboutPHI1R1n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI1R1_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI1R2"||
	output=="vmstuboutPHI1R2n1"||
	output=="vmstuboutPHI1R2n2"||
	output=="vmstuboutPHI1R2n3"||
	output=="vmstuboutPHI1R2n4"||
	output=="vmstuboutPHI1R2n5"||
	output=="vmstuboutPHI1R2n6"||
	output=="vmstuboutPHI1R2n7"||
	output=="vmstuboutPHI1R2n8"||
	output=="vmstuboutPHI1R2n9"||
	output=="vmstuboutPHI1R2n10"||
	output=="vmstuboutPHI1R2n11"||
	output=="vmstuboutPHI1R2n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI1R2_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI2R1"||
	output=="vmstuboutPHI2R1n1"||
	output=="vmstuboutPHI2R1n2"||
	output=="vmstuboutPHI2R1n3"||
	output=="vmstuboutPHI2R1n4"||
	output=="vmstuboutPHI2R1n5"||
	output=="vmstuboutPHI2R1n6"||
	output=="vmstuboutPHI2R1n7"||
	output=="vmstuboutPHI2R1n8"||
	output=="vmstuboutPHI2R1n9"||
	output=="vmstuboutPHI2R1n10"||
	output=="vmstuboutPHI2R1n11"||
	output=="vmstuboutPHI2R1n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI2R1_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI2R2"||
	output=="vmstuboutPHI2R2n1"||
	output=="vmstuboutPHI2R2n2"||
	output=="vmstuboutPHI2R2n3"||
	output=="vmstuboutPHI2R2n4"||
	output=="vmstuboutPHI2R2n5"||
	output=="vmstuboutPHI2R2n6"||
	output=="vmstuboutPHI2R2n7"||
	output=="vmstuboutPHI2R2n8"||
	output=="vmstuboutPHI2R2n9"||
	output=="vmstuboutPHI2R2n10"||
	output=="vmstuboutPHI2R2n11"||
	output=="vmstuboutPHI2R2n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI2R2_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI3R1"||
	output=="vmstuboutPHI3R1n1"||
	output=="vmstuboutPHI3R1n2"||
	output=="vmstuboutPHI3R1n3"||
	output=="vmstuboutPHI3R1n4"||
	output=="vmstuboutPHI3R1n5"||
	output=="vmstuboutPHI3R1n6"||
	output=="vmstuboutPHI3R1n7"||
	output=="vmstuboutPHI3R1n8"||
	output=="vmstuboutPHI3R1n9"||
	output=="vmstuboutPHI3R1n10"||
	output=="vmstuboutPHI3R1n11"||
	output=="vmstuboutPHI3R1n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI3R1_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI3R2"||
	output=="vmstuboutPHI3R2n1"||
	output=="vmstuboutPHI3R2n2"||
	output=="vmstuboutPHI3R2n3"||
	output=="vmstuboutPHI3R2n4"||
	output=="vmstuboutPHI3R2n5"||
	output=="vmstuboutPHI3R2n6"||
	output=="vmstuboutPHI3R2n7"||
	output=="vmstuboutPHI3R2n8"||
	output=="vmstuboutPHI3R2n9"||
	output=="vmstuboutPHI3R2n10"||
	output=="vmstuboutPHI3R2n11"||
	output=="vmstuboutPHI3R2n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI3R2_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI4R1"||
	output=="vmstuboutPHI4R1n1"||
	output=="vmstuboutPHI4R1n2"||
	output=="vmstuboutPHI4R1n3"||
	output=="vmstuboutPHI4R1n4"||
	output=="vmstuboutPHI4R1n5"||
	output=="vmstuboutPHI4R1n6"||
	output=="vmstuboutPHI4R1n7"||
	output=="vmstuboutPHI4R1n8"||
	output=="vmstuboutPHI4R1n9"||
	output=="vmstuboutPHI4R1n10"||
	output=="vmstuboutPHI4R1n11"||
	output=="vmstuboutPHI4R1n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI4R1_.push_back(tmp);
      return;
    }
    if (output=="vmstuboutPHI4R2"||
	output=="vmstuboutPHI4R2n1"||
	output=="vmstuboutPHI4R2n2"||
	output=="vmstuboutPHI4R2n3"||
	output=="vmstuboutPHI4R2n4"||
	output=="vmstuboutPHI4R2n5"||
	output=="vmstuboutPHI4R2n6"||
	output=="vmstuboutPHI4R2n7"||
	output=="vmstuboutPHI4R2n8"||
	output=="vmstuboutPHI4R2n9"||
	output=="vmstuboutPHI4R2n10"||
	output=="vmstuboutPHI4R2n11"||
	output=="vmstuboutPHI4R2n12") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubsPHI4R2_.push_back(tmp);
      return;
    }
    cout << "Could not find : "<<output<<endl;
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="stubinLink1"||input=="stubinLink2"||input=="stubinLink3"){
      FPGAStubLayer* tmp1=dynamic_cast<FPGAStubLayer*>(memory);
      FPGAStubDisk* tmp2=dynamic_cast<FPGAStubDisk*>(memory);
      assert(tmp1!=0||tmp2!=0);
      if (tmp1!=0){
	stubinputs_.push_back(tmp1);
      }
      if (tmp2!=0){
	stubinputsdisk_.push_back(tmp2);
      }
      return;
    }
    cout << "Could not find input : "<<input<<endl;
    assert(0);
  }

  void execute(){

    //only one of these can be filled!
    assert(stubinputsdisk_.size()*stubinputs_.size()==0);

    if (stubinputs_.size()!=0){
      for(unsigned int j=0;j<stubinputs_.size();j++){
	for(unsigned int i=0;i<stubinputs_[j]->nStubs();i++){
	  std::pair<FPGAStub*,L1TStub*> stub=stubinputs_[j]->getStub(i);
	  //FIXME Next few lines should be member data in FPGAStub...
	  int iz=4+(stub.first->z().value()>>(stub.first->z().nbits()-3));
	  int iphitmp=stub.first->phi().value();
	  int layer=stub.first->layer().value()+1;
	  if ((layer%2)==1) iphitmp-=(1<<(stub.first->phi().nbits()-3));  
	  assert(iphitmp>=0);
	  int iphi=iphitmp>>(stub.first->phi().nbits()-2);

	  //cout << "iphi iz : "<<iphi<<" "<<iz<<endl;
	  assert(iz>=0);
	  assert(iz<=7);
	  iz=iz%2;
	  assert(iphi>=0);
	  assert(iphi<=3);

	  stub.first->setAllStubIndex(allstubs_[0]->nStubs());
	  stub.second->setAllStubIndex(allstubs_[0]->nStubs());

	  for (unsigned int l=0;l<allstubs_.size();l++){
	    allstubs_[l]->addStub(stub);
	  }

	  bool insert=false;

	  if (iphi==0&&iz==0) {
	    for (unsigned int l=0;l<vmstubsPHI1Z1_.size();l++){
	      vmstubsPHI1Z1_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI1Z1_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  if (iphi==0&&iz==1) {
	    for (unsigned int l=0;l<vmstubsPHI1Z2_.size();l++){
	      vmstubsPHI1Z2_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI1Z2_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  
	  if (iphi==1&&iz==0) {
	    for (unsigned int l=0;l<vmstubsPHI2Z1_.size();l++){
	      vmstubsPHI2Z1_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI2Z1_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  if (iphi==1&&iz==1) {
	    for (unsigned int l=0;l<vmstubsPHI2Z2_.size();l++){
	      vmstubsPHI2Z2_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI2Z2_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  
	  if (iphi==2&&iz==0) {
	    for (unsigned int l=0;l<vmstubsPHI3Z1_.size();l++){
	      vmstubsPHI3Z1_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI3Z1_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  if (iphi==2&&iz==1) {
	    for (unsigned int l=0;l<vmstubsPHI3Z2_.size();l++){
	      vmstubsPHI3Z2_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI3Z2_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  
	  if (iphi==3&&iz==0) {
	    for (unsigned int l=0;l<vmstubsPHI4Z1_.size();l++){
	      vmstubsPHI4Z1_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI4Z1_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  if (iphi==3&&iz==1) {
	    for (unsigned int l=0;l<vmstubsPHI4Z2_.size();l++){
	      vmstubsPHI4Z2_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI4Z2_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  assert(insert);
	}
      }

      if (writeVMOccupancy) {
	static ofstream out("vmoccupancy.txt");
	out<<vmstubsPHI1Z1_[0]->getName()<<" "<<vmstubsPHI1Z1_[0]->nStubs()<<endl;
	out<<vmstubsPHI1Z2_[0]->getName()<<" "<<vmstubsPHI1Z2_[0]->nStubs()<<endl;
	out<<vmstubsPHI2Z1_[0]->getName()<<" "<<vmstubsPHI2Z1_[0]->nStubs()<<endl;
	out<<vmstubsPHI2Z2_[0]->getName()<<" "<<vmstubsPHI2Z2_[0]->nStubs()<<endl;
	out<<vmstubsPHI3Z1_[0]->getName()<<" "<<vmstubsPHI3Z1_[0]->nStubs()<<endl;
	out<<vmstubsPHI3Z2_[0]->getName()<<" "<<vmstubsPHI3Z2_[0]->nStubs()<<endl;
	if (vmstubsPHI3Z2_[0]->getName()[5]-'0'%2==1) {
	  out<<vmstubsPHI4Z1_[0]->getName()<<" "<<vmstubsPHI4Z1_[0]->nStubs()<<endl;
	  out<<vmstubsPHI4Z2_[0]->getName()<<" "<<vmstubsPHI4Z2_[0]->nStubs()<<endl;
	}
      }

    }
    if (stubinputsdisk_.size()>0) {
      //cout << "Routing stubs in disk" <<endl;
      for(unsigned int j=0;j<stubinputsdisk_.size();j++){
	for(unsigned int i=0;i<stubinputsdisk_[j]->nStubs();i++){
	  //cout << "Found stub in disk in "<<getName()<<endl;
	  std::pair<FPGAStub*,L1TStub*> stub=stubinputsdisk_[j]->getStub(i);
	  //FIXME Next few lines should be member data in FPGAStub...
	  int irtmp=stub.first->r().value(); 
	  int ir=irtmp>>(stub.first->r().nbits()-2);
	  int iphitmp=stub.first->phi().value();
	  int disk=stub.first->disk().value();
	  if ((disk%2)==0) iphitmp-=(1<<(stub.first->phi().nbits()-3));  
	  assert(iphitmp>=0);
	  int iphi=iphitmp>>(stub.first->phi().nbits()-2);

	  //cout << "iphi ir : "<<iphi<<" "<<ir<<endl;
	  assert(ir>=0);
	  assert(ir<=3);
	  ir=ir%2;
	  assert(iphi>=0);
	  assert(iphi<=3);

	  stub.first->setAllStubIndex(allstubs_[0]->nStubs());
	  stub.second->setAllStubIndex(allstubs_[0]->nStubs());
	  
	  for (unsigned int l=0;l<allstubs_.size();l++){
	    allstubs_[l]->addStub(stub);
	  }

	  bool insert=false;
	  
	  if (iphi==0&&ir==0) {
	    for (unsigned int l=0;l<vmstubsPHI1R1_.size();l++){
	      vmstubsPHI1R1_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI1R1_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  if (iphi==0&&ir==1) {
	    for (unsigned int l=0;l<vmstubsPHI1R2_.size();l++){
	      vmstubsPHI1R2_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI1R2_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  
	  if (iphi==1&&ir==0) {
	    for (unsigned int l=0;l<vmstubsPHI2R1_.size();l++){
	      vmstubsPHI2R1_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI2R1_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  if (iphi==1&&ir==1) {
	    for (unsigned int l=0;l<vmstubsPHI2R2_.size();l++){
	      vmstubsPHI2R2_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI2R2_[l]->getName() << endl;	      
	      insert=true;
	    }
	  }
	  
	  if (iphi==2&&ir==0) {
	    for (unsigned int l=0;l<vmstubsPHI3R1_.size();l++){
	      vmstubsPHI3R1_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI3R1_[l]->getName() << endl;
	      insert=true;

	    }
	  }
	  if (iphi==2&&ir==1) {
	    for (unsigned int l=0;l<vmstubsPHI3R2_.size();l++){
	      vmstubsPHI3R2_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI3R2_[l]->getName() << endl;
	      insert=true;
	    }
	  }
	  
	  if (iphi==3&&ir==0) {
	    for (unsigned int l=0;l<vmstubsPHI4R1_.size();l++){
	      vmstubsPHI4R1_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI4R1_[l]->getName() << endl;	   
	      insert=true; 
	    }
	  }
	  if (iphi==3&&ir==1) {
	    for (unsigned int l=0;l<vmstubsPHI4R2_.size();l++){
	      vmstubsPHI4R2_[l]->addStub(stub);
	      //cout << "Adding stub in " << vmstubsPHI4R1_[l]->getName() << endl;	 
	      insert=true;
	    }
	  }
	  assert(insert);
	}
      }     
    }
  }



private:

  vector<FPGAStubLayer*> stubinputs_;
  vector<FPGAStubDisk*> stubinputsdisk_;
  vector<FPGAAllStubs*> allstubs_;

  vector<FPGAVMStubs*> vmstubsPHI1Z1_;
  vector<FPGAVMStubs*> vmstubsPHI1Z2_;
  vector<FPGAVMStubs*> vmstubsPHI2Z1_;
  vector<FPGAVMStubs*> vmstubsPHI2Z2_;
  vector<FPGAVMStubs*> vmstubsPHI3Z1_;
  vector<FPGAVMStubs*> vmstubsPHI3Z2_;
  vector<FPGAVMStubs*> vmstubsPHI4Z1_;
  vector<FPGAVMStubs*> vmstubsPHI4Z2_;

  vector<FPGAVMStubs*> vmstubsPHI1R1_;
  vector<FPGAVMStubs*> vmstubsPHI1R2_;
  vector<FPGAVMStubs*> vmstubsPHI2R1_;
  vector<FPGAVMStubs*> vmstubsPHI2R2_;
  vector<FPGAVMStubs*> vmstubsPHI3R1_;
  vector<FPGAVMStubs*> vmstubsPHI3R2_;
  vector<FPGAVMStubs*> vmstubsPHI4R1_;
  vector<FPGAVMStubs*> vmstubsPHI4R2_;


};

#endif

