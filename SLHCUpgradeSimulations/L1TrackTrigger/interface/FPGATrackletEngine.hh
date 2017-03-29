//This class implementes the tracklet engine
#ifndef FPGATRACKLETENGINE_H
#define FPGATRACKLETENGINE_H

#include "FPGAProcessBase.hh"
#include "FPGATETable.hh"
#include "FPGATETableDisk.hh"
#include "FPGATETableOverlap.hh"

using namespace std;

class FPGATrackletEngine:public FPGAProcessBase{

public:

  FPGATrackletEngine(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    double dphi=two_pi/NSector;
    phimin_=iSector*dphi;
    phimax_=phimin_+dphi;
    if (phimin_>0.5*two_pi) phimin_-=two_pi;
    if (phimax_>0.5*two_pi) phimax_-=two_pi;
    if (phimin_>phimax_)  phimin_-=two_pi;
    //cout << "phimin_ phimax_ "<<phimin_<<" "<<phimax_<<endl;
    assert(phimax_>phimin_);
    stubpairs_=0;
    innervmstubs_=0;
    outervmstubs_=0;
    table_=0;
    layer1_=0;
    layer2_=0;
    disk1_=0;
    disk2_=0;
    dct1_=0;
    dct2_=0;
    phi1_=0;
    phi2_=0;
    z1_=0;
    z2_=0;
    r1_=0;
    r2_=0;
    if (name[3]=='L') {
      layer1_=name[4]-'0';
      z1_=name[12]-'0';
    }
    if (name[3]=='F') {
      disk1_=name[4]-'0';
      r1_=name[12]-'0';
    }
    if (name[3]=='B') {
      disk1_=-(name[4]-'0');
      r1_=name[12]-'0';
    }
    if (name[14]=='L') {
      layer2_=name[15]-'0';
      z2_=name[23]-'0';
    }
    if (name[14]=='F') {
      disk2_=name[15]-'0';
      r2_=name[23]-'0';
    }
    if (name[14]=='B') {
      disk2_=-(name[15]-'0');
      r2_=name[23]-'0';
    }
    
    phi1_=name[10]-'0';
    phi2_=name[21]-'0';

    dct1_=name[6]-'0';
    dct2_=name[17]-'0';


    if (layer1_!=0 && layer2_!=0) {

      int index=1000000*layer1_
	+100000*dct1_
	+10000*dct2_
	+1000*phi1_
	+100*phi2_
	+10*z1_
	+z2_;

      static map<int, FPGATETable*> tetables_;

      map<int, FPGATETable*>::iterator it=tetables_.find(index);

      if (it!=tetables_.end()){
	table_=it->second;
      }
      else {
        // cout << "name : "<<name<<" "<<layer1_<<"/"<<disk1_<<" "<<dct1_<<" "
      	//    <<phi1_<<" "<<z1_<<"/"<<r1_<<" _  "<<layer2_<<"/"<<disk2_<<" "
        //    <<dct2_<<" "<<phi2_<<" "<<z2_<<"/"<<r2_<<endl;
	// cout << "Building table for "<<getName()<<endl;
      

	int i1=phi1_;
	int i2=phi2_;

	int j1=(dct1_-1)*2+z1_-1;
	int j2=(dct2_-1)*2+z2_-1;
	
	assert(i2==i1||i2==i1+1);
	double phioffset=-0.5*(phimax_-phimin_)/3.0;
	if (i2==i1+1) phioffset=-phioffset;
	
	table_=new FPGATETable();
	table_->init(3,3,4,3,
		     4,4,2,2,
		     (phimax_-phimin_)/3.0,
		     phioffset,
		     -zlength+j1*(2*zlength)/L1Nz,
		     -zlength+(j1+1)*(2*zlength)/L1Nz,
		     -zlength+j2*(2*zlength)/L2Nz,
		     -zlength+(j2+1)*(2*zlength)/L2Nz,
		     rmin[layer1_-1],
		     rmin[layer1_-1]+2*drmax,
		     rmin[layer2_-1],
		     rmin[layer2_-1]+2*drmax,
		     i1,
		     i2,
		     j1,
		     j2);
	
	tetables_[index]=table_;

	if(writeTETables){
	  std::ostringstream oss;
	  oss << "LUT_"<<getName();
	  std::string fnamephi=oss.str();
	  fnamephi+="_phi.dat";
	  std::string fnamez=oss.str();
	  fnamez+="_z.dat";
	  table_->writephi(fnamephi);
	  table_->writez(fnamez);
	}

      }
    }




    if (disk1_!=0 && disk2_!=0) {

      int index=1000000*disk1_
	+100000*dct1_
	+10000*dct2_
	+1000*phi1_
	+100*phi2_
	+10*r1_
	+r2_;

      static map<int, FPGATETableDisk*> tetablesdisk_;

      map<int, FPGATETableDisk*>::iterator it=tetablesdisk_.find(index);

      if (it!=tetablesdisk_.end()){
	tabledisk_=it->second;
      }
      else {
         // cout << "name : "<<name<<" "<<layer1_<<"/"<<disk1_<<" "<<dct1_<<" "
      	 //   <<phi1_<<" "<<z1_<<"/"<<r1_<<" _  "<<layer2_<<"/"<<disk2_<<" "
         //   <<dct2_<<" "<<phi2_<<" "<<z2_<<"/"<<r2_<<endl;

	 // cout << "Building table for "<<getName()<<endl;
      

	assert(dct1_==5||dct1_==7);
	assert(dct2_==5||dct2_==7);

	int i1=phi1_;
	int i2=phi2_;

	int j1=r1_-1;
	int j2=r2_-1;
	
	assert(i1==i2||i1==i2+1);
	double phioffset=0.5*(phimax_-phimin_)/3.0;
	if (i1==i2+1) phioffset=-phioffset;

	int sign=1;
	if (disk1_<0) sign=-1;

	tabledisk_=new FPGATETableDisk();
	if (disk1_>0) {
	  tabledisk_->init(3,3,4,4,
			   nzbitsdiskvm,nzbitsdiskvm,nrbitsdiskvm,nrbitsdiskvm,
			   (phimax_-phimin_)/3.0,
			   phioffset,
			   sign*(zmean[abs(disk1_)-1]-dzmax),
			   sign*(zmean[abs(disk1_)-1]+dzmax),
			   sign*(zmean[abs(disk2_)-1]-dzmax),
			   sign*(zmean[abs(disk2_)-1]+dzmax),
			   rmindisk+j1*drdisk/L1Nr,
			   rmindisk+(j1+1)*drdisk/L1Nr,
			   rmindisk+j2*drdisk/L1Nr,
			   rmindisk+(j2+1)*drdisk/L1Nr,
			   i1,
			   i2,
			   j1,
			   j2);
	} else {
	  tabledisk_->init(3,3,4,4,
			   nzbitsdiskvm,nzbitsdiskvm,nrbitsdiskvm,nrbitsdiskvm,
			   (phimax_-phimin_)/3.0,
			   phioffset,
			   sign*(zmean[abs(disk1_)-1]+dzmax),
			   sign*(zmean[abs(disk1_)-1]-dzmax),
			   sign*(zmean[abs(disk2_)-1]+dzmax),
			   sign*(zmean[abs(disk2_)-1]-dzmax),
			   rmindisk+j1*drdisk/L1Nr,
			   rmindisk+(j1+1)*drdisk/L1Nr,
			   rmindisk+j2*drdisk/L1Nr,
			   rmindisk+(j2+1)*drdisk/L1Nr,
			   i1,
			   i2,
			   j1,
			   j2);
	}

	tetablesdisk_[index]=tabledisk_;

	if(writeTETables){
	  std::ostringstream oss;
	  oss << "LUT_"<<getName();
	  std::string fnamephi=oss.str();
	  fnamephi+="_phi.dat";
	  std::string fnamez=oss.str();
	  fnamez+="_z.dat";
	  tabledisk_->writephi(fnamephi);
	  tabledisk_->writer(fnamez);
	}

      }
    }





    if (disk1_!=0 && layer2_!=0) {


      assert(abs(disk1_)==1);
      assert(layer2_==1||layer2_==2);

      int index=10000000*disk1_
	+1000000*layer2_
	+100000*dct1_
	+10000*dct2_
	+1000*phi1_
	+100*phi2_
	+10*r1_
	+z2_;

      static map<int, FPGATETableOverlap*> tetablesoverlap_;

      map<int, FPGATETableOverlap*>::iterator it=tetablesoverlap_.find(index);

      if (it!=tetablesoverlap_.end()){
	tableoverlap_=it->second;
      }
      else {
        // cout << "name : "<<name<<" "<<layer1_<<"/"<<disk1_<<" "<<dct1_<<" "
       	//    <<phi1_<<" "<<z1_<<"/"<<r1_<<" _  "<<layer2_<<"/"<<disk2_<<" "
        //    <<dct2_<<" "<<phi2_<<" "<<z2_<<"/"<<r2_<<endl;     
	// cout << "Building table for "<<getName()<<endl;


	assert(dct1_==5||dct1_==7);
	
	int j1=r1_-1;
	int j2=(dct2_-1)*2+z2_-1;

	
	int i1=phi1_;
	int i2=phi2_;

	assert(i1==i2||i1==i2+1);
	double phioffset=-0.5*(phimax_-phimin_)/3.0;
	//if (i1==i2+1) phioffset=-phioffset;
	assert(layer2_==1); //have not debugged layer2_==2!
	if (layer2_==2) phioffset=0.0; //both have four phi divisions
	
	int sign=1;
	if (disk1_<0) sign=-1;

	tableoverlap_=new FPGATETableOverlap();
	tableoverlap_->init(3,3,4,3,
			    nrbitsdiskvm,nzbitsdiskvm,4,2,
			    (phimax_-phimin_)/3.0,
			    phioffset,
			    rmindisk+j1*drdisk/L1Nr,
			    rmindisk+(j1+1)*drdisk/L1Nr,
			    sign*(zmean[abs(disk1_)-1]-dzmax),
			    sign*(zmean[abs(disk1_)-1]+dzmax),
			    -zlength+j2*(2*zlength)/L2Nz,
			    -zlength+(j2+1)*(2*zlength)/L2Nz,
			    rmin[layer2_-1],
			    rmin[layer2_-1]+2*drmax,
			    i1,
			    i2,
			    j1,
			    j2);
	
	tetablesoverlap_[index]=tableoverlap_;
	if(writeTETables){
	  std::ostringstream oss;
	  oss << "LUT_"<<getName();
	  std::string fnamephi=oss.str();
	  fnamephi+="_phi.dat";
	  std::string fnamez=oss.str();
	  fnamez+="_z.dat";
	  tableoverlap_->writephi(fnamephi);
	  tableoverlap_->writer(fnamez);
	}

      }
    }



    
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="stubpairout") {
      FPGAStubPairs* tmp=dynamic_cast<FPGAStubPairs*>(memory);
      assert(tmp!=0);
      stubpairs_=tmp;
      return;
    }
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="innervmstubin") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      innervmstubs_=tmp;
      return;
    }
    if (input=="outervmstubin") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      outervmstubs_=tmp;
      return;
    }
    cout << "Could not find input : "<<input<<endl;
    assert(0);
  }

  void execute() {
    
    if (!((doL1L2&&(layer1_==1)&&(layer2_==2))||
	  (doL3L4&&(layer1_==3)&&(layer2_==4))||
	  (doL5L6&&(layer1_==5)&&(layer2_==6))||
	  (doF1F2&&(disk1_==1)&&(disk2_==2))||
	  (doF3F4&&(disk1_==3)&&(disk2_==4))||
	  (doB1B2&&(disk1_==-1)&&(disk2_==-2))||
	  (doB3B4&&(disk1_==-3)&&(disk2_==-4))||
	  (doL1F1&&(disk1_==1)&&(layer2_==1))||
	  (doL2F1&&(disk1_==1)&&(layer2_==2))||
	  (doL1B1&&(disk1_==-1)&&(layer2_==1))||
	  (doL2B1&&(disk1_==-1)&&(layer2_==2)))) return;

    

    //if (disk1_==0||disk2_==0) return;

    //cout << getName()<<" disk1_ disk2_ "<<disk1_<<" "<<disk2_<<endl;


    //if (getName().substr(0,5)!="TE_L1") return;

   
    //cout << "In FPGATrackletEngine::execute : "<<getName()<<endl;
    //cout <<" "<<innervmstubs_->nStubs()
    // 	 <<" "<<outervmstubs_->nStubs()
    //	 <<endl;

    unsigned int countall=0;
    unsigned int countpass=0;

    //cout << "layer/disk : "<<layer1_<<" "<<layer2_<<" "
    //	 <<disk1_<<" "<<disk2_<<endl;

    assert(innervmstubs_!=0);
    assert(outervmstubs_!=0);
    for(unsigned int i=0;i<innervmstubs_->nStubs();i++){
      std::pair<FPGAStub*,L1TStub*> innerstub=innervmstubs_->getStub(i);
      for(unsigned int j=0;j<outervmstubs_->nStubs();j++){
	std::pair<FPGAStub*,L1TStub*> outerstub=outervmstubs_->getStub(j);
	countall++;

	if (layer1_!=0 && layer2_!=0) {
	  assert(table_!=0);

	  int istubpt1=innerstub.first->stubpt().value();
	  int iphivm1=innerstub.first->phivm().value();
	  FPGAWord iphi1=innerstub.first->phi();
	  FPGAWord iz1=innerstub.first->z();
	  FPGAWord ir1=innerstub.first->r();
	  int irvm1=innerstub.first->rvm().value();
	  int izvm1=innerstub.first->zvm().value();

	  int istubpt2=outerstub.first->stubpt().value();
	  int iphivm2=outerstub.first->phivm().value();
	  FPGAWord iphi2=outerstub.first->phi();
	  FPGAWord iz2=outerstub.first->z();
	  FPGAWord ir2=outerstub.first->r();
	  int irvm2=outerstub.first->rvm().value();
	  int izvm2=outerstub.first->zvm().value();

	  int ideltaphi=iphivm2-iphivm1;
	  int ideltar=irvm2-irvm1;

	  if (ideltar<0) ideltar+=8;
	  assert(ideltar>=0);
	  if (ideltaphi<0) ideltaphi+=16;
	  assert(ideltaphi>=0);

	  assert(istubpt1>=0);
	  assert(istubpt2>=0);


	  int address=(istubpt1<<10)+(istubpt2<<7)+(ideltaphi<<3)+ideltar;
	  int zaddress=(izvm1<<8)+(izvm2<<4)+(irvm1<<2)+irvm2;

	  bool phimatch=table_->phicheck(address);
	  bool zmatch=table_->zcheck(zaddress);


	  //cout << "layer matches "<<layer1_<<" "<<phimatch<<" "<<zmatch<<endl;

	  if (!(phimatch&&zmatch)) continue;

	} else if (disk1_!=0 && disk2_!=0) {
	  assert(tabledisk_!=0);
	  
	  //cout << "disk trying "
	  //     <<innerstub.second->phi()<<" "
	  //     <<innerstub.second->r()<<" "
	  //     <<outerstub.second->phi()<<" "
	  //     <<outerstub.second->r()<<" "
	  //     <<2*sin(outerstub.second->phi()-innerstub.second->phi())/
	  //  (innerstub.second->r()-outerstub.second->r())<<endl;


	  int istubpt1=innerstub.first->stubpt().value();
	  int iphivm1=innerstub.first->phivm().value();
	  FPGAWord iphi1=innerstub.first->phi();
	  FPGAWord iz1=innerstub.first->z();
	  FPGAWord ir1=innerstub.first->r();
	  int irvm1=innerstub.first->rvm().value();
	  int izvm1=innerstub.first->zvm().value();

	  int istubpt2=outerstub.first->stubpt().value();
	  int iphivm2=outerstub.first->phivm().value();
	  FPGAWord iphi2=outerstub.first->phi();
	  FPGAWord iz2=outerstub.first->z();
	  FPGAWord ir2=outerstub.first->r();
	  int irvm2=outerstub.first->rvm().value();
	  int izvm2=outerstub.first->zvm().value();

	  int ideltaphi=iphivm2-iphivm1;
	  int ideltar=irvm2-irvm1;

	  ideltar>>=2;

	  if (ideltar<0) ideltar+=16;
	  assert(ideltar>=0);
	  if (ideltaphi<0) ideltaphi+=16;
	  assert(ideltaphi>=0);

	  assert(istubpt1>=0);
	  assert(istubpt2>=0);


	  int address=(istubpt1<<11)+(istubpt2<<8)+(ideltaphi<<4)+ideltar;

	  bool phimatch=tabledisk_->phicheck(address);
	  bool zmatch=tabledisk_->zcheck(izvm1,izvm2,irvm1,irvm2);

	  //cout << "disk " << getName()<<" "<<ideltaphi<<" "<<ideltar 
	  //     <<" "<<irvm1<<" "<<irvm2<<" "<<outerstub.first->rvm().nbits()
	  //     <<endl;
	  //cout << "disk matches "<<address<<" "<<disk1_<<" "<<phimatch<<" "<<zmatch<<endl;

	  //if (!phimatch) continue;

	  if (!(phimatch&&zmatch)) continue;

	} else if (disk1_!=0 && layer2_!=0) {


	  //cout << "overlap trying "
	  //     <<innerstub.second->phi()<<" "
	  //     <<innerstub.second->r()<<" "
	  //     <<outerstub.second->phi()<<" "
	  //     <<outerstub.second->r()<<" "
	  //     <<2*sin(outerstub.second->phi()-innerstub.second->phi())/
	  //       (innerstub.second->r()-outerstub.second->r())<<endl;



      
	  assert(tableoverlap_!=0);

	  int istubpt1=innerstub.first->stubpt().value();
	  int iphivm1=innerstub.first->phivm().value();
	  FPGAWord iphi1=innerstub.first->phi();
	  FPGAWord iz1=innerstub.first->z();
	  FPGAWord ir1=innerstub.first->r();
	  int irvm1=innerstub.first->rvm().value();
	  int izvm1=innerstub.first->zvm().value();

	  assert(innerstub.first->rvm().nbits()==5);

	  int istubpt2=outerstub.first->stubpt().value();
	  int iphivm2=outerstub.first->phivm().value();
	  FPGAWord iphi2=outerstub.first->phi();
	  FPGAWord iz2=outerstub.first->z();
	  FPGAWord ir2=outerstub.first->r();
	  int irvm2=outerstub.first->rvm().value();
	  int izvm2=outerstub.first->zvm().value();

	  //cout << "outerstub.first->rvm().nbits() : "
	  //     << outerstub.first->rvm().nbits()<<endl;

	  assert(outerstub.first->rvm().nbits()==2);

	  
	  int ideltaphi=iphivm2-iphivm1;

	  //cout << "overlap phi :"<<ideltaphi<<" "<<iphivm2<<" "<<iphivm1<<endl;

	  int ideltar=(irvm1>>2)-(irvm2>>1);

	  //cout << "overlap :"<<ideltar<<" "<<irvm2<<" "<<irvm1<<endl;

	  if (ideltar<0) ideltar=0; 
	  if (ideltaphi<0) ideltaphi+=8;
	  assert(ideltaphi>=0);

	  assert(istubpt1>=0);
	  assert(istubpt2>=0);


	  int address=(istubpt1<<10)+(istubpt2<<7)+(ideltaphi<<3)+ideltar;

	  bool phimatch=tableoverlap_->phicheck(address);
	  bool zmatch=tableoverlap_->zcheck(izvm1,izvm2,irvm1,irvm2);


	  //cout << "overlap matches "<<getName()<<" "<<disk1_<<" "<<layer2_
	  //     <<" "<<phimatch<<" "<<zmatch
	  //     <<" | "<<address<<" "<<istubpt1<<" "<<istubpt2
	  //     <<" "<<ideltaphi<<" "<<ideltar<<endl;

	  if (!(phimatch&&zmatch)) {
	    //cout << "Failed : "<<phimatch<<" "<<zmatch<<endl;
	    continue;
	  }

	} else {
	  assert(0);
	}

	//cout << "Adding stub pair in "<<getName()<<endl;
	assert(stubpairs_!=0);
	countpass++;
	stubpairs_->addStubPair(innerstub,outerstub);

	if (countall>=NMAXTE) break;
      }
      if (countall>=NMAXTE) break;
    }

    if (writeTE) {
      static ofstream out("trackletengine.txt");
      out << getName()<<" "<<countall<<" "<<countpass<<endl;
    }

  }

private:

  double phimax_;
  double phimin_;

  FPGATETable* table_;
  FPGATETableDisk* tabledisk_;
  FPGATETableOverlap* tableoverlap_;
  int layer1_;
  int layer2_;
  int disk1_;
  int disk2_;
  int dct1_;
  int dct2_;
  int phi1_;
  int phi2_;
  int z1_;
  int z2_;
  int r1_;
  int r2_;

  FPGAVMStubs* innervmstubs_;
  FPGAVMStubs* outervmstubs_;

  FPGAStubPairs* stubpairs_;


};

#endif
