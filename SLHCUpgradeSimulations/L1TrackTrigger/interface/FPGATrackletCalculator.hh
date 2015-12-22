//This class implementes the tracklet engine
#ifndef FPGATRACKLETCALCULATOR_H
#define FPGATRACKLETCALCULATOR_H

#include "FPGAProcessBase.hh"
#include "FPGAInverseTable.hh"

using namespace std;

class FPGATrackletCalculator:public FPGAProcessBase{

public:

  FPGATrackletCalculator(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    double dphi=two_pi/NSector;
    phimin_=iSector*dphi;
    phimax_=phimin_+dphi;
    if (phimin_>0.5*two_pi) phimin_-=two_pi;
    if (phimax_>0.5*two_pi) phimax_-=two_pi;
    if (phimin_>phimax_)  phimin_-=two_pi;
    phioffset_=phimin_-dphi/6.0;

   trackletproj_L1D1_=0;
   trackletproj_L1D2_=0;
   trackletproj_L1D3_=0;
   trackletproj_L1D4_=0;
   trackletproj_L1Minus_=0;
   trackletproj_L1Plus_=0;

   trackletproj_L2D1_=0;
   trackletproj_L2D2_=0;
   trackletproj_L2D3_=0;
   trackletproj_L2D4_=0;
   trackletproj_L2Minus_=0;
   trackletproj_L2Plus_=0;

   trackletproj_L3D1_=0;
   trackletproj_L3D2_=0;
   trackletproj_L3D3_=0;
   trackletproj_L3D4_=0;
   trackletproj_L3Minus_=0;
   trackletproj_L3Plus_=0;

   trackletproj_L4D1_=0;
   trackletproj_L4D2_=0;
   trackletproj_L4D3_=0;
   trackletproj_L4D4_=0;
   trackletproj_L4Minus_=0;
   trackletproj_L4Plus_=0;

   trackletproj_L5D1_=0;
   trackletproj_L5D2_=0;
   trackletproj_L5D3_=0;
   trackletproj_L5D4_=0;
   trackletproj_L5Minus_=0;
   trackletproj_L5Plus_=0;

   trackletproj_L6D1_=0;
   trackletproj_L6D2_=0;
   trackletproj_L6D3_=0;
   trackletproj_L6D4_=0;
   trackletproj_L6Minus_=0;
   trackletproj_L6Plus_=0;

   trackletproj_D1Di_=0;
   trackletproj_D1Do_=0;
   trackletproj_D1Minus_=0;
   trackletproj_D1Plus_=0;

   trackletproj_D2Di_=0;
   trackletproj_D2Do_=0;
   trackletproj_D2Minus_=0;
   trackletproj_D2Plus_=0;
  
   trackletproj_D3Di_=0;
   trackletproj_D3Do_=0;
   trackletproj_D3Minus_=0;
   trackletproj_D3Plus_=0;

   trackletproj_D4Di_=0;
   trackletproj_D4Do_=0;
   trackletproj_D4Minus_=0;
   trackletproj_D4Plus_=0;

   trackletproj_D5Di_=0;
   trackletproj_D5Do_=0;
   trackletproj_D5Minus_=0;
   trackletproj_D5Plus_=0;

   layer_=0;
   disk_=0;

   if (name_[3]=='L') layer_=name_[4]-'0';    
   if (name_[3]=='F') disk_=name_[4]-'0';    
   if (name_[3]=='B') disk_=-(name_[4]-'0');

   assert((layer_!=0)||(disk_!=0));

   if (layer_!=0){
     invTable_.init(9,round_int((rmean[layer_]-rmean[layer_-1])/kr));
     if (writeInvTable) {
       string fname="InvTable_"+name+".dat";     
       invTable_.write(fname);
     }
   }

  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="trackpar"){
      FPGATrackletParameters* tmp=dynamic_cast<FPGATrackletParameters*>(memory);
      assert(tmp!=0);
      trackletpars_=tmp;
      return;
    }

    if (output=="projout_F1D5"||output=="projout_B1D7"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D1Di_=tmp;
      return;
    }
    if (output=="projout_F1D6"||output=="projout_B1D8"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D1Do_=tmp;
      return;
    }

    if (output=="projout_F2D5"||output=="projout_B2D7"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D2Di_=tmp;
      return;
    }
    if (output=="projout_F2D6"||output=="projout_B2D8"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D2Do_=tmp;
      return;
    }

    if (output=="projout_F3D5"||output=="projout_B3D7"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D3Di_=tmp;
      return;
    }
    if (output=="projout_F3D6"||output=="projout_B3D8"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D3Do_=tmp;
      return;
    }

    if (output=="projout_F4D5"||output=="projout_B4D7"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D4Di_=tmp;
      return;
    }
    if (output=="projout_F4D6"||output=="projout_B4D8"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D4Do_=tmp;
      return;
    }

    if (output=="projout_F5D5"||output=="projout_B5D7"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D5Di_=tmp;
      return;
    }
    if (output=="projout_F5D6"||output=="projout_B5D8"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D5Do_=tmp;
      return;
    }

    if (output=="projout_L1D1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L1D1_=tmp;
      return;
    }

    if (output=="projout_L1D2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L1D2_=tmp;
      return;
    }

    if (output=="projout_L1D3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L1D3_=tmp;
      return;
    }

    if (output=="projout_L1D4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L1D4_=tmp;
      return;
    }


    if (output=="projout_L2D1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L2D1_=tmp;
      return;
    }

    if (output=="projout_L2D2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L2D2_=tmp;
      return;
    }

    if (output=="projout_L2D3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L2D3_=tmp;
      return;
    }

    if (output=="projout_L2D4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L2D4_=tmp;
      return;
    }


    if (output=="projout_L3D1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3D1_=tmp;
      return;
    }

    if (output=="projout_L3D2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3D2_=tmp;
      return;
    }

    if (output=="projout_L3D3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3D3_=tmp;
      return;
    }

    if (output=="projout_L3D4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3D4_=tmp;
      return;
    }


    if (output=="projout_L4D1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4D1_=tmp;
      return;
    }

    if (output=="projout_L4D2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4D2_=tmp;
      return;
    }

    if (output=="projout_L4D3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4D3_=tmp;
      return;
    }

    if (output=="projout_L4D4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4D4_=tmp;
      return;
    }


    if (output=="projout_L5D1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5D1_=tmp;
      return;
    }

    if (output=="projout_L5D2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5D2_=tmp;
      return;
    }

    if (output=="projout_L5D3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5D3_=tmp;
      return;
    }

    if (output=="projout_L5D4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5D4_=tmp;
      return;
    }


    if (output=="projout_L6D1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6D1_=tmp;
      return;
    }

    if (output=="projout_L6D2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6D2_=tmp;
      return;
    }

    if (output=="projout_L6D3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6D3_=tmp;
      return;
    }

    if (output=="projout_L6D4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6D4_=tmp;
      return;
    }


    if (output=="projoutToPlus_F1"||output=="projoutToPlus_B1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D1Plus_=tmp;
      return;
    }

    if (output=="projoutToMinus_F1"||output=="projoutToMinus_B1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D1Minus_=tmp;
      return;
    }

    if (output=="projoutToPlus_F2"||output=="projoutToPlus_B2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D2Plus_=tmp;
      return;
    }

    if (output=="projoutToMinus_F2"||output=="projoutToMinus_B2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D2Minus_=tmp;
      return;
    }

    if (output=="projoutToPlus_F3"||output=="projoutToPlus_B3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D3Plus_=tmp;
      return;
    }

    if (output=="projoutToMinus_F3"||output=="projoutToMinus_B3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D3Minus_=tmp;
      return;
    }

    if (output=="projoutToPlus_F4"||output=="projoutToPlus_B4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D4Plus_=tmp;
      return;
    }

    if (output=="projoutToMinus_F4"||output=="projoutToMinus_B4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D4Minus_=tmp;
      return;
    }

    if (output=="projoutToPlus_F5"||output=="projoutToPlus_B5"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D5Plus_=tmp;
      return;
    }

    if (output=="projoutToMinus_F5"||output=="projoutToMinus_B5"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_D5Minus_=tmp;
      return;
    }


    if (output=="projoutToPlus_L1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L1Plus_=tmp;
      return;
    }

    if (output=="projoutToMinus_L1"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L1Minus_=tmp;
      return;
    }



    if (output=="projoutToPlus_L2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L2Plus_=tmp;
      return;
    }

    if (output=="projoutToMinus_L2"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L2Minus_=tmp;
      return;
    }



    if (output=="projoutToPlus_L3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3Plus_=tmp;
      return;
    }

    if (output=="projoutToMinus_L3"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L3Minus_=tmp;
      return;
    }



    if (output=="projoutToPlus_L4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4Plus_=tmp;
      return;
    }

    if (output=="projoutToMinus_L4"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L4Minus_=tmp;
      return;
    }



    if (output=="projoutToPlus_L5"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5Plus_=tmp;
      return;
    }

    if (output=="projoutToMinus_L5"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L5Minus_=tmp;
      return;
    }



    if (output=="projoutToPlus_L6"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6Plus_=tmp;
      return;
    }

    if (output=="projoutToMinus_L6"){
      FPGATrackletProjections* tmp=dynamic_cast<FPGATrackletProjections*>(memory);
      assert(tmp!=0);
      trackletproj_L6Minus_=tmp;
      return;
    }
    cout << "Could not find output : "<<output<<endl;
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="innerallstubin"){
      FPGAAllStubs* tmp=dynamic_cast<FPGAAllStubs*>(memory);
      assert(tmp!=0);
      innerallstubs_=tmp;
      return;
    }
    if (input=="outerallstubin"){
      FPGAAllStubs* tmp=dynamic_cast<FPGAAllStubs*>(memory);
      assert(tmp!=0);
      outerallstubs_=tmp;
      return;
    }
    if (input=="stubpair1in"||
	input=="stubpair2in"||
	input=="stubpair3in"||
	input=="stubpair4in"||
	input=="stubpair5in"||
	input=="stubpair6in"||
	input=="stubpair7in"||
	input=="stubpair8in"||
	input=="stubpair9in"||
	input=="stubpair10in"||
	input=="stubpair11in"||
	input=="stubpair12in"||
	input=="stubpair13in"||
	input=="stubpair14in"||
	input=="stubpair15in"||
	input=="stubpair16in"||
	input=="stubpair17in"||
	input=="stubpair18in"||
	input=="stubpair19in"||
	input=="stubpair20in"||
	input=="stubpair21in"||
	input=="stubpair22in"||
	input=="stubpair23in"||
	input=="stubpair24in"||
	input=="stubpair25in"||
	input=="stubpair26in"||
	input=="stubpair27in"||
	input=="stubpair28in"||
	input=="stubpair29in"||
	input=="stubpair30in"||
	input=="stubpair31in"||
	input=="stubpair32in"||
	input=="stubpair33in"||
	input=="stubpair34in"||
	input=="stubpair35in"||
	input=="stubpair36in"||
	input=="stubpair37in"||
	input=="stubpair38in"||
	input=="stubpair39in"||
	input=="stubpair40in"||
	input=="stubpair41in"||
	input=="stubpair42in"){
      FPGAStubPairs* tmp=dynamic_cast<FPGAStubPairs*>(memory);
      assert(tmp!=0);
      stubpairs_.push_back(tmp);
      return;
    }
    assert(0);
  }


  void exacttracklet(double r1, double z1, double phi1,
		     double r2, double z2, double phi2, double sigmaz,
		     double& rinv, double& phi0,
		     double& t, double& z0,
		     double phiproj[4], double zproj[4], 
		     double phider[4], double zder[4],
		     double phiprojdisk[5], double rprojdisk[5], 
		     double phiderdisk[5], double rderdisk[5]) {

    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
    
    rinv=2*sin(deltaphi)/dist;

    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;    

    phi0=phi1tmp+asin(0.5*r1*rinv);
    
    if (phi0>0.5*two_pi) phi0-=two_pi;
    if (phi0<-0.5*two_pi) phi0+=two_pi;
    assert(fabs(phi0)<0.5*two_pi);
    
    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
    t=(z1-z2)/(rhopsi1-rhopsi2);
    
    z0=z1-t*rhopsi1;

    for (int i=0;i<4;i++) {
      exactproj(rproj_[i],rinv,phi0,t,z0,
		phiproj[i],zproj[i],phider[i],zder[i]);
    }

    for (int i=0;i<5;i++) {
      int sign=1;
      if (t<0) sign=-1;
      exactprojdisk(sign*zmean[i],rinv,phi0,t,z0,
		phiprojdisk[i],rprojdisk[i],phiderdisk[i],rderdisk[i]);
    }



  }


  void exacttrackletdisk(double r1, double z1, double phi1,
			 double r2, double z2, double phi2, double sigmaz,
			 double& rinv, double& phi0,
			 double& t, double& z0,
			 double phiprojLayer[3], double zprojLayer[3], 
			 double phiderLayer[3], double zderLayer[3],
			 double phiproj[3], double rproj[3], 
			 double phider[3], double rder[3]) {

    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
    
    rinv=2*sin(deltaphi)/dist;

    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;    

    //cout << "phi1 phi2 phi1tmp : "<<phi1<<" "<<phi2<<" "<<phi1tmp<<endl;

    phi0=phi1tmp+asin(0.5*r1*rinv);
    
    if (phi0>0.5*two_pi) phi0-=two_pi;
    if (phi0<-0.5*two_pi) phi0+=two_pi;
    if (!(fabs(phi0)<0.5*two_pi)) {
      cout << "phi1tmp r1 rinv phi0 deltaphi dist: "
	   <<phi1tmp<<" "<<r1<<" "<<rinv<<" "<<phi0
	   <<" "<<deltaphi<<" "<<dist<<endl;
      exit(1);
    }
    
    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
    t=(z1-z2)/(rhopsi1-rhopsi2);
    
    z0=z1-t*rhopsi1;


    if (disk_==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "DUMPPARS0:" 
	     <<" dz= "<<z2-z1
	     <<" rinv= "<<rinv
	     <<" phi0= "<<phi0
	     <<" t= "<<t
	     <<" z0= "<<z0
	     <<endl;
      }
    }


    for (int i=0;i<3;i++) {
      exactprojdisk(zproj_[i],rinv,phi0,t,z0,
		    phiproj[i],rproj[i],
		    phider[i],rder[i]);
    }


    for (int i=0;i<3;i++) {
      exactproj(rmean[i],rinv,phi0,t,z0,
		    phiprojLayer[i],zprojLayer[i],
		    phiderLayer[i],zderLayer[i]);
    }


  }






  void approxtracklet(double r1, double z1, double phi1,
		      double r2, double z2, double phi2, double sigmaz,
		      double &rinv, double &phi0,
		      double &t, double &z0,
		      double phiproj[4], double zproj[4], 
		      double phider[4], double zder[4],
		      double phiprojdisk[5], double rprojdisk[5], 
		      double phiderdisk[5], double rderdisk[5]) {

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }


    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (phi1<0.0) phi1+=two_pi;
    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;
    //cout << "phi1 phimin_ phimax_:"<<phi1<<" "<<phimin_<<" "<<phimax_<<endl;
    assert(phi1tmp>-1e-10);

    double dr=r2-r1;
    double dz=z1-z2;
    double drinv=1.0/dr;
    double t2=deltaphi*drinv;
    double delta=0.5*r1*r2*t2*t2;//*(1+deltaphi*deltaphi/12.0);
    double t5=1.0-delta+1.5*delta*delta;//-2.5*delta*delta*delta;
    double deltainv=t5*drinv;
    rinv=2.0*deltaphi*deltainv;//*(1-deltaphi*deltaphi/6.0);
    t=-dz*deltainv;//*(1.0-deltaphi*deltaphi/6.0); 
    double t7=0.5*r1*rinv;
    double t9=1+t7*t7/6.0;//+3.0*t7*t7*t7*t7/40.0;
    phi0=phi1tmp+t7*t9;
    double t12=t*r1*t9;
    z0=z1-t12;


    if (layer_==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "DUMPPARS1:" 
	     << -deltaphi
	     <<" "<<z2-z1
	     <<" "<<r2-r1
	     <<" "<<1.0/(r2-r1)
	     <<" "<<delta
	     <<" "<<t5
	     <<" "<<deltainv
	     <<" | "<<t
	     <<" | "<<r1
	     <<" "<<t7
	     <<" | "<<t9
	     <<" "<<phi1tmp
	     <<" "<<phi0
	     <<" * "<<t12
	     <<" "<<z0
	     <<endl;
      }
      /*

      cout << "Approx tracklet: dphi="<<-deltaphi<<" dz="<<z2-z1
	   << " dr="<<r2-r1<<" drinv="<<1.0/(r2-r1)
	   <<" delta="<<delta
	   <<" t5="<<t5
           <<" deltainv="<<deltainv
	   <<" rinv="<<rinv
	   <<" t="<<t
	   <<" r1abs="<<r1
	   <<" t7="<<t7
	   <<" t9="<<t9
	   <<" phi1="<<phi1tmp
	   <<" ***phi0="<<phi0
	   <<" t12="<<t12
	   <<" z1="<<z1
	   <<" z0="<<z0
	   <<endl;

    */

    }
   

    //calculate projection


    static ofstream out1;
    if (writeNeighborProj) out1.open("neighborproj.txt");
    for (int i=0;i<4;i++) {
      approxproj(rproj_[i],rinv,phi0,t,z0,
		 phiproj[i],zproj[i],phider[i],zder[i]);
      if (writeNeighborProj) {
	if (fabs(z0)<15.0&&fabs(rinv)<0.0057) {
	  if ((fabs(zproj[i])<270.0)&&(phiproj[i]<(phimax_-phimin_)/6)){
	    out1<<layer_<<" -1 "<<phiproj[i]<<endl;
	  } else  if ((fabs(zproj[i])<270.0)&&(phiproj[i]>7.0*(phimax_-phimin_)/6)){
	    out1<<layer_<<" +1 "<<phiproj[i]<<endl;
	  } else if (fabs(zproj[i])<270.0){
	    out1<<layer_<<" 0 "<<phiproj[i]<<endl;
	  }
	}
      }
    }

    for (int i=0;i<5;i++) {
      int sign=1;
      if (t<0) sign=-1;
      approxprojdisk(sign*zmean[i],rinv,phi0,t,z0,
		     phiprojdisk[i],rprojdisk[i],phiderdisk[i],rderdisk[i]);
      //cout << "DUMPDISKPROJ1: "<<i<<" "<<rprojdisk[i]
      //	   <<" t="<<t<<" z0="<<z0<<" zdisk="<<zmean[i]<<endl;

    }

  }


  bool binarytracklet(FPGAStub* innerFPGAStub, 
		      FPGAStub* outerFPGAStub,
		      double sigmaz,
		      int& irinv, int& iphi0,
		      int& it, int& iz0,
		      int iphiproj[4], int izproj[4],
		      int iphider[4], int izder[4],
		      bool minusNeighbor[4], bool plusNeighbor[4],
		      int iphiprojdisk[5], int irprojdisk[5],
		      int iphiderdisk[5], int irderdisk[5],
		      bool minusNeighborDisk[5], bool plusNeighborDisk[5]){

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

    //cout << "Layer : "<<layer_<<" "<<innerFPGAStub->str()
    //	 <<" "<<outerFPGAStub->str()<<endl;

    //double phi1tmp=innerFPGAStub.phitmp();
    //double phimintmp=innerFPGAStub.phimin();
 
    int ir1=innerFPGAStub->ir();
    int iphi1=innerFPGAStub->iphi();
    int iz1=innerFPGAStub->iz();

    int ir2=outerFPGAStub->ir();
    int iphi2=outerFPGAStub->iphi();
    int iz2=outerFPGAStub->iz();

    //Simplify to work in common number of bits for all layers
    if (layer_<4) iphi1<<=(nbitsphistubL456-nbitsphistubL123);
    if (layer_<3) iphi2<<=(nbitsphistubL456-nbitsphistubL123);

    if (layer_<4) ir1<<=(nbitsrL456-nbitsrL123);
    if (layer_<3) ir2<<=(nbitsrL456-nbitsrL123);

    if (layer_>3) iz1<<=(nbitszL123-nbitszL456);
    if (layer_>2) iz2<<=(nbitszL123-nbitszL456);


    //Here is where the actual calculation starts
    //Can the first few steps be combined?
    //step 1
    int ideltaphi=iphi2-iphi1; 
    assert(abs(ideltaphi)<(1<<15));
    //step 2
    int idz=iz2-iz1;
    assert(abs(idz)<(1<<11));
    //step 3
    int idrrel=ir2-ir1;
    assert(abs(idrrel)<(1<<8));
    //step 4
    int ir1abs=round_int(rmean[layer_-1]/kr)+ir1;
    assert(ir1abs<(1<<13));
    //step 5
    int ir2abs=round_int(rmean[layer_]/kr)+ir2;
    assert(ir2abs<(1<<13));
    //step 6
    int idrinv=invTable_.lookup(idrrel&((1<<9)-1)); //Take top 9 bits

    //int idr=round_int((rmean[layer_]-rmean[layer_-1])/kr)+idrrel; 
    //int idrinvtmp=round_int((1<<idrinvbits)/(1.0*idr));  //not actually using idrinvbits since 
                                     //idr is largish. Implement in lookup 
                                     //table, just a fcn of ir1-ir2
                                     //for a given layer
    //cout << "idrrel idrinv= "<<idrrel<<" "<<(idrrel&((1<<9)-1))
    //	 <<" "<<idrinv<<" "<<idrinvtmp<<endl;
    //assert(idrinv==idrinvtmp);

    assert(idrinv<(1<<11));
    //step 7
    int it1=(ir1abs*ir2abs)>>it1shift;
    assert(it1<(1<<2));
    //step 8 
    int it2=(idrinv*ideltaphi)>>it2shift;
    assert(abs(it2)<(1<<10));
    //step 9
    int it3=(it2*it2)>>it3shift;
    assert(it3<(1<<6));
    assert(it3>=0);
    //step 10
    int idelta=0.5*it1*it3;
    assert(idelta<(1<<6));
    assert(idelta>=0);
    //step 11
    int ideltatmp=round_int(idelta*(kdelta*(1<<it4bits)));
    assert(ideltatmp<(1<<6));
    //step 12
    int it4=(1<<it4bits)-3*(ideltatmp>>1);
    assert(it4<(1<<(it4bits+1)));
    //step 13    
    int it5=(1<<it5bits)-((ideltatmp*it4)>>(2*it4bits-it5bits));
    assert(it5<(1<<(it5bits+1)));
    //step 14
    int iDeltainv=(idrinv*it5)>>it5bits;  
    assert(fabs(iDeltainv)<(1<<11));
    //step 15
    assert(rinvbitshift>0);
    irinv=-((ideltaphi*iDeltainv)>>(rinvbitshift-1)); //-1 because of *2 
    assert(fabs(irinv)<(1<<14));
    //step 16
    it=(idz*iDeltainv)>>tbitshift;
    assert(fabs(it)<(1<<12));
    //step 17
    assert(it7shift+irinvshift-rinvbitshift>=0);
    int it7=((ir1abs>>1)*irinv)>>(it7shift+irinvshift-rinvbitshift);
    assert(fabs(it7)<(1<<17));
    //step 18
    int it7tmp=(it7*it7tmpfactor)>>it7tmpshift;
    assert(fabs(it7tmp)<(1<<7));
    //step 19
    int it9=(1<<it9bits)+((it7tmp*it7tmp)>>(2*it7tmpbits-2*it7tmpshift-2*it7shift-it9bits));
    assert(fabs(it9)<(1<<13));
    //step 20
    int shifttmp1=it9bits+idrinvbits-irinvshift-it7shift-it7shift2;
    iphi0=(iphi1+(((it7>>it7shift2)*it9)>>shifttmp1))>>phi0bitshift;
    assert(fabs(iphi0)<(1<<17));  
    //step 21
    assert(it12shift+itshift-tbitshift>=0);
    int it12=(it*ir1abs)>>(it12shift+itshift-tbitshift);
    assert(fabs(it12)<(1<<18)); 
    //step 22
    int shifttmp2=it9bits+idrinvbits-itshift-it12shift;
    iz0=(iz1-((it12*it9)>>shifttmp2))>>z0bitshift;
    assert(fabs(iz0)<(1<<10));


    if (layer_==1) {
      
      if (dumppars) {
	cout << "DUMPPARS2: " 
	     << ideltaphi*kphi1
	     <<" "<<idz*kz
	     <<" "<<idrinv*kdrinv
	     <<" "<<idelta*kdelta
	     <<" "<<it5*kt5
	     <<" "<<iDeltainv*kdrinv
	     <<" | "<<it*ktpars
	     <<" | "<<ir1abs*kr
	     <<" "<<it7*kt7
	     <<" | "<<((it7tmp*it7tmp)>>(2*it7tmpbits-2*it7tmpshift-2*it7shift-it9bits))*kt9
	     <<" "<<it9*kt9
	     <<" "<<iphi1*kphi1
	     <<" "<<iphi0*kphi0pars
	     <<" * "<<it12*kt12<<" ("<<it12<<")" 
	     <<" "<<iz0*kzpars
	     <<endl;
      }
	
    }


    if (fabs(irinv*krinvpars)>rinvcut) {
      //cout << "Failed tracklet pt cut"<<endl;
      return false;
    }
    if (fabs(iz0*kzpars)>z0cut) {
      //cout << "Failed tracklet z0 cut"<<endl;
      return false;
    }


    assert(rinvbitshift>=0);
    assert(phi0bitshift>=0);
    assert(tbitshift>=0);
    assert(z0bitshift>=0);
    
    assert(fabs(iz0)<(1<<(nbitsz0-1)));

    if (iz0>=(1<<(nbitsz0-1))) iz0=(1<<(nbitsz0-1))-1; 
    if (iz0<=-(1<<(nbitsz0-1))) iz0=1-(1<<(nbitsz0-1));
    if (irinv>=(1<<(nbitsrinv-1))) irinv=(1<<(nbitsrinv-1))-1;
    if (irinv<=-(1<<(nbitsrinv-1))) irinv=1-(1<<(nbitsrinv-1));

    //calculate projections

    for (int i=0;i<4;i++) {
      binaryproj(rproj_[i],irinv,iphi0,it,iz0,
		 iphiproj[i],izproj[i],iphider[i],izder[i],
		 minusNeighbor[i], plusNeighbor[i]);
    }


    for (int i=0;i<5;i++) {
      int sign=1;
      if (it<0) sign=-1;
      binaryprojdisk(sign*zmean[i],irinv,iphi0,it,iz0,
		     iphiprojdisk[i],irprojdisk[i],iphiderdisk[i],irderdisk[i],
		     minusNeighborDisk[i], plusNeighborDisk[i]);
      //cout << "DUMPDISKPROJ2: "<<i<<" "<<irprojdisk[i]*krprojshiftdisk<<endl;
    }



    //cout << "irinv iphi0 it iz0: "<<irinv<<" "<<iphi0<<" "<<it<<" "<<iz0<<endl;

    return true;

  }


  void approxproj(double rproj,double rinv,double phi0,
		  double t, double z0,
		  double &phiproj, double &zproj,
		  double &phider, double &zder) {

    //This code was written when traveling across the
    //northpole A.R. 2014-04-09

    double s1=0.5*rproj*rinv;

    double s2=s1*s1;

    double s3=1.0+s2/6.0;
    
    double s4=s1*s3;

    phiproj=phi0-s4;

    double s5=t*rproj;

    double s6=s5*s3;

    zproj=z0+s6;

    phider=-0.5*rinv; //-0.25*rinv*s2;

    zder=t; //+0.5*t*s2;

   
    if (dumpproj) {
      if (fabs(rproj-50.0)<10.0) {
	cout << "DUMPPROJ: "
	     << rproj
	     << " "<<rinv
	     << " "<<s1
	     << " "<<s2
	     << " "<<s3
	     << " "<<s4
	     << " "<<phi0
	     << " "<<phiproj
	     << " * "<<t
	     << " "<<s5
	     << " "<<s6
	     << " # "<<zproj
	     << " "<<phider
	     << " "<<zder
	     << endl;
      }
    }

  }


  void exactproj(double rproj,double rinv,double phi0,
		  double t, double z0,
		  double &phiproj, double &zproj,
		  double &phider, double &zder) {

    phiproj=phi0-asin(0.5*rproj*rinv);
    zproj=z0+(2*t/rinv)*asin(0.5*rproj*rinv);

    phider=-0.5*rinv/sqrt(1-pow(0.5*rproj*rinv,2));
    zder=t/sqrt(1-pow(0.5*rproj*rinv,2));

  }



  void binaryproj(double rproj,int irinv, int iphi0, int it, int iz0,
		  int &iphiproj, int &izproj, int &iphider, int &izder,
		  bool &minusNeighbor, bool &plusNeighbor) {


    int irproj=rproj/kr;  //fixed constant


    int is1=((irproj*irinv)>>1)>>is1shift;

    assert(abs(is1)<(1<<16));

    int is2=(is1*is1)>>is2shift;

    assert(is2<(1<<8));
   
    assert(is2>=0);

    int is3=(1<<is3bits)+is2*((ks2/6.0)*(1<<is3bits));

    assert(is2<(1<<(is3bits+1)));

    int shifttmp=idrinvbits+phi0bitshift-is1shift+is3bits-rinvbitshift;
    
    int is4=(is1*is3)>>shifttmp;

    assert(abs(is2)<(1<<17));
 
    iphiproj=iphi0-is4; 
    
    int is5=(it*irproj)>>is5shift;

    assert(abs(is5)<(1<<15));

    int bitshift=is3bits+idrinvbits-tbitshift-is5shift;

    izproj=((iz0<<bitshift)+is5*is3)>>bitshift;

    //cout << "is1 ... is5 : "<<is1<<" "<<is2<<" "<<is3<<" "
    //	 <<is4<<" "<<is5<<endl;


    //cout << "bitshift = "<<bitshift<<endl;

    //izproj=iz0+((is6+(1<<(bitshift-1)))>>bitshift); //This fixes a
                                                      //bias in the
                                                      //calcualted t
    //izproj=iz0+(is6>>bitshift); //has a bias

    iphider=-0.5*irinv;

    izder=it;
    
    iphiproj<<=1; //correct for L456 to match with stubs. Not good... FIXME
    minusNeighbor=false;
    plusNeighbor=false;

    if (iphiproj<(1<<nbitsphistubL456)/8) {
      minusNeighbor=true;
      iphiproj+=3*(1<<nbitsphistubL456)/4;
    }
    if (iphiproj>=7*(1<<nbitsphistubL456)/8) {
      plusNeighbor=true;
      iphiproj-=3*(1<<nbitsphistubL456)/4;
    }

    //cout << "iphiproj (1<<nbitsphistubL456) "
    //	 <<iphiproj<<" "<<(1<<nbitsphistubL456)<<endl;

    assert(iphiproj>=0);
    assert(iphiproj<(1<<nbitsphistubL456));

    if (rproj<60.0) iphiproj>>=(nbitsphistubL456-nbitsphistubL123);

    //cout << " izproj "<<izproj*kz<<endl;

    if (dumpproj) {
      double kphiproj=kphiproj123;
      if (rproj>60.0) kphiproj=kphiproj456;
      if (fabs(rproj-50.0)<10.0) {
	//cout << "kphi0pars kphiproj "<<kphi0pars<<" "<<kphiproj<<endl;
	cout << "DUMPPROJ2 :"<<irproj*kr
	     << " "<<irinv*krinvpars
	     << " "<<is1*ks1
	     << " "<<is2*ks2
	     << " "<<is3*ks3
	     << " "<<is4*ks4
	     << " "<<iphi0*kphi0pars
	     << " "<<iphiproj*kphiproj
	     << " * "<<it*ktpars 
	     << " "<<is5*ks5
	     << " # "<<izproj*kz
	     << " "<<iphider*krinvpars
	     << " "<<izder*ktpars
	     <<endl;
      }
    }


    if (izproj<-(1<<(nbitszprojL123-1))) izproj=-(1<<(nbitszprojL123-1));
    if (izproj>=(1<<(nbitszprojL123-1))) izproj=(1<<(nbitszprojL123-1))-1;

    if (rproj>60.) {
      izproj>>=(nbitszprojL123-nbitszprojL456);
    }
    iphider>>=phiderbitshift;
    izder>>=zderbitshift;

  }



  void approxprojdisk(double zproj,double rinv,double phi0,
		      double t, double z0,
		      double &phiproj, double &rproj,
		      double &phider, double &rder) {


    //double tmp=rinv*(zproj-z0)/(2.0*t);
    //double rprojexact=(2.0/rinv)*sin(tmp);
  
    double t1=zproj-z0;

    double t2=1.0/t;

    double t3=t1*t2;
    
    double t4=t3*rinv;

    phiproj=phi0-t4/2.0;

    double t5=t4*t4;

    double t6=1.0-t5/24.0;

    rproj=t3*t6;

    //cout << "rresid rprojexact rproj "<<rprojexact<<" "<<rproj<<endl;

    rder=t2; 

    phider=-0.5*t2*rinv; 

    //assert(fabs(phider)<0.1);

    //cout << "disk_ zproj "<<disk_<<" "<<zproj<<endl;

    if (dumpproj) {
      if (fabs(zproj+300.0)<10.0) {
	cout << "DUMPPROJDISK: "
	     << " "<<phi0
	     << " "<<zproj
	     << " "<<z0
	     << " "<<rinv
	     << " "<<t1
	     << " "<<t2
	     << " "<<t3
	     << " "<<t4
	     << " phi="<<phiproj
	     << " "<<t5
	     << " "<<t6
	     << " r="<<rproj
	     << " "<<phider
	     << " "<<rder
	     << endl;
      }
    }
  }



  void exactprojdisk(double zproj,double rinv,double phi0,
		     double t, double z0,
		     double &phiproj, double &rproj,
		     double &phider, double &rder) {

    double tmp=rinv*(zproj-z0)/(2.0*t);
    rproj=(2.0/rinv)*sin(tmp);
    phiproj=phi0-tmp;


    //if (fabs(1.0/rinv)>180.0&&fabs(z0)<15.0) {
    //  cout << "phiproj phi0 tmp zproj z0 t: "<<phiproj<<" "<<phi0
    //	   <<" "<<tmp<<" "<<zproj<<" "<<z0<<" "<<t<<endl;
    //}

    if (dumpproj) {
      if (fabs(zproj+300.0)<10.0) {
	cout << "DUMPPROJDISK1: "
	       << " phi="<<phiproj
	       << " r="<<rproj
	       << endl;
	}
      }


    phider=-rinv/(2*t);
    rder=cos(tmp)/t;

    //assert(fabs(phider)<0.1);

  }



  void binaryprojdisk(double zproj,int irinv, int iphi0, int it, int iz0,
		      int &iphiproj, int &irproj, int &iphider, int &irder,
		      bool &minusNeighbor, bool &plusNeighbor) {

    assert(fabs(zproj)>100.0);

    //cout << "it "<<it<<" "<<it*ktparsdisk<<endl;

    //Check if track can hit disk
    if (fabs(it*ktparsdisk)<0.7) {
      irproj=0;
      iphiproj=0;
      iphider=0;
      irder=0;
      return;
    }

    int izproj=zproj/kzdisk;  //fixed constant

    int t1=izproj-iz0;

    int t2=(1<<t2bits)/it;

    int t3=(t1*t2)>>t3shift;
    
    int t4=(t3*irinv)>>t4shift;

    //cout << "kt4disk/kphi0parsdisk :"<<kt4disk/kphi0parsdisk<<" "
    //	 << t3shift+t4shift+rinvbitshiftdisk-t2bits-tbitshift-phi0bitshiftdisk
    //	 << endl;

    //The +1 is for division by 2
    int tmpshift=1+t2bits+tbitshift+phi0bitshiftdisk-t3shift-t4shift-rinvbitshiftdisk;

    iphiproj=iphi0-(t4>>tmpshift);

  
    int t5=(t4>>t4shift2)*(t4>>t4shift2);

    //cout << "t4 = "<<t4<<endl;

    assert(t5>=0);


    int t6=(1<<t6bits)-(t5/24.0)*kst5disk*(1<<t6bits);

    irproj=(t3*t6)>>t6bits;

    irder=t2; 
 
    iphider=-0.5*t2*irinv; 

    iphiproj<<=1; //bit that was shifted away...

    minusNeighbor=false;
    plusNeighbor=false;
    if (iphiproj<(1<<nbitsphistubL456)/8) {
      minusNeighbor=true;
      iphiproj+=3*(1<<nbitsphistubL456)/4;
    }
    if (iphiproj>=7*(1<<nbitsphistubL456)/8) {
      plusNeighbor=true;
      iphiproj-=3*(1<<nbitsphistubL456)/4;
    }

    if (iphiproj<0) iphiproj=0;
    if (iphiproj>=(1<<nbitsphistubL456)) iphiproj=(1<<nbitsphistubL456)-1;


    assert(iphiproj<(1<<nbitsphistubL456));
   
    //cout << "irproj :"<<irproj<<" "<<irproj*krprojdisk<<endl;

    if (dumpproj) {
      if (fabs(zproj+300.0)<10.0) {
	cout << "DUMPPROJDISK2: "
	     << " "<<iphi0*kphi0parsdisk  
	     << " "<<izproj*kzdisk
	     << " "<<iz0*kzdisk
	     << " "<<irinv*krinvparsdisk
	     << " "<<t1*kzdisk
	     << " "<<t2*kt2disk
	     << " "<<t3*kt3disk
	     << " "<<t4*kt4disk
	     << " phi="<<iphiproj*kphi0parsdisk*0.5
	     << " "<<t5*kst5disk
	     << " "<<t6*kt6disk
	     << " r="<<irproj*krprojdisk
	     << " "<<iphider*kphiprojderdisk
	     << " "<<irder*krprojderdisk
	     << endl;
      }
    }


    iphiproj>>=(nbitsphistubL456-nbitsphistubL123);

    irproj>>=rprojdiskbitshift;
    iphider>>=phiderdiskbitshift;
    irder>>=rderdiskbitshift;

    if (irproj<=0) {
      irproj=0;
      iphiproj=0;
      iphider=0;
      irder=0;
      return;      
    }

    assert(irproj>0);

    if (irproj*krprojshiftdisk>120.0) {
      irproj=0;
      iphiproj=0;
      iphider=0;
      irder=0;
      return;      
    }
    


    //cout <<"iphiproj : "<<iphiproj<<endl;

    //assert(irproj*krprojshiftdisk<116);

    //cout << "FPGADisk projection: irproj="<<irproj<<" "<<irproj*krprojshiftdisk
    //	 <<"   iphiproj="<<iphiproj<<" "<<iphiproj*kphi0parsdisk<<endl;


  }



  bool binarytrackletdisk(FPGAStub* innerFPGAStub, 
			  FPGAStub* outerFPGAStub,
			  double sigmaz,
			  int& irinv, int& iphi0,
			  int& it, int& iz0,
			  int iphiprojLayer[6], int izprojLayer[6],
			  int iphiderLayer[6], int izderLayer[6],
			  bool minusNeighborLayer[6], bool plusNeighborLayer[6],
			  int iphiproj[4], int izproj[4],
			  int iphider[4], int izder[4],
			  bool minusNeighbor[4], bool plusNeighbor[4]){
    
    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

 
    int ir1=innerFPGAStub->ir();
    int iphi1=innerFPGAStub->iphi();
    int iz1=innerFPGAStub->iz();

    int ir2=outerFPGAStub->ir();
    int iphi2=outerFPGAStub->iphi();
    int iz2=outerFPGAStub->iz();

    //To get same precission as for layers.
    iphi1<<=(nbitsphistubL456-nbitsphistubL123);
    iphi2<<=(nbitsphistubL456-nbitsphistubL123);


    //Here is where the actual calculation starts
    //Can the first few steps be combined?
    //step 1
    int ideltaphi=iphi2-iphi1;  //should make this positive and keep sign bit?
    //step 2
    int sign=1;
    if (disk_<0) {
      sign=-1;
    }

    int idz=sign*(zmean[abs(disk_)]-zmean[abs(disk_)-1])/kzdisk+iz2-iz1;
    //step 3
    int idr=ir2-ir1;  
    //step 4
    int ir1abs=rmindisk/krdisk+ir1;
    //step 5
    int ir2abs=rmindisk/krdisk+ir2;
    //step 6
    if (idr==0) idr=1;
    int idrinv=(1<<idrinvbits)/idr;  //not actually using 15 bits since idr is largish.
                             //implement in lookup table, just a fcn of ir1-ir2

    //step 7
    int it1=(ir1abs*ir2abs)>>it1shiftdisk;
    //step 8 
    int it2=(idrinv*ideltaphi)>>it2shiftdisk;
    //step 9
    int it3=(it2*it2)>>it3shiftdisk;
    //step 10
    int idelta=0.5*it1*it3;
    //step 11
    int ideltatmp=idelta*(kdeltadisk*(1<<it4bitsdisk));
    //step 12
    int it4=(1<<it4bitsdisk)-3*(ideltatmp>>1);
    //step 13    
    int it5=(1<<it5bitsdisk)-((ideltatmp*it4)>>(2*it4bitsdisk-it5bitsdisk));
    //step 14
    int iDeltainv=(idrinv*it5)>>it5bitsdisk;  
    //step 15
    assert(rinvbitshiftdisk>0);
    irinv=-((ideltaphi*iDeltainv)>>(rinvbitshiftdisk-1)); //-1 because of *2 
    //step 16
    it=(idz*iDeltainv)>>tbitshift;
    //step 17
    assert(it7shiftdisk+irinvshiftdisk-rinvbitshiftdisk>=0);
    int it7=((ir1abs>>1)*irinv)>>(it7shiftdisk+irinvshiftdisk-rinvbitshiftdisk);
    //step 18
    int it7tmp=(it7*it7tmpfactordisk)>>it7tmpshiftdisk;
    //step 19
    int it9=(1<<it9bitsdisk)+((it7tmp*it7tmp)>>(2*it7tmpbitsdisk-2*it7tmpshiftdisk-2*it7shiftdisk-it9bitsdisk));
    //step 20
    int shifttmp1=it9bitsdisk+idrinvbits-irinvshiftdisk-it7shiftdisk-it7shift2disk;
    iphi0=(iphi1+(((it7>>it7shift2disk)*it9)>>shifttmp1))>>phi0bitshiftdisk;
    //step 21
    assert(it12shiftdisk+itshift-tbitshift>=0);
    int it12=(it*ir1abs)>>(it12shiftdisk+itshift-tbitshift);
    //step 23
    iz1+=sign*zmean[abs(disk_)-1]/kzdisk;
    //step 24
    int shifttmp2=it9bitsdisk+idrinvbits-itshift-it12shiftdisk;
    iz0=(iz1-((it12*it9)>>shifttmp2))>>z0bitshift;

 
    if (abs(disk_)==1) {
      
       if (dumppars) {
	cout << "DUMPPARS2:" 
	     << ideltaphi*kphi1
	     <<" "<<idz*kzdisk
	     <<" "<<idr*krdisk
	     <<" > "<<ir1abs*krdisk
	     <<" "<<ir2abs*krdisk
	     <<" "<<idrinv*kdrinvdisk
	     <<" "<<idelta*kdeltadisk
	     <<" "<<it5*kt5disk
	     <<" "<<iDeltainv*kdrinvdisk
	     <<" "<<irinv*krinvpars
	     <<" t= "<<it*ktpars
	     <<" | "<<ir1abs*krdisk
	     <<" "<<it7*kt7disk
	     <<" | "<<it9*kt9
	     <<" "<<iphi1*kphi1
	     <<" phi0= "<<iphi0*kphi0pars
	     <<" * "<<it12*kt12disk
	     <<" "<<iz1*kzdisk
	     <<" z0= "<<iz0*kzpars
	     <<endl;
      }
   

    }

    if (fabs(iz0*kzdisk)>z0cut) {
      //cout << "DUMP iz0 too large: "<<iz0*kzdisk<<endl;
      return false;
    }

    //if (fabs(irinv*krinvparsdisk)>0.0057) {
    if (fabs(irinv*krinvparsdisk)>rinvcut) {
      //cout << "DUMP irinv too large: "<<irinv*krinvparsdisk<<endl;
      return false;
    }


    assert(fabs(iz0)<(1<<(nbitsz0-1)));

    if (iz0>=(1<<(nbitsz0-1))) iz0=(1<<(nbitsz0-1))-1; 
    if (iz0<=-(1<<(nbitsz0-1))) iz0=1-(1<<(nbitsz0-1));
    if (irinv>=(1<<(nbitsrinv-1))) irinv=(1<<(nbitsrinv-1))-1;
    if (irinv<=-(1<<(nbitsrinv-1))) irinv=1-(1<<(nbitsrinv-1));


    //calculate projections

    //cout << "DUMP calling projectbinary disk_="<<disk_<<endl;

    for (int i=0;i<3;i++) {
      binaryprojdisk(zproj_[i],irinv,iphi0,it,iz0,
		     iphiproj[i],izproj[i],iphider[i],izder[i],
		     minusNeighbor[i], plusNeighbor[i]);
    }


    for (int i=0;i<3;i++) {
      binaryproj(rmean[i],irinv,iphi0,it,iz0,
		 iphiprojLayer[i],izprojLayer[i],
		 iphiderLayer[i],izderLayer[i],
		 minusNeighborLayer[i], plusNeighborLayer[i]);
      //cout << "iphiprojLayer : "<<iphiprojLayer[i]<<endl;
    }



    




    //cout << "irinv iphi0 it iz0: "<<irinv<<" "<<iphi0<<" "<<it<<" "<<iz0<<endl;
    
    return true;

  }


  

  bool binarytrackletOverlap(FPGAStub* innerFPGAStub, 
			     FPGAStub* outerFPGAStub,
			     double sigmaz,
			     int& irinv, int& iphi0,
			     int& it, int& iz0,
			     int iphiprojLayer[6], int izprojLayer[6],
			     int iphiderLayer[6], int izderLayer[6],
			     bool minusNeighborLayer[6], 
			     bool plusNeighborLayer[6],
			     int iphiproj[4], int izproj[4],
			     int iphider[4], int izder[4],
			     bool minusNeighbor[4], bool plusNeighbor[4]){

    //cout << "In binarytrackletoverlap : "<<disk_<<endl;

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }
 
    int ir1=innerFPGAStub->ir();
    int iphi1=innerFPGAStub->iphi();
    int iz1=innerFPGAStub->iz();

    int ir2=outerFPGAStub->ir();
    int iphi2=outerFPGAStub->iphi();
    int iz2=outerFPGAStub->iz();

    //To get same precission as for disks.
    iphi1<<=3;
    iphi2<<=3;

    //Radial position in layers are less precise than disks.
    //to make them the same we need to add a bit. ???
    ir2<<=1;

    //Here is where the actual calculation starts
    //Can the first few steps be combined?
    //step 1
    int ideltaphi=iphi2-iphi1;  //should make this positive and keep sign bit?
    //step 2
    int sign=1;
    if (disk_<0) {
      sign=-1;
    }

    //cout << "disk_ = "<<disk_<<endl;
    assert(abs(disk_)==1);
    int idz=-sign*zmean[abs(disk_)-1]/kzdisk+iz2-iz1;
    //step 3
    int idr=rmean[outerFPGAStub->layer().value()]/kr+ir2-rmindisk/krdisk-ir1;  
    //step 4
    int ir1abs=rmindisk/krdisk+ir1;
    //step 5
    int ir2abs=rmean[outerFPGAStub->layer().value()]/kr+ir2;
    //step 6
    if (idr==0) idr=1;
    int idrinv=(1<<idrinvbits)/idr;  //not actually using 15 bits since idr is largish.
                             //implement in lookup table, just a fcn of ir1-ir2

    //step 7
    int it1=(ir1abs*ir2abs)>>it1shiftdisk;
    //step 8 
    int it2=(idrinv*ideltaphi)>>it2shiftdisk;
    //step 9
    int it3=(it2*it2)>>it3shiftdisk;
    //step 10
    int idelta=0.5*it1*it3;
    //step 11
    int ideltatmp=idelta*(kdeltadisk*(1<<it4bitsdisk));
    //step 12
    int it4=(1<<it4bitsdisk)-3*(ideltatmp>>1);
    //step 13    
    int it5=(1<<it5bitsdisk)-((ideltatmp*it4)>>(2*it4bitsdisk-it5bitsdisk));
    //step 14
    int iDeltainv=(idrinv*it5)>>it5bitsdisk;  
    //step 15
    assert(rinvbitshiftdisk>0);
    irinv=-((ideltaphi*iDeltainv)>>(rinvbitshiftdisk-1)); //-1 because of *2 
    //step 16
    it=(idz*iDeltainv)>>tbitshift;
    //step 17
    assert(it7shiftdisk+irinvshiftdisk-rinvbitshiftdisk>=0);
    int it7=((ir1abs>>1)*irinv)>>(it7shiftdisk+irinvshiftdisk-rinvbitshiftdisk);
    //step 18
    int it7tmp=(it7*it7tmpfactordisk)>>it7tmpshiftdisk;
    //step 19
    int it9=(1<<it9bitsdisk)+((it7tmp*it7tmp)>>(2*it7tmpbitsdisk-2*it7tmpshiftdisk-2*it7shiftdisk-it9bitsdisk));
    //step 20
    int shifttmp1=it9bitsdisk+idrinvbits-irinvshiftdisk-it7shiftdisk-it7shift2disk;
    iphi0=(iphi1+(((it7>>it7shift2disk)*it9)>>shifttmp1))>>phi0bitshiftdisk;
    //step 22
    assert(it12shiftdisk+itshift-tbitshift>=0);
    int it12=(it*ir1abs)>>(it12shiftdisk+itshift-tbitshift);
    //step 23
    iz1+=sign*zmean[abs(disk_)-1]/kzdisk;
    //step 24
    int shifttmp2=it9bitsdisk+idrinvbits-itshift-it12shiftdisk;
    iz0=(iz1-((it12*it9)>>shifttmp2))>>z0bitshift;

    if (dumppars) {   //overlap
      cout << "DUMPPARS2:"
	   << ideltaphi*kphi1
	   <<" "<<idz*kzdisk
	   <<" "<<idr*krdisk
	   <<" > "<<ir1abs*krdisk
	   <<" "<<ir2abs*krdisk
	   <<" "<<idrinv*kdrinvdisk
	   <<" "<<idelta*kdeltadisk
	   <<" "<<it5*kt5disk
	   <<" "<<iDeltainv*kdrinvdisk
	   <<" "<<irinv*krinvpars
	   <<" t= "<<it*ktpars
	   <<" | "<<ir1abs*krdisk
	   <<" "<<it7*kt7disk
	   <<" | "<<it9*kt9disk
	   <<" "<<iphi1*kphi1
	   <<" phi0= "<<iphi0*kphi0pars
	   <<" * "<<it12*kt12disk
	   <<" "<<iz1*kzdisk
	   <<" z0= "<<iz0*kzpars
	   <<endl;
      
    }

    //cout << "z0 and rinv : "<<iz0*kzdisk<<" "<<irinv*krinvparsdisk<<endl;

    if (fabs(iz0*kzdisk)>z0cut) {
      //cout << "DUMP overlap iz0 too large: "<<iz0*kzdisk<<endl;
      return false;
    }

    if (fabs(irinv*krinvparsdisk)>rinvcut) {
      //cout << "DUMP overlap irinv too large: "<<irinv*krinvparsdisk<<endl;
      return false;
    }

    assert(fabs(iz0)<(1<<(nbitsz0-1)));

    if (iz0>=(1<<(nbitsz0-1))) iz0=(1<<(nbitsz0-1))-1; 
    if (iz0<=-(1<<(nbitsz0-1))) iz0=1-(1<<(nbitsz0-1));
    if (irinv>=(1<<(nbitsrinv-1))) irinv=(1<<(nbitsrinv-1))-1;
    if (irinv<=-(1<<(nbitsrinv-1))) irinv=1-(1<<(nbitsrinv-1));

    //calculate projections

    //cout << "DUMP calling projectbinary disk_="<<disk_<<endl;

    for (int i=0;i<4;i++) {
      binaryprojdisk(zprojoverlap_[i],irinv,iphi0,it,iz0,
		     iphiproj[i],izproj[i],iphider[i],izder[i],
		     minusNeighbor[i], plusNeighbor[i]);
      //cout << "zproj 1 der : "<<izder[i]<<" "<<it<<endl;
    }


    for (int i=0;i<1;i++) {
      binaryproj(rmean[i],irinv,iphi0,it,iz0,
		 iphiprojLayer[i],izprojLayer[i],
		 iphiderLayer[i],izderLayer[i],
		 minusNeighborLayer[i], plusNeighborLayer[i]);
      //cout << "iphiprojLayer : "<<iphiprojLayer[i]<<endl;
      //cout << "zproj 2 der : "<<izderLayer[i]<<" "<<it<<endl;
    }



    




    //cout << "irinv iphi0 it iz0: "<<irinv<<" "<<iphi0<<" "<<it<<" "<<iz0<<endl;
    
    return true;

  }



  void approxtrackletdisk(double r1, double z1, double phi1,
			  double r2, double z2, double phi2, double sigmaz,
			  double &rinv, double &phi0,
			  double &t, double &z0,
			  double phiprojLayer[4], double rprojLayer[4], 
			  double phiderLayer[4], double rderLayer[4],
			  double phiproj[4], double rproj[4], 
			  double phider[4], double rder[4]) {

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }


    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (phi1<0.0) phi1+=two_pi;
    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;
    //cout << "phi1 phimin_ phimax_:"<<phi1<<" "<<phimin_<<" "<<phimax_<<endl;
    assert(phi1tmp>0.0);

    double dr=r2-r1;
    double dz=z1-z2;
    double drinv=1.0/dr;
    double t2=deltaphi*drinv;
    double delta=0.5*r1*r2*t2*t2;// *(1+deltaphi*deltaphi/12.0);
    double t5=1.0-delta+1.5*delta*delta;// -2.5*delta*delta*delta;
    double deltainv=t5*drinv;
    rinv=2.0*deltaphi*deltainv;// *(1-deltaphi*deltaphi/6.0);
    t=-dz*deltainv;// *(1.0-deltaphi*deltaphi/6.0); 
    double t7=0.5*r1*rinv;
    double t9=1+t7*t7/6.0;// +3.0*t7*t7*t7*t7/40.0;
    phi0=phi1tmp+t7*t9;
    double t12=t*r1*t9;
    z0=z1-t12;


    if (abs(disk_)==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "t7*t9 : "<<t7*t9<<endl;
	cout << "DUMPPARS1:" 
	     << -deltaphi
	     <<" "<<z2-z1
	     <<" > "<<r1
	     <<" "<<r2
	     <<" "<<1.0/(r2-r1)
	     <<" "<<delta
	     <<" "<<t5
	     <<" "<<deltainv
	     <<" "<<rinv
	     <<" t= "<<t
	     <<" | "<<r1
	     <<" "<<t7
	     <<" | "<<t9
	     <<" "<<phi1tmp
	     <<" phi0= "<<phi0
	     <<" * "<<t12
	     <<" "<<z1
	     <<" z0= "<<z0
	     <<endl;
      }
      

//      cout << "Approx tracklet: dphi="<<-deltaphi<<" dz="<<z2-z1
//	   << " dr="<<r2-r1<<" drinv="<<1.0/(r2-r1)
//	   <<" delta="<<delta
//	   <<" t5="<<t5
//           <<" deltainv="<<deltainv
//	   <<" rinv="<<rinv
//	   <<" t="<<t
//	   <<" r1abs="<<r1
//	   <<" t7="<<t7
//	   <<" t9="<<t9
//	   <<" phi1="<<phi1tmp
//	   <<" ***phi0="<<phi0
//	   <<" t12="<<t12
//	   <<" z1="<<z1
//	   <<" z0="<<z0
//	   <<endl;

   

    }
   

    //calculate projection


    for (int i=0;i<3;i++) {
      approxproj(rmean[i],rinv,phi0,t,z0,
		 phiprojLayer[i],rprojLayer[i],
		 phiderLayer[i],rderLayer[i]);
    }


    //static ofstream out1("neighborproj.txt");

    for (int i=0;i<3;i++) {
      approxprojdisk(zproj_[i],rinv,phi0,t,z0,
		     phiproj[i],rproj[i],phider[i],rder[i]);
      /*
      if (fabs(z0)<15.0&&fabs(rinv)<0.0057) {
	if ((fabs(rproj[i])<100.0)&&(phiproj[i]<(phimax_-phimin_)/6)){
	  out1<<disk_<<" -1 "<<phiproj[i]<<endl;
	} else  if ((fabs(rproj[i])<100.0)&&(phiproj[i]>7.0*(phimax_-phimin_)/6)){
	  out1<<disk_<<" +1 "<<phiproj[i]<<endl;
	} else if (fabs(rproj[i])<100.0){
	  out1<<disk_<<" 0 "<<phiproj[i]<<endl;
	}
      }
      */
    }

  }



 

  void execute() {

    //we are taking a shortcut here by using the information to
    //find the stubs from the stub pairs list...

    unsigned int countall=0;
    unsigned int countsel=0;

    for(unsigned int l=0;l<stubpairs_.size();l++){
      for(unsigned int i=0;i<stubpairs_[l]->nStubPairs();i++){

	countall++;
	
	L1TStub* innerStub=stubpairs_[l]->getL1TStub1(i);
	FPGAStub* innerFPGAStub=stubpairs_[l]->getFPGAStub1(i);

	L1TStub* outerStub=stubpairs_[l]->getL1TStub2(i);
	FPGAStub* outerFPGAStub=stubpairs_[l]->getFPGAStub2(i);
	
	if (innerFPGAStub->isBarrel()){

	  assert(outerFPGAStub->isBarrel());

	  //FIXME - should be set at initialization
	  assert(layer_==innerFPGAStub->layer().value()+1);

	  assert(layer_==1||layer_==3||layer_==5);

	  int lproj[4];

	  if (layer_==1) {
	    rproj_[0]=rmeanL3;
	    rproj_[1]=rmeanL4;
	    rproj_[2]=rmeanL5;
	    rproj_[3]=rmeanL6;
	    lproj[0]=3;
	    lproj[1]=4;
	    lproj[2]=5;
	    lproj[3]=6;
	  }
      
	  if (layer_==3) {
	    rproj_[0]=rmeanL1;
	    rproj_[1]=rmeanL2;
	    rproj_[2]=rmeanL5;
	    rproj_[3]=rmeanL6;
	    lproj[0]=1;
	    lproj[1]=2;
	    lproj[2]=5;
	    lproj[3]=6;
	  }
	  
	  if (layer_==5) {
	    rproj_[0]=rmeanL1;
	    rproj_[1]=rmeanL2;
	    rproj_[2]=rmeanL3;
	    rproj_[3]=rmeanL4;
	    lproj[0]=1;
	    lproj[1]=2;
	    lproj[2]=3;
	    lproj[3]=4;
	  }
	  


	  //cout << "Calculating tracklet parameters in layer "<<layer_<<endl;
	
      	  
	  FPGAWord iphi1=innerFPGAStub->phi();
	  FPGAWord iz1=innerFPGAStub->z();
	  FPGAWord ir1=innerFPGAStub->r();

	  FPGAWord iphi2=outerFPGAStub->phi();
	  FPGAWord iz2=outerFPGAStub->z();
	  FPGAWord ir2=outerFPGAStub->r();
	  
	  
	  double r1=innerStub->r();
	  double z1=innerStub->z();
	  double phi1=innerStub->phi();
	  
	  double r2=outerStub->r();
	  double z2=outerStub->z();
	  double phi2=outerStub->phi();
	  
	  
	  double rinv,phi0,t,z0;
	  
	  double phiproj[4],zproj[4],phider[4],zder[4];
	  double phiprojdisk[5],rprojdisk[5],phiderdisk[5],rderdisk[5];
	  
	  exacttracklet(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
			rinv,phi0,t,z0,
			phiproj,zproj,phider,zder,
			phiprojdisk,rprojdisk,phiderdisk,rderdisk);
	  
	  
	  if (1) {
	    int lphi=1;
	    int lz=1;
	    int lr=2;
	    if (layer_>3) {
	      lphi=8;
	      lz=16;
	      lr=1;
	    }
	    
	    double dphi1=phi1-phimin_+(phimax_-phimin_)/6.0-iphi1.value()*kphi/lphi;

	    //cout << "dphi1 phi1 phimin_ phimax_ : "
	    //	 <<dphi1<<" "<<phi1<<" "<<phimin_<<" "<<phimax_<<endl;
	    
	    if (dphi1<-0.5*two_pi) dphi1+=two_pi;
	    //cout << "layer dphi1 phi1 : "<<layer_<<" "<<dphi1<<" "<<phi1<<endl;
	    assert(fabs(dphi1)<1e-4);
	    phi1-=dphi1;
	    double dz=z1-iz1.value()*kz*lz;
	    assert(fabs(dz)<1.0*lz);
	    z1-=dz;
	    //cout <<r1<<" "<<lr*ir1.value()*kr<<" "<<rmean[layer_-1]<<endl;
	    double dr=r1-lr*ir1.value()*kr-rmean[layer_-1];
	    assert(fabs(dr)<0.1);
	    r1-=dr;
	  }
	  
	  


	  if (1) {
	    int lphi=1;
	    int lz=1;
	    int lr=2;
	    if (layer_>2) {
	      lphi=8;
	      lz=16;
	      lr=1;
	    }
	    double dphi2=phi2-phimin_+(phimax_-phimin_)/6.0-iphi2.value()*kphi/lphi;
	    if (dphi2<-0.5*two_pi) dphi2+=two_pi;
	    //cout << "layer dphi2 phi2 iphi2: "<<layer_<<" "<<dphi2<<" "
	    //	 <<phi2<<" "<< iphi2.value()<<endl;
	    assert(fabs(dphi2)<1e-4);
	    phi2-=dphi2;
	    //cout <<"z2 iz2: "<<z2<<" "<<iz2.value()*kz*lz<<endl;
	    double dz=z2-iz2.value()*kz*lz;
	    assert(fabs(dz)<1.0*lz);
	    z2-=dz;
	    //cout <<r2<<" "<<2*ir2.value()*kr<<" "<<rmin[layer_]<<endl;
	    double dr=r2-lr*ir2.value()*kr-rmean[layer_];
	    assert(fabs(dr)<0.1);
	    r2-=dr;
	  }
	  
	  double rinvapprox,phi0approx,tapprox,z0approx;
	  double phiprojapprox[4],zprojapprox[4],phiderapprox[4],zderapprox[4];
	  double phiprojdiskapprox[5],rprojdiskapprox[5];
	  double phiderdiskapprox[5],rderdiskapprox[5];
	  
	  approxtracklet(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
			 rinvapprox,phi0approx,tapprox,z0approx,
			 phiprojapprox,zprojapprox,phiderapprox,zderapprox,
			 phiprojdiskapprox,rprojdiskapprox,
			 phiderdiskapprox,rderdiskapprox);
	  
	  int irinv,iphi0,it,iz0;
	  int iphiproj[4],izproj[4],iphider[4],izder[4];
	  bool minusNeighbor[4],plusNeighbor[4];
	  int iphiprojdisk[5],irprojdisk[5],iphiderdisk[5],irderdisk[5];
	  bool minusNeighborDisk[5],plusNeighborDisk[5];
	  
	  bool success=binarytracklet(innerFPGAStub,outerFPGAStub,
				      outerStub->sigmaz(),
				      irinv,iphi0,it,iz0,
				      iphiproj,izproj,iphider,izder,
				      minusNeighbor,plusNeighbor,
				      iphiprojdisk,irprojdisk,
				      iphiderdisk,irderdisk,
				      minusNeighborDisk,plusNeighborDisk);
	  
	  if (!success) continue;
	  
	  for(unsigned int j=0;j<5;j++){
	    if (minusNeighborDisk[j]) {
	      phiprojdiskapprox[j]+=dphisector;
	      phiprojdisk[j]+=dphisector;
	    }
	    if (plusNeighborDisk[j]) {
	      phiprojdiskapprox[j]-=dphisector;
	      phiprojdisk[j]-=dphisector;
	    }
	  }
	  
	  for(unsigned int j=0;j<4;j++){
	    if (minusNeighbor[j]) {
	      phiprojapprox[j]+=dphisector;
	    phiproj[j]+=dphisector;
	    }
	    if (plusNeighbor[j]) {
	      phiprojapprox[j]-=dphisector;
	      phiproj[j]-=dphisector;
	    }	    
	  }
	  
	  
	  if ((layer_==1)&&(fabs(iz0*kzpars)>z0cut)) {
	    continue;
	  }
	  
	  if ((layer_>=3)&&(fabs(iz0*kzpars)>25.0)) {
	    continue;
	  }
	  
	  //Can be done more accurate now
	  //limit on the number of processed matches in each TE
	  //++processed_matches;
	  //if ( processed_matches >= NMAXTE) {
	  //	    cout<<i<<" "<<processed_matches<<"\n";
	  //continue; 
	  //}
	  
	  if (writeTrackletPars) {
	    static ofstream out("trackletpars.txt");
	    out <<"Trackpars "<<layer_
		<<"   "<<rinv<<" "<<rinvapprox<<" "<<irinv*krinvpars
		<<"   "<<phi0<<" "<<phi0approx<<" "<<iphi0*kphi0pars
		<<"   "<<t<<" "<<tapprox<<" "<<it*ktpars
		<<"   "<<z0<<" "<<z0approx<<" "<<iz0*kzpars
		<<endl;
	  }	    

	  if (writeTrackProj) {
	    static ofstream out1("trackproj.txt");
	    for (int i=0;i<4;i++) {
	      double kphiproj=kphiproj123;
	      int lz=1;
	      if (rproj_[i]>60.0) {
		kphiproj=kphiproj456;
		lz=16;
	      }
	      out1 <<"Trackproj "<<layer_<<" "<<rproj_[i]
		   <<"   "<<phiproj[i]<<" "<<phiprojapprox[i]
		   <<" "<<iphiproj[i]*kphiproj
		   <<"   "<<zproj[i]<<" "<<zprojapprox[i]
		   <<" "<<izproj[i]*kzproj*lz
		   <<"   "<<phider[i]<<" "<<phiderapprox[i]
		   <<" "<<iphider[i]*kphider
		   <<"   "<<zder[i]<<" "<<zderapprox[i]
		   <<" "<<izder[i]*kzder
		   <<endl;
	    }
	    
	  }
	  


	  //Can also be done more accurately now
	  //counter for steps 4&5: number of tracklets to route and project into VM
	  //counter is per sector
	  //++Ntracklets_;
	  //if(Ntracklets_ < NMAXproj){
	  
	  FPGATracklet* tracklet=new FPGATracklet(innerStub,outerStub,
						  innerFPGAStub,outerFPGAStub,
						  phioffset_,
						  rinv,phi0,z0,t,
						  rinvapprox,phi0approx,
						  z0approx,tapprox,
						  irinv,iphi0,iz0,it,
						  iphiproj,izproj,iphider,izder,
						  minusNeighbor,plusNeighbor,
						  phiproj,zproj,phider,zder,
						  phiprojapprox,zprojapprox,
						  phiderapprox,zderapprox,
						  iphiprojdisk,irprojdisk,
						  iphiderdisk,irderdisk,
						  minusNeighborDisk,
						  plusNeighborDisk,
						  phiprojdisk,rprojdisk,
						  phiderdisk,rderdisk,
						  phiprojdiskapprox,
						  rprojdiskapprox,
						  phiderdiskapprox,
						  rderdiskapprox,
						  false);
	  
	  //cout << "Found tracklet in layer = "<<layer_<<" "
	  //<<iSector_<<" "<<tracklet<<endl;

	  countsel++;

	  trackletpars_->addTracklet(tracklet);

	  for(unsigned int j=0;j<5;j++){
	    int disk=j+1;
	    addDiskProj(tracklet,disk);
	  }
	  
	  for(unsigned int j=0;j<4;j++){
	    addLayerProj(tracklet,lproj[j]);
	  }
	  



	  
	}  else {

	  if (outerFPGAStub->isDisk()) {

	    //FIXME - should be set at initialization
	    disk_=innerFPGAStub->disk().value();
	    int dproj[3];

	    assert(abs(disk_)==1||abs(disk_)==3);
       
	    if (disk_==1) {
	      zproj_[0]=zmeanD3;
	      zproj_[1]=zmeanD4;
	      zproj_[2]=zmeanD5;
	      dproj[0]=3;
	      dproj[1]=4;
	      dproj[2]=5;
	    }
      
	    if (disk_==3) {
	      zproj_[0]=zmeanD1;
	      zproj_[1]=zmeanD2;
	      zproj_[2]=zmeanD5;
	      dproj[0]=1;
	      dproj[1]=2;
	      dproj[2]=5;
	    }
	    
	    if (disk_==-1) {
	      zproj_[0]=-zmeanD3;
	      zproj_[1]=-zmeanD4;
	      zproj_[2]=-zmeanD5;
	      dproj[0]=-3;
	      dproj[1]=-4;
	      dproj[2]=-5;
	    }
	    
	    if (disk_==-3) {
	      zproj_[0]=-zmeanD1;
	      zproj_[1]=-zmeanD2;
	      zproj_[2]=-zmeanD5;
	      dproj[0]=-1;
	      dproj[1]=-2;
	      dproj[2]=-5;
	    }
	    
	    
	    if (innerStub->r()>60.0) continue;
	    if (outerStub->r()>60.0) continue;
	    
	    int istubpt1=innerFPGAStub->stubpt().value();
	    int iphivm1=innerFPGAStub->phivm().value();
	    FPGAWord iphi1=innerFPGAStub->phi();
	    FPGAWord iz1=innerFPGAStub->z();
	    FPGAWord ir1=innerFPGAStub->r();
	    int irvm1=innerFPGAStub->rvm().value();
	    //int izvm1=innerFPGAStub->zvm().value();
	    
	    int istubpt2=outerFPGAStub->stubpt().value();
	    int iphivm2=outerFPGAStub->phivm().value();
	    FPGAWord iphi2=outerFPGAStub->phi();
	    FPGAWord iz2=outerFPGAStub->z();
	    FPGAWord ir2=outerFPGAStub->r();
	    int irvm2=outerFPGAStub->rvm().value();
	    //int izvm2=outerFPGAStub->zvm().value();
	    
	  
	    
	    int ideltaphi=iphivm2-iphivm1;
	    int ideltar=irvm2-irvm1;

	    ideltar>>=2;
	    
	    if (ideltar<0) ideltar+=8;
	    assert(ideltar>=0);
	    if (ideltaphi<0) ideltaphi+=16;
	    assert(ideltaphi>=0);
	    
	    //cout << "istubpt1 istubpt2 : "<<istubpt1<<" "<<istubpt2<<endl;
	    assert(istubpt1>=0);
	    assert(istubpt2>=0);
	    
	    //int address=(istubpt1<<10)+(istubpt2<<7)+(ideltaphi<<3)+ideltar;
	    
	    //int i1=TEs_[i].first.first;
	    //int i2=TEs_[i].second.first;
	    //int j1=TEs_[i].first.second;
	    //int j2=TEs_[i].second.second;
	    
	    
	    //bool phimatch=(*TETables_)[i].phicheck(address,i1,i2,j1,j2);
	    
	    //bool zmatch=(*TETables_)[i].zcheck(izvm1,izvm2,irvm1,irvm2);
	    
	    
	    double r1=innerStub->r();
	    double z1=innerStub->z();
	    double phi1=innerStub->phi();
	    
	    double r2=outerStub->r();
	    double z2=outerStub->z();
	    double phi2=outerStub->phi();
	    
	    if (r2<r1+2.0) continue; //Protection... Should be handled cleaner
	                             //to avoid problem with floating point 
	                             //calculation
	      
	    //cout << "r1 r2 z1 z2 "<<r1<<" "<<r2<<" "<<z1<<" "<<z2<<endl;

	    double rinv,phi0,t,z0;

	    double phiproj[3],zproj[3],phider[3],zder[3];
	    double phiprojdisk[3],rprojdisk[3],phiderdisk[3],rderdisk[3];
	    
	    exacttrackletdisk(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
			      rinv,phi0,t,z0,
			      phiproj,zproj,phider,zder,
			      phiprojdisk,rprojdisk,phiderdisk,rderdisk);


	    //Truncates floating point positions to integer
	    //representation precision
	    if (1) {
	      int lphi=1;
	      int lz=1;
	      int lr=1;
	      //if (layer_>3) {
	      //  lphi=8;
	      //  lz=16;
	      //  lr=1;
	      //}
	      double dphi1=phi1-phimin_+(phimax_-phimin_)/6.0-iphi1.value()*kphi/lphi;
	      if (dphi1<-0.5*two_pi) dphi1+=two_pi;
	      //cout << "layer dphi1 phi1 : "<<layer_<<" "<<dphi1<<" "<<phi1<<endl;
	      assert(fabs(dphi1)<1e-4);
	      phi1-=dphi1;
	      int sign=1;
	      if (disk_<0) sign=-1;
	      double dz=z1-(iz1.value()*kzdisk*lz+sign*zmean[abs(disk_)-1]);
	      //cout << "z1 iz1.value() kz : "<<z1<<" "<<iz1.value()<<" "<<kz<<" "
	      //	 <<iz1.value()*kz<<" "<<sign*zmean[abs(disk_)-1]<<endl;
	      assert(fabs(dz)<1.0*lz);
	      z1-=dz;
	      //cout <<"r1 : "<<r1<<" "<<lr*ir1.value()*krdisk+rmindisk<<endl;
	      double dr=r1-lr*ir1.value()*krdisk-rmindisk;
	      assert(fabs(dr)<0.75);
	      r1-=dr;
	      //cout << "dz1 dr1 : "<<dz<<" "<<dr<<endl;
	    }
	    



	    if (1) {
	      int lphi=1;
	      int lz=1;
	      int lr=1;
	      
	      double dphi2=phi2-phimin_+(phimax_-phimin_)/6.0-iphi2.value()*kphi/lphi;
	      if (dphi2<-0.5*two_pi) dphi2+=two_pi;
	      //cout << "layer dphi2 phi2 iphi2: "<<layer_<<" "<<dphi2<<" "
	      //	 <<phi2<<" "<< iphi2.value()<<endl;
	      assert(fabs(dphi2)<1e-4);
	      phi2-=dphi2;
	      //cout <<"z2 iz2: "<<z2<<" "<<iz2.value()*kz*lz<<endl;
	      int sign=1;
	      if (disk_<0) sign=-1;
	      double dz=z2-iz2.value()*kzdisk*lz-sign*zmean[abs(disk_)];
	      assert(fabs(dz)<1.0*lz);
	      z2-=dz;
	      //cout <<r2<<" "<<2*ir2.value()*kr<<" "<<rmin[layer_]<<endl;
	      double dr=r2-lr*ir2.value()*krdisk-rmindisk;
	      assert(fabs(dr)<0.1);
	      r2-=dr;
	      //cout << "dz2 dr2 : "<<dz<<" "<<dr<<endl;
	    }
	    
	    
	    
	    //FPGATrackletCand trackletcand(innerStub,outerStub);
	    //candlist_[i].addTrackletCand(trackletcand);
	    
	    
	    double rinvapprox,phi0approx,tapprox,z0approx;
	    double phiprojapprox[3],zprojapprox[3],phiderapprox[3],zderapprox[3];
	    double phiprojapproxdisk[3],rprojapproxdisk[3],
	      phiderapproxdisk[3],rderapproxdisk[3];
	  
	    approxtrackletdisk(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
			       rinvapprox,phi0approx,tapprox,z0approx,
			       phiprojapprox,zprojapprox,
			       phiderapprox,zderapprox,
			       phiprojapproxdisk,rprojapproxdisk,
			       phiderapproxdisk,rderapproxdisk);
	    
	    int irinv,iphi0,it,iz0;
	    int iphiproj[3],izproj[3],iphider[3],izder[3];
	    bool minusNeighbor[3],plusNeighbor[3];
	    
	    int iphiprojdisk[3],irprojdisk[3],iphiderdisk[3],irderdisk[3];
	    bool minusNeighbordisk[3],plusNeighbordisk[3];
	    
	    bool success=binarytrackletdisk(innerFPGAStub,outerFPGAStub,
					    outerStub->sigmaz(),
					    irinv,iphi0,it,iz0,
					    iphiproj,izproj,
					    iphider,izder,
					    minusNeighbor,plusNeighbor,
					    iphiprojdisk,irprojdisk,
					    iphiderdisk,irderdisk,
					    minusNeighbordisk,plusNeighbordisk);
	    
	    if (!success) continue;
	  
	    for(unsigned int j=0;j<3;j++){
	      if (minusNeighbordisk[j]) {
		phiprojapproxdisk[j]+=dphisector;
		phiprojdisk[j]+=dphisector;
	      }
	      if (plusNeighbordisk[j]) {
		phiprojapproxdisk[j]-=dphisector;
		phiprojdisk[j]-=dphisector;
	      }	    
	    }
	    
	    for(unsigned int j=0;j<3;j++){
	      if (minusNeighbor[j]) {
		phiprojapprox[j]+=dphisector;
		phiproj[j]+=dphisector;
	      }
	      if (plusNeighbor[j]) {
		phiprojapprox[j]-=dphisector;
		phiproj[j]-=dphisector;
	      }
	    }
	    
	    
	    /* 
	       if (!phimatch) {
	       //cout << "Rejected due to phimatch "<<istubpt1<<" "
	       //	 <<istubpt2<<" "<<rinv<<" "<<t<<endl;
	       continue;
	       }
	       
	       if (!zmatch) { 
	       //cout << "Rejected due to zmatch : "<<z0approx<<" "
	       //	 <<izvm1<<" "<<izvm2<<" "<<irvm1<<" "<<irvm2<<" "
	       //	 <<z1<<" "<<z2<<" "<<r1<<" "<<r2<<endl;
	       continue;
	       }
	    */
	    
	    /*
	    //limit on the number of processed matches in each TE
	    ++processed_matches;
	    if ( processed_matches >= NMAXTE) {
	    //	    cout<<i<<" "<<processed_matches<<"\n";
	    continue; 
	    }
	    */

	    if (writeTrackletParsDisk) {
	      static ofstream out("trackletparsdisk.txt");
	      out <<"Trackpars "<<disk_
		  <<"   "<<rinv<<" "<<rinvapprox<<" "<<irinv*krinvparsdisk
		  <<"   "<<phi0<<" "<<phi0approx<<" "<<iphi0*kphi0parsdisk
		  <<"   "<<t<<" "<<tapprox<<" "<<it*ktparsdisk
		  <<"   "<<z0<<" "<<z0approx<<" "<<iz0*kzdisk
		  <<endl;
	    }

	    if (writeTrackProj) {
	      static ofstream out1("trackproj.txt");
	      for (int i=0;i<3;i++) {
		double kphiproj=kphiproj123;
		out1 <<"Trackproj "<<disk_<<" "<<zproj_[i]
		     <<"   "<<phiprojdisk[i]<<" "<<phiprojapproxdisk[i]
		     <<" "<<iphiprojdisk[i]*kphiproj
		     <<"   "<<rprojdisk[i]<<" "<<rprojapproxdisk[i]
		     <<" "<<irprojdisk[i]*krprojshiftdisk
		     <<"   "<<phiderdisk[i]<<" "<<phiderapproxdisk[i]
		     <<" "<<iphiderdisk[i]*kphider
		     <<"   "<<rderdisk[i]<<" "<<rderapproxdisk[i]
		     <<" "<<irderdisk[i]*krprojderdiskshift
		     <<endl;
	      }	      
	    }
	    
	    //counter is per sector
	    //++Ntracklets_;
	    //if(Ntracklets_ < NMAXproj){

	    FPGATracklet* tracklet=new FPGATracklet(innerStub,outerStub,
						    innerFPGAStub,outerFPGAStub,
						    phioffset_,
						    rinv,phi0,z0,t,
						    rinvapprox,phi0approx,
						    z0approx,tapprox,
						    irinv,iphi0,iz0,it,
						    iphiproj,izproj,iphider,izder,
						    minusNeighbor,plusNeighbor,	
						    phiproj,zproj,phider,zder,
						    phiprojapprox,zprojapprox,
						    phiderapprox,zderapprox,
						    iphiprojdisk,irprojdisk,
						    iphiderdisk,irderdisk,
						    minusNeighbordisk,
						    plusNeighbordisk,
						    phiprojdisk,rprojdisk,
						    phiderdisk,rderdisk,
						    phiprojapproxdisk,
						    rprojapproxdisk,
						    phiderapproxdisk,
						    rderapproxdisk,
						    true);
	    

	    //cout << "Found tracklet in disk = "<<disk_<<" "<<tracklet
	    //<<" "<<iSector_<<endl;


	    countsel++;

	    //cout << "Found tracklet "<<tracklet<<endl;

	    trackletpars_->addTracklet(tracklet);

	    
	    addLayerProj(tracklet,1);
	    addLayerProj(tracklet,2);

	  
	    for(unsigned int j=0;j<3;j++){
	      addDiskProj(tracklet,dproj[j]);
	    }
	  

	  } else {


	    //Deal with overlap stubs here
	    assert(outerFPGAStub->isBarrel());
	    assert(innerFPGAStub->isDisk());

	    disk_=innerFPGAStub->disk().value();

	    //cout << "trying to make overlap tracklet disk_ = "<<disk_<<endl;

	    int sign=1;
	    if (disk_<0) sign=-1;
	    
	    zprojoverlap_[0]=sign*zmeanD2;
	    zprojoverlap_[1]=sign*zmeanD3;
	    zprojoverlap_[2]=sign*zmeanD4;
	    zprojoverlap_[3]=sign*zmeanD5;


	    int istubpt1=innerFPGAStub->stubpt().value();
	    int iphivm1=innerFPGAStub->phivm().value();
	    FPGAWord iphi1=innerFPGAStub->phi();
	    FPGAWord iz1=innerFPGAStub->z();
	    FPGAWord ir1=innerFPGAStub->r();
	    int irvm1=innerFPGAStub->rvm().value();
	    //int izvm1=innerFPGAStub->zvm().value();
	    
	    int istubpt2=outerFPGAStub->stubpt().value();
	    int iphivm2=outerFPGAStub->phivm().value();
	    FPGAWord iphi2=outerFPGAStub->phi();
	    FPGAWord iz2=outerFPGAStub->z();
	    FPGAWord ir2=outerFPGAStub->r();
	    int irvm2=outerFPGAStub->rvm().value();
	    //int izvm2=outerFPGAStub->zvm().value();

	  

	    int ideltaphi=iphivm2-iphivm1;
	    int ideltar=irvm2-irvm1;
	    
	    if (ideltar<0) ideltar+=32;
	    assert(ideltar>=0);
	    if (ideltaphi<0) ideltaphi+=16;
	    assert(ideltaphi>=0);

	    //cout << "istubpt1 istubpt2 : "<<istubpt1<<" "<<istubpt2<<endl;
	    assert(istubpt1>=0);
	    assert(istubpt2>=0);
	    
	   
	    //int i1=TEOverlaps_[lindex][i].first.first;
	    //int i2=TEOverlaps_[lindex][i].second.first;
	    //int j1=TEOverlaps_[lindex][i].first.second;
	    //int j2=TEOverlaps_[lindex][i].second.second;
	    

	    //bool phimatch=(*TETablesOverlap_[lindex])[i].phicheck(address,i1,i2,j1,j2);

	    //bool zmatch=(*TETablesOverlap_[lindex])[i].zcheck(izvm1,izvm2,irvm1,irvm2*2); //the *2 is a hack since we need one more bit to match the irvm1 precision
       

	    double r1=innerStub->r();
	    double z1=innerStub->z();
	    double phi1=innerStub->phi();
	    
	    double r2=outerStub->r();
	    double z2=outerStub->z();
	    double phi2=outerStub->phi();
	    
	    //Protection... Should be handled cleaner
	    //to avoid problem with floating point 
	    //calculation and with overflows
	    //in the integer calculation
	    if (r1<r2+1.5) {
	      //cout << "in overlap tracklet: radii wrong"<<endl;
	      continue;
	    }


	    double rinv,phi0,t,z0;
	    
	    double phiproj[3],zproj[3],phider[3],zder[3];
	    double phiprojdisk[4],rprojdisk[4],phiderdisk[4],rderdisk[4];

	    exacttrackletOverlap(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
				 rinv,phi0,t,z0,
				 phiproj,zproj,phider,zder,
				 phiprojdisk,rprojdisk,phiderdisk,rderdisk);


	    //Truncates floating point positions to integer
	    //representation precision
	    if (1) {
	      int lphi=1;
	      int lz=1;
	      int lr=1;
	      //if (layer_>3) {
	      //  lphi=8;
	      //  lz=16;
	      //  lr=1;
	      //}
	      double dphi1=phi1-phimin_+(phimax_-phimin_)/6.0-iphi1.value()*kphi/lphi;
	      if (dphi1<-0.5*two_pi) dphi1+=two_pi;
	      //cout << "layer dphi1 phi1 : "<<layer_<<" "<<dphi1<<" "<<phi1<<endl;
	      assert(fabs(dphi1)<1e-4);
	      phi1-=dphi1;
	      int sign=1;
	      if (disk_<0) sign=-1;
	      double dz=z1-(iz1.value()*kzdisk*lz+sign*zmean[abs(disk_)-1]);
	      //cout << "z1 iz1.value() kz : "<<z1<<" "<<iz1.value()<<" "<<kz<<" "
	      //	 <<iz1.value()*kz<<" "<<sign*zmean[abs(disk_)-1]<<endl;
	      assert(fabs(dz)<1.0*lz);
	      z1-=dz;
	      //cout <<"r1 : "<<r1<<" "<<lr*ir1.value()*krdisk+rmindisk<<endl;
	      double dr=r1-lr*ir1.value()*krdisk-rmindisk;
	      assert(fabs(dr)<0.75);
	      r1-=dr;
	      //cout << "dz1 dr1 : "<<dz<<" "<<dr<<endl;
	    }



	    if (1) {
	      int lphi=1;
	      int lz=1;
	      int lr=2;
	      
	      double dphi2=phi2-phimin_+(phimax_-phimin_)/6.0-iphi2.value()*kphi/lphi;
	      if (dphi2<-0.5*two_pi) dphi2+=two_pi;
	      //cout << "layer dphi2 phi2 iphi2: "<<layer_<<" "<<dphi2<<" "
	      //	 <<phi2<<" "<< iphi2.value()<<endl;
	      assert(fabs(dphi2)<1e-4);
	      phi2-=dphi2;
	      //cout <<"z2 iz2: "<<z2<<" "<<iz2.value()*kz*lz<<endl;
	      double dz=z2-iz2.value()*kz*lz;
	      assert(fabs(dz)<1.0*lz);
	      z2-=dz;
	      int lindex=outerFPGAStub->layer().value();
	      //cout <<lindex<<" "<<r2<<" "<<2*ir2.value()*kr<<" "<<rmin[lindex]<<endl;
	      double dr=r2-lr*ir2.value()*kr-rmean[lindex]; 
	      assert(fabs(dr)<0.1);
	      r2-=dr;
	    }
	    

	    double rinvapprox,phi0approx,tapprox,z0approx;
	    double phiprojapprox[3],zprojapprox[3],phiderapprox[3],zderapprox[3];
	    double phiprojapproxdisk[4],rprojapproxdisk[4],
	      phiderapproxdisk[4],rderapproxdisk[4];
	  
	    approxtrackletoverlap(r1,z1,phi1,r2,z2,phi2,outerStub->sigmaz(),
				  rinvapprox,phi0approx,tapprox,z0approx,
				  phiprojapprox,zprojapprox,
				  phiderapprox,zderapprox,
				  phiprojapproxdisk,rprojapproxdisk,
				  phiderapproxdisk,rderapproxdisk);
	    
	    int irinv,iphi0,it,iz0;
	    int iphiproj[3],izproj[3],iphider[3],izder[3];
	    bool minusNeighbor[3],plusNeighbor[3];
	    
	    int iphiprojdisk[4],irprojdisk[4],iphiderdisk[4],irderdisk[4];
	    bool minusNeighbordisk[4],plusNeighbordisk[4];

	    //cout << "Will call binarytrackletoverlap : "<<disk_<<endl;

	    bool success=binarytrackletOverlap(innerFPGAStub,outerFPGAStub,
					       outerStub->sigmaz(),
					       irinv,iphi0,it,iz0,
					       iphiproj,izproj,
					       iphider,izder,
					       minusNeighbor,plusNeighbor,
					       iphiprojdisk,irprojdisk,
					       iphiderdisk,irderdisk,
					       minusNeighbordisk,
					       plusNeighbordisk);
	    
	  

	    if (!success) {
	      //cout << "binarytrackletoverlap failed"<<endl;
	      continue;
	    }

	    //cout << "Trying stub pair 3" << endl;
	  

	    for(unsigned int j=0;j<3;j++){
	      if (minusNeighbordisk[j]) {
		phiprojapproxdisk[j]+=dphisector;
		phiprojdisk[j]+=dphisector;
	      }
	      if (plusNeighbordisk[j]) {
		phiprojapproxdisk[j]-=dphisector;
		phiprojdisk[j]-=dphisector;
	      }	    
	    }

	    for(unsigned int j=0;j<3;j++){
	      if (minusNeighbor[j]) {
		phiprojapprox[j]+=dphisector;
		phiproj[j]+=dphisector;
	      }
	      if (plusNeighbor[j]) {
		phiprojapprox[j]-=dphisector;
		phiproj[j]-=dphisector;
	      }
	    }
	    
	  
	    //if (fabs(z0)>15.0) continue;
	    //if (fabs(rinv)>0.0057) continue;
	    
	    //if (!pass) continue;
	    

	    //if (!phimatch) { //Have to fix the fact that L1 and D1 are not offset
	    //cout << "Rejected due to phimatch "<<istubpt1<<" "
	    //	 <<istubpt2<<" "<<rinv<<" "<<t<<endl;
	    //continue;
	    //}
	    
	    //if (!zmatch) { 
	      //cout << "Rejected due to zmatch : "<<z0approx<<" "
	      //	 <<izvm1<<" "<<izvm2<<" "<<irvm1<<" "<<irvm2<<" "
	      // 	 <<z1<<" "<<z2<<" "<<r1<<" "<<r2<<endl;
	    //  continue;
	    //}

	    //limit on the number of processed matches in each TE
	    //++processed_matches;
	    //if ( processed_matches >= NMAXTE) {
	    //	    cout<<i<<" "<<processed_matches<<"\n";
	    //  continue; 
	    //}


	    if (writeTrackletParsOverlap) {
	      static ofstream out("trackletparsoverlap.txt");
	      out <<"Trackpars "<<disk_
		  <<"   "<<rinv<<" "<<rinvapprox<<" "<<irinv*krinvparsdisk
		  <<"   "<<phi0<<" "<<phi0approx<<" "<<iphi0*kphi0parsdisk
		  <<"   "<<t<<" "<<tapprox<<" "<<it*ktparsdisk
		  <<"   "<<z0<<" "<<z0approx<<" "<<iz0*kzdisk
		  <<endl;
	    }

	    if (writeTrackProj) {
	      static ofstream out1("trackproj.txt");
	      for (int i=0;i<3;i++) {
		double kphiproj=kphiproj123;
		out1 <<"Trackproj "<<disk_<<" "<<zproj_[i]
		     <<"   "<<phiprojdisk[i]<<" "<<phiprojapproxdisk[i]
		     <<" "<<iphiprojdisk[i]*kphiproj
		     <<"   "<<rprojdisk[i]<<" "<<rprojapproxdisk[i]
		     <<" "<<irprojdisk[i]*krprojshiftdisk
		     <<"   "<<phiderdisk[i]<<" "<<phiderapproxdisk[i]
		     <<" "<<iphiderdisk[i]*kphider
		     <<"   "<<rderdisk[i]<<" "<<rderapproxdisk[i]
		     <<" "<<irderdisk[i]*krprojderdiskshift
		     <<endl;
	      }
	      
	    }

	    //cout << "disktracklet "<<iphi0*kphi0parsdisk<<endl;
	    
	    //cout << "Found new FPGATracklet : " 
	    //     <<iphider[0]<<" "<<phider[0]<<" "<<phiderapprox[0]
	    //     <<" <> "<<irinv<<" "<<rinv<<" "<<rinvapprox<<endl;
	    
	    //++NtrackletsOverlap_;
	    //if(NtrackletsOverlap_ < NMAXproj){
	      
	    FPGATracklet* tracklet=new FPGATracklet(innerStub,outerStub,
						    innerFPGAStub,outerFPGAStub,
						    phioffset_,
						    rinv,phi0,z0,t,
						    rinvapprox,phi0approx,
						    z0approx,tapprox,
						    irinv,iphi0,iz0,it,
						    iphiproj,izproj,iphider,izder,
						    minusNeighbor,plusNeighbor,		
						    phiproj,zproj,phider,zder,
						    phiprojapprox,zprojapprox,
						    phiderapprox,zderapprox,
						    iphiprojdisk,irprojdisk,
						    iphiderdisk,irderdisk,
						    minusNeighbordisk,
						    plusNeighbordisk,
						    phiprojdisk,rprojdisk,
						    phiderdisk,rderdisk,
						    phiprojapproxdisk,
						    rprojapproxdisk,
						    phiderapproxdisk,
						    rderapproxdisk,
						    false,true);

	    //cout << "Found tracklet in overlap = "<<layer_<<" "<<disk_
	    //	 <<" "<<tracklet<<" "<<iSector_<<endl;


	    //cout << "Adding overlap tracklet" << endl;

	    countsel++;

	    trackletpars_->addTracklet(tracklet);
	    //FIXME  need to stick projection in correct place

	    int layer=outerFPGAStub->layer().value()+1;

	    if (layer==2) {
	      addLayerProj(tracklet,1);
	    }

	  
	    for(unsigned int disk=2;disk<6;disk++){
	      addDiskProj(tracklet,disk);
	    }


	  }
	}
	if (countall>=NMAXTC) break;
      }
      if (countall>=NMAXTC) break;
    }

    if (writeTrackletCalculator) {
      static ofstream out("trackletcalculator.txt");
      out << getName()<<" "<<countall<<" "<<countsel<<endl;
    }


  }



  void exacttrackletOverlap(double r1, double z1, double phi1,
			    double r2, double z2, double phi2, double sigmaz,
			    double& rinv, double& phi0,
			    double& t, double& z0,
			    double phiprojLayer[3], double zprojLayer[3], 
			    double phiderLayer[3], double zderLayer[3],
			    double phiproj[3], double rproj[3], 
			    double phider[3], double rder[3]) {

    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }

    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
    
    rinv=2*sin(deltaphi)/dist;

    if (r1>r2) rinv=-rinv;

    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;    

    //cout << "phi1 phi2 phi1tmp : "<<phi1<<" "<<phi2<<" "<<phi1tmp<<endl;

    phi0=phi1tmp+asin(0.5*r1*rinv);
    
    if (phi0>0.5*two_pi) phi0-=two_pi;
    if (phi0<-0.5*two_pi) phi0+=two_pi;
    if (!(fabs(phi0)<0.5*two_pi)) {
      cout << "phi1tmp r1 rinv phi0 deltaphi dist: "
	   <<phi1tmp<<" "<<r1<<" "<<rinv<<" "<<phi0
	   <<" "<<deltaphi<<" "<<dist<<endl;
      exit(1);
    }
    
    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
    t=(z1-z2)/(rhopsi1-rhopsi2);
    
    z0=z1-t*rhopsi1;


    if (disk_==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "DUMPPARS0:" 
	     <<" dz= "<<z2-z1
	     <<" rinv= "<<rinv
	     <<" phi0= "<<phi0
	     <<" t= "<<t
	     <<" z0= "<<z0
	     <<endl;
      }
    }


    for (int i=0;i<4;i++) {
      exactprojdisk(zprojoverlap_[i],rinv,phi0,t,z0,
		    phiproj[i],rproj[i],
		    phider[i],rder[i]);
    }


    for (int i=0;i<1;i++) {
      exactproj(rmean[i],rinv,phi0,t,z0,
		    phiprojLayer[i],zprojLayer[i],
		    phiderLayer[i],zderLayer[i]);
    }


  }


  void approxtrackletoverlap(double r1, double z1, double phi1,
			     double r2, double z2, double phi2, double sigmaz,
			     double &rinv, double &phi0,
			     double &t, double &z0,
			     double phiprojLayer[4], double rprojLayer[4], 
			     double phiderLayer[4], double rderLayer[4],
			     double phiproj[4], double rproj[4], 
			     double phider[4], double rder[4]) {

    if (sigmaz<-10.0) {
      cout << "Negative sigmaz"<<endl;
    }


    double deltaphi=phi1-phi2;

    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
    assert(fabs(deltaphi)<0.5*two_pi);

    if (phi1<0.0) phi1+=two_pi;
    double phi1tmp=phi1-phimin_+(phimax_-phimin_)/6.0;
    if (phi1tmp>two_pi) phi1-=two_pi;
    //cout << "phi1 phimin_ phimax_:"<<phi1<<" "<<phimin_
    //	 <<" "<<phimax_<<" "<<phi1tmp<<endl;
    assert(phi1tmp>-1e-10);

    //cout << "DUMPPARS01 : "<<r1<<" "<<r2<<endl;

    double dr=r2-r1;
    double dz=z1-z2;
    double drinv=1.0/dr;
    double t2=deltaphi*drinv;
    double delta=0.5*r1*r2*t2*t2;// *(1+deltaphi*deltaphi/12.0);
    double t5=1.0-delta+1.5*delta*delta;// -2.5*delta*delta*delta;
    double deltainv=t5*drinv;
    rinv=2.0*deltaphi*deltainv;// *(1-deltaphi*deltaphi/6.0);
    t=-dz*deltainv;// *(1.0-deltaphi*deltaphi/6.0); 
    double t7=0.5*r1*rinv;
    double t9=1+t7*t7/6.0;// +3.0*t7*t7*t7*t7/40.0;
    phi0=phi1tmp+t7*t9;
    double t12=t*r1*t9;
    z0=z1-t12;


    if (abs(disk_)==1) {
      if (dumppars) {
	cout << "------------------------------------------------"<<endl;
	cout << "DUMPPARS1:"  //overlap 
	     << -deltaphi
	     <<" "<<z2-z1
	     <<" "<<r2-r1
	     <<" > "<<r1
	     <<" "<<r2
	     <<" "<<1.0/(r2-r1)
	     <<" "<<delta
	     <<" "<<t5
	     <<" "<<deltainv
	     <<" "<<rinv
	     <<" t= "<<t
	     <<" | "<<r1
	     <<" "<<t7
	     <<" | "<<t9
	     <<" "<<phi1tmp
	     <<" phi0= "<<phi0
	     <<" * "<<t12
	     <<" "<<z1
	     <<" z0= "<<z0
	     <<endl;
      }
      

//      cout << "Approx tracklet: dphi="<<-deltaphi<<" dz="<<z2-z1
//	   << " dr="<<r2-r1<<" drinv="<<1.0/(r2-r1)
//	   <<" delta="<<delta
//	   <<" t5="<<t5
//           <<" deltainv="<<deltainv
//	   <<" rinv="<<rinv
//	   <<" t="<<t
//	   <<" r1abs="<<r1
//	   <<" t7="<<t7
//	   <<" t9="<<t9
//	   <<" phi1="<<phi1tmp
//	   <<" ***phi0="<<phi0
//	   <<" t12="<<t12
//	   <<" z1="<<z1
//	   <<" z0="<<z0
//	   <<endl;

   

    }
   

    //calculate projection



    for (int i=0;i<1;i++) {
      approxproj(rmean[i],rinv,phi0,t,z0,
		 phiprojLayer[i],rprojLayer[i],
		 phiderLayer[i],rderLayer[i]);
    }


    //static ofstream out1("neighborproj.txt");

    for (int i=0;i<4;i++) {
      approxprojdisk(zprojoverlap_[i],rinv,phi0,t,z0,
		     phiproj[i],rproj[i],phider[i],rder[i]);
      /*
      if (fabs(z0)<15.0&&fabs(rinv)<0.0057) {
	if ((fabs(rproj[i])<100.0)&&(phiproj[i]<(phimax_-phimin_)/6)){
	  out1<<disk_<<" -1 "<<phiproj[i]<<endl;
	} else  if ((fabs(rproj[i])<100.0)&&(phiproj[i]>7.0*(phimax_-phimin_)/6)){
	  out1<<disk_<<" +1 "<<phiproj[i]<<endl;
	} else if (fabs(rproj[i])<100.0){
	  out1<<disk_<<" 0 "<<phiproj[i]<<endl;
	}
      }
      */
    }


  }



  void addDiskProj(FPGATracklet* tracklet, int disk){

    //cout << "Trying to add projection to disk = "<<disk<<endl;

    FPGAWord fpgar=tracklet->fpgarprojdisk(disk);

    
    if (fpgar.value()*krprojshiftdisk<rmindisk) return;
    if (fpgar.value()*krprojshiftdisk>rmaxdisk) return;
	    
    //FIXME should not use the floats...
    int ir=2*(fpgar.value()*krprojshiftdisk-rmindisk)/(rmaxdisk-rmindisk)+1;

    //cout << "irproj ir : "<<fpgar.value()<<" "
    //	 <<fpgar.value()<<" "<<ir<<endl;
    
    assert(ir>0);
    assert(ir<=2);

    if (tracklet->plusNeighborDisk(disk)) {
      //cout << "Plus projection"<<endl;
      if (disk<0) disk=-disk; //hack...
      if (disk==1) {
	static bool firstD1=true;
	if (trackletproj_D1Plus_==0) {
	  if (firstD1) cout << "In "<<getName()<<" projection to disk 1 plus not used"<<endl;
	  firstD1=false;
	  return;
	}
	trackletproj_D1Plus_->addProj(tracklet);
      }
      if (disk==2) {
	static bool firstD2=true;
	if (trackletproj_D2Plus_==0) {
	  if (firstD2) cout << "In "<<getName()<<" projection to disk 2 plus not used"<<endl;
	  firstD2=false;
	  return;
	}
	//cout << "Added to D2Plus" <<endl;
	trackletproj_D2Plus_->addProj(tracklet);
      }
      if (disk==3) {
	static bool firstD3=true;
	if (trackletproj_D3Plus_==0) {
	  if (firstD3) cout << "In "<<getName()<<" projection to disk 3 plus not used"<<endl;
	  firstD3=false;
	  return;
	}
	//cout << "Added to D3Plus" <<endl;
	trackletproj_D3Plus_->addProj(tracklet);
      }
      if (disk==4) {
	static bool firstD4=true;
	if (trackletproj_D4Plus_==0) {
	  if (firstD4) cout << "In "<<getName()<<" projection to disk 4 plus not used"<<endl;
	  firstD4=false;
	  return;
	}
	//cout << "Added to D4Plus" <<endl;
	trackletproj_D4Plus_->addProj(tracklet);
      }
      if (disk==5) {
	static bool firstD5=true;
	if (trackletproj_D5Plus_==0) {
	  if (firstD5) cout << "In "<<getName()<<" projection to disk 5 plus not used"<<endl;
	  firstD5=false;
	  return;
	}
	//cout << "Added to D5Plus" <<endl;
	trackletproj_D5Plus_->addProj(tracklet);
      }
      return;
    }

    if (tracklet->minusNeighborDisk(disk)) {
      //cout << "Minus projection"<<endl;
      if (disk<0) disk=-disk; //hack...
      if (disk==1) {
	static bool firstD1=true;
	if (trackletproj_D1Minus_==0) {
	  if (firstD1) cout << "In "<<getName()<<" projection to disk 1 minus not used"<<endl;
	  firstD1=false;
	  return;
	}
	trackletproj_D1Minus_->addProj(tracklet);
      }
      if (disk==2) {
	static bool firstD2=true;
	if (trackletproj_D2Minus_==0) {
	  if (firstD2) cout << "In "<<getName()<<" projection to disk 2 minus not used"<<endl;
	  firstD2=false;
	  return;
	}
	//cout << "Added to D2Minus" <<endl;
	trackletproj_D2Minus_->addProj(tracklet);
      }
      if (disk==3) {
	static bool firstD3=true;
	if (trackletproj_D3Minus_==0) {
	  if (firstD3) cout << "In "<<getName()<<" projection to disk 3 minus not used"<<endl;
	  firstD3=false;
	  return;
	}
	//cout << "Added to D3Minus" <<endl;
	trackletproj_D3Minus_->addProj(tracklet);
      }
      if (disk==4) {
	static bool firstD4=true;
	if (trackletproj_D4Minus_==0) {
	  if (firstD4) cout << "In "<<getName()<<" projection to disk 4 minus not used"<<endl;
	  firstD4=false;
	  return;
	}
	//cout << "Added to D4Minus" <<endl;
	trackletproj_D4Minus_->addProj(tracklet);
      }
      if (disk==5) {
	static bool firstD5=true;
	if (trackletproj_D5Minus_==0) {
	  if (firstD5) cout << "In "<<getName()<<" projection to disk 5 minus not used"<<endl;
	  firstD5=false;
	  return;
	}
	//cout << "Added to D5Minus" <<endl;
	trackletproj_D5Minus_->addProj(tracklet);
      }
      return;
    }

    //cout << "Projections to same sector disk = "<<disk<<" ir = "<<ir<<endl;

    if (disk<0) disk=-disk; //hack...

    if (disk==1&&ir==1) {
      static bool first=true;
      if (trackletproj_D1Di_==0) {
	if (first) cout << "In "<<getName()<<" projection to inner disk 1 not used"<<endl;
	first=false;
	return;
      }
      trackletproj_D1Di_->addProj(tracklet);
      return;
    }

    if (disk==1&&ir==2) {
      static bool first=true;
      if (trackletproj_D1Do_==0) {
	if (first) cout << "In "<<getName()<<" projection to outer disk 1 not used"<<endl;
	first=false;
	return;
      }
      trackletproj_D1Do_->addProj(tracklet);
      return;
    }


    if (disk==2&&ir==1) {
      static bool first=true;
      if (trackletproj_D2Di_==0) {
	if (first) cout << "In "<<getName()<<" projection to inner disk 2 not used"<<endl;
	first=false;
	return;
      }
      trackletproj_D2Di_->addProj(tracklet);
      return;
    }

    if (disk==2&&ir==2) {
      static bool first=true;
      if (trackletproj_D2Do_==0) {
	if (first) cout << "In "<<getName()<<" projection to outer disk 2 not used"<<endl;
	first=false;
	return;
      }
      trackletproj_D2Do_->addProj(tracklet);
      return;
    }

    
    if (disk==3) {
      static bool first=true;
      if (trackletproj_D3Di_==0||trackletproj_D3Do_==0) {
	if (first) cout << "In "<<getName()<<" projection to disk 3 not used"<<endl;
	first=false;
	return;
      }
      if (ir==1) {
	//cout << "Adding projection in D3Di to "<<trackletproj_D3Di_->getName()<<endl;
	trackletproj_D3Di_->addProj(tracklet);
      }
      if (ir==2) {
	//cout << "Adding projection in D3Do to "<<trackletproj_D3Do_->getName()<<endl;
	trackletproj_D3Do_->addProj(tracklet);
      }
      return;
    }
    
    if (disk==4) {
      static bool first=true;
      if (trackletproj_D4Di_==0||trackletproj_D4Do_==0) {
	if (first) cout << "In "<<getName()<<" projection to disk 4 not used"<<endl;
	first=false;
	return;
      }
      if (ir==1) {
	//cout << "Adding projection in D4Di to "<<trackletproj_D4Di_->getName()<<endl;	
	trackletproj_D4Di_->addProj(tracklet);
      }
      if (ir==2) {
	//cout << "Adding projection in D4Do to "<<trackletproj_D4Do_->getName()<<endl;	
	trackletproj_D4Do_->addProj(tracklet);
      }
      return;
    }
    
    if (disk==5) {
      static bool first=true;
      if (trackletproj_D5Di_==0||trackletproj_D5Do_==0) {
	if (first) cout << "In "<<getName()<<" projection to disk 5 not used"<<endl;
	first=false;
	return;
      }
      if (ir==1) {
	//cout << "Adding projection in D5Di to "<<trackletproj_D5Di_->getName()<<endl;	
	trackletproj_D5Di_->addProj(tracklet);
      }
      if (ir==2) {
	//cout << "Adding projection in D5D0 to "<<trackletproj_D5Do_->getName()<<endl;	
	trackletproj_D5Do_->addProj(tracklet);
      }
      return;
    }

  }


  void addLayerProj(FPGATracklet* tracklet, int layer){

    assert(layer>0);

    FPGAWord fpgaz=tracklet->fpgazproj(layer);

    if (fpgaz.atExtreme()) return;

    if (fabs(fpgaz.value()*kz)>zlength) return;

    int iz=4+(fpgaz.value()>>(fpgaz.nbits()-3));
    iz=iz/2+1;

    assert(iz>0);
    assert(iz<=4);

    //cout << "layer_ iz "<<layer_<<" "<<iz<<" "<<getName()<<endl;

    //This will protect to not fill the projections
    //to neighboring sectors if we work with a subset
    //of the detector. Is this fully safe?
    if (layer_==1) {
      if (!((iz==1&&trackletproj_L3D1_!=0)||
	    (iz==2&&trackletproj_L3D2_!=0)|| 
	    (iz==3&&trackletproj_L3D3_!=0)|| 
	    (iz==4&&trackletproj_L3D4_!=0))) {
	return;
      } 
    }
    if (layer_==3) {
      if (!((iz==1&&trackletproj_L1D1_!=0)||
	    (iz==2&&trackletproj_L1D2_!=0)|| 
	    (iz==3&&trackletproj_L1D3_!=0)|| 
	    (iz==4&&trackletproj_L1D4_!=0))) {
	return;
      } 
    }
    if (layer_==5) {
      if (!((iz==1&&trackletproj_L1D1_!=0)||
	    (iz==2&&trackletproj_L1D2_!=0)|| 
	    (iz==3&&trackletproj_L1D3_!=0)|| 
	    (iz==4&&trackletproj_L1D4_!=0))) {
	return;
      } 
    }



    if (tracklet->plusNeighbor(layer)) {
      if (layer==1) {
	static bool firstL1=true;
	if (trackletproj_L1Plus_==0) {
	  if (firstL1) cout << "In "<<getName()<<" projection to L1 plus not used"<<endl;
	  firstL1=false;
	  return;
	}
	trackletproj_L1Plus_->addProj(tracklet);
      }
      if (layer==2) {
	static bool firstL2=true;
	if (trackletproj_L2Plus_==0) {
	  if (firstL2) cout << "In "<<getName()<<" projection to L2 plus not used"<<endl;
	  firstL2=false;
	  return;
	}
	trackletproj_L2Plus_->addProj(tracklet);
      }
      if (layer==3) trackletproj_L3Plus_->addProj(tracklet);
      if (layer==4) trackletproj_L4Plus_->addProj(tracklet);
      if (layer==5) trackletproj_L5Plus_->addProj(tracklet);
      if (layer==6) trackletproj_L6Plus_->addProj(tracklet);
      return;
    }

    if (tracklet->minusNeighbor(layer)) {
      if (layer==1) {
	static bool firstL1=true;
	if (trackletproj_L1Minus_==0) {
	  if (firstL1) cout << "In "<<getName()<<" projection to L1 minus not used"<<endl;
	  firstL1=false;
	  return;
	}
        trackletproj_L1Minus_->addProj(tracklet);
      }
      if (layer==2) {
	static bool firstL2=true;
	if (trackletproj_L2Minus_==0) {
	  if (firstL2) cout << "In "<<getName()<<" projection to L2 minus not used"<<endl;
	  firstL2=false;
	  return;
	}
	trackletproj_L2Minus_->addProj(tracklet);
      }
      if (layer==3) trackletproj_L3Minus_->addProj(tracklet);
      if (layer==4) trackletproj_L4Minus_->addProj(tracklet);
      if (layer==5) trackletproj_L5Minus_->addProj(tracklet);
      if (layer==6) trackletproj_L6Minus_->addProj(tracklet);
      return;
    }

    if (layer==1) {

      static bool firstL1D1=true;
      if (iz==1&&trackletproj_L1D1_==0) {
	if (firstL1D1) cout << "In "<<getName()<<" projection to L1D1 not used"<<endl;
	firstL1D1=false;
	return;
      }
      if (iz==1) {
	//cout << "Adding L1D1 projection " <<trackletproj_L1D1_->getName()<<endl;
	trackletproj_L1D1_->addProj(tracklet);
      }
      static bool firstL1D2=true;
      if (iz==2&&trackletproj_L1D2_==0) {
	if (firstL1D2) cout << "In "<<getName()<<" projection to L1D2 not used"<<endl;
	firstL1D2=false;
	return;
      }
      if (iz==2) trackletproj_L1D2_->addProj(tracklet);
      static bool firstL1D3=true;
      if (iz==3&&trackletproj_L1D3_==0) {
	if (firstL1D3) cout << "In "<<getName()<<" projection to L1D3 not used"<<endl;
	firstL1D3=false;
	return;
      }
      if (iz==3) trackletproj_L1D3_->addProj(tracklet);
      static bool firstL1D4=true;
      if (iz==4&&trackletproj_L1D4_==0) {
	if (firstL1D4) cout << "In "<<getName()<<" projection to L1D4 not used"<<endl;
	firstL1D4=false;
	return;
      }
      if (iz==4) trackletproj_L1D4_->addProj(tracklet);
      return;
    }

    if (layer==2) {
      static bool firstL2D1=true;
      if (iz==1&&trackletproj_L2D1_==0) {
	if (firstL2D1) cout << "In "<<getName()<<" projection to L2D1 not used"<<endl;
	firstL2D1=false;
	return;
      }
      if (iz==1) trackletproj_L2D1_->addProj(tracklet);
      static bool firstL2D2=true;
      if (iz==2&&trackletproj_L2D2_==0) {
	if (firstL2D2) cout << "In "<<getName()<<" projection to L2D2 not used"<<endl;
	firstL2D2=false;
	return;
      }
      if (iz==2) trackletproj_L2D2_->addProj(tracklet);
      static bool firstL2D3=true;
      if (iz==3&&trackletproj_L2D3_==0) {
	if (firstL2D3) cout << "In "<<getName()<<" projection to L2D3 not used"<<endl;
	firstL2D3=false;
	return;
      }
      if (iz==3) trackletproj_L2D3_->addProj(tracklet);
      static bool firstL2D4=true;
      if (iz==4&&trackletproj_L2D4_==0) {
	if (firstL2D4) cout << "In "<<getName()<<" projection to L2D4 not used"<<endl;
	firstL2D4=false;
	return;
      }
      if (iz==4) trackletproj_L2D4_->addProj(tracklet);
      return;
    }

    if (layer==3) {
      //cout << "layer==3 iz = "<<iz
      //	   <<" "<<trackletproj_L3D1_->getName()
      //	   <<" "<<trackletproj_L3D2_->getName()
      //   <<" "<<trackletproj_L3D3_->getName()
      //   <<" "<<trackletproj_L3D4_->getName()
      //   <<endl;

      if (iz==1&&trackletproj_L3D1_!=0) trackletproj_L3D1_->addProj(tracklet);
      if (iz==2&&trackletproj_L3D2_!=0) trackletproj_L3D2_->addProj(tracklet);
      if (iz==3&&trackletproj_L3D3_!=0) trackletproj_L3D3_->addProj(tracklet);
      if (iz==4&&trackletproj_L3D4_!=0) trackletproj_L3D4_->addProj(tracklet);
      return;
    }

    if (layer==4) {
      if (iz==1&&trackletproj_L4D1_!=0) trackletproj_L4D1_->addProj(tracklet);
      if (iz==2&&trackletproj_L4D2_!=0) trackletproj_L4D2_->addProj(tracklet);
      if (iz==3&&trackletproj_L4D3_!=0) trackletproj_L4D3_->addProj(tracklet);
      if (iz==4&&trackletproj_L4D4_!=0) trackletproj_L4D4_->addProj(tracklet);
      return;
    }

    if (layer==5) {
      if (iz==1&&trackletproj_L5D1_!=0) trackletproj_L5D1_->addProj(tracklet);
      if (iz==2&&trackletproj_L5D2_!=0) trackletproj_L5D2_->addProj(tracklet);
      if (iz==3&&trackletproj_L5D3_!=0) trackletproj_L5D3_->addProj(tracklet);
      if (iz==4&&trackletproj_L5D4_!=0) trackletproj_L5D4_->addProj(tracklet);
      return;
    }

    if (layer==6) {
      if (iz==1&&trackletproj_L6D1_!=0) trackletproj_L6D1_->addProj(tracklet);
      if (iz==2&&trackletproj_L6D2_!=0) trackletproj_L6D2_->addProj(tracklet);
      if (iz==3&&trackletproj_L6D3_!=0) trackletproj_L6D3_->addProj(tracklet);
      if (iz==4&&trackletproj_L6D4_!=0) trackletproj_L6D4_->addProj(tracklet);
      return;
    }



  }


  int round_int( double r ) {
    return (r > 0.0) ? (r + 0.5) : (r - 0.5); 
  }
 
    
private:

  int layer_;
  int disk_;
  double phimin_;
  double phimax_;
  double phioffset_;
  double rproj_[4];
  double zproj_[3];
  double zprojoverlap_[4];

  FPGAAllStubs* innerallstubs_;
  FPGAAllStubs* outerallstubs_;
  vector<FPGAStubPairs*> stubpairs_;

  FPGAInverseTable invTable_;

  FPGATrackletParameters* trackletpars_;

  FPGATrackletProjections* trackletproj_L1D1_;
  FPGATrackletProjections* trackletproj_L1D2_;
  FPGATrackletProjections* trackletproj_L1D3_;
  FPGATrackletProjections* trackletproj_L1D4_;
  FPGATrackletProjections* trackletproj_L1Minus_;
  FPGATrackletProjections* trackletproj_L1Plus_;

  FPGATrackletProjections* trackletproj_L2D1_;
  FPGATrackletProjections* trackletproj_L2D2_;
  FPGATrackletProjections* trackletproj_L2D3_;
  FPGATrackletProjections* trackletproj_L2D4_;
  FPGATrackletProjections* trackletproj_L2Minus_;
  FPGATrackletProjections* trackletproj_L2Plus_;

  FPGATrackletProjections* trackletproj_L3D1_;
  FPGATrackletProjections* trackletproj_L3D2_;
  FPGATrackletProjections* trackletproj_L3D3_;
  FPGATrackletProjections* trackletproj_L3D4_;
  FPGATrackletProjections* trackletproj_L3Minus_;
  FPGATrackletProjections* trackletproj_L3Plus_;

  FPGATrackletProjections* trackletproj_L4D1_;
  FPGATrackletProjections* trackletproj_L4D2_;
  FPGATrackletProjections* trackletproj_L4D3_;
  FPGATrackletProjections* trackletproj_L4D4_;
  FPGATrackletProjections* trackletproj_L4Minus_;
  FPGATrackletProjections* trackletproj_L4Plus_;

  FPGATrackletProjections* trackletproj_L5D1_;
  FPGATrackletProjections* trackletproj_L5D2_;
  FPGATrackletProjections* trackletproj_L5D3_;
  FPGATrackletProjections* trackletproj_L5D4_;
  FPGATrackletProjections* trackletproj_L5Minus_;
  FPGATrackletProjections* trackletproj_L5Plus_;

  FPGATrackletProjections* trackletproj_L6D1_;
  FPGATrackletProjections* trackletproj_L6D2_;
  FPGATrackletProjections* trackletproj_L6D3_;
  FPGATrackletProjections* trackletproj_L6D4_;
  FPGATrackletProjections* trackletproj_L6Minus_;
  FPGATrackletProjections* trackletproj_L6Plus_;

  FPGATrackletProjections* trackletproj_D1Di_;
  FPGATrackletProjections* trackletproj_D1Do_;
  FPGATrackletProjections* trackletproj_D1Minus_;
  FPGATrackletProjections* trackletproj_D1Plus_;

  FPGATrackletProjections* trackletproj_D2Di_;
  FPGATrackletProjections* trackletproj_D2Do_;
  FPGATrackletProjections* trackletproj_D2Minus_;
  FPGATrackletProjections* trackletproj_D2Plus_;
  
  FPGATrackletProjections* trackletproj_D3Di_;
  FPGATrackletProjections* trackletproj_D3Do_;
  FPGATrackletProjections* trackletproj_D3Minus_;
  FPGATrackletProjections* trackletproj_D3Plus_;

  FPGATrackletProjections* trackletproj_D4Di_;
  FPGATrackletProjections* trackletproj_D4Do_;
  FPGATrackletProjections* trackletproj_D4Minus_;
  FPGATrackletProjections* trackletproj_D4Plus_;

  FPGATrackletProjections* trackletproj_D5Di_;
  FPGATrackletProjections* trackletproj_D5Do_;
  FPGATrackletProjections* trackletproj_D5Minus_;
  FPGATrackletProjections* trackletproj_D5Plus_;





  
};

#endif
