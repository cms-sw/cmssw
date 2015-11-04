#ifndef FPGATRACKLET_H
#define FPGATRACKLET_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
#include "L1TStub.hh"
#include "FPGAStub.hh"
#include "FPGAWord.hh"
#include "FPGATrack.hh"

using namespace std;

class FPGATracklet{

public:

  FPGATracklet(L1TStub* innerStub, L1TStub* outerStub,
	       FPGAStub* innerFPGAStub, FPGAStub* outerFPGAStub,
	       double phioffset,
	       double rinv, double phi0, double z0, double t,
	       double rinvapprox, double phi0approx, 
	       double z0approx, double tapprox,
	       int irinv, int iphi0, 
	       int iz0, int it,
	       int iphiproj[4], int izproj[4],
	       int iphider[4], int izder[4],
	       bool minusNeighbor[4], bool plusNeighbor[4],
	       double phiproj[4], double zproj[4],
	       double phider[4], double zder[4],
	       double phiprojapprox[4], double zprojapprox[4],
	       double phiderapprox[4], double zderapprox[4],
	       int iphiprojDisk[5], int irprojDisk[5],
	       int iphiderDisk[5], int irderDisk[5],
	       bool minusNeighborDisk[5], bool plusNeighborDisk[5],
	       double phiprojDisk[5], double rprojDisk[5],
	       double phiderDisk[5], double rderDisk[5],
	       double phiprojapproxDisk[5], double rprojapproxDisk[5],
	       double phiderapproxDisk[5], double rderapproxDisk[5],
	       bool disk, bool overlap=false
	       ){

    //static int count=0;
    //count++;
    //if (count%100==0) {
    //  cout << "Creating FPGATracklet : "<<count<<endl;
    //}

    overlap_=overlap;
    disk_=disk;
    assert(!(disk&&overlap));
    barrel_=(!disk)&&(!overlap);

    phioffset_=phioffset;

    assert(disk_||barrel_||overlap_);
  
    innerStub_=innerStub;
    outerStub_=outerStub;
    innerFPGAStub_=innerFPGAStub;
    outerFPGAStub_=outerFPGAStub;
    rinv_=rinv;
    phi0_=phi0;
    z0_=z0;
    t_=t;
    rinvapprox_=rinvapprox;
    phi0approx_=phi0approx;
    z0approx_=z0approx;
    tapprox_=tapprox;

    fpgarinv_.set(irinv,nbitsrinv,false,__LINE__,__FILE__); 
    fpgaphi0_.set(iphi0,nbitsphi0,false,__LINE__,__FILE__); 
    fpgaz0_.set(iz0,nbitsz0,false,__LINE__,__FILE__);
    fpgat_.set(it,nbitst,false,__LINE__,__FILE__);       

    assert(innerStub_->layer()<6||innerStub_->disk()<5);
    assert(outerStub_->layer()<6||outerStub_->disk()<5);

    for (unsigned int i=0;i<4;i++){
      fpgastubid_[i].set(63,9);  //FIXME dummy values...
		fpgastubiddisk_[i].set(63,9);  //FIXME dummy values...
    }

    if (innerStub_->layer()==0) {
      rproj_[0]=rmeanL3;
      rproj_[1]=rmeanL4;
      rproj_[2]=rmeanL5;
      rproj_[3]=rmeanL6;
      projlayer_[0]=3;
      projlayer_[1]=4;
      projlayer_[2]=5;
      projlayer_[3]=6;
    }
    
    if (innerStub_->layer()==2) {
      rproj_[0]=rmeanL1;
      rproj_[1]=rmeanL2;
      rproj_[2]=rmeanL5;
      rproj_[3]=rmeanL6;
      projlayer_[0]=1;
      projlayer_[1]=2;
      projlayer_[2]=5;
      projlayer_[3]=6;
    }

    if (innerStub_->layer()==4) {
      rproj_[0]=rmeanL1;
      rproj_[1]=rmeanL2;
      rproj_[2]=rmeanL3;
      rproj_[3]=rmeanL4;
      projlayer_[0]=1;
      projlayer_[1]=2;
      projlayer_[2]=3;
      projlayer_[3]=4;
    }

    if (innerStub_->layer()>999) {
      
      assert((innerStub->disk()==1)||(innerStub->disk()==3)||
	     (innerStub->disk()==-1)||(innerStub->disk()==-3));

      rproj_[0]=rmeanL1;
      rproj_[1]=rmeanL2;
      projlayer_[0]=1;
      projlayer_[1]=2;
      
      if (overlap_) {
	assert((innerStub->disk()==1)||(innerStub->disk()==-1));
	if (innerStub->disk()==1) {
	  zprojdisk_[0]=zmeanD1;
	  zprojdisk_[1]=zmeanD2;
	  zprojdisk_[2]=zmeanD3;
	  zprojdisk_[3]=zmeanD4;
	  zprojdisk_[4]=zmeanD5;
	} else {
	  zprojdisk_[0]=-zmeanD1;
	  zprojdisk_[1]=-zmeanD2;
	  zprojdisk_[2]=-zmeanD3;
	  zprojdisk_[3]=-zmeanD4;
	  zprojdisk_[4]=-zmeanD5;
	}
	projdisk_[0]=1;
	projdisk_[1]=2;
	projdisk_[2]=3;
	projdisk_[3]=4;
	projdisk_[4]=5;
      } else {
	
	if (innerStub->disk()==1) {
	  zprojdisk_[0]=zmeanD3;
	  zprojdisk_[1]=zmeanD4;
	  zprojdisk_[2]=zmeanD5;
	  projdisk_[0]=3;
	  projdisk_[1]=4;
	  projdisk_[2]=5;
	}
	if (innerStub->disk()==3) {
	  zprojdisk_[0]=zmeanD1;
	  zprojdisk_[1]=zmeanD2;
	  zprojdisk_[2]=zmeanD5;
	  projdisk_[0]=1;
	  projdisk_[1]=2;
	  projdisk_[2]=5;
	}
	
	if (innerStub->disk()==-1) {
	  zprojdisk_[0]=-zmeanD3;
	  zprojdisk_[1]=-zmeanD4;
	  zprojdisk_[2]=-zmeanD5;
	  projdisk_[0]=-3;
	  projdisk_[1]=-4;
	  projdisk_[2]=-5;
	}
	if (innerStub->disk()==-3) {
	  zprojdisk_[0]=-zmeanD1;
	  zprojdisk_[1]=-zmeanD2;
	  zprojdisk_[2]=-zmeanD5;
	  projdisk_[0]=-1;
	  projdisk_[1]=-2;
	  projdisk_[2]=-5;
	}	
      }

    } else {
      
      int sign=1;
      if (it<0) sign=-1;

      zprojdisk_[0]=sign*zmeanD1;
      zprojdisk_[1]=sign*zmeanD2;
      zprojdisk_[2]=sign*zmeanD3;
      zprojdisk_[3]=sign*zmeanD4;
      zprojdisk_[4]=sign*zmeanD5;
      projdisk_[0]=sign*1;
      projdisk_[1]=sign*2;
      projdisk_[2]=sign*3;
      projdisk_[3]=sign*4;
      projdisk_[4]=sign*5;

    }

    std::pair<FPGAStub*, L1TStub*> zeropair;
    zeropair.first=0;
    zeropair.second=0;

    for(int i=0;i<4;i++) {
      alphadisk_[i]=0.0;
      phiresid_[i]=1e30;
      zresid_[i]=1e30;
      phiresidapprox_[i]=1e30;
      zresidapprox_[i]=1e30;
      fpgaphiresid_[i].set((1<<(phiresidbits-1))-1,phiresidbits,false);
      assert(fpgaphiresid_[i].atExtreme());
      fpgazresid_[i].set((1<<(zresidbits-1))-1,zresidbits,false);
      stubptrs_[i]=zeropair;
    }
    

    for(int i=0;i<5;i++) {
      phiresiddisk_[i]=1e30;
      rresiddisk_[i]=1e30;
      phiresidapproxdisk_[i]=1e30;
      rresidapproxdisk_[i]=1e30;
      fpgaphiresiddisk_[i].set((1<<(phiresidbits-1))-1,phiresidbits,false);
      assert(fpgaphiresiddisk_[i].atExtreme());
      fpgarresiddisk_[i].set((1<<(rresidbits-1))-1,rresidbits,false);
      stubptrsdisk_[i]=zeropair;
    }
    

    if (barrel_) {
      //barrel
      for (int i=0;i<4;i++) {
	phiproj_[i]=phiproj[i];
	zproj_[i]=zproj[i];
	phiprojder_[i]=phider[i];
	zprojder_[i]=zder[i];
	phiprojapprox_[i]=phiprojapprox[i];
	zprojapprox_[i]=zprojapprox[i];
	phiprojderapprox_[i]=phiderapprox[i];
	zprojderapprox_[i]=zderapprox[i];
	minusNeighbor_[i]=minusNeighbor[i];
	plusNeighbor_[i]=plusNeighbor[i];
	if (iphiproj[i]<0) iphiproj[i]=0; //FIXME should be assert?
	if (rproj_[i]<60.0) {
	  fpgaphiproj_[i].set(iphiproj[i],nbitsphiprojL123,true,__LINE__,__FILE__);
	  int iphivm=(iphiproj[i]>>(nbitsphiprojL123-5))&0x7;
	  if ((projlayer_[i]%2)==1) {
	    iphivm^=4;
	  }
	  fpgaphiprojvm_[i].set(iphivm,3,true,__LINE__,__FILE__);
	  fpgazproj_[i].set(izproj[i],nbitszprojL123,false,__LINE__,__FILE__);
	  int izvm=izproj[i]>>(12-7)&0xf;
	  fpgazprojvm_[i].set(izvm,4,true,__LINE__,__FILE__);
	  fpgaphiprojder_[i].set(iphider[i],nbitsphiprojderL123,false,__LINE__,__FILE__);
	  fpgazprojder_[i].set(izder[i],nbitszprojderL123,false,__LINE__,__FILE__);
	} else {
	  fpgaphiproj_[i].set(iphiproj[i],nbitsphiprojL456,true,__LINE__,__FILE__);
	  int iphivm=(iphiproj[i]>>(nbitsphiprojL456-5))&0x7;
	  if ((projlayer_[i]%2)==1) {
	    iphivm^=4;
	  }
	  fpgaphiprojvm_[i].set(iphivm,3,true,__LINE__,__FILE__);
	  fpgazproj_[i].set(izproj[i],nbitszprojL456,false,__LINE__,__FILE__);
	  int izvm=izproj[i]>>(8-7)&0xf;
	  fpgazprojvm_[i].set(izvm,4,true,__LINE__,__FILE__);
	  fpgaphiprojder_[i].set(iphider[i],nbitsphiprojderL456,false,__LINE__,__FILE__);
	  fpgazprojder_[i].set(izder[i],nbitszprojderL456,false,__LINE__,__FILE__); 
	}
      }
      //Now handle projections to the disks
      for (int i=0;i<5;i++) {
	phiprojdisk_[i]=phiprojDisk[i];
	rprojdisk_[i]=rprojDisk[i];
	phiprojderdisk_[i]=phiderDisk[i];
	rprojderdisk_[i]=rderDisk[i];
	phiprojapproxdisk_[i]=phiprojapproxDisk[i];
	rprojapproxdisk_[i]=rprojapproxDisk[i];
	phiprojderapproxdisk_[i]=phiderapproxDisk[i];
	rprojderapproxdisk_[i]=rderapproxDisk[i];
	minusNeighbordisk_[i]=minusNeighborDisk[i];
	plusNeighbordisk_[i]=plusNeighborDisk[i];
	if (iphiprojDisk[i]<0) iphiprojDisk[i]=0; //FIXME should be assert?

	fpgaphiprojdisk_[i].set(iphiprojDisk[i],nbitsphiprojL123,true,__LINE__,__FILE__);
	int iphivm=(iphiprojDisk[i]>>(nbitsphiprojL123-5))&0x7;
	if ((abs(projdisk_[i])%2)==1) {
	  iphivm^=4;
	}
	fpgaphiprojvmdisk_[i].set(iphivm,3,true,__LINE__,__FILE__);
	fpgarprojdisk_[i].set(irprojDisk[i],nrbitsprojdisk,false,__LINE__,__FILE__);
	int irvm=irprojDisk[i]>>(13-7)&0xf;
	fpgarprojvmdisk_[i].set(irvm,4,true,__LINE__,__FILE__);
	fpgaphiprojderdisk_[i].set(iphiderDisk[i],nbitsphiprojderL123,false,__LINE__,__FILE__);
	fpgarprojderdisk_[i].set(irderDisk[i],nrbitsprojderdisk,false,__LINE__,__FILE__);
      }
    } 

    if (disk_) {
      //disk stub 
      for (int i=0;i<3;i++) {
	phiprojdisk_[i]=phiprojDisk[i];
	rprojdisk_[i]=rprojDisk[i];
	phiprojderdisk_[i]=phiderDisk[i];
	assert(fabs(phiprojderdisk_[i])<0.1);
	rprojderdisk_[i]=rderDisk[i];
	phiprojapproxdisk_[i]=phiprojapproxDisk[i];
	rprojapproxdisk_[i]=rprojapproxDisk[i];
	phiprojderapproxdisk_[i]=phiderapproxDisk[i];
	rprojderapproxdisk_[i]=rderapproxDisk[i];
	minusNeighbordisk_[i]=minusNeighborDisk[i];
	plusNeighbordisk_[i]=plusNeighborDisk[i];
	if (iphiprojDisk[i]<0) iphiprojDisk[i]=0; //FIXME should be assert?
	
	fpgaphiprojdisk_[i].set(iphiprojDisk[i],nbitsphiprojL123,true,__LINE__,__FILE__);
	int iphivm=(iphiprojDisk[i]>>(nbitsphiprojL123-5))&0x7;
	if ((abs(projdisk_[i])%2)==1) {
	  iphivm^=4;
	}
	fpgaphiprojvmdisk_[i].set(iphivm,3,true,__LINE__,__FILE__);
	fpgarprojdisk_[i].set(irprojDisk[i],nrbitsprojdisk,false,__LINE__,__FILE__);
	int irvm=irprojDisk[i]>>(13-7)&0xf;
	fpgarprojvmdisk_[i].set(irvm,4,true,__LINE__,__FILE__);
	fpgaphiprojderdisk_[i].set(iphiderDisk[i],nbitsphiprojderL123,false,__LINE__,__FILE__);
	fpgarprojderdisk_[i].set(irderDisk[i],nrbitsprojderdisk,false,__LINE__,__FILE__);
      }
      for (int i=0;i<2;i++) {
	phiproj_[i]=phiproj[i];
	zproj_[i]=zproj[i];
	phiprojder_[i]=phider[i];
	zprojder_[i]=zder[i];
	phiprojapprox_[i]=phiprojapprox[i];
	zprojapprox_[i]=zprojapprox[i];
	phiprojderapprox_[i]=phiderapprox[i];
	zprojderapprox_[i]=zderapprox[i];
	minusNeighbor_[i]=minusNeighbor[i];
	plusNeighbor_[i]=plusNeighbor[i];
	if (iphiproj[i]<0) iphiproj[i]=0; //FIXME should be assert?
	assert(rproj_[i]<60.0);
	fpgaphiproj_[i].set(iphiproj[i],nbitsphiprojL123,true,__LINE__,__FILE__);
	int iphivm=(iphiproj[i]>>(nbitsphiprojL123-5))&0x7;
	if ((projlayer_[i]%2)==1) {
	  iphivm^=4;
	}
	fpgaphiprojvm_[i].set(iphivm,3,true,__LINE__,__FILE__);
	fpgazproj_[i].set(izproj[i],nbitszprojL123,false,__LINE__,__FILE__);
	int izvm=izproj[i]>>(12-7)&0xf;
	fpgazprojvm_[i].set(izvm,4,true,__LINE__,__FILE__);
	fpgaphiprojder_[i].set(iphider[i],nbitsphiprojderL123,false,__LINE__,__FILE__);
	fpgazprojder_[i].set(izder[i],nbitszprojderL123,false,__LINE__,__FILE__);
      }
    }

    if (overlap_) {
      //projections to layers
      for (int i=0;i<1;i++) {
	phiproj_[i]=phiproj[i];
	zproj_[i]=zproj[i];
	phiprojder_[i]=phider[i];
	zprojder_[i]=zder[i];
	phiprojapprox_[i]=phiprojapprox[i];
	zprojapprox_[i]=zprojapprox[i];
	phiprojderapprox_[i]=phiderapprox[i];
	zprojderapprox_[i]=zderapprox[i];
	minusNeighbor_[i]=minusNeighbor[i];
	plusNeighbor_[i]=plusNeighbor[i];
	if (iphiproj[i]<0) iphiproj[i]=0; //FIXME should be assert?
	assert(rproj_[i]<60.0);

	fpgaphiproj_[i].set(iphiproj[i],nbitsphiprojL123,true,__LINE__,__FILE__);
	int iphivm=(iphiproj[i]>>(nbitsphiprojL123-5))&0x7;
	if ((projlayer_[i]%2)==1) {
	  iphivm^=4;
	}
	fpgaphiprojvm_[i].set(iphivm,3);
	fpgazproj_[i].set(izproj[i],nbitszprojL123,false,__LINE__,__FILE__);
	int izvm=izproj[i]>>(12-7)&0xf;
	fpgazprojvm_[i].set(izvm,4,true,__LINE__,__FILE__);
	fpgaphiprojder_[i].set(iphider[i],nbitsphiprojderL123,false,__LINE__,__FILE__);
	fpgazprojder_[i].set(izder[i],nbitszprojderL123,false,__LINE__,__FILE__);
      }
      //Now handle projections to the disks
      for (int i=0;i<4;i++) {
	phiprojdisk_[i+1]=phiprojDisk[i];
	rprojdisk_[i+1]=rprojDisk[i];
	phiprojderdisk_[i+1]=phiderDisk[i];
	rprojderdisk_[i+1]=rderDisk[i];
	phiprojapproxdisk_[i+1]=phiprojapproxDisk[i];
	rprojapproxdisk_[i+1]=rprojapproxDisk[i];
	phiprojderapproxdisk_[i+1]=phiderapproxDisk[i];
	rprojderapproxdisk_[i+1]=rderapproxDisk[i];
	minusNeighbordisk_[i+1]=minusNeighborDisk[i];
	plusNeighbordisk_[i+1]=plusNeighborDisk[i];
	if (iphiprojDisk[i]<0) iphiprojDisk[i]=0; //FIXME should be assert?

	//cout << "iphiprojDisk : "<<i<<" "<<iphiprojDisk[i]<<endl;

	fpgaphiprojdisk_[i+1].set(iphiprojDisk[i],nbitsphiprojL123,true,__LINE__,__FILE__);
	int iphivm=(iphiprojDisk[i]>>(nbitsphiprojL123-5))&0x7;
	if ((abs(projdisk_[i+1])%2)==1) {
	  iphivm^=4;
	}
	fpgaphiprojvmdisk_[i+1].set(iphivm,3,true,__LINE__,__FILE__);
	fpgarprojdisk_[i+1].set(irprojDisk[i],nrbitsprojdisk,false,__LINE__,__FILE__);
	int irvm=irprojDisk[i]>>(13-7)&0xf;
	fpgarprojvmdisk_[i+1].set(irvm,4,true,__LINE__,__FILE__);
	fpgaphiprojderdisk_[i+1].set(iphiderDisk[i],nbitsphiprojderL123,false,__LINE__,__FILE__);
	fpgarprojderdisk_[i+1].set(irderDisk[i],nrbitsprojderdisk,false,__LINE__,__FILE__);
      }
    } 


    ichisqfit_.set(-1,8,false);

  }




  ~FPGATracklet() {

    if (writeTrackletParameters) {
      static ofstream out("trackletparameters.txt");

      out << innerStub_->r() 
	  << " " << rinv_
	  << " " << phi0_
	  << " " << z0_
	  << " " << t_
	  << "     " << rinvapprox_
	  << " " << phi0approx_
	  << " " << z0approx_
	  << " " << tapprox_<<endl;
    }

  }



  L1TStub* innerStub() {return innerStub_;}
  L1TStub* outerStub() {return outerStub_;}

  std::string addressstr() {
    std::ostringstream oss;
    oss << innerFPGAStub_->fedregionaddressstr()<<"|" 
	<< outerFPGAStub_->fedregionaddressstr();

    return oss.str();

  }

  std::string trackletparstr() {
    std::ostringstream oss;
    oss << fpgarinv_.str()<<"|"
	<< fpgaphi0_.str()<<"|"
	<< fpgaz0_.str()<<"|"
	<< fpgat_.str();

    return oss.str();

  }

  std::string trackletprojstr(int i) const {
    std::ostringstream oss;
    oss << "yyyy|yyyyyy|"
	<< fpgaphiproj_[i].str()<<"|"
	<< fpgazproj_[i].str()<<"|"
	<< fpgaphiprojder_[i].str()<<"|"
	<< fpgazprojder_[i].str();

    return oss.str();

  }

  std::string vmstrlayer(int layer) {
    std::ostringstream oss;
    int count=0;
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	count++;
	FPGAWord index;
	if (projnum_[i]>=(1<<6)) {
	  cout << "Warning projection number too large!"<<endl;
	  projnum_[i]=(1<<6)-1;
	}
	index.set(projnum_[i],6);
	oss << "xxx|"<<index.str()<<"|"<<fpgaphiprojvm_[i].str()<<"|"
	    << fpgazprojvm_[i].str();
      }
    }
    assert(count==1);
    return oss.str();

  }


  std::string trackletprojstrlayer(int layer) const {
    std::ostringstream oss;
    if (disk_||overlap_) return trackletprojstr(layer-1);
    int count=0;
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	count++;
	return trackletprojstr(i); 
      }
    }

    cout << "In trackletprojstrlayer "<<layer<<endl;
    for (int i=0;i<4;i++) {
      cout << "projlayer = "<<projlayer_[i]<<endl;
    }
    
    
    assert(count==1);
    return oss.str();

  }


  double zproj(int layer) const {
    if (disk_||overlap_) return zproj_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return zproj_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double zprojapprox(int layer) {
    if (disk_||overlap_) return zprojapprox_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return zprojapprox_[i];
      }
    }
    assert(0);
    return 0.0;

  }


  double zprojder(int layer) {
    if (disk_||overlap_) return zprojder_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return zprojder_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double zprojderapprox(int layer) {
    if (disk_||overlap_) return zprojderapprox_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return zprojderapprox_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double phiprojder(int layer) {
    if (disk_||overlap_) return phiprojder_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return phiprojder_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double phiprojderapprox(int layer) {
    if (disk_||overlap_) return phiprojderapprox_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return phiprojderapprox_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  FPGAWord fpgazprojder(int layer) {
    if (disk_||overlap_) return fpgazprojder_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return fpgazprojder_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }

  FPGAWord fpgaphiprojder(int layer) {
    if (disk_||overlap_) return fpgaphiprojder_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return fpgaphiprojder_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }


  double phiproj(int layer) {
    if (disk_||overlap_) return phiproj_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return phiproj_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double phiprojapprox(int layer) {
    if (disk_||overlap_) return phiprojapprox_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return phiprojapprox_[i];
      }
    }
    assert(0);
    return 0.0;

  }


  FPGAWord fpgazproj(int layer) {
    if (disk_||overlap_) return fpgazproj_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return fpgazproj_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }

  double rproj(int layer) {
    if (disk_||overlap_) return rproj_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return rproj_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double rstub(int layer) {
    if (disk_||overlap_) return rstub_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return rstub_[i];
      }
    }
    assert(0);
    return 0.0;

  }


  bool minusNeighbor(int layer) {
    if (disk_||overlap_) return minusNeighbor_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return minusNeighbor_[i];
      }
    }
    assert(0);
    return false;
  }


  bool plusNeighbor(int layer) {
    if (disk_||overlap_) return plusNeighbor_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return plusNeighbor_[i];
      }
    }
    assert(0);
    return false;
  }


  FPGAWord fpgaphiproj(int layer) {
    if (disk_||overlap_) return fpgaphiproj_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return fpgaphiproj_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }

  void setProjNum(int projnum,int layer){
    if (disk_||overlap_) {
      projnum_[layer-1]=projnum;
      return;
    }
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	projnum_[i]=projnum;
	return;
      }
    }
    assert(0);
  }


  int getProjNum(int layer) const {
    if (disk_||overlap_) return projnum_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return projnum_[i];
      }
    }
    assert(0);
    return 0;
  }

  double zproj(int layer) {
    if (disk_||overlap_) return zproj_[layer-1];
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return zproj_[i];
      }
    }
    assert(0);
    return 0.0;

  }




  //Disks


  FPGAWord fpgaphiresiddisk(int disk) {
    if (!disk_) return fpgaphiresiddisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return fpgaphiresiddisk_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }

  FPGAWord fpgarresiddisk(int disk) {
    if (!disk_) return fpgarresiddisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return fpgarresiddisk_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }

  double phiresiddisk(int disk) {
    if (!disk_) return phiresiddisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return phiresiddisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double rresiddisk(int disk) {
    if (!disk_) return rresiddisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return rresiddisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double phiresidapproxdisk(int disk) {
    if (!disk_) return phiresidapproxdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return phiresidapproxdisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double rresidapproxdisk(int disk) {
    if (!disk_) return rresidapproxdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return rresidapproxdisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }


  double zstubdisk(int disk) {
    if (!disk_) return zstub_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return zstub_[i];
      }
    }
    assert(0);
    return 0.0;

  }


  double zprojdisk(int disk) const {
    if (!disk_) return zprojdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return zprojdisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }


  double rprojdisk(int disk) const {
    if (!disk_) return rprojdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return rprojdisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double alphadisk(int disk) const {
    if (!disk_) return alphadisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return alphadisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double rprojapproxdisk(int disk) {
    if (!disk_) return rprojapproxdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return rprojapproxdisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }


  double rprojderdisk(int disk) {
    if (!disk_) return rprojderdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return rprojderdisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double rprojderapproxdisk(int disk) {
    if (!disk_) return rprojderapproxdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return rprojderapproxdisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double phiprojderdisk(int disk) {
    if (!disk_) return phiprojderdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return phiprojderdisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double phiprojderapproxdisk(int disk) {
    if (!disk_) return phiprojderapproxdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return phiprojderapproxdisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  FPGAWord fpgarprojderdisk(int disk) {
    if (!disk_) return fpgarprojderdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return fpgarprojderdisk_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }

  FPGAWord fpgaphiprojderdisk(int disk) {
    if (!disk_) return fpgaphiprojderdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return fpgaphiprojderdisk_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }

  FPGAWord fpgaphiprojdisk(int disk) {
    if (!disk_) return fpgaphiprojdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return fpgaphiprojdisk_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }


  double phiprojdisk(int disk) {
    if (!disk_) return phiprojdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return phiprojdisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  double phiprojapproxdisk(int disk) {
    if (!disk_) return phiprojapproxdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return phiprojapproxdisk_[i];
      }
    }
    assert(0);
    return 0.0;

  }


  FPGAWord fpgarprojdisk(int disk) {
    if (!disk_) return fpgarprojdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return fpgarprojdisk_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }

  bool minusNeighborDisk(int disk) {
    if (!disk_) return minusNeighbordisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return minusNeighbordisk_[i];
      }
    }
    assert(0);
    return false;
  }


  bool plusNeighborDisk(int disk) {
    if (!disk_) return plusNeighbordisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return plusNeighbordisk_[i];
      }
    }
    assert(0);
    return false;
  }


  void setProjNumDisk(int projnum,int disk){
    if (!disk_) {
      projnumdisk_[abs(disk)-1]=projnum;
      return;
    }
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	projnumdisk_[i]=projnum;
	return;
      }
    }
    assert(0);
  }


  int getProjNumDisk(int disk) const {
    if (!disk_) return projnumdisk_[abs(disk)-1];
    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	return projnumdisk_[i];
      }
    }
    assert(0);
    return 0;
  }

  bool matchdisk(int disk) {
    if (!disk_) return !fpgaphiresiddisk_[abs(disk)-1].atExtreme();
    for (int i=0;i<3;i++) {
      //cout << "disk projdisk : "<<disk<<" "<<projdisk_[i]<<endl;
      if (projdisk_[i]==disk) {
	return !fpgaphiresiddisk_[i].atExtreme();
      }
    } 
    assert(0);
    return false;
  }





  void addMatch(int layer, int ideltaphi, int ideltaz, 
		double dphi, double dz, 
		double dphiapprox, double dzapprox, 
		int stubid,double rstub,
		std::pair<FPGAStub*,L1TStub*> stubptrs){
    
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	assert(abs(ideltaphi)<(1<<(phiresidbits-1)));
	assert(abs(ideltaz)<(1<<(zresidbits-1)));
	if (fabs(ideltaphi)<fabs(fpgaphiresid_[i].value())) {
	  phiresid_[i]=dphi;
	  zresid_[i]=dz;
	  phiresidapprox_[i]=dphiapprox;
	  zresidapprox_[i]=dzapprox;
	  rstub_[i]=rstub;
	  stubptrs_[i]=stubptrs;
	  fpgaphiresid_[i].set(ideltaphi,phiresidbits,false);
	  fpgazresid_[i].set(ideltaz,zresidbits,false);
	  fpgastubid_[i].set(stubid,9);
	  assert(!fpgaphiresid_[i].atExtreme());
	}
	else {
	  if (fpgaphiresid_[i].atExtreme()) {
	    static int count=0;
	    count++;
	    if (count<3) {
	      cout << "WARNING trying to add too large residual"<<endl;
	    }
	  }
	}
      }
    }
 

  }



  void addMatchDisk(int disk, int ideltaphi, int ideltar, 
		    double dphi, double dr, 
		    double dphiapprox, double drapprox, double alpha,
		    int stubid,double zstub,
		    std::pair<FPGAStub*,L1TStub*> stubptrs){

    //cout << "addMatchDisk1 "<<disk<<" "<<ideltaphi<<endl; 

    if (!disk_) {
      assert(abs(ideltaphi)<(1<<(phiresidbits-1)));
      assert(abs(ideltar)<(1<<(rresidbits-1)));
      if (fabs(ideltaphi)<fabs(fpgaphiresiddisk_[abs(disk)-1].value())) {
	//cout << "addMatchDisk2 "<<disk<<" "<<ideltaphi<<endl; 
	phiresiddisk_[abs(disk)-1]=dphi;
	rresiddisk_[abs(disk)-1]=dr;
	assert(dphiapprox!=0.0);
	assert(drapprox!=0.0);
	stubptrsdisk_[abs(disk)-1]=stubptrs;
	phiresidapproxdisk_[abs(disk)-1]=dphiapprox;
	rresidapproxdisk_[abs(disk)-1]=drapprox;
	zstub_[abs(disk)-1]=zstub;
	alphadisk_[abs(disk)-1]=alpha;
	fpgaphiresiddisk_[abs(disk)-1].set(ideltaphi,phiresidbits,false,__LINE__,__FILE__);
	fpgarresiddisk_[abs(disk)-1].set(ideltar,rresidbits,false,__LINE__,__FILE__);
	if (stubid<0) stubid=(1<<9)-1;
	fpgastubiddisk_[abs(disk)-1].set(stubid,9,true,__LINE__,__FILE__);
	assert(!fpgaphiresiddisk_[abs(disk)-1].atExtreme());
      }
      else {
	if (fpgaphiresiddisk_[abs(disk)-1].atExtreme()) {
	  static int count=0;
	  count++;
	  if (count<3) {
	    cout << "WARNING trying to add too large residual"<<endl;
	  }
	}
      }

      return;

    }
    

    for (int i=0;i<3;i++) {
      if (projdisk_[i]==disk) {
	assert(abs(ideltaphi)<(1<<(phiresidbits-1)));
	assert(abs(ideltar)<(1<<(rresidbits-1)));
	if (fabs(ideltaphi)<fabs(fpgaphiresiddisk_[i].value())) {
	  phiresiddisk_[i]=dphi;
	  rresiddisk_[i]=dr;
	  assert(dphiapprox!=0.0);
	  assert(drapprox!=0.0);
	  phiresidapproxdisk_[i]=dphiapprox;
	  rresidapproxdisk_[i]=drapprox;
	  stubptrsdisk_[i]=stubptrs;
	  zstub_[i]=zstub;
	  alphadisk_[i]=alpha;
	  fpgaphiresiddisk_[i].set(ideltaphi,phiresidbits,false,__LINE__,__FILE__);
	  fpgarresiddisk_[i].set(ideltar,rresidbits,false,__LINE__,__FILE__);
	  if (stubid<0) stubid=(1<<9)-1;
	  fpgastubiddisk_[i].set(stubid,9,true,__LINE__,__FILE__);
	  assert(!fpgaphiresiddisk_[i].atExtreme());
	}
	else {
	  if (fpgaphiresiddisk_[i].atExtreme()) {
	    static int count=0;
	    count++;
	    if (count<3) {
	      cout << "WARNING trying to add too large residual"<<endl;
	    }
	  }
	}
      }
    }
 

  }


  int nMatches() {

    int nmatches=0;

    for (int i=0;i<4;i++) {
      if (!fpgaphiresid_[i].atExtreme()) {
	//cout << "Match in layer : "<<i+1<<endl; 
	nmatches++;
      }
    }


    return nmatches;
    
  }

  int nMatchesDisk() {

    int nmatches=0;

    if (!disk_) {
      //cout << "nMatchesDisk : ";
      for (int i=0;i<5;i++) {
	//cout <<" "<<!fpgaphiresiddisk_[i].atExtreme();
	if (!fpgaphiresiddisk_[i].atExtreme()) {
	  //cout << "Match1 in disk : "<<i+1<<endl; 
	  nmatches++;
	}
      }
      //cout << endl;
      return nmatches;
    }

    for (int i=0;i<3;i++) {
      if (!fpgaphiresiddisk_[i].atExtreme()) {
	nmatches++;
	//cout << "Match2 in disk : "<<projdisk_[i]<<endl; 
      }
    }

    return nmatches;
    
  }


  int nMatchesOverlap() {

    int nmatches=0;

    assert(overlap_);

    for (int i=1;i<5;i++) {
	//cout <<" "<<!fpgaphiresiddisk_[i].atExtreme();
	if (!fpgaphiresiddisk_[i].atExtreme()) nmatches++;
    }

    for (int i=0;i<2;i++) {
      if (!fpgaphiresid_[i].atExtreme()) nmatches++;
    }
    return nmatches;
    
  }



  bool match(int layer) {
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	//cout << "atExtreme: "<<fpgaphiresid_[i].atExtreme()<<endl;
	return !fpgaphiresid_[i].atExtreme();
      }
    } 
    assert(0);
    return false;
  }

  std::string fullmatchstr(int layer) {
    std::ostringstream oss;
    int count=0;
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	count++;
	FPGAWord tmp;
	if (projnum_[i]<0||projnum_[i]>63) {
	  cout << "projnum_[i] = "<<projnum_[i]<<endl;
	  continue;
	}
	tmp.set(projnum_[i],6);
	oss << "yyyy|"<<tmp.str()<<"|"
	    << fpgastubid_[i].str()<<"|"
	    << fpgaphiresid_[i].str()<<"|"
	    << fpgazresid_[i].str();
      }
    }
    //assert(count==1);
    return oss.str();

  }

  double phiresid(int layer) {
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return phiresid_[i];
      }
    }
    assert(0);
    return 0.0;
  }

  double phiresidapprox(int layer) {
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return phiresidapprox_[i];
      }
    }
    assert(0);
    return 0.0;
  }

  double zresid(int layer) {
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return zresid_[i];
      }
    }
    assert(0);
    return 0.0;
  }

  double zresidapprox(int layer) {
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return zresidapprox_[i];
      }
    }
    assert(0);
    return 0.0;

  }

  FPGAWord fpgaphiresid(int layer) {
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return fpgaphiresid_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }

  FPGAWord fpgazresid(int layer) {
    for (int i=0;i<4;i++) {
      if (projlayer_[i]==layer) {
	return fpgazresid_[i];
      }
    }
    assert(0);
    FPGAWord tmp;
    return tmp;

  }
/*
    int getStubID(int layer) {

     //printf("Track Type Barrel %i  Disk %i  Overlap %i \n",barrel_,disk_,overlap_);
     int stubID = 63; //this is no stub code value
	  
     bool found = false;
	  int i=0;
	  while(!found && i<4) {
	    if(projlayer_[i]-1 == layer) {
		    stubID = fpgastubid_[i].value();
			 found = true;     
		 }
	    i++;
	  }
	  
	  //if not found this must be a tracklet layer
	  if(!found && barrel_) {
	    if(innerFPGAStub_.layer().value() == layer) {
		    stubID = ((innerFPGAStub_.fedregion()-1)<<6)+innerFPGAStub_.stubindex().value();
		 } else if (outerFPGAStub_.layer().value() == layer) {
		    stubID = ((outerFPGAStub_.fedregion()-1)<<6)+outerFPGAStub_.stubindex().value();
		 }
	  }
	  
     return stubID;
  }

*/

  std::vector<L1TStub*> getL1Stubs() {

    std::vector<L1TStub*> tmp;

    tmp.push_back(innerStub_);
    tmp.push_back(outerStub_);

    for (unsigned int i=0;i<4;i++) {
      if (stubptrs_[i].second!=0) tmp.push_back(stubptrs_[i].second);
    }

    for (unsigned int i=0;i<5;i++) {
      if (stubptrsdisk_[i].second!=0) tmp.push_back(stubptrsdisk_[i].second);
    }

    return tmp;

  }

  std::map<int, int> getStubIDs() {


    std::map<int, int> stubIDs;
    
    //printf("Track Type Barrel %i   Disk %i   Overlap %i  \n",barrel_,disk_,overlap_);
    //printf(" irinv %i,  iphi %i  it %i \n",irinvfit().value(),iphi0fit().value(),itfit().value());
    //printf(" Matches:  barrel  %i   disks %i \n",nMatches(),nMatchesDisk());
    //for(int i=0; i<16; i++)  stubIDs[i] = 63; //this is no stub code value
	  
	  
    if(barrel_) {
	   
	     
      for(int i=0; i<4; i++) {
		  
	//get barrel stubs
	if(fpgastubid_[i].value() != 63) stubIDs[projlayer_[i]] = fpgastubid_[i].value();			  		  
		  
	//check disk (probably only really need to check first 1 or 2)
	if(fpgastubiddisk_[i].value() != 63) {
	  //printf("Disk values for barrel track proj  %i   stud ID %i \n",10+i+1,fpgastubiddisk_[i].value());
	  if(itfit().value() < 0) {
	    stubIDs[-10-i-1] = fpgastubiddisk_[i].value();
	  } else {
	    stubIDs[10+i+1] = fpgastubiddisk_[i].value();
	  }
	}	 			  		  		  
      }
		  
		  
      //get stubs making up tracklet
      //printf(" inner %i  outer %i layers \n",innerFPGAStub_.layer().value(),outerFPGAStub_.layer().value());
      stubIDs[innerFPGAStub_->layer().value()+1] = ((innerFPGAStub_->fedregion()-1)<<6)+innerFPGAStub_->stubindex().value();
      stubIDs[outerFPGAStub_->layer().value()+1] = ((outerFPGAStub_->fedregion()-1)<<6)+outerFPGAStub_->stubindex().value();
		  
		  		  
    } else if (disk_) {

      for(int i=0; i<3; i++) {
	
	//check inner two layers of barrel
	if(fpgastubid_[i].value() != 63) stubIDs[projlayer_[i]] = fpgastubid_[i].value();			  
	
	//get disks
	if(fpgastubiddisk_[i].value() != 63) {
	  if(projdisk_[i] < 0) {
	    stubIDs[projdisk_[i]-10] = fpgastubiddisk_[i].value();
	  } else {
	    stubIDs[projdisk_[i]+10] = fpgastubiddisk_[i].value();
	  }
	}	 			  
      }
		  
      //get stubs making up tracklet
      //printf(" inner %i  outer %i disks \n",innerFPGAStub_.disk().value(),outerFPGAStub_.disk().value());
      if(innerFPGAStub_->disk().value() < 0) { //negative side runs 6-10
	stubIDs[innerFPGAStub_->disk().value()-10] = ((innerFPGAStub_->fedregion()-1)<<6)+innerFPGAStub_->stubindex().value();
	stubIDs[outerFPGAStub_->disk().value()-10] = ((outerFPGAStub_->fedregion()-1)<<6)+outerFPGAStub_->stubindex().value();
      } else { // positive side runs 11-15]
	stubIDs[innerFPGAStub_->disk().value()+10] = ((innerFPGAStub_->fedregion()-1)<<6)+innerFPGAStub_->stubindex().value();
	stubIDs[outerFPGAStub_->disk().value()+10] = ((outerFPGAStub_->fedregion()-1)<<6)+outerFPGAStub_->stubindex().value();
      }		  

    } else if (overlap_) {
      //printf("Overlap Track...\n");
      
      
      for(int i=0; i<4; i++) {
	
	//check inner two layers of barrel
	if(i<2 && fpgastubid_[i].value() != 63) stubIDs[projlayer_[i]] = fpgastubid_[i].value();			  
	
	//get disks
	if(fpgastubiddisk_[i].value() != 63) {
	  if(innerStub_->disk() < 0) {
	    stubIDs[-projdisk_[i]-10] = fpgastubiddisk_[i].value();
	  } else {
	    stubIDs[projdisk_[i]+10] = fpgastubiddisk_[i].value();
	  }
	}	 			  
      }
      
      //get stubs making up tracklet
      //printf(" inner %i  outer %i layers \n",innerFPGAStub_.layer().value(),outerFPGAStub_.layer().value());
      //printf(" inner %i  outer %i disks \n",innerFPGAStub_.disk().value(),outerFPGAStub_.disk().value());
      if(innerFPGAStub_->disk().value() < 0) { //negative side runs -11 - -15
	stubIDs[innerFPGAStub_->disk().value()-10] = ((innerFPGAStub_->fedregion()-1)<<6)+innerFPGAStub_->stubindex().value();
	stubIDs[outerFPGAStub_->layer().value()+1] = ((outerFPGAStub_->fedregion()-1)<<6)+outerFPGAStub_->stubindex().value();
      } else { // positive side runs 11-15]
	stubIDs[innerFPGAStub_->disk().value()+10] = ((innerFPGAStub_->fedregion()-1)<<6)+innerFPGAStub_->stubindex().value();
	stubIDs[outerFPGAStub_->layer().value()+1] = ((outerFPGAStub_->fedregion()-1)<<6)+outerFPGAStub_->stubindex().value();
      }		  
		  
    }


    //cout << "New track layer = "<<innerFPGAStub_->layer().value()+1<<endl;
    //for(std::map<int,int>::iterator i=stubIDs.begin(); i!=stubIDs.end(); i++) {
    //      printf("Layer/Disk %i  ID  %i \n",i->first, i->second);
    //}
    
    return stubIDs;
  }


  double rinv() const { return rinv_; }
  double phi0() const { return phi0_; }
  double t() const { return t_; }
  double z0() const { return z0_; }

  double rinvapprox() const { return rinvapprox_; }
  double phi0approx() const { return phi0approx_; }
  double tapprox() const { return tapprox_; }
  double z0approx() const { return z0approx_; }


  FPGAWord fpgarinv() const { return fpgarinv_; }
  FPGAWord fpgaphi0() const { return fpgaphi0_; }
  FPGAWord fpgat() const { return fpgat_; }
  FPGAWord fpgaz0() const { return fpgaz0_; }

  double rinvfit() const { return rinvfit_; }
  double phi0fit() const { return phi0fit_; }
  double tfit() const { return tfit_; }
  double z0fit() const { return z0fit_; }

  double rinvfitexact() const { return rinvfitexact_; }
  double phi0fitexact() const { return phi0fitexact_; }
  double tfitexact() const { return tfitexact_; }
  double z0fitexact() const { return z0fitexact_; }

  FPGAWord irinvfit() const { return irinvfit_; }
  FPGAWord iphi0fit() const { return iphi0fit_; }
  FPGAWord itfit() const { return itfit_; }
  FPGAWord iz0fit() const { return iz0fit_; }
  FPGAWord ichiSqfit() const { return ichisqfit_; }


  void setFitPars(double rinvfit, double phi0fit, double tfit,
		  double z0fit, double chisqfit,
		  double rinvfitexact, double phi0fitexact, double tfitexact,
		  double z0fitexact, double chisqfitexact,
		  int irinvfit, int iphi0fit, int itfit,
		  int iz0fit, int ichisqfit){

    rinvfit_=rinvfit;
    phi0fit_=phi0fit;
    tfit_=tfit;
    z0fit_=z0fit;
    chisqfit_=chisqfit;

    rinvfitexact_=rinvfitexact;
    phi0fitexact_=phi0fitexact;
    tfitexact_=tfitexact;
    z0fitexact_=z0fitexact;
    chisqfitexact_=chisqfitexact;
    
    if (irinvfit>(1<<14)) irinvfit=(1<<14);
    if (irinvfit<=-(1<<14)) irinvfit=-(1<<14)+1;
    irinvfit_.set(irinvfit,15,false,__LINE__,__FILE__);
    iphi0fit_.set(iphi0fit,19,false,__LINE__,__FILE__);
    itfit_.set(itfit,14,false,__LINE__,__FILE__);

    if (iz0fit>=(1<<10)) {
      iz0fit=(1<<10)-1;  //FIXME should use some number of bits
      cout << "WARNING: truncating iz0fit"<<endl;
    }

    if (iz0fit<=-(1<<10)) {
      iz0fit=1-(1<<10); //FIXME should use some number of bits
      cout << "WARNING: truncating iz0fit"<<endl;
    }

    iz0fit_.set(iz0fit,11,false,__LINE__,__FILE__);
    ichisqfit_.set(ichisqfit,5,true,__LINE__,__FILE__);

  }

  std::string trackfitstr() {
    std::ostringstream oss;
    oss << irinvfit_.str()<<"|"
	<< iphi0fit_.str()<<"|"
        << "xxxxxxxxxxx|"
	<< itfit_.str()<<"|"
	<< iz0fit_.str()<<"|"
	<< ichisqfit_.str()<< "|"
	<< innerFPGAStub_->fedregionaddressstr()<< "|"
	<< outerFPGAStub_->fedregionaddressstr()
	<< "|"<<fpgastubid_[0].str()
	<< "|"<<fpgastubid_[1].str()
	<< "|"<<fpgastubid_[2].str()
	<< "|"<<fpgastubid_[3].str();
    return oss.str();

  }

  FPGATrack getTrack() {
    assert(fit());
    FPGATrack tmpTrack(irinvfit_.value(),
		       iphi0fit_.value(),
		       itfit_.value(),
		       iz0fit_.value(),
		       getStubIDs(),
		       getL1Stubs());

    return tmpTrack;
    
  }
  

  bool fit() const { return ichisqfit_.value()!=-1; }

  int layer() const { 
    if (isOverlap()) return outerStub_->layer()+1;
    return innerStub_->layer()+1; 
  }

  int disk() const {
    return innerStub_->disk(); 
  }

  int disk2() const {
    if (innerStub_->disk()>0) {
      return innerStub_->disk()+1;
    }
    return innerStub_->disk()-1;
  }

  int overlap() const {
    return innerStub_->layer()+21;
  }

  bool isBarrel() const { 
    return barrel_;
  }

  bool isOverlap() const { 
    return overlap_;
  }

  int isDisk() const { 
    return disk_;
  }

  bool foundTrack(L1SimTrack simtrk){

    double deta=simtrk.eta()-asinh(itfit().value()*ktpars);
    double dphi=simtrk.phi()-(iphi0fit().value()*kphi0pars+phioffset_);

    if (dphi>0.5*two_pi) dphi-=two_pi;
    if (dphi<-0.5*two_pi) dphi+=two_pi;

    //cout << "deta dphi : "<<deta<<" "<<dphi<<endl;

    return (fabs(deta)<0.02)&&(fabs(dphi)<0.01);

  }

  double phioffset() const {return phioffset_;}

private:

  //Three types of tracklets... Overly complicated
  bool barrel_;
  bool disk_;
  bool overlap_;

  double phioffset_;

  FPGAStub* innerFPGAStub_;
  FPGAStub* outerFPGAStub_;


  L1TStub* innerStub_;
  L1TStub* outerStub_;

  double rinv_;
  double phi0_;
  double z0_;
  double t_;
  double rinvapprox_;
  double phi0approx_;
  double z0approx_;
  double tapprox_;

  double rproj_[6];

  FPGAWord fpgarinv_;
  FPGAWord fpgaphi0_;
  FPGAWord fpgaz0_;
  FPGAWord fpgat_;


  int      projlayer_[4];
  FPGAWord fpgaphiproj_[4];
  FPGAWord fpgazproj_[4];
  FPGAWord fpgaphiprojder_[4];
  FPGAWord fpgazprojder_[4];
  bool minusNeighbor_[4];
  bool plusNeighbor_[4];

  double rstub_[4];


  //Virtual module
  FPGAWord fpgaphiprojvm_[4];
  FPGAWord fpgazprojvm_[4];


  double phiproj_[4];
  double zproj_[4];
  double phiprojder_[4];
  double zprojder_[4];

  double phiprojapprox_[4];
  double zprojapprox_[4];
  double phiprojderapprox_[4];
  double zprojderapprox_[4];
  
  int projnum_[4];

  double phiresid_[4];
  double zresid_[4];
  double phiresidapprox_[4];
  double zresidapprox_[4];
  FPGAWord fpgaphiresid_[4];
  FPGAWord fpgazresid_[4];
  FPGAWord fpgastubid_[4];


  
  int      projdisk_[5];
  FPGAWord fpgaphiprojdisk_[5];
  FPGAWord fpgarprojdisk_[5];
  FPGAWord fpgaphiprojderdisk_[5];
  FPGAWord fpgarprojderdisk_[5];
  bool minusNeighbordisk_[5];
  bool plusNeighbordisk_[5];

  double zstub_[5];


  //Virtual module
  FPGAWord fpgaphiprojvmdisk_[5];
  FPGAWord fpgarprojvmdisk_[5];


  double phiprojdisk_[5];
  double rprojdisk_[5];
  double zprojdisk_[5];
  double phiprojderdisk_[5];
  double rprojderdisk_[5];
 
  double phiprojapproxdisk_[5];
  double rprojapproxdisk_[5];
  double phiprojderapproxdisk_[5];
  double rprojderapproxdisk_[5];
  
  int projnumdisk_[5];

  double phiresiddisk_[5];
  double rresiddisk_[5];
  double phiresidapproxdisk_[5];
  double rresidapproxdisk_[5];
  FPGAWord fpgaphiresiddisk_[5];
  FPGAWord fpgarresiddisk_[5];
  FPGAWord fpgastubiddisk_[5];

  double alphadisk_[5];


  double rinvfit_;
  double phi0fit_;
  double tfit_;
  double z0fit_;
  double chisqfit_;

  double rinvfitexact_;
  double phi0fitexact_;
  double tfitexact_;
  double z0fitexact_;
  double chisqfitexact_;

  FPGAWord irinvfit_;
  FPGAWord iphi0fit_;
  FPGAWord itfit_;
  FPGAWord iz0fit_;
  FPGAWord ichisqfit_;

  std::pair<FPGAStub*,L1TStub*> stubptrs_[4];
  std::pair<FPGAStub*,L1TStub*> stubptrsdisk_[5];

};



#endif



