//This class implementes the match calculator
#ifndef FPGAMATCHCALCULATOR_H
#define FPGAMATCHCALCULATOR_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAMatchCalculator:public FPGAProcessBase{

public:

  FPGAMatchCalculator(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    double dphi=two_pi/NSector;
    phimin_=iSector*dphi;
    phimax_=phimin_+dphi;
    if (phimin_>0.5*two_pi) phimin_-=two_pi;
    if (phimax_>0.5*two_pi) phimax_-=two_pi;
    phioffset_=phimin_-(phimax_-phimin_)/6.0;
    string subname=name.substr(8,2);
    fullmatchesToPlus_=0;
    fullmatchesToMinus_=0;
    layer_=0;
    if (subname=="L1") layer_=1;
    if (subname=="L2") layer_=2;
    if (subname=="L3") layer_=3;
    if (subname=="L4") layer_=4;
    if (subname=="L5") layer_=5;
    if (subname=="L6") layer_=6;
    disk_=0;
    if (subname=="F1") disk_=1;
    if (subname=="F2") disk_=2;
    if (subname=="F3") disk_=3;
    if (subname=="F4") disk_=4;
    if (subname=="F5") disk_=5;
    if (subname=="B1") disk_=-1;
    if (subname=="B2") disk_=-2;
    if (subname=="B3") disk_=-3;
    if (subname=="B4") disk_=-4;
    if (subname=="B5") disk_=-5;
    if (layer_==0 && disk_==0) {
      cout << "name subname "<<name<<" "<<subname<<endl;
      assert(0);
    }
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="matchout1"||
	output=="matchout2"||
	output=="matchout3"||
	output=="matchout4"||
	output=="matchout5"||
	output=="matchout6"||
	output=="matchout7"||
	output=="matchout8"
	){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      fullmatches_.push_back(tmp);
      return;
    }
    if (output=="matchoutplus"){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      assert(fullmatchesToPlus_==0);
      fullmatchesToPlus_=tmp;
      return;
    }
    if (output=="matchoutminus"){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      assert(fullmatchesToMinus_==0);
      fullmatchesToMinus_=tmp;
      return;
    }
    cout << "Count not fined output = "<<output<<endl;
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="allstubin"){
      FPGAAllStubs* tmp=dynamic_cast<FPGAAllStubs*>(memory);
      assert(tmp!=0);
      allstubs_=tmp;
      return;
    }
    if (input=="allprojin"){
      FPGAAllProjections* tmp=dynamic_cast<FPGAAllProjections*>(memory);
      assert(tmp!=0);
      allprojs_=tmp;
      return;
    }
    if (input=="match1in"||
	input=="match2in"||
	input=="match3in"||
	input=="match4in"||
	input=="match5in"||
	input=="match6in"||
	input=="match7in"||
	input=="match8in"){
      FPGACandidateMatch* tmp=dynamic_cast<FPGACandidateMatch*>(memory);
      assert(tmp!=0);
      matches_.push_back(tmp);
      return;
    }
    assert(0);
  }

  void execute() {

    //again here we will cheat a little and use the information in matches

    assert(fullmatches_.size()!=0);
    assert(fullmatchesToPlus_!=0);
    assert(fullmatchesToMinus_!=0);

    unsigned int countall=0;

    for(unsigned int j=0;j<matches_.size();j++){
      for(unsigned int i=0;i<matches_[j]->nMatches();i++){

	countall++;
	
	L1TStub* stub=matches_[j]->getStub(i).second;
	FPGAStub* fpgastub=matches_[j]->getStub(i).first;
	FPGATracklet* tracklet=matches_[j]->getFPGATracklet(i);
	
	if (layer_!=0) {

	  double pttracklet=0.3*3.8/(tracklet->rinv()*100);
	  bool keep=fabs(1.0/pttracklet-1.0/stub->pt())<ptstubconsistencymatching;
            
	  if (!keep) continue;
      
	  int seedlayer=tracklet->layer();
	
	  double phi=stub->phi();
	  if (phi<0) phi+=two_pi;
	  phi-=phioffset_;
	  
	  double dr=stub->r()-tracklet->rproj(layer_);
	  assert(fabs(dr)<drmax);

	  int ir=fpgastub->r().value();
	
	  int iphi=tracklet->fpgaphiproj(layer_).value();
	  int icorr=0;
	  
	  int lr=1;    

	  if (layer_<4) {
	    lr=2;
	    icorr=(ir*tracklet->fpgaphiprojder(layer_).value())>>(idrinvbits+1+phi0bitshift-rinvbitshift-phiderbitshift);
	    assert(idrinvbits+1+phi0bitshift-rinvbitshift-phiderbitshift>=0);
	  } else {
	    icorr=(ir*tracklet->fpgaphiprojder(layer_).value())>>(idrinvbits-1+phi0bitshift-rinvbitshift-phiderbitshift);
	    assert(idrinvbits-1+phi0bitshift-rinvbitshift-phiderbitshift>=0);
	  }
	  
	  iphi+=icorr;
	    
	  int iz=tracklet->fpgazproj(layer_).value();

	  int izcor=0;

	  if (layer_<4) {
	    izcor=(ir*tracklet->fpgazprojder(layer_).value())>>(idrinvbits-1-zderbitshift-tbitshift);
	  } else {
	    izcor=(ir*tracklet->fpgazprojder(layer_).value())>>(idrinvbits-zderbitshift-tbitshift+(nbitszprojL123-nbitszprojL456));
	  }

	  iz+=izcor;

	  int ideltaz=fpgastub->z().value()-iz;


	  int ideltaphi=0;
	  if (layer_<4) {  //FIXME are we loosing precission here? Yes!!!!
	    ideltaphi=(fpgastub->phi().value()<<3)-(iphi<<(phi0bitshift+2)); 
	  } else {
	    assert(phi0bitshift-1>=0);
	    ideltaphi=fpgastub->phi().value()-(iphi<<(phi0bitshift-1));  
	  }


	  if (dumpmatch) {
	    int l=1;
            int fact=1;
	    if (layer_>3) {
	      l=8; 
	      fact=(1<<(nbitszprojL123-nbitszprojL456));
	    }

	    cout << "DUMPMATCH : layer = "<<layer_<<endl;
	    cout << "DUMPMATCH : phiproj  = "<<tracklet->phiproj(layer_)<<" "
		 << tracklet->fpgaphiproj(layer_).value()*kphiproj123/l<<endl;
	    cout << "DUMPMATCH : phi corr = "<< dr*tracklet->phiprojder(layer_)
		 << " "<<icorr*kphiproj123/l<<endl;
	    cout << "DUMPMATCH : dr       = "<<dr<<" "<<ir*kr*lr<<endl;
	    cout << "DUMPMATCH : zproj    = "<<tracklet->zproj(layer_)<<" "
		 <<  tracklet->fpgazproj(layer_).value()*kz*fact<<" "
		 <<  tracklet->zproj(layer_)-tracklet->fpgazproj(layer_).value()*kz*fact
		 << endl;
	    cout << "DUMPMATCH : z der   = "<<tracklet->zprojder(layer_)
		 << " "<<tracklet->fpgazprojder(layer_).value()*kz/(kr*(1<<(idrinvbits-zderbitshift-tbitshift)))
		 <<" ("<<tracklet->fpgazprojder(layer_).value()<<")"<<endl;  
	    cout << "DUMPMATCH : z corr   = "<<dr*tracklet->zprojder(layer_)
		 << " "<<izcor*kz*fact<<endl;	      
	  }
	  
	  assert(fabs(dr)<3.0);
	  
	  double dphi=phi-(tracklet->phiproj(layer_)+
			   dr*tracklet->phiprojder(layer_));
	  double dz=stub->z()-(tracklet->zproj(layer_)+
			       dr*tracklet->zprojder(layer_));
	  
	  double dphiapprox=phi-(tracklet->phiprojapprox(layer_)+
				 dr*tracklet->phiprojderapprox(layer_));
	  double dzapprox=stub->z()-(tracklet->zprojapprox(layer_)+
				     dr*tracklet->zprojderapprox(layer_));
	  
	  bool imatch=false;


	  if (layer_<4){
	    if (seedlayer==1) {
	      imatch=(fabs(ideltaphi)<0.1/(kphi1*stub->r()))&&(fabs(ideltaz)<0.5/kz);
	    }
	    else {
	      imatch=(fabs(ideltaphi)<0.095/(kphi1*stub->r()))&&(fabs(ideltaz)<14.0/kz);
	    }

	  }else{
	    int fact=(1<<(nbitszprojL123-nbitszprojL456));
	    if (seedlayer==1) {
	      imatch=(fabs(ideltaphi)<0.28/(kphi1*stub->r()))&&(fabs(fact*ideltaz)<3.0/kz);
	    }
	    else {
	      imatch=(fabs(ideltaphi)<0.2/(kphi1*stub->r()))&&(fabs(fact*ideltaz)<14.0/kz);
	    }
	  }
	  
	  
	  if (imatch) {
	    //cout << "1: ideltaphi = "<<ideltaphi<<" "<<ideltaphi*kphi1*stub->r()<<endl;
	    std::pair<FPGAStub*,L1TStub*> tmp(fpgastub,stub);

	    tracklet->addMatch(layer_,ideltaphi,ideltaz,
			       dphi,dz,dphiapprox,dzapprox,
			       fpgastub->stubindex().value()+(1<<6)*(fpgastub->fedregion()-1),
			       stub->r(),tmp);
	    
	   

	    if (tracklet->plusNeighbor(layer_)){
	      fullmatchesToMinus_->addMatch(tracklet,tmp);
	    } else if (tracklet->minusNeighbor(layer_)) {
	      fullmatchesToPlus_->addMatch(tracklet,tmp);
	    } else {
	      for (unsigned int l=0;l<fullmatches_.size();l++){
		fullmatches_[l]->addMatch(tracklet,tmp);
	      }
	    }

	  }
	  
	} else {

	  //cout << "In disk MatchCalculator disk="<<disk_<<" "<<getName()<<endl;
	  assert(disk_!=0);
	  
	  double pttracklet=0.3*3.8/(tracklet->rinv()*100);
	  bool keep=fabs(1.0/pttracklet-1.0/stub->pt())<ptstubconsistencydiskmatching;

	  if (!keep) continue;

	  double phi=stub->phi();
	  if (phi<0) phi+=two_pi;
	  phi-=phioffset_;
	  
	  double dz=stub->z()-tracklet->zprojdisk(disk_);

	  assert(fabs(dz)<dzmax);

	  int iz=fpgastub->z().value();

	  int iphi=tracklet->fpgaphiprojdisk(disk_).value();

	  int shifttmp=t2bits+tbitshift+phi0bitshift+2-rinvbitshiftdisk-phiderdiskbitshift;

	  assert(shifttmp>=0);
	  int iphicorr=(iz*tracklet->fpgaphiprojderdisk(disk_).value())>>shifttmp;

	  iphi+=iphicorr;
	  
	  double phicorr=dz*tracklet->phiprojderdisk(disk_);

	  assert(fabs(tracklet->phiprojderdisk(disk_))<0.1);
	  assert(fabs(phicorr)<0.1);

	  double phiproj=tracklet->phiprojdisk(disk_)+phicorr;
	
	  int ir=tracklet->fpgarprojdisk(disk_).value();

	  int shifttmp2=rprojdiskbitshift+t3shift-rderdiskbitshift;

	  assert(shifttmp2>=0);
	  int ircorr=(iz*tracklet->fpgarprojderdisk(disk_).value())>>shifttmp2;


	  ir+=ircorr;

	  double rcorr=dz*tracklet->rprojderdisk(disk_);

	  double rproj=tracklet->rprojdisk(disk_)+rcorr;

	  int ideltaphi=fpgastub->phi().value()*kphi/kphiproj123-iphi; 

	  double deltar=stub->r()-rproj;

	  int ideltar=(fpgastub->r().value()*krdisk+rmindisk)/krprojshiftdisk-ir;


	  double dr=stub->r()-(tracklet->rprojdisk(disk_)+
			      dz*tracklet->rprojderdisk(disk_));

	  double dphi=phi-(tracklet->phiprojdisk(disk_)+
			   dz*tracklet->phiprojderdisk(disk_));


	  double dphiapprox=phi-(tracklet->phiprojapproxdisk(disk_)+
				 dz*tracklet->phiprojderapproxdisk(disk_));

	  double drapprox=stub->r()-(tracklet->rprojapproxdisk(disk_)+
				    dz*tracklet->rprojderapproxdisk(disk_));
	
	  double alpha=0.0;
	
	  if (stub->r()>60.0) {
	    alpha=stub->alpha(); 	
	    dphi+=dr*alpha;
	    dphiapprox+=drapprox*alpha;
	    ideltaphi+=ideltar*fpgastub->alpha().value()*krprojshiftdisk*kalpha/kphiproj123;  
	  }	



	  double drphicut=0.20;
	  double drcut=0.75;
	  if (stub->r()>60.0) {
	    drcut=3.0;
	    drphicut=0.3;
	  }

	  bool match=(fabs(dphi)<drphicut/stub->r())&&(fabs(deltar)<drcut);

	  bool imatch=(fabs(ideltaphi)<drphicut/(kphiproj123*stub->r()))&&(fabs(ideltar)<drcut/krprojshiftdisk);


	  if (writeDiskMatch1) {

	    static ofstream out1("diskmatch1.txt");

	    out1 << disk_<<" "
		 << phiproj<<" "
		 << rproj<<" "
		 << dphi<<" "
		 << deltar<<"    "
		 << iphi*kphiprojdisk<<" "
		 << ir*krprojshiftdisk<<"  "
		 << ideltaphi*kphiprojdisk<<" "
		 << ideltar*krprojshiftdisk<<" "
		 << endl;

	  }
	  
	  if (dumpmatch) {

	    cout << "DUMPMATCHDISK1 : "<<disk_<<" "
		 <<tracklet->phiprojdisk(disk_)<<" "
		 <<dz<<" "
	      //<<tracklet->phiprojderdisk(disk_)<<" "
		 <<phicorr<<" "
		 <<phiproj<<" | "
		 <<tracklet->rprojdisk(disk_)<<" "
		 <<tracklet->rprojderdisk(disk_)<<" "
		 <<rcorr<<" "
		 <<rproj<<" dphi= "
		 <<dphi<<" dr= "
		 <<deltar<<" "
		 <<match
		 <<endl;
	    cout << "DUMPMATCHDISK2 : "<<disk_<<" "
		 <<tracklet->fpgaphiprojdisk(disk_).value()*kphiproj123<<" "
		 <<iz*kzdisk<<" "
	      //<<tracklet->fpgaphiprojderdisk(disk_).value()*kphiprojderdiskshift<<" "
		 <<iphicorr*kphiproj123<<" ("<<iphicorr<<") "
		 <<iphi*kphiproj123<<" | "
		 <<tracklet->fpgarprojdisk(disk_).value()*krprojshiftdisk<<" "
		 <<tracklet->fpgarprojderdisk(disk_).value()*krprojderdiskshift<<" "
		 <<ircorr*krprojshiftdisk<<" "
		 <<ir*krprojshiftdisk<<" <> "
		 <<ideltaphi*kphiproj123<<" "
		 <<ideltar*krprojshiftdisk<<" "
		 <<imatch
		 <<endl;
	  }
	  
	  
	  if (imatch) {

	    //cout << "2: ideltaphi = "<<ideltaphi<<endl;

	    std::pair<FPGAStub*,L1TStub*> tmp(fpgastub,stub);

	    tracklet->addMatchDisk(disk_,ideltaphi,ideltar,
				   dphi,dr,dphiapprox,drapprox,
				   stub->alphatruncated(),
				   fpgastub->stubindex().value()+((fpgastub->fedregion()-1)<<6),
				   stub->z(),tmp);

	    //cout << "Accepted full match in disk " <<getName()<<" "<<tracklet<<endl;
	    
	    if (tracklet->plusNeighborDisk(disk_)){
	      fullmatchesToMinus_->addMatch(tracklet,tmp);
	    } else if (tracklet->minusNeighborDisk(disk_)) {
	      fullmatchesToPlus_->addMatch(tracklet,tmp);
	    } else {
	      for (unsigned int l=0;l<fullmatches_.size();l++){
		fullmatches_[l]->addMatch(tracklet,tmp);
	      }
	    }
	  }
	}
 	if (countall>=NMAXMC) break;
      }
      if (countall>=NMAXMC) break;
    }

  }
    
private:

  int layer_;
  int disk_;
  double phimin_;
  double phimax_;
  double phioffset_;

  FPGAAllStubs* allstubs_;
  FPGAAllProjections* allprojs_;

  vector<FPGACandidateMatch*> matches_;

  vector<FPGAFullMatch*> fullmatches_;
  FPGAFullMatch* fullmatchesToPlus_;
  FPGAFullMatch* fullmatchesToMinus_;

};

#endif
