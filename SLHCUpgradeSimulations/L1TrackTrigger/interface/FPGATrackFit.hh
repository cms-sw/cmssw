//This class stores the track fit
#ifndef FPGATRACKFIT_H
#define FPGATRACKFIT_H

#include "FPGATracklet.hh"
#include "FPGAMemoryBase.hh"

using namespace std;

class FPGATrackFit:public FPGAMemoryBase{

public:

  FPGATrackFit(string name, unsigned int iSector, 
	       double phimin, double phimax):
    FPGAMemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addTrack(FPGATracklet* tracklet) {
    tracks_.push_back(tracklet);
  }

  unsigned int nTracks() const {return tracks_.size();}

  void clean() {
    //cout << "Cleaning tracks : "<<tracks_.size()<<endl;
    tracks_.clear();
  }

  bool foundTrack(ofstream& outres, L1SimTrack simtrk){
    bool match=false;
    double phioffset=phimin_-(phimax_-phimin_)/6.0;
    for(unsigned int i=0;i<tracks_.size();i++){
      match=match||tracks_[i]->foundTrack(simtrk);
      if (tracks_[i]->foundTrack(simtrk)) {
	FPGATracklet* tracklet=tracks_[i];
	int charge = simtrk.id()/abs(simtrk.id());
	if(abs(simtrk.id())<100) charge = -charge; 
	double simphi=simtrk.phi();
	if (simphi<0.0) simphi+=two_pi; 
	int irinv=tracklet->irinvfit().value();
	if (irinv==0) irinv=1;
	int layerordisk=-1;
	if (tracklet->isBarrel()) {
	  layerordisk=tracklet->layer();
	} else {
	  layerordisk=tracklet->disk();
	}
	outres << layerordisk
	       <<" "<<tracklet->nMatches()
	       <<" "<<simtrk.pt()*charge
	       <<" "<<simphi
	       <<" "<<simtrk.eta()
	       <<" "<<simtrk.vz()
	       <<"   "
	       <<(0.3*3.8/100.0)/tracklet->rinvfit()
	       <<" "<<tracklet->phi0fit()+phioffset
	       <<" "<<asinh(tracklet->tfit())
	       <<" "<<tracklet->z0fit()
	       <<"   "
	       <<(0.3*3.8/100.0)/tracklet->rinvfitexact()
	       <<" "<<tracklet->phi0fitexact()+phioffset
	       <<" "<<asinh(tracklet->tfitexact())
	       <<" "<<tracklet->z0fitexact()
		 <<"   "
	       <<(0.3*3.8/100.0)/(irinv*krinvpars)
	       <<" "<<tracklet->iphi0fit().value()*kphi0pars+phioffset
	       <<" "<<asinh(tracklet->itfit().value()*ktpars)
	       <<" "<<tracklet->iz0fit().value()*kz
	       <<"   "
	       <<(0.3*3.8/100.0)/(1e-20+tracklet->fpgarinv().value()*krinvpars)
	       <<" "<<tracklet->fpgaphi0().value()*kphi0pars+phioffset
	       <<" "<<asinh(tracklet->fpgat().value()*ktpars)
	       <<" "<<tracklet->fpgaz0().value()*kz
	       <<"               "
	       <<(0.3*3.8/100.0)/(1e-20+tracklet->rinvapprox())
	       <<" "<<tracklet->phi0approx()+phioffset
	       <<" "<<asinh(tracklet->tapprox())
	       <<" "<<tracklet->z0approx()
	       <<endl;
      }
    }
    return match;
  }
  void writeTF(bool first) {

    std::string fname="TrackFit_";
    fname+=getName();
    fname+="_";
    ostringstream oss;
    oss << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=oss.str();
    fname+=".dat";
    if (first) {
      bx_=0;
      event_=1;
      out_.open(fname.c_str());
    }
    else
      out_.open(fname.c_str(),std::ofstream::app);

    out_ << "BX = "<<(bitset<3>)bx_ << " Event : " << event_ << endl;

    unsigned long int uu;
    for (unsigned int j=0;j<tracks_.size();j++){
      uu = (((long int)tracks_[j]->irinvfit().value()&32767)<<44)|
	(((long int)tracks_[j]->iphi0fit().value()&524287)<<25)|
	(((long int)tracks_[j]->itfit().value()&16383)<<11)|
	((long int)tracks_[j]->iz0fit().value()&2047);
      out_<<"0000000000000000";
      out_.fill('0');
      out_.width(16);
      out_<<std::hex<<uu;
      out_<<"\n";
    }
    out_.close();

    bx_++;
    event_++;
    if (bx_>7) bx_=0;

  }

private:

  double phimin_;
  double phimax_;
  std::vector<FPGATracklet*> tracks_;

};

#endif
