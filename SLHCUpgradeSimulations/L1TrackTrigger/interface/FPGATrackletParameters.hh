// This class holds the tracklet parameters for the selected stub pairs 
//This class owns the tracklets. Furhter modules only holds pointers
#ifndef FPGATRACKLETPARAMETERS_H
#define FPGATRACKLETPARAMETERS_H

#include "FPGATracklet.hh"
#include "FPGAMemoryBase.hh"

using namespace std;

class FPGATrackletParameters:public FPGAMemoryBase{

public:

  FPGATrackletParameters(string name, unsigned int iSector, 
			 double phimin, double phimax):
    FPGAMemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addTracklet(FPGATracklet* tracklet) {
    //static int count=0;
    //count++;
    //cout <<"count = "<<count<<" "<<sizeof(FPGATracklet)
    //	 <<" "<<count*sizeof(FPGATracklet)<<endl;
    tracklets_.push_back(tracklet);
  }

  unsigned int nTracklets() const {return tracklets_.size();}

  FPGATracklet* getFPGATracklet(unsigned int i) const {return tracklets_[i];}

  void clean() {
    for(unsigned int i=0;i<tracklets_.size();i++){
      delete tracklets_[i];
    }
    tracklets_.clear();
  }

  void writeTPAR(bool first) {

    std::string fname="TrackletParameters_";
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

    for (unsigned int j=0;j<tracklets_.size();j++){
      string tpar=tracklets_[j]->trackletparstr();
      if (j<16) out_ <<"0";
      out_ << hex << j << dec ;
      out_ <<" "<<tpar<< endl;
    }
    out_.close();

    bx_++;
    event_++;
    if (bx_>7) bx_=0;

  }


private:

  double phimin_;
  double phimax_;
  std::vector<FPGATracklet*> tracklets_;

};

#endif
