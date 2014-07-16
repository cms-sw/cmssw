#include "SimDataFormats/Track/interface/CoreSimTrack.h"

const float onethird = 1./3.;
const float twothird = 2./3.;
const float chg[109] = {
 -onethird,twothird,-onethird,twothird,-onethird,twothird,-onethird,twothird,0,0,
 -1,0,-1,0,-1,0,-1,0,0,0,
  0,0, 0,1, 0,0, 0,0,0,0,
  1,0, 1,2, 0,0, 1,2,0,0,
 -onethird,twothird,-onethird,twothird,-onethird,twothird,0,0,0,0,
 -1,0,-1,0,-1,0, 0,0,0,0,
 -onethird,twothird,-onethird,twothird,-onethird,twothird,0,0,0,0,
 -1,0,-1,0,-1,0, 1,1,0,0,
  0,0, 0,0, 0,0, 0,0,0,0,
  0,0, 0,0, 0,0, 0,0,0,0,
  0,0, 0,0, 0,0, 0,0,0};

float CoreSimTrack::charge() const { 
  float hepchg = 0;
  if(thePID != 0) {
    int kqa = std::abs(thePID);
    if(kqa < 10000000) {
      //... direct translation
      if(kqa<=100) {hepchg = chg[kqa-1];}
      //... deuteron or tritium
      else if(kqa==100 || kqa==101) {hepchg = -1;}
      //... alpha or He3
      else if(kqa==102 || kqa==104) {hepchg = -2;}
      else if(kqa%10 != 0) {
	int kqx=kqa/1000000%10;
	int kq3=kqa/1000%10;
	int kq2=kqa/100%10;
	int kq1=kqa/10%10;
	int irt=kqa%10000;
	if(kqx>0 && irt<100) {
	  hepchg = chg[irt-1];
	  if(kqa==5100061 || kqa==5100062) {hepchg = 2;}
	} else if(kq3==0) {
	  //   Construction from quark content for heavy meson, 
	  //   diquark, baryon, mesons.
	  hepchg = chg[kq2-1]-chg[kq1-1];
	  //...Strange or beauty mesons.
	  if((kq2==3) || (kq2==5)) {hepchg = chg[kq1-1]-chg[kq2-1];}
	} else if(kq1 == 0) {
	  //...Diquarks.
	  hepchg = chg[kq3-1] + chg[kq2-1];
	} else {
	  //...Baryons
	  hepchg = chg[kq3-1]+chg[kq2-1]+chg[kq1-1];
	}
      }
      //... fix sign of charge
      if(thePID<0) {hepchg = -hepchg;}
    }
  }
  return hepchg;
}

std::ostream & operator <<(std::ostream & o , const CoreSimTrack& t) 
{
    o << t.type() << ", ";
    o << t.momentum();
    return o;
}
