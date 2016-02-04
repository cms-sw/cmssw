//babar #include "BaBar/BaBar.hh"
//babar #include <math.h>
//babar #include "DcxReco/Dcxprobab.hh"
#include <cmath>
#include "RecoTracker/RoadSearchHelixMaker/interface/Dcxprobab.hh"
extern float Dcxprobab(int& ndof, float& chisq){

  //constants
  static float srtopi=2.0/sqrt(2.0*M_PI);
  static float upl=100.0;

  float prob=0.0;
  if(ndof<=0) {return prob;}
  if(chisq<0.0) {return prob;}
  if(ndof<=60) {
    //full treatment
    if(chisq>upl) {return prob;}
    float sum=exp(-0.5*chisq);
    float term=sum;

    int m=ndof/2;
    if(2*m==ndof){
      if(m==1){return sum;}
      for(int i=2; i<=m;i++){
	term=0.5*term*chisq/(i-1);
	sum+=term;
      }//(int i=2; i<=m)
      return sum;
      //even

    }else{
      //odd
      float srty=sqrt(chisq);
      float temp=srty/M_SQRT2;
      prob=erfc(temp);
      if(ndof==1) {return prob;}
      if(ndof==3) {return (srtopi*srty*sum+prob);}
      m=m-1;
      for(int i=1; i<=m; i++){
	term=term*chisq/(2*i+1);
	sum+=term;
      }//(int i=1; i<=m; i++)
      return (srtopi*srty*sum+prob);

    }//(2*m==ndof)

  }else{
    //asymtotic Gaussian approx
    float srty=sqrt(chisq)-sqrt(ndof-0.5);
    if(srty<12.0) {prob=0.5*erfc(srty);};
    return prob;

  }//ndof<30

  // cannot reach following line; warning on OSF; so comment it out
  // return prob;
}//endof Dcxprobab
