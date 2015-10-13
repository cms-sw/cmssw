#ifndef _FPGATIMER_H_
#define _FPGATIMER_H_
#include <math.h>
#include <sys/time.h>

class FPGATimer{
    
 public:
    
    FPGATimer(){ntimes_=0; ttot_=0.0;ttotsq_=0.0;};
    virtual ~FPGATimer(){};

    void start() {gettimeofday(&tstart_,0);}
    void stop() {
	timeval tstop;
	gettimeofday(&tstop,0);
	float tsec=tstop.tv_sec-tstart_.tv_sec;
	float tusec=tstop.tv_usec-tstart_.tv_usec;
	if (tusec<0){
	    tusec+=1000000.0;
	    tsec-=1.0;
	}
	double tmp=tsec+tusec/1000000.0;
	ttot_+=tmp;
	ttotsq_+=tmp*tmp;
	ntimes_++;
    }

    unsigned int ntimes() const {return ntimes_;}
    double  avgtime() const {return ttot_/ntimes_;}
    double  rms() const {return sqrt((ttot_*ttot_-ttotsq_))/ntimes_;}    
    double tottime() const {return ttot_;}

 private:

    unsigned int ntimes_;
    double ttot_;
    double ttotsq_;

    timeval tstart_;

};

#endif
