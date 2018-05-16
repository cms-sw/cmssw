#ifndef __SimFastTiming_FastTimingCommon_MTDShapeBase_h__
#define __SimFastTiming_FastTimingCommon_MTDShapeBase_h__

#include<vector>
#include <array>

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
  

class MTDShapeBase : public CaloVShape
{
   public:

      typedef std::vector<double> DVec ;
  
      MTDShapeBase() ;

      ~MTDShapeBase() override ;

      double operator() ( double aTime ) const override ;

      unsigned int indexOfMax() const;
      double       timeOfMax()  const;
      double       timeToRise() const override { return 0.; }

      std::array<float,3> timeAtThr(const float Npe, 
				    float threshold1, 
				    float threshold2) const;


      enum { kReadoutTimeInterval = 31  , // in nsec
	     kNBinsPerNSec        = 100 , // granularity of internal array
	     k1NSecBinsTotal      = kReadoutTimeInterval*kNBinsPerNSec };


   protected:

      unsigned int timeIndex( double aTime ) const;

      void buildMe() ;

      virtual void fillShape( DVec& aVec ) const = 0;


   private:
      
      const double qNSecPerBin_;
      unsigned int indexOfMax_;
      double       timeOfMax_ ;
      DVec         shape_;

};

#endif
