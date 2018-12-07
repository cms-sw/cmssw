#ifndef __SimFastTiming_FastTimingCommon_MTDShapeBase_h__
#define __SimFastTiming_FastTimingCommon_MTDShapeBase_h__

#include <vector>
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

      std::array<float,3> timeAtThr(const float scale, 
				    const float threshold1, 
				    const float threshold2) const;

      static constexpr unsigned int kReadoutTimeInterval = 28;    // in nsec
      static constexpr unsigned int kNBinsPerNSec        = 100;   // granularity of internal array
      static constexpr unsigned int k1NSecBinsTotal      = kReadoutTimeInterval*kNBinsPerNSec;


   protected:

      unsigned int timeIndex( double aTime ) const;

      void buildMe() ;

      virtual void fillShape( DVec& aVec ) const = 0;


   private:
      
      double linear_interpolation(const double& y,
				  const double& x1, const double& x2,
				  const double& y1, const double& y2) const;

      const double qNSecPerBin_;
      unsigned int indexOfMax_;
      double       timeOfMax_ ;
      DVec         shape_;

};

#endif
