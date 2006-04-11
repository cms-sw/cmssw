#ifndef TrackingTools_FakeField_h
#define TrackingTools_FakeField_h
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/Math/interface/Point3D.h"

namespace  TrackingTools{
  namespace FakeField {
    class Field{
      public:
//       Field( MagneticField* aField): theField(aField) {};
      static  GlobalVector inTesla( const GlobalPoint& pos)
      	 {return GlobalVector(0, 0, 4);}
      static  GlobalVector inTesla( const math::XYZPoint& pos)
       	 {return GlobalVector(0, 0, 4);}
      static  MagneticField* field() {return theField;}
      static  GlobalVector inGeVPerCentimeter( const GlobalPoint& pos)
      	 {return inTesla(pos) * 2.99792458e-3;}
      static  GlobalVector inGeVPerCentimeter( const math::XYZPoint& pos)
       	 {return inTesla(pos) * 2.99792458e-3;}
     private:
      static MagneticField* theField;
    };
  }
}
#endif
