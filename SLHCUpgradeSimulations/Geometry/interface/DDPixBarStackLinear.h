#ifndef DD_PixBarStackLinear_h
#define DD_PixBarStackLinear_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDPixBarStackLinear : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDPixBarStackLinear(); 
  virtual ~DDPixBarStackLinear();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string              idNameSpace; //Namespace of this and ALL sub-parts
  std::string              childName;   //Child name
  int                      number;      //Number of positioning
  double                   theta;       //Direction of translation
  double                   phi;         //  ......
  double                   offset;      //Offset    along (theta,phi) direction
  double                   delta;       //Increment     ................
  std::vector<double>      centre;      //Centre
  std::string              rotMat;      //Rotation matrix
  double                   zoffset;     //Offset of modules in y
  double		   stackoffset; //Offset of modules to compensate for eta
  int			   stackoffsetT;//Period of offset of modules to compensate for eta
};

#endif
