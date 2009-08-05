#ifndef DDPixFwdInnerDisks_h
#define DDPixFwdInnerDisks_h

/* 

== CMS Phase 1 Forward Pixels Geometry ==

 Author:  Pratima Jindal, Purdue University Calumet
         July 2009


== ALGORITHM DESCRIPTION: ==

  Algorithm for placing blades in Inner Disks

== Parameters : ==

  "Endcap" - +1 if placing the child volume into +Z disk, -1 if placing into -Z disk.
  "Child" - name of a child volume being places (should be in the form "file:volume")
            In no child name is given, the algorithm simply calculates Nipple parameters.
  "ChildRotation" - rotation of the child volume with respect to the "blade frame". [OPTIONAL]
  "ChildTranslation" - vector defining translation of the child volume with respect to the 
                       "blade frame". [OPTIONAL]
  "FlagString" - string of 24 characters, used to indicate blades into which the child volume 
                 should be placed. [OPTIONAL]
  "FlagSelector" - 1 character string, key to interpreting "FlagString".
                   Positions in "BladeFlag" that have this character will get the child volume.
                   
  
  Blade frame: origin on the axis of the blade at a distance "ancorRadius" from the beam line
  (it therefore coincides with the ancor point of a blade). 
  Y along blade axis pointing away from beam line, Z perpendicular to blade plane and pointing away from IP.
  (That assumes the axes of ZPlus disk are aligned with CMS global reference frame, and ZMinus disk
  is rotated around Y by 180 degrees.)

== Example of use : ==

<Algorithm name="track:DDPixFwdBlades">
  <rParent name="pixfwdDisk:PixelForwardDiskZMinus"/>
  <Numeric name="Endcap"        value="-1." />
  <String  name="Child"         value="pixfwdPanel:PixelForwardPanel4Left"/>
  <Vector  name="ChildTranslation" type="numeric" nEntries="3"> 0., -[pixfwdPanel:AncorY], [zPanel] </Vector>
  <String  name="ChildRotation" value="pixfwdCommon:Y180"/>
  <String  name="FlagString"    value="LRRRRLRRRRRRLRRRRLRRRRRR" />  <!-- Panel Layout ZMinus 4  -->
  <String  name="FlagSelector"  value="L" />
</Algorithm>

*/

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"

class DDPixFwdInnerDisks : public DDAlgorithm {
 
public:

  // Constructors & Destructor :  --------------------------------------------------------

  DDPixFwdInnerDisks(); 
  virtual ~DDPixFwdInnerDisks();
  
  // Initialization & Execution :  -------------------------------------------------------
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute();
  
  // -------------------------------------------------------------------------------------

private:

  // -- Input geometry parameters :  -----------------------------------------------------
 
  static const int     nBlades;            // Number of blades
  static const double  bladeAngle;    // Angle of blade rotation around axis perpendicular to beam
  static const double  zPlane;             // Common shift in Z for all blades (with respect to disk center plane)
  static const double  bladeZShift;     // Shift in Z between the axes of two adjacent blades
  
  static const double  ancorRadius; // Distance from beam line to ancor point defining center of "blade frame"
  

  // -- Algorithm parameters :  ----------------------------------------------------------

  double        endcap;          // +1 for Z Plus endcap disks, -1 for Z Minus endcap disks

  std::string   flagString;         // String of flags
  std::string   flagSelector;       // Character that means "yes" in flagString
  
  std::string   childName;          // Child volume name
  
  std::vector<double> childTranslationVector; // Child translation with respect to "blade frame"
  std::string   childRotationName;            // Child rotation with respect to "blade frame"

  // -------------------------------------------------------------------------------------

  std::string   idNameSpace;    //Namespace of this and ALL sub-parts
  
  static std::map<std::string, int> copyNumbers;

  // -- Helper functions :  --------------------------------------------------------------
  
  int issueCopyNumber();
  
 // -------------------------------------------------------------------------------------

};

#endif
