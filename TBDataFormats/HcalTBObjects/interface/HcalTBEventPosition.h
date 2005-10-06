#ifndef HCALTBEVENTPOSITION_H
#define HCALTBEVENTPOSITION_H 1

#include <string>
#include <iostream>
#include <vector>
#include "boost/cstdint.hpp"

  /** \class HcalTBEventPosition
      
  $Date: 2005/08/23 01:07:18 $
  $Revision: 1.1 $
  \author P. Dudero - Minnesota
  */
  class HcalTBEventPosition {
  public:
    HcalTBEventPosition();

    // Getter methods
    double hfTableX()       const { return hfTableX_;       }
    double hfTableY()       const { return hfTableY_;       }
    double hfTableV()       const { return hfTableV_;       }
    double hbheTableEta()   const { return hbheTableEta_;   }
    double hbheTablePhi()   const { return hbheTablePhi_;   }

    void   getChamberHits     ( char chamberch, // 'A','B','C','D', or 'E'
				std::vector<double>& xvec,
				std::vector<double>& yvec  ) const;
				      
    // Setter methods
    void   setHFtableCoords   ( double x, 
				double y,
				double v                        );
    void   setHBHEtableCoords ( double eta,
				double phi                      );
    void   setChamberHits     ( char chamberch,
				const std::vector<double>& xvec,
				const std::vector<double>& yvec  );

  private:
    double hfTableX_, hfTableY_, hfTableV_;
    double hbheTableEta_, hbheTablePhi_;

    std::vector<double> ax_,ay_,bx_,by_,cx_,cy_,dx_,dy_,ex_,ey_;
  };

  std::ostream& operator<<(std::ostream& s, const HcalTBEventPosition& htbep);

#endif
