

#include <map>
#include <cmath>

using namespace std;

void rangeAngle(int wheel, int station, int sl, float& min, float& max) {

  
  //   W       St
  map<int, map<int, float> > m_min;
  map<int, map<int, float> > m_max;

// map<float, map<float, float> > m_min;
//  map<float, map<float, float> > m_max;

  if (sl !=2 ) { // phi
    m_min[2][3] = -0.18;
    m_min[2][2] = -0.26;
    m_min[2][1] = -0.1;
    m_min[1][3] = -0.2;
    m_min[1][2] = -0.25;
    m_min[1][1] = -0.1;
    m_min[0][3] = -0.2;
    m_min[0][2] = -0.2;
    m_min[0][1] = -0.2;
   

    m_max[2][3] = 0.22;
    m_max[2][2] = 0.16;
    m_max[2][1] = 0.28;
    m_max[1][3] = 0.25;
    m_max[1][2] = 0.18;
    m_max[1][1] = 0.28;
    m_max[0][3] = 0.2;
    m_max[0][2] = 0.2;
    m_max[0][1] = 0.2;
   

  } else { //theta
    
    m_min[0][1] = -0.25;
    m_min[0][2] = -0.2;
    m_min[0][3] = -0.18;
    m_min[1][1] =  0.36;
    m_min[1][2] =  0.3;
    m_min[1][3] =  0.25;
    m_min[2][1] =  0.78;
    m_min[2][2] =  0.7;
    m_min[2][3] =  0.6;

    m_max[0][1] = 0.25;
    m_max[0][2] = 0.2;
    m_max[0][3] = 0.18;
    m_max[1][1] = 0.7;
    m_max[1][2] = 0.63;
    m_max[1][3] = 0.55;
    m_max[2][1] = 0.96;
    m_max[2][2] = 0.88;
    m_max[2][3] = 0.79;

 
 }
 
    min = m_min[wheel][station];
    max = m_max[wheel][station];
    

  //  std::cout << "min" << min << endl;
  //std::cout << "max" << max << endl;

}

