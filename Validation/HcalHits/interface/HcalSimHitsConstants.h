/* 
   Defines constants used to define SimHits histograms 
   September 2016 - K. Call
*/

//Defines constants used for ieta plots. Centered on integers.
//Give space at high ieta to make it clear that all bins are plotted.

const int    IETA_HIGH_ =  42;
const int    IETA_LOW_  = -42;

const int    IPHI_HIGH_ =  72;
const int    IPHI_LOW_  =   1;

const int    IETA_BINS_ =  IETA_HIGH_ - IETA_LOW_ + 1; //  85
const double IETA_MIN_  =  (double)IETA_LOW_  - 0.5;   // -42.5
const double IETA_MAX_  =  (double)IETA_HIGH_ + 0.5;   //  42.5

const int    IPHI_BINS_ =  IPHI_HIGH_ - IPHI_LOW_ + 1; //  72
const double IPHI_MIN_  =  (double)IPHI_LOW_  - 0.5;   //   0.5 
const double IPHI_MAX_  =  (double)IPHI_HIGH_ + 0.5;   //  72.5
