#ifndef SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2TimeReconstructor_h
#define SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2TimeReconstructor_h

#include <vector>
#include <cstdint>

class EcalEBPhase2TPGTimeWeightIdMap;
class EcalTPGWeightGroup;

/** \class EcalEBPhase2TimeReconstructor
\author L. Lutton, N. Marinelli - Univ. of Notre Dame
 Description: forPhase II 
 Measures the timing of a xTal signal
*/

class EcalEBPhase2TimeReconstructor {
private:
  static const int maxSamplesUsed_ = 12;
  bool debug_;
  int inputsAlreadyIn_;
  int buffer_[maxSamplesUsed_];
  int weights_[maxSamplesUsed_];
  uint64_t ampIn_[maxSamplesUsed_];
  int shift_;
  bool extraShift_[2] = {false, false};
  int setInput(int input);
  void process();
  int processedOutput_;

  // The array invAmpPar is pre-calulated, at least for now since it has shown to be stable. We might decide at a later stage
  // to make the calculation dynamic in CMSSW.
  // Here some explnation of what this LUT is.

  //The sum of the digis multiplied by the time weight coefficients gives dT*A, where A is the amplitude.
  //So to get dT the amplitude, calculated in EcalEBPhase2AmplitudeReconstructor, must be divided off of the result here.
  //However, when this is implemented into the BCP hardware (of which we are emulating the behaviour),
  //division is not a trivial operation, often requiring relatively large
  //amounts of time and resources to complete.
  //To optimize this operation, we instead use an approximation of that division via a lookup table (LUT).
  //Think about the division as a multiplication of 1/A, then the LUT is filled with all the possible values of 1/A,
  //precalculated, in the range that A takes and A itself is used to index the LUT.
  //The element received is then multiplied by the dT*A result in order to get a measurement of the timing.

  //Another limitation of hardware is that we do not use floating point numbers.
  //So, instead, each element of the LUT is bit shifted to the left by some amount sufficient to make every element
  //of the LUT large enough such that converting it to an integer doesn't lead to a loss in performance
  //(this bit shift is undone after the multiplication by a corresponding bit shift to the right).
  //Another method of approximation used here is that not every element in A's range is included in the LUT.
  //Instead, every 8th element is used, since the difference between, e.g., dividing by 1000 and 1008 is generally
  //small whereas it can be a significant save in time and resources in hardware to use a smaller LUT.
  //Note that this requires the indexing amplitude to be bit shifted by 3 to the right to compensate for the smaller size of the LUT.
  //Finally, dT will be in units of ns, but to convert it to ps each element of the LUT is multiplied by 1000.

  //The pre-calculation of the LUT is given by:
  //invAmpAr_ = [1000*invMult] #Since 1/0 is undefined, the array starts with 1/1. The rest of the elements will be filled in the following loop:
  //for i in range(8,4096,8): #loop is set to do every 8th number, so 1/8, 1/16, 1/24, etc. 4096 is the expected range of A
  //invAmpAr_.append(round((ns_to_ps_conv/i)*invMult))
  //Where ns_to_ps_conv = 1000 #this is to convert the resulting dT from units of ns to units of ps
  //invMult = 2**18 #Acts as a shift by 18 bits that is done to the fractions to make them integers instead of floats.

  uint64_t invAmpAr_[512] = {
      262144000, 32768000, 16384000, 10922667, 8192000, 6553600, 5461333, 4681143, 4096000, 3640889, 3276800, 2978909,
      2730667,   2520615,  2340571,  2184533,  2048000, 1927529, 1820444, 1724632, 1638400, 1560381, 1489455, 1424696,
      1365333,   1310720,  1260308,  1213630,  1170286, 1129931, 1092267, 1057032, 1024000, 992970,  963765,  936229,
      910222,    885622,   862316,   840205,   819200,  799220,  780190,  762047,  744727,  728178,  712348,  697191,
      682667,    668735,   655360,   642510,   630154,  618264,  606815,  595782,  585143,  574877,  564966,  555390,
      546133,    537180,   528516,   520127,   512000,  504123,  496485,  489075,  481882,  474899,  468114,  461521,
      455111,    448877,   442811,   436907,   431158,  425558,  420103,  414785,  409600,  404543,  399610,  394795,
      390095,    385506,   381023,   376644,   372364,  368180,  364089,  360088,  356174,  352344,  348596,  344926,
      341333,    337814,   334367,   330990,   327680,  324436,  321255,  318136,  315077,  312076,  309132,  306243,
      303407,    300624,   297891,   295207,   292571,  289982,  287439,  284939,  282483,  280068,  277695,  275361,
      273067,    270810,   268590,   266407,   264258,  262144,  260063,  258016,  256000,  254016,  252062,  250137,
      248242,    246376,   244537,   242726,   240941,  239182,  237449,  235741,  234057,  232397,  230761,  229147,
      227556,    225986,   224438,   222912,   221405,  219919,  218453,  217007,  215579,  214170,  212779,  211406,
      210051,    208713,   207392,   206088,   204800,  203528,  202272,  201031,  199805,  198594,  197398,  196216,
      195048,    193893,   192753,   191626,   190512,  189410,  188322,  187246,  186182,  185130,  184090,  183061,
      182044,    181039,   180044,   179060,   178087,  177124,  176172,  175230,  174298,  173376,  172463,  171560,
      170667,    169782,   168907,   168041,   167184,  166335,  165495,  164663,  163840,  163025,  162218,  161419,
      160627,    159844,   159068,   158300,   157538,  156785,  156038,  155299,  154566,  153840,  153121,  152409,
      151704,    151005,   150312,   149626,   148945,  148271,  147604,  146942,  146286,  145636,  144991,  144352,
      143719,    143092,   142470,   141853,   141241,  140635,  140034,  139438,  138847,  138262,  137681,  137105,
      136533,    135967,   135405,   134848,   134295,  133747,  133203,  132664,  132129,  131598,  131072,  130550,
      130032,    129518,   129008,   128502,   128000,  127502,  127008,  126517,  126031,  125548,  125069,  124593,
      124121,    123653,   123188,   122727,   122269,  121814,  121363,  120915,  120471,  120029,  119591,  119156,
      118725,    118296,   117871,   117448,   117029,  116612,  116199,  115788,  115380,  114975,  114573,  114174,
      113778,    113384,   112993,   112605,   112219,  111836,  111456,  111078,  110703,  110330,  109960,  109592,
      109227,    108864,   108503,   108145,   107789,  107436,  107085,  106736,  106390,  106045,  105703,  105363,
      105026,    104690,   104357,   104025,   103696,  103369,  103044,  102721,  102400,  102081,  101764,  101449,
      101136,    100825,   100515,   100208,   99902,   99599,   99297,   98997,   98699,   98402,   98108,   97815,
      97524,     97234,    96947,    96661,    96376,   96094,   95813,   95534,   95256,   94980,   94705,   94432,
      94161,     93891,    93623,    93356,    93091,   92827,   92565,   92304,   92045,   91787,   91531,   91276,
      91022,     90770,    90519,    90270,    90022,   89775,   89530,   89286,   89043,   88802,   88562,   88323,
      88086,     87850,    87615,    87381,    87149,   86918,   86688,   86459,   86232,   86005,   85780,   85556,
      85333,     85112,    84891,    84672,    84454,   84237,   84021,   83806,   83592,   83379,   83168,   82957,
      82747,     82539,    82332,    82125,    81920,   81716,   81512,   81310,   81109,   80909,   80709,   80511,
      80314,     80117,    79922,    79727,    79534,   79341,   79150,   78959,   78769,   78580,   78392,   78205,
      78019,     77834,    77649,    77466,    77283,   77101,   76920,   76740,   76561,   76382,   76205,   76028,
      75852,     75677,    75502,    75329,    75156,   74984,   74813,   74642,   74473,   74304,   74136,   73968,
      73802,     73636,    73471,    73306,    73143,   72980,   72818,   72656,   72496,   72336,   72176,   72018,
      71860,     71702,    71546,    71390,    71235,   71080,   70926,   70773,   70621,   70469,   70318,   70167,
      70017,     69868,    69719,    69571,    69424,   69277,   69131,   68985,   68840,   68696,   68552,   68409,
      68267,     68125,    67983,    67843,    67702,   67563,   67424,   67285,   67148,   67010,   66873,   66737,
      66602,     66467,    66332,    66198,    66065,   65932,   65799,   65667,   65536,   65405,   65275,   65145,
      65016,     64887,    64759,    64631,    64504,   64377,   64251,   64125};

public:
  EcalEBPhase2TimeReconstructor(bool debug);
  virtual ~EcalEBPhase2TimeReconstructor();
  virtual void process(std::vector<int> &addout, std::vector<int> &ampRecoOutput, std::vector<int64_t> &output);
  void setParameters(uint32_t raw,
                     const EcalEBPhase2TPGTimeWeightIdMap *ecaltpgTimeWeightMap,
                     const EcalTPGWeightGroup *ecaltpgWeightGroup);
};

#endif
