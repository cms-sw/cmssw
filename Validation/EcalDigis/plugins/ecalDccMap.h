#include <iostream>

template <class T>
T mod(const T& a, const T& b) {
  T c = a % b;
  return c < 0 ? c + b : c;
}

static const char endcapDccMap[401] = {
    "       777777       "
    "    666777777888    "
    "   66667777778888   "
    "  6666667777888888  "
    " 666666677778888888 "
    " 566666677778888880 "  //    Z
    " 555666667788888000 "  //     x-----> X
    "55555566677888000000"  //     |
    "555555566  880000000"  //     |
    "55555555    00000000"  //_          //     |
    "55555554    10000000"  //     V Y
    "554444444  111111100"
    "44444444332211111111"
    " 444444333222111111 "
    " 444443333222211111 "
    " 444433333222221111 "
    "  4443333322222111  "
    "   43333332222221   "
    "    333333222222    "
    "       333222       "};

/** Gets the phi index of the DCC reading a RU (SC or TT)
 * @param iDet 0 for EE-, 1 for EB, 2 for EE+
 * @param i iEta or iX
 * @param j iPhi or iY
 * @return DCC phi index between 0 and 8 for EE
 * and between 0 and 17 for EB
 */
inline int dccPhiIndexOfRU(int iDet, int i, int j) {
  if (iDet == 1) {  //barrel
    //iEta=i, iPhi=j
    //phi edge of a SM is 4 TT
    return (j + 2) / 4;
  }
  char flag = endcapDccMap[i + j * 20];
  return (flag == ' ') ? -1 : (flag - '0');
}

/** Gets the phi index of the DCC reading a crystal
 * @param iDet 0 for EE-, 1 for EB, 2 for EE+
 * @param i iEta or iX
 * @param j iPhi or iY
 * @return DCC phi index between 0 and 8 for EE
 * and between 0 and 17 for EB
 */
inline int dccPhiIndex(int iDet, int i, int j) { return dccPhiIndexOfRU(iDet, i / 5, j / 5); }

/** Gets the index of the DCC reading a crystal
 * @param iDet 0 for EE-, 1 for EB, 2 for EE+
 * @param i iEta or iX
 * @param j iPhi or iY
 * @return DCC index between 0 and 53 
 */
inline int dccIndex(int iDet, int i, int j) {
  if (iDet == 1) {  //barrel
    //a SM is 85 crystal long:
    int iEtaSM = i / 85;
    //a SM is 20 crystal wide:
    int iPhiSM = (j + 10) / 20;
    //DCC numbers start at 9 in the barrel and there 18 DCC/SM
    return 9 + 18 * iEtaSM + iPhiSM;
  }
  int iPhi = dccPhiIndex(iDet, i, j);
  if (iPhi < 0)
    return -1;
  //34 DCCs in barrel and 8 in EE-=>in EE+ DCC numbering starts at 45,
  //iDet/2 is 0 for EE- and 1 for EE+:
  return iPhi + iDet / 2 * 45;
}

/** Gets the index of the DCC reading a crystal
 * @param iDet 0 for EE-, 1 for EB, 2 for EE+
 * @param i iEta (staring at eta=-1.48)  or iX
 * @param j iPhi or iY
 * @return DCC index between 0 and 53 
 */
inline int dccIndexOfRU(int iDet, int i, int j) {
  if (iDet == 1) {  //barrel
    //a SM is 17 RU long:
    int iEtaSM = i / 17;
    //a SM is 4 RU wide:
    int iPhiSM = (j + 2) / 4;
    //DCC numbers start at 9 in the barrel and there 18 DCC/SM
    return 9 + 18 * iEtaSM + iPhiSM;
  }
  int iPhi = dccPhiIndexOfRU(iDet, i, j);
  if (iPhi < 0)
    return -1;
  //34 DCCs in barrel and 8 in EE-=>in EE+ DCC numbering starts at 45,
  //iDet/2 is 0 for EE- and 1 for EE+:
  return iPhi + iDet / 2 * 45;
}

inline int abOfDcc(int iDCC) {
  if (iDCC < 0 || iDCC > 54)
    return -1;
  if (iDCC < 9) {  //EE-
    return iDCC / 3;
  } else if (iDCC < 27) {  //EB-
    //an EB AB is made of 6 DCCs,
    //first EB- AB is numbered 3
    //and "1st" DCC of AB 3 is DCC 26
    //(AB 3 made of DCCs 26,9,10,11,12,13):
    return 3 + mod(iDCC - 26, 18) / 6;
  } else if (iDCC < 45) {  //EB+
    //an EB AB is made of 6 DCCs,
    //first EB+ AB is numbered 6
    //and "1st" DCC of AB6 is DCC 44
    //(AB 6 made of DCCs 44,27,28,29,30,31):
    return 6 + mod(iDCC - 44, 18) / 6;
  } else {  //EE+
    //AB numbering starts at DCC=45 and runs along phi in increasing phi
    //first EE+ AB is numbered 9:
    return 9 + (iDCC - 45) / 3;
  }
}
