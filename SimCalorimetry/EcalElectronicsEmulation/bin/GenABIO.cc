// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-

/** \file
 * GenABIO is a standalone program to produce individual SRP card trigger tower
 * and selective readout action flags from TTF.txt and SRF.txt global flag
 * files.
 * Run 'GenABIO -h' for usage.
 */

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "ecalDccMap.h"

#if !defined(__linux__) && !(defined(__APPLE__) && __DARWIN_C_LEVEL >= 200809L)
#include <errno.h>
/* getline implementation is copied from glibc. */

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif
#ifndef SSIZE_MAX
#define SSIZE_MAX ((ssize_t)(SIZE_MAX / 2))
#endif
namespace {
  ssize_t getline(char **lineptr, size_t *n, FILE *fp) {
    ssize_t result;
    size_t cur_len = 0;

    if (lineptr == NULL || n == NULL || fp == NULL) {
      errno = EINVAL;
      return -1;
    }

    if (*lineptr == NULL || *n == 0) {
      *n = 120;
      *lineptr = (char *)malloc(*n);
      if (*lineptr == NULL) {
        result = -1;
        goto end;
      }
    }

    for (;;) {
      int i;

      i = getc(fp);
      if (i == EOF) {
        result = -1;
        break;
      }

      /* Make enough space for len+1 (for final NUL) bytes.  */
      if (cur_len + 1 >= *n) {
        size_t needed_max = SSIZE_MAX < SIZE_MAX ? (size_t)SSIZE_MAX + 1 : SIZE_MAX;
        size_t needed = 2 * *n + 1; /* Be generous. */
        char *new_lineptr;

        if (needed_max < needed)
          needed = needed_max;
        if (cur_len + 1 >= needed) {
          result = -1;
          goto end;
        }

        new_lineptr = (char *)realloc(*lineptr, needed);
        if (new_lineptr == NULL) {
          result = -1;
          goto end;
        }

        *lineptr = new_lineptr;
        *n = needed;
      }

      (*lineptr)[cur_len] = i;
      cur_len++;

      if (i == '\n')
        break;
    }
    (*lineptr)[cur_len] = '\0';
    result = cur_len ? (ssize_t)cur_len : result;

  end:
    return result;
  }
}  // namespace
#endif

using namespace std;

/** Range of the x-index of endcap crystals (xman-xmin+1).
 */
const static int nEndcapXBins = 100;
/** Range of the y-index of endcap crystals (yman-ymin+1).
 */
const static int nEndcapYBins = 100;
/** Edge size of a supercrystal. A supercrystal is a tower of 5x5 crystals.
 */
const static int supercrystalEdge = 5;
/** Range of endcap supercrystal x-index (xmax-xmin+1)
 */
const static int nSupercrystalXBins = nEndcapXBins / supercrystalEdge;
/** Range of endcap supercrystal y-index (ymay-ymin+1)
 */
const static int nSupercrystalYBins = nEndcapYBins / supercrystalEdge;
/** Number of endcap, obviously tow
 */
const static int nEndcaps = 2;
/** Number of trigger towers along eta in one endcap
 */
const static int nEndcapTTInEta = 11;
/** Number of barrel trigger towers along eta
 */
const static int nBarrelTTInEta = 34;
/** Number of trigger towers along eta for the whole ECAL
 */
const static int nTTInEta = 2 * nEndcapTTInEta + nBarrelTTInEta;
/** Number of trigger towers in an eta ring
 */
const static int nTTInPhi = 72;
/** Number of ABs in a phi-sector
 */
const int nABInEta = 4;
/** Number of ABs in an eta slice
 */
const int nABInPhi = 3;
/** Number of DCCs in an endcap
 */
const int nDCCEE = 9;
const int nABABCh = 8;    // nbr of AB input/output ch. on an AB
const int nABTCCCh = 12;  // nbr of TCC inputs on an AB
const int nDCCCh = 12;    // nbr of DCC outputs on an AB
const int nTCCInEta = 6;  // nbr of TCC bins along eta
const int nAB = nABInPhi * nABInEta;
const int nTTInABAlongPhi = nTTInPhi / nABInPhi;
const int iTTEtaMin[nABInEta] = {0, 11, 28, 45};
const int iTTEtaMax[nABInEta] = {10, 27, 44, 55};
const int iTTEtaSign[nABInEta] = {-1, -1, 1, 1};

// Eta bounds for TCC partionning
// a TCC covers from iTCCEtaBounds[iTCCEta] included to
// iTCCEtaBounds[iTCCEta+1] excluded.
const int iTCCEtaBounds[nTCCInEta + 1] = {0, 7, 11, 28, 45, 49, 56};

const char *abTTFFilePrefix = "TTF_AB";
const char *abTTFFilePostfix = ".txt";
const char *abSRFFilePrefix = "AF_AB";
const char *abSRFFilePostfix = ".txt";
const char *abIOFilePrefix = "IO_AB";
const char *abIOFilePostfix = ".txt";

const char *srfFilename = "SRF.txt";
const char *ttfFilename = "TTF.txt";
const char *xconnectFilename = "xconnect_universal.txt";

const char srpFlagMarker[] = {'.', 'S', 'N', 'C', '4', '5', '6', '7'};
const char tccFlagMarker[] = {'.', 'S', '?', 'C', '4', '5', '6', '7'};

char srp2roFlags[128];

typedef enum { suppress = 0, sr2, sr1, full, fsuppress, fsr2, fsr1, ffull } roAction_t;
char roFlagMarker[] = {
    /*suppress*/ '.',
    /*sr1*/ 'z',
    /*sr1*/ 'Z',
    /*full*/ 'F',
    /*fsuppress*/ '4',
    /*fsr2*/ '5',
    /*fsr1*/ '6',
    /*ffull*/ '7'};

const int nactions = 8;
// can be overwritten according by cmd line arguments
roAction_t actions[nactions] = {
    /*LI->*/ sr2,
    /*S->*/ full,
    /*N->*/ full,
    /*C->*/ full,
    /*fLI->*/ sr2,
    /*fS->*/ sr2,
    /*fN->*/ sr2,
    /*fC->*/ sr2};

// list of SC deserves by an endcap DCC [0(EE-)|1(EE+)][iDCCPhi]
vector<pair<int, int>> ecalDccSC[nEndcaps][nDCCEE];

void fillABTTFFiles(const char ttFlags[nTTInEta][nTTInPhi], ofstream files[]);
void fillABSRPFiles(const char barrelSrFlags[nBarrelTTInEta][nTTInPhi],
                    const char endcapSrFlags[nEndcaps][nSupercrystalXBins][nSupercrystalYBins],
                    ofstream files[]);
void fillABIOFiles(const char ttFlags[nTTInEta][nTTInPhi],
                   const char barrelSrFlags[nBarrelTTInEta][nTTInPhi],
                   const char endcapSrFlags[nEndcaps][nSupercrystalXBins][nSupercrystalYBins],
                   ofstream files[]);
inline int abNum(int iABEta, int iABPhi) { return 3 * iABEta + iABPhi; }

bool readTTF(FILE *file, char ttFlags[nTTInEta][nTTInPhi]);
bool readSRF(FILE *file,
             char barrelSrFlags[nBarrelTTInEta][nTTInPhi],
             char endcapSrFlags[nEndcaps][nSupercrystalXBins][nSupercrystalYBins]);

void writeABTTFFileHeader(ofstream &f, int abNum);
void writeABSRFFileHeader(ofstream &f, int abNum);
void writeABIOFileHeader(ofstream &f, int abNum);
string getFlagStream(char flags[nTTInEta][nTTInPhi], int iEtaMin, int iEtaMax, int iPhiMin, int iPhiMax);
string getABTCCInputStream(const char tccFlags[nTTInEta][nTTInPhi], int iABEta, int iABPhi, int iTCCCh);
void getABTTPhiBounds(int iABPhi, int &iTTPhiMin, int &iTTPhiMax);
string getABABOutputStream(const char tccFlags[nTTInEta][nTTInPhi], int iABEta, int iABPhi, int iABCh);
string getABABInputStream(const char tccFlags[nTTInEta][nTTInPhi], int iABEta, int iABPhi, int iABCh);
string getABDCCOutputStream(const char barrelSrFlags[nBarrelTTInEta][nTTInPhi],
                            const char endcapSrFlags[nEndcaps][nSupercrystalXBins][nSupercrystalYBins],
                            int iABEta,
                            int iABPhi,
                            int DCCCh);
void abConnect(int iAB, int iABCh, int &iOtherAB, int &iOtherABCh);

int iEvent = 0;

int theAB = -1;

int main(int argc, char *argv[]) {
  char barrelSrFlags[nBarrelTTInEta][nTTInPhi];
  char endcapSrFlags[nEndcaps][nEndcapXBins / 5][nEndcapYBins / 5];
  char ttFlags[nTTInEta][nTTInPhi];
  ofstream abTTFFiles[nAB];
  ofstream abSRFFiles[nAB];
  ofstream abIOFiles[nAB];

  int iarg = 0;
  while (++iarg < argc) {
    if (strcmp(argv[iarg], "-h") == 0 || strcmp(argv[iarg], "--help") == 0) {
      cout << "Usage: GenABIO [OPTIONS]\n\n"
              "Produces TT and SR flag files for each SRP board from TTF.txt "
              "and "
              "SRF.txt global flag files. Requires the SRP cross-connect "
              "description"
              " description file (xconnect_universal.txt). TTF.txt, SRF.txt and "
              "xconnect_universal.txt must be in the directory the command is "
              "launched.\n\n"
              "OPTIONS:\n"
              "  -A, --actions IJKLMNOP. IJKLMNOP I..P integers from 0 to 7.\n"
              "                I: action flag for low interest RUs\n"
              "                J: action flag for single RUs\n"
              "                K: action flag for neighbour RUs\n"
              "                L: action flag for centers RUs\n"
              "                M: action flag for forced low interest RUs\n"
              "                N: action flag for forced single RUs\n"
              "                O: action flag for forced neighbour RUs\n"
              "                P: action flag for forced centers RUs\n\n"
              " -h, --help display this help\n"
              " -a n, --ab n specifies indices of the AB whose file must be "
              "produced. The ab number runs from 1 to 12. Use -1 to produce "
              "files "
              "for every AB\n\n";

      return 0;
    }

    if (!strcmp(argv[iarg], "-A") || !strcmp(argv[iarg], "-A")) {  // actions
      if (++iarg >= argc) {
        cout << "Option error. Try -h\n";
        return 1;
      }
      for (int i = 0; i < 8; ++i) {
        int act = argv[iarg][i] - '0';
        if (act < 0 || act >= nactions) {
          cout << "Error. Action argument is invalid.\n";
          return 1;
        } else {
          actions[i] = (roAction_t)act;
        }
      }
      continue;
    }
    if (!strcmp(argv[iarg], "-a") || !strcmp(argv[iarg], "--ab")) {
      if (++iarg >= argc) {
        cout << "Option error. Try -h\n";
        return 1;
      }
      theAB = strtoul(argv[iarg], nullptr, 0);
      if (theAB >= 0)
        --theAB;
      if (theAB < -1 || theAB > 11) {
        cout << "AB number is incorrect. Try -h option to get help.\n";
      }
      continue;
    }
  }

  for (size_t i = 0; i < sizeof(srp2roFlags) / sizeof(srp2roFlags[0]); srp2roFlags[i++] = '?')
    ;
  for (size_t i = 0; i < sizeof(actions) / sizeof(actions[0]); ++i) {
    srp2roFlags[(int)srpFlagMarker[i]] = roFlagMarker[actions[i]];
  }

  for (int iEE = 0; iEE < nEndcaps; ++iEE) {
    for (int iY = 0; iY < nSupercrystalXBins; ++iY) {
      for (int iX = 0; iX < nSupercrystalYBins; ++iX) {
        int iDCCPhi = dccPhiIndexOfRU(iEE == 0 ? 0 : 2, iX, iY);
        if (iDCCPhi >= 0) {  // SC exists
          ecalDccSC[iEE][iDCCPhi].push_back(pair<int, int>(iX, iY));
        }
      }
    }
  }

  stringstream s;
  for (int iAB = 0; iAB < nAB; ++iAB) {
    if (theAB != -1 && theAB != iAB)
      continue;
    s.str("");
    s << abTTFFilePrefix << (iAB < 9 ? "0" : "") << iAB + 1 << abTTFFilePostfix;
    abTTFFiles[iAB].open(s.str().c_str(), ios::out);
    writeABTTFFileHeader(abTTFFiles[iAB], iAB);
    s.str("");
    s << abSRFFilePrefix << (iAB < 9 ? "0" : "") << iAB + 1 << abSRFFilePostfix;
    abSRFFiles[iAB].open(s.str().c_str(), ios::out);
    writeABSRFFileHeader(abSRFFiles[iAB], iAB);
    s.str("");
    s << abIOFilePrefix << (iAB < 9 ? "0" : "") << iAB + 1 << abIOFilePostfix;
    abIOFiles[iAB].open(s.str().c_str(), ios::out);
    writeABIOFileHeader(abIOFiles[iAB], iAB);
  }

  FILE *srfFile = fopen(srfFilename, "r");
  if (srfFile == nullptr) {
    cerr << "Failed to open SRF file, " << srfFilename << endl;
    exit(EXIT_FAILURE);
  }

  FILE *ttfFile = fopen(ttfFilename, "r");
  if (ttfFile == nullptr) {
    cerr << "Failed to open TTF file, " << ttfFilename << endl;
    exit(EXIT_FAILURE);
  }

  iEvent = 0;
  while (readSRF(srfFile, barrelSrFlags, endcapSrFlags) && readTTF(ttfFile, ttFlags)) {
    if (iEvent % 100 == 0) {
      cout << "Event " << iEvent << endl;
    }
    fillABTTFFiles(ttFlags, abTTFFiles);
    fillABSRPFiles(barrelSrFlags, endcapSrFlags, abSRFFiles);
    fillABIOFiles(ttFlags, barrelSrFlags, endcapSrFlags, abIOFiles);
    ++iEvent;
  }

  return 0;
}

/** Produces one file per AB. Each file contains the TT flags
 * the AB receives from its inputs.
 */
void fillABTTFFiles(const char ttFlags[nTTInEta][nTTInPhi], ofstream files[]) {
  for (int iABEta = 0; iABEta < nABInEta; ++iABEta) {
    for (int iABPhi = 0; iABPhi < nABInPhi; ++iABPhi) {
      int iAB = abNum(iABEta, iABPhi);
      int iTTPhiMin;
      int iTTPhiMax;
      getABTTPhiBounds(iABPhi, iTTPhiMin, iTTPhiMax);
      //      writeEventHeader(files[iAB], iEvent, nTTInABAlongPhi);
      files[iAB] << "# Event " << iEvent << "\n";

      for (int i = 0; i <= iTTEtaMax[iABEta] - iTTEtaMin[iABEta]; ++i) {
        int iTTEta;
        if (iTTEtaSign[iABEta] > 0) {
          iTTEta = iTTEtaMin[iABEta] + i;
        } else {
          iTTEta = iTTEtaMax[iABEta] - i;
        }
        for (int iTTPhi = iTTPhiMin; mod(iTTPhiMax - iTTPhi, nTTInPhi) < nTTInABAlongPhi;
             iTTPhi = mod(++iTTPhi, nTTInPhi)) {
          files[iAB] << ttFlags[iTTEta][iTTPhi];
        }
        files[iAB] << "\n";
      }
      files[iAB] << "#\n";
      // writeEventTrailer(files[iAB], nTTInABAlongPhi);
    }
  }
}

void fillABSRPFiles(const char barrelSrFlags[nBarrelTTInEta][nTTInPhi],
                    const char endcapSrFlags[nEndcaps][nSupercrystalXBins][nSupercrystalYBins],
                    ofstream files[]) {
  // event headers:
  for (int iAB = 0; iAB < nAB; ++iAB) {
    files[iAB] << "# Event " << iEvent << "\n";
  }

  bool lineAppended[nAB];
  for (int i = 0; i < nAB; lineAppended[i++] = false) /*empty*/
    ;

  // EE:
  for (int iEE = 0; iEE < nEndcaps; ++iEE) {
    for (int iX = 0; iX < nSupercrystalXBins; ++iX) {
      for (int iY = 0; iY < nSupercrystalYBins; ++iY) {
        //        int iDCC = dccIndex(iEE==0?0:2,iX*5,iY*5);
        int iDCC = dccIndexOfRU(iEE == 0 ? 0 : 2, iX, iY);
        if (iDCC >= 0) {
          int iAB = abOfDcc(iDCC);
          if (!lineAppended[iAB]) {
            for (int i = 0; i < iY; ++i)
              files[iAB] << ' ';
          }
          files[iAB] << srp2roFlags[(int)endcapSrFlags[iEE][iX][iY]];
          lineAppended[iAB] = true;
        }
      }  // next iY
      for (int iFile = 0; iFile < nAB; ++iFile) {
        if (lineAppended[iFile]) {
          files[iFile] << "\n";
          lineAppended[iFile] = false;
        }
      }
    }  // next iX
  }

  // EB:
  for (int iABEta = 1; iABEta < 3; ++iABEta) {
    for (int iABPhi = 0; iABPhi < nABInPhi; ++iABPhi) {
      int iAB = abNum(iABEta, iABPhi);
      int iTTPhiMin;
      int iTTPhiMax;
      getABTTPhiBounds(iABPhi, iTTPhiMin, iTTPhiMax);
      // writeEventHeader(files[iAB], iEvent, nTTInABAlongPhi);
      for (int i = 0; i <= iTTEtaMax[iABEta] - iTTEtaMin[iABEta]; ++i) {
        int iTTEta;
        if (iTTEtaSign[iABEta] > 0) {
          iTTEta = iTTEtaMin[iABEta] + i;
        } else {
          iTTEta = iTTEtaMax[iABEta] - i;
        }
        for (int iTTPhi = iTTPhiMin; mod(iTTPhiMax - iTTPhi, nTTInPhi) < nTTInABAlongPhi;
             iTTPhi = mod(++iTTPhi, nTTInPhi)) {
          files[iAB] << srp2roFlags[(int)barrelSrFlags[iTTEta - nEndcapTTInEta][iTTPhi]];
        }
        files[iAB] << "\n";
      }
      //      writeEventTrailer(files[iAB], nTTInABAlongPhi);
      files[iAB] << "#\n";
    }
  }

  // file trailers
  for (int iAB = 0; iAB < nAB; ++iAB) {
    files[iAB] << "#\n";
  }
}

void fillABIOFiles(const char ttFlags[nTTInEta][nTTInPhi],
                   const char barrelSrFlags[nBarrelTTInEta][nTTInPhi],
                   const char endcapSrFlags[nEndcaps][nSupercrystalXBins][nSupercrystalYBins],
                   ofstream files[]) {
  for (int iABEta = 0; iABEta < nABInEta; ++iABEta) {
    for (int iABPhi = 0; iABPhi < nABInPhi; ++iABPhi) {
      int iAB = abNum(iABEta, iABPhi);
      //      writeABIOFileHeader(files[iAB], iAB);
      files[iAB] << "# Event " << iEvent << "\n";
      // TCC inputs:
      for (int iTCC = 0; iTCC < nABTCCCh; ++iTCC) {
        files[iAB] << "ITCC" << iTCC + 1 << ":" << getABTCCInputStream(ttFlags, iABEta, iABPhi, iTCC) << "\n";
      }
      // AB inputs:
      for (int iABCh = 0; iABCh < nABABCh; ++iABCh) {
        files[iAB] << "IAB" << iABCh + 1 << ":" << getABABInputStream(ttFlags, iABEta, iABPhi, iABCh) << "\n";
      }
      // AB outputs:
      for (int iABCh = 0; iABCh < nABABCh; ++iABCh) {
        files[iAB] << "OAB" << iABCh + 1 << ":" << getABABOutputStream(ttFlags, iABEta, iABPhi, iABCh) << "\n";
      }
      // DCC output:
      for (int iDCCCh = 0; iDCCCh < nDCCCh; ++iDCCCh) {
        files[iAB] << "ODCC";
        files[iAB] << (iDCCCh <= 8 ? "0" : "") << iDCCCh + 1 << ":"
                   << getABDCCOutputStream(barrelSrFlags, endcapSrFlags, iABEta, iABPhi, iDCCCh) << "\n";
      }
      files[iAB] << "#\n";
    }
  }
}

/*
  stringstream filename;
  filename.str("");
  filename << abTTFFilePrefix << abNum(iABEta, iABPhi) <<abTTFFilePostfix;
  ofstream file(filename.str(), ios::ate);

*/

bool readTTF(FILE *f, char ttFlags[nTTInEta][nTTInPhi]) {
  char *buffer = nullptr;
  size_t bufferSize = 0;
  int read;
  if (f == nullptr)
    exit(EXIT_FAILURE);
  int line = 0;
  int iEta = 0;
  while (iEta < nTTInEta && (read = getline(&buffer, &bufferSize, f)) != -1) {
    ++line;
    char *pos = buffer;
    while (*pos == ' ' || *pos == '\t')
      ++pos;                            // skip spaces
    if (*pos != '#' && *pos != '\n') {  // not a comment line nor an empty line
      if (read - 1 != nTTInPhi) {
        cerr << "Error: line " << line << " of file " << ttfFilename
             << " has incorrect length"
             //             << " (" << read-1 << " instead of " << nTTInPhi <<
             //             ")"
             << endl;
        exit(EXIT_FAILURE);
      }
      for (int iPhi = 0; iPhi < nTTInPhi; ++iPhi) {
        ttFlags[iEta][iPhi] = buffer[iPhi];
        //         if(ttFlags[iEta][iPhi]!='.'){
        //           cout << __FILE__ << ":" << __LINE__ << ": "
        //                << iEta << "," << iPhi
        //                << " " << ttFlags[iEta][iPhi] << "\n";
        //         }
      }
      ++iEta;
    }
  }
  // returns true if all TT were read (not at end of file)
  return (iEta == nTTInEta) ? true : false;
}

bool readSRF(FILE *f,
             char barrelSrFlags[nBarrelTTInEta][nTTInPhi],
             char endcapSrFlags[nEndcaps][nSupercrystalXBins][nSupercrystalYBins]) {
  char *buffer = nullptr;
  size_t bufferSize = 0;
  int read;
  if (f == nullptr)
    exit(EXIT_FAILURE);
  int line = 0;
  int iEta = 0;
  int iXm = 0;
  int iXp = 0;
  int iReadLine = 0;  // number of read line, comment lines excluded
  // number of non-comment lines to read:
  const int nReadLine = nBarrelTTInEta + nEndcaps * nSupercrystalXBins;
  while (iReadLine < nReadLine && (read = getline(&buffer, &bufferSize, f)) != -1) {
    ++line;
    char *pos = buffer;
    while (*pos == ' ' || *pos == '\t')
      ++pos;                            // skip spaces
    if (*pos != '#' && *pos != '\n') {  // not a comment line nor an empty line
      // go back to beginning of line:
      pos = buffer;
      if (iReadLine < nSupercrystalXBins) {  // EE- reading
        if (read - 1 != nSupercrystalYBins) {
          cerr << "Error: line " << line << " of file " << srfFilename << " has incorrect length"
               << " (" << read - 1 << " instead of " << nSupercrystalYBins << ")" << endl;
          exit(EXIT_FAILURE);
        }
        for (int iY = 0; iY < nSupercrystalYBins; ++iY) {
          endcapSrFlags[0][iXm][iY] = buffer[iY];
        }
        ++iXm;
      } else if (iReadLine < nSupercrystalYBins + nBarrelTTInEta) {  // EB
                                                                     // reading
        if (read - 1 != nTTInPhi) {
          cerr << "Error: line " << line << " of file " << srfFilename << " has incorrect length"
               << " (" << read - 1 << " instead of " << nTTInPhi << ")" << endl;
          exit(EXIT_FAILURE);
        }
        for (int iPhi = 0; iPhi < nTTInPhi; ++iPhi) {
          barrelSrFlags[iEta][iPhi] = buffer[iPhi];
        }
        ++iEta;
      } else if (iReadLine < 2 * nSupercrystalXBins + nBarrelTTInEta) {  // EE+ reading
        if (read - 1 != nSupercrystalYBins) {
          cerr << "Error: line " << line << " of file " << srfFilename << " has incorrect length"
               << " (" << read - 1 << " instead of " << nSupercrystalYBins << ")" << endl;
          exit(EXIT_FAILURE);
        }
        for (int iY = 0; iY < nSupercrystalYBins; ++iY) {
          endcapSrFlags[1][iXp][iY] = buffer[iY];
        }
        ++iXp;
      }
      ++iReadLine;
    }  // not a comment or empty line
  }
  // returns 0 if all TT were read:
  return (iReadLine == nReadLine) ? true : false;
}

// void writeEventHeader(ofstream& f, int iEvent, int nPhi){
//   //event header:
//   stringstream header;
//   header.str("");
//   header << " event " << iEvent << " ";
//   f << "+";
//   for(int iPhi = 0; iPhi < nPhi; ++iPhi){
//     if(iPhi == (int)(nPhi-header.str().size())/2){
//       f << header.str();
//       iPhi += header.str().size()-1;
//     } else{
//       f << "-";
//     }
//   }
//   f << "+\n";
// }

// void writeEventTrailer(ofstream& f, int nPhi){
//   f << "+";
//   for(int iPhi = 0; iPhi < nPhi; ++iPhi) f << "-";
//   f << "+\n";
// }

void writeABTTFFileHeader(ofstream &f, int abNum) {
  time_t t;
  time(&t);
  const char *date = ctime(&t);
  f << "# TTF flag map covered by AB " << abNum + 1
    << "\n#\n"
       "# Generated on : "
    << date
    << "#\n"
       "# +---> Phi          "
    << srpFlagMarker[0]
    << ": 000 (low interest)\n"
       "# |                  "
    << srpFlagMarker[1]
    << ": 001 (single)\n"
       "# |                  "
    << srpFlagMarker[2]
    << ": 010 (neighbour)\n"
       "# V |Eta|            "
    << srpFlagMarker[3]
    << ": 011 (center)\n"
       "#\n";
}

void writeABSRFFileHeader(ofstream &f, int abNum) {
  time_t t;
  time(&t);
  const char *date = ctime(&t);
  const char *xLabel;
  const char *yLabel;
  if (abNum < 3 || abNum > 8) {  // endcap
    xLabel = "Y  ";
    yLabel = "X    ";
  } else {  // barrel
    xLabel = "Phi";
    yLabel = "|Eta|";
  }
  f << "# SRF flag map covered by AB " << abNum + 1
    << "\n#\n"
       "# Generated on : "
    << date
    << "#\n"
       "# +---> "
    << xLabel << "          " << roFlagMarker[0]
    << ": 000 (suppress)\n"
       "# |                  "
    << roFlagMarker[1]
    << ": 010 (SR Threshold 2)\n"
       "# |                  "
    << roFlagMarker[2]
    << ": 001 (SR Threshold 1)\n"
       "# V "
    << yLabel << "            " << roFlagMarker[3]
    << ": 011 (Full readout)\n"
       "#\n"
       "# action table (when forced):\n"
       "# LI-> "
    << roFlagMarker[actions[0]] << " (" << roFlagMarker[actions[4]] << ")"
    << "\n"
       "# S -> "
    << roFlagMarker[actions[1]] << " (" << roFlagMarker[actions[5]] << ")"
    << "\n"
       "# N -> "
    << roFlagMarker[actions[2]] << " (" << roFlagMarker[actions[6]] << ")"
    << "\n"
       "# C -> "
    << roFlagMarker[actions[3]] << " (" << roFlagMarker[actions[7]] << ")"
    << "\n";
}

void writeABIOFileHeader(ofstream &f, int abNum) {
  time_t t;
  time(&t);
  const char *date = ctime(&t);
  f << "# AB " << abNum + 1
    << " I/O \n#\n"
       "# Generated on : "
    << date
    << "#\n"
       "# "
    << srpFlagMarker[0] << ": 000 (low interest)   " << tccFlagMarker[0] << ": 000 (low interest)   " << roFlagMarker[0]
    << ": 000 (suppress)\n"
       "# "
    << srpFlagMarker[1] << ": 001 (single)         " << tccFlagMarker[1] << ": 001 (mid interest)   " << roFlagMarker[1]
    << ": 010 (SR Threshold 2)\n"
       "# "
    << srpFlagMarker[2] << ": 010 (neighbour)      " << tccFlagMarker[2] << ": 010 (not valid)      " << roFlagMarker[2]
    << ": 001 (SR Threshold 1)\n"
       "# "
    << srpFlagMarker[3] << ": 011 (center)         " << tccFlagMarker[3] << ": 011 (high interest)  " << roFlagMarker[3]
    << ": 011 (Full readout)\n"
       "#\n"
       "# action table (when forced):\n"
       "# LI-> "
    << roFlagMarker[actions[0]] << " (" << roFlagMarker[actions[4]] << ")"
    << "\n"
       "# S -> "
    << roFlagMarker[actions[1]] << " (" << roFlagMarker[actions[5]] << ")"
    << "\n"
       "# N -> "
    << roFlagMarker[actions[2]] << " (" << roFlagMarker[actions[6]] << ")"
    << "\n"
       "# C -> "
    << roFlagMarker[actions[3]] << " (" << roFlagMarker[actions[7]] << ")"
    << "\n"
       "#\n";
}

string getFlagStream(const char flags[nTTInEta][nTTInPhi], int iEtaMin, int iEtaMax, int iPhiMin, int iPhiMax) {
  assert(0 <= iEtaMin && iEtaMin <= iEtaMax && iEtaMax < nTTInEta);
  if (iEtaMin <= nTTInEta / 2 && iEtaMax > nTTInEta) {
    cerr << "Implementation Errror:" << __FILE__ << ":" << __LINE__
         << ": A flag stream cannot covers parts of both half-ECAL!" << endl;
    exit(EXIT_FAILURE);
  }

  bool zPos = (iEtaMin >= nTTInEta / 2);

  stringstream buffer;
  buffer.str("");
  for (int jEta = 0; jEta <= iEtaMax - iEtaMin; ++jEta) {
    // loops on iEta in |eta| increasing order:
    int iEta;
    if (zPos) {
      iEta = iEtaMin + jEta;
    } else {
      iEta = iEtaMax - jEta;
    }

    for (int iPhi = mod(iPhiMin, nTTInPhi); mod(iPhiMax + 1 - iPhi, nTTInPhi) != 0; iPhi = mod(++iPhi, nTTInPhi)) {
      buffer << flags[iEta][iPhi];
    }
  }

  return buffer.str();
}

string getABTCCInputStream(const char tccFlags[nTTInEta][nTTInPhi], int iABEta, int iABPhi, int iTCCCh) {
  // gets eta bounds for this tcc channel:
  int iTCCEta;
  if (iABEta == 1 || iABEta == 2) {  // barrel
    if (iTCCCh > 5)
      return "";  // only 6 TCCs per AB for barrel
    iTCCEta = 1 + iABEta;
  } else {              // endcap
    if (iABEta == 0) {  // EE-
      iTCCEta = (iTCCCh < 6) ? 1 : 0;
    } else {  // EE+
      iTCCEta = (iTCCCh < 6) ? 4 : 5;
    }
  }
  int iEtaMin = iTCCEtaBounds[iTCCEta];
  int iEtaMax = iTCCEtaBounds[iTCCEta + 1] - 1;

  // gets phi bounds:
  int iPhiMin;
  int iPhiMax;
  getABTTPhiBounds(iABPhi, iPhiMin, iPhiMax);
  // phi is increasing with TTC channel number
  // a TTC covers a 4TT-wide phi-sector
  //=>iPhiMin(iTTCCh) = iPhiMin(AB) + 4*iTTCCh for iTCCCh<6
  iPhiMin += 4 * (iTCCCh % 6);
  iPhiMax = iPhiMin + 4 - 1;

  return getFlagStream(tccFlags, iEtaMin, iEtaMax, iPhiMin, iPhiMax);
}

string getABABOutputStream(const char tccFlags[nTTInEta][nTTInPhi], int iABEta, int iABPhi, int iABCh) {
  stringstream buffer;
  buffer.str("");
  bool barrel = (iABEta == 1 || iABEta == 2);  // true for barrel, false for endcap
  switch (iABCh) {
    case 0:
      // to AB ch #0 are sent the 16 1st TCC flags received on TCC input Ch. 0
      buffer << getABTCCInputStream(tccFlags, iABEta, iABPhi, 0).substr(0, 16);
      break;
    case 1:
      // to AB ch #1 are sent the 16 1st TCC flags received on TCC input Ch. 0 to
      // 5:
      for (int iTCCCh = 0; iTCCCh < 6; ++iTCCCh) {
        buffer << getABTCCInputStream(tccFlags, iABEta, iABPhi, iTCCCh).substr(0, 16);
      }
      break;
    case 2:
      // to AB ch #2 are sent the 16 1st TCC flags received on TCC input Ch. 5:
      buffer << getABTCCInputStream(tccFlags, iABEta, iABPhi, 5).substr(0, 16);
      break;
    case 3:
      // to AB ch #3 are sent TCC flags received on TCC input Ch. 0 and 6:
      buffer << getABTCCInputStream(tccFlags, iABEta, iABPhi, 0);
      buffer << getABTCCInputStream(tccFlags, iABEta, iABPhi, 6);
      break;
    case 4:
      // to AB ch #4 are sent TCC flags received on TCC input Ch 5 and 11:
      buffer << getABTCCInputStream(tccFlags, iABEta, iABPhi, 5);
      buffer << getABTCCInputStream(tccFlags, iABEta, iABPhi, 11);
      break;
    case 5:
      // for endcaps AB output ch 5 is not used.
      // for barrel, to AB ch #5 are sent the 16 last TCC flags received on TCC
      // input Ch. 0:
      if (barrel) {  // in barrel
        string s = getABTCCInputStream(tccFlags, iABEta, iABPhi, 0);
        assert(s.size() >= 16);
        buffer << s.substr(s.size() - 16, 16);
      }
      break;
    case 6:
      // for endcaps AB output ch 6 is not used.
      // for barrel, to AB ch #6 are sent the 16 last TCC flags received on TCC
      // input Ch. 0 to 5:
      if (barrel) {  // in barrel
        for (int iTCCCh = 0; iTCCCh < 6; ++iTCCCh) {
          string s = getABTCCInputStream(tccFlags, iABEta, iABPhi, iTCCCh);
          buffer << s.substr(s.size() - 16, 16);
        }
      }
      break;
    case 7:
      // for endcaps AB output ch 7 is not used.
      // for barrel, to AB ch #7 are sent the 16 last TCC flags received on TCC
      // input Ch. 5:
      if (barrel) {  // in barrel
        string s = getABTCCInputStream(tccFlags, iABEta, iABPhi, 5);
        assert(s.size() >= 16);
        buffer << s.substr(s.size() - 16, 16);
      }
      break;
    default:
      assert(false);
  }
  return buffer.str();
}

string getABABInputStream(const char tccFlags[nTTInEta][nTTInPhi], int iABEta, int iABPhi, int iABCh) {
  int iAB = abNum(iABEta, iABPhi);
  int iOtherAB;    // AB which this channel is connected to
  int iOtherABCh;  // ch # on the other side of the AB-AB link
  abConnect(iAB, iABCh, iOtherAB, iOtherABCh);
  int iOtherABEta = iOtherAB / 3;
  int iOtherABPhi = iOtherAB % 3;
  return getABABOutputStream(tccFlags, iOtherABEta, iOtherABPhi, iOtherABCh);
}

void getABTTPhiBounds(int iABPhi, int &iTTPhiMin, int &iTTPhiMax) {
  iTTPhiMin = mod(-6 + iABPhi * nTTInABAlongPhi, nTTInPhi);
  iTTPhiMax = mod(iTTPhiMin + nTTInABAlongPhi - 1, nTTInPhi);
}

void abConnect(int iAB, int iABCh, int &iOtherAB, int &iOtherABCh) {
  static bool firstCall = true;
  static int xconnectMap[nAB][nABABCh][2];
  if (firstCall) {
    FILE *f = fopen(xconnectFilename, "r");
    if (f == nullptr) {
      cerr << "Error. Failed to open xconnect definition file," << xconnectFilename << endl;
      exit(EXIT_FAILURE);
    }
    // skips two first lines:
    for (int i = 0; i < 2; ++i) {
      int c;
      while ((c = getc(f)) != '\n' && c >= 0)
        ;
    }
    int ilink = 0;
    while (!feof(f)) {
      int abIn;
      int pinIn;
      int abOut;
      int pinOut;
      if (4 == fscanf(f, "%d\t%d\t%d\t%d", &abIn, &pinIn, &abOut, &pinOut)) {
        xconnectMap[abIn][pinIn][0] = abOut;
        xconnectMap[abIn][pinIn][1] = pinOut;
        ++ilink;
      }
    }
    if (ilink != nAB * nABABCh) {
      cerr << "Error cross-connect definition file " << xconnectFilename
           << " contains an unexpected number of link definition." << endl;
      exit(EXIT_FAILURE);
    }
    firstCall = false;
  }

  iOtherAB = xconnectMap[iAB][iABCh][0];
  iOtherABCh = xconnectMap[iAB][iABCh][1];
}

string getABDCCOutputStream(const char barrelSrFlags[nBarrelTTInEta][nTTInPhi],
                            const char endcapSrFlags[nEndcaps][nSupercrystalXBins][nSupercrystalYBins],
                            int iABEta,
                            int iABPhi,
                            int iDCCCh) {
  bool barrel = (iABEta == 1 || iABEta == 2);
  if (barrel) {
    // same as TCC with same ch number but with TCC flags replaced by SRP flags:
    string stream = getABTCCInputStream(barrelSrFlags - nEndcapTTInEta, iABEta, iABPhi, iDCCCh);
    // converts srp flags to readout flags:
    for (size_t i = 0; i < stream.size(); ++i) {
      stream[i] = srp2roFlags[(int)stream[i]];
    }
    return stream;
  } else {             // endcap
    if (iDCCCh < 3) {  // used DCC output channel
      // endcap index:
      int iEE = (iABEta == 0) ? 0 : 1;
      stringstream buffer("");
      // 3 DCC per AB and AB DCC output channel in
      // increasing DCC phi position:
      int iDCCPhi = iABPhi * 3 + iDCCCh;
      for (size_t iSC = 0; iSC < ecalDccSC[iEE][iDCCPhi].size(); ++iSC) {
        pair<int, int> sc = ecalDccSC[iEE][iDCCPhi][iSC];
        buffer << srp2roFlags[(int)endcapSrFlags[iEE][sc.first][sc.second]];
      }
      return buffer.str();
    } else {  // unused output channel
      return "";
    }
  }
}
