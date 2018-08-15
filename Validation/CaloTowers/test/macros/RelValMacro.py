#!/usr/bin/env python
#RelValMacro.py

import sys
import json
import pprint
import RelValMacro

def spaceEscape(myString):
#return "\"" + str(myString) + "\""
    return str(myString)

if len(sys.argv) == 6:
    ref_vers = sys.argv[1]
    val_vers = sys.argv[2]
    rfname = sys.argv[3]
    vfname = sys.argv[4]
    range = sys.argv[5]
	
    with open('InputRelVal.json', 'r') as fp:
        inputFile = json.load(fp)
        for histogram in inputFile:
            if range in inputFile[histogram]:
                if inputFile[histogram]['drawSwitch']:
					#Call RelValMacro.exe with histogram parameters
                    histName = spaceEscape(histogram)
                    ofileName = spaceEscape(inputFile[histogram]['outLabel'])
                    nRebin = spaceEscape(inputFile[histogram][range]['rebin'])
                    xAxisMin = spaceEscape(inputFile[histogram][range]['xAxisMin'])
                    xAxisMax = spaceEscape(inputFile[histogram][range]['xAxisMax'])
                    yAxisMin = spaceEscape(inputFile[histogram][range]['yAxisMin'])
                    yAxisMax = spaceEscape(inputFile[histogram][range]['yAxisMax'])
                    dimFlag = spaceEscape(inputFile[histogram]['dimFlag'])
                    statFlag = spaceEscape(inputFile[histogram]['statFlag'])
                    chi2Flag = spaceEscape(inputFile[histogram]['chi2Flag'])
                    logFlag = spaceEscape(inputFile[histogram]['logFlag'])
                    ratioFlag = spaceEscape(inputFile[histogram]['ratioFlag'])
                    refColor = spaceEscape(inputFile[histogram]['refColor'])
                    valColor = spaceEscape(inputFile[histogram]['valColor'])
                    xAxisTitle = spaceEscape(inputFile[histogram]['xAxisTitle'])
                    histName2 = spaceEscape(inputFile[histogram]['histName2'])
                    normFlag = spaceEscape(inputFile[histogram]['normFlag'])
                    cmd = ref_vers + "|" + val_vers + "|" + rfname + "|" + vfname + "|" + histName + "|" + ofileName + "|" + nRebin + "|" + xAxisMin + "|" + xAxisMax + "|" + yAxisMin + "|" + yAxisMax + "|" + dimFlag + "|" + statFlag + "|" + chi2Flag + "|" + logFlag + "|" + ratioFlag + "|" + refColor + "|" + valColor + "|" + xAxisTitle + "|" + histName2 + "|" + normFlag
                    RelValMacro.RelValMacro(cmd)
else:
    print ("Usage: ./RelValMacro.py ref_vers val_vers ref_file_name val_file_name range[High/Medium/Low]")
#std::string ref_vers, std::string val_vers, std::string rfname, std::string vfname, std::string histName, std::string ofileName, int nRebin, double xAxisMin, double xAxisMax, double yAxisMin, double yAxisMax, std::string dimFlag, std::string statFlag, std::string chi2Flag, std::string logFlag, int refCol, int valCol, std::string xAxisTitle, std::string histName2, std::string normFlag
