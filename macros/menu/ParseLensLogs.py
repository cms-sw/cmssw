#!/usr/bin/env python 
from sys import stderr, exit
import os
import commands


def parseFile(logfile, Table):
 print " ... parsing file: ",logfile
 TheLog = open(logfile)
 gl="go"
 line=0
 l0=-1
 idx=-1
 while gl:
   gl = TheLog.readline()
   line=line+1
   if len(gl)>0:
	if gl.find("rigReport Events total =")>=0:
		total = gl.split()[4]
		evtpass = gl.split()[7]
		#print total, evtpass
		if  Table.has_key("TOTAL"):
		   Table["TOTAL"] = int(Table["TOTAL"]) + int(total)
		else:
		   Table["TOTAL"] = total
                if  Table.has_key("PASS"):
                   Table["PASS"] = int(Table["PASS"]) + int(evtpass)
                else:
                   Table["PASS"] = evtpass

	
	if gl.find("eport for L1 menu")>=0:
	  l0  = line
# --- the numbers below (esp. the "l0 + 88") may need to be changed 
# --- depending on how many L1 bits were emulated in the menu...
        if l0 > 0 and line >= l0+3 and line <= l0+88:
	    idx = idx+1
	    L1bit = gl.split()[0]
	    nbevts = gl.split()[3]
	    if  Table.has_key(L1bit):
		Table[L1bit] = int(Table[L1bit]) + int(nbevts)
	    else:
		Table[L1bit] = nbevts
 TheLog.close()


def parseLogs(directory, Table):
  cmd='ls -1 ' + directory+'*.stdout   > list.txt'
  print cmd
  os.system('ls -1 ' + directory+'*.stdout   > list.txt')

  ll = open('list.txt')
  fl="go"
  while fl:
	fl=ll.readline()
 	if len(fl)>0:
		logfile = fl.split()[0]
		#logfile = directory+logfile
		parseFile(logfile, Table)
  os.system('rm list.txt')


# -- for skim v3, only the PU10 files were skimmed.

TablePU={}

# --- Len's log files are sitting in lxplus442

# -- skim over DATA :
# directory="/tmp/apana/cern_r179828_ZeroBiasHPF0/res/"
# parseLogs(directory,TablePU)
# directory="/tmp/apana/cern_r179828_ZeroBiasHPF1/res/"
# parseLogs(directory,TablePU)
# directory="/tmp/apana/cern_r179828_ZeroBiasHPF2/res/"
# parseLogs(directory,TablePU)
# directory="/tmp/apana/cern_r179828_ZeroBiasHPF3/res/"
# parseLogs(directory,TablePU)

# -- skim over the 8 TeV MC
directory="/tmp/apana/cern_MinBias_Fall11-Ave23_8TeV_50ns-v1/res/"

# -- skim over the 7 TeV MC 
#directory="/tmp/apana/cern_MinBias_Fall11-E7TeV_Ave32_50ns-v2/res/"

parseLogs(directory,TablePU)

for bit in TablePU.keys():
        n1=TablePU[bit]
        print bit,"\t",n1




