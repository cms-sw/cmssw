#! /usr/bin/env python3
################################################################################
# RelMon: a tool for automatic Release Comparison
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
#
#
# Danilo Piparo CERN - danilo.piparo@cern.ch
#
################################################################################

from __future__ import print_function
import subprocess as sub
import sys
from optparse import OptionParser

parser = OptionParser(usage="usage: %prog cmssw_release [options]")


parser.add_option("-d","--data ",
                  action="store",
                  dest="data",
                  default=False,
                  help="Fetch data relvals")

parser.add_option("-m","--mc ",
                  action="store",
                  dest="mc",
                  default=False,
                  help="Fetch Monte Carlo relvals")

parser.add_option("--p1","--path1 ",
                  action="store",
                  dest="path1",
                  default="",
                  help="Additional path to match in relvals")

parser.add_option("--p2","--path2 ",
                  action="store",
                  dest="path2",
                  default="",
                  help="Additional path to match in relvals")

(options, args) = parser.parse_args()

#if len(args)!=1:
#  print "Specify one and only one release!"
#  print args
#  sys.exit(2)

cmssw_release = args[0]

# Specify the directory of data or MC
relvaldir="RelVal"
if options.data:
  relvaldir+="Data"

# Specify the directory of the release
releasedir=cmssw_release[:10]+"x"

#fetch!
thepath=cmssw_release
if len(options.path1)>0:
  thepath="%s.*%s"%(options.path1,thepath)
if len(options.path2)>0:
  thepath="%s.*%s"%(thepath,options.path2)  
command='relmon_rootfiles_spy.py ROOT/%s/%s/ -u -g -p %s'%(relvaldir,releasedir,thepath)
print(command)
sub.call(command.split(" "))

# Main tree:
# https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/


