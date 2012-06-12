#! /usr/bin/env bash
################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
# $Author: dpiparo $
# $Date: 2011/05/10 13:14:55 $
# $Revision: 1.2 $
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
# Transform all html files into a directory into .htmlgz files and add the 
# .htaccess file to tell Apache to serve the new compressed data.
# Moves also the pickles away.
#
################################################################################


mkdir FastSim
mv *FastSim*root FastSim

mkdir START
mv *START* START

mkdir MC
mv *MC* MC

cd FastSim

mkdir START
mv *START* START

mkdir MC
mv *MC* MC

cd ..