#! /usr/bin/env python3
################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
# Transform all html files into a directory into .htmlgz files and add the 
# .htaccess file to tell Apache to serve the new compressed data.
# Moves also the pickles away.
#
################################################################################

from __future__ import print_function
htaccess_content="""
RewriteEngine on

RewriteCond %{HTTP:Accept-Encoding} gzip
RewriteRule (.*)\.html$ $1\.htmlgz [L]
RewriteRule (.*)\.png$ $1\.pnggz [L]

AddType "text/html;charset=UTF-8" .htmlgz
AddEncoding gzip .htmlgz

AddType "image/png" .pnggz
AddEncoding gzip .pnggz

DirectoryIndex RelMonSummary.htmlgz

"""


from os.path import exists
from os import system
from sys import argv,exit

argc=len(argv)

if argc!=2:
  print("Usage: %prog directoryname")
  exit(-1)

directory =  argv[1]

while directory[-1]=="/": directory= directory[:-1]

if not exists(directory):
  print("Directory %s does not exist: aborting!"%directory)
  exit(-1)  

print("Moving pkls away...")
pkl_dir="%s_pkls" %directory
system("mkdir %s" %pkl_dir)
system("mv %s/*pkl %s" %(directory,pkl_dir))
print("All pkls moved in directory %s" %pkl_dir)

print("Backupping directory %s" %directory)
system("cp -r %s %s_back"%(directory,directory)) # i know, it should be better..
print("Backupped!")

print("Gzipping content of %s" %directory)
system("time gzip -r -S gz %s"%directory) # i know, it should be better..
print("Content of %s zipped!" %directory)

print("Adding .htaccess file...")
htaccess=open("%s/.htaccess"%directory,"w")
htaccess.write(htaccess_content)
htaccess.close()
print("Apache .htaccess file successfully added!")
