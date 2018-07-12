#!/bin/sh
#
# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
#

echo '-->' $cc $cppflags $cppdebugflags -frtti -c $CheckerGccPlugins_args  -fplugin-arg-libchecker_gccplugins-checkers=all  $1
$cc $cppflags $cppdebugflags -frtti -c $CheckerGccPlugins_args  -fplugin-arg-libchecker_gccplugins-checkers=all  $1

