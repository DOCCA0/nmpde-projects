#!/bin/bash
source /u/sw/lmod/8.5.8/init/bash
export MODULEPATH=/u/sw/toolchains/gcc-glibc/11.2.0/modules
export PATH=/u/sw/toolchains/gcc-glibc/11.2.0/base/bin:$PATH
module load dealii/9.5.1
