#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef CH_LANG_FORT
#  ifdef INSIDE
#    undef INSIDE
#  else
#    error Matching NamespaceHeader.H not in this scope.  You probably have a NamespaceHeader.H followed by another NamespaceHeader.H, instead of followed by a NamespaceFooter.H.
#  endif

#  ifdef CH_MULTIDIM
       }
       }
       namespace Dummy
       {
       }
#  else
#    ifdef CH_NAMESPACE
       }
       namespace Dummy
       {
       }
#    endif
#  endif
#endif
