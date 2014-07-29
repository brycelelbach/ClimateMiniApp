#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include "DataIterator.H"
#include "IntVect.H"
#include "AsyncCopier.H"
#include "MayDay.H"
#include "LayoutIterator.H"
#include "NeighborIterator.H"

#include <vector>
#include "NamespaceHeader.H"

void AsyncCopier::unilateralDefine(const BoxLayout& a_src,
                                   const BoxLayout& a_dest,
                                   const ProblemDomain& a_domain,
                                   const IntVect& a_ghost,
                                   IntVect a_shift,
                                   int a_procID)
{
  CH_assert(a_src.isClosed());
  CH_assert(a_dest.isClosed());

  clear();

  m_src  = a_src;
  m_dest = a_dest;

  m_procID = a_procID;

  const BoxLayout& level = a_src;
  const BoxLayout& dest = a_dest;

  // set up vector of dataIndexes to keep track of which
  // "to" boxes are not completely contained within the primary
  // domain.  these boxes are then candidates for filling by
  // periodic images of the "from" data.
  Vector<DataIndex> periodicallyFilledToVect;

  // in order to cull which "from" data may be needed to
  // fill the "to" data, keep track of the radius around the
  // primary domain in which all these cells lie.
  // do this by incrementally growing the domain box and
  // keeping track of what this radius is.
  // just to make things simpler, start off with a radius of one
  Box grownDomainCheckBox = a_domain.domainBox();
  grownDomainCheckBox.grow(1);
  int periodicCheckRadius = 1;

  // since valid regions of the "from" DBL may also be outside
  // the primary domain, need to keep track of whether any of these
  // need to be checked separately.
  Vector<DataIndex> periodicFromVect;
  // use same domain trick here as well
  Box grownFromDomainCheckBox = a_domain.domainBox();
  int periodicFromCheckRadius = 1;

  Box domainBox(a_domain.domainBox());
  bool isPeriodic = false;
  if (!domainBox.isEmpty())
    isPeriodic = a_domain.isPeriodic();

  // (dfm -- 9/13/05) as currently written, the AsyncCopier won't correctly
  // handle periodic cases where the number of ghost cells is greater
  // than the width of the domain.  We _should_ do multiple wraparounds,
  // but we don't. So, put in this assertion. We can revisit this if it
  // becomes an issue
  if (isPeriodic)
    {
      for (int dir = 0; dir < SpaceDim; dir++)
        {
          if (a_domain.isPeriodic(dir))
            {
              CH_assert (a_ghost[dir] <= domainBox.size(dir));
            }
        }
    }

  // The following 4 for loops are the result of a performance optimization.
  // When increasing the size of the problem, we found that the code was
  // looping over every destination box for every source box which was N1*N2
  // loop iterations (essentially an N-squared approach).
  // The following code attempts to simply reduce N1 and N2 by first separating
  // the boxes (or LayoutIndexes to boxes) that reside on the current processor.
  // Then the loop to determine which boxes of the first list intersect with
  // which boxes of the second list can be done in N1' * N2' iterations,
  // where N1' is the reduced N1 and N2' is the reduced N2.
  // We have to break up the assigning of AsyncSendInstructionss into two separate
  // loops and be careful about the local copies.  These 4 loops are
  // significantly faster than the original for loop -- _especially_
  // for large problems.  (ndk)

  // make a vector of boxes (or LayoutIndexes to boxes) from destination layout
  // that are known to reside on this processor.
  vector<DataIndex> vectorDestDI;
  vector<DataIndex> vectorDestOnProcDI;
  for (LayoutIterator to(a_dest.layoutIterator()); to.ok(); ++to)
  {
    vectorDestDI.push_back(DataIndex(to()));
    if (m_procID == dest.procID(to()))
    {
      vectorDestOnProcDI.push_back(DataIndex(to()));
    }
  }

  // make a vector of boxes (or LayoutIndexes to boxes) from "level"/src layout
  // that are known to reside on this processor.
  vector<DataIndex> vectorLevelDI;
  vector<DataIndex> vectorLevelOnProcDI;
  for (LayoutIterator from(a_src.layoutIterator()); from.ok(); ++from)
  {
    vectorLevelDI.push_back(DataIndex(from()));
    if (m_procID == level.procID(from()))
    {
      vectorLevelOnProcDI.push_back(DataIndex(from()));
    }
  }

  bool isSorted = (a_src.isSorted() && a_dest.isSorted());

  /////////////////////////////////////////////////////////////////////////////

  // loop over all dest/to DI's on my processor
  for (vector<DataIndex>::iterator vdi = vectorDestOnProcDI.begin();
      vdi != vectorDestOnProcDI.end(); ++vdi)
  {
    // at this point, i know m_procID == toProcID
    const DataIndex todi(*vdi);

    Box ghost(dest[todi]);
    ghost -= a_shift;

    ghost.grow(a_ghost);

    //bool isSorted = (a_src.isSorted() && a_dest.isSorted());
    // then for each level/from DI, see if they intersect
    for (vector<DataIndex>::iterator vli = vectorLevelDI.begin();
        vli != vectorLevelDI.end(); ++vli)
    {
      const DataIndex fromdi(*vli);
      const unsigned int fromProcID = level.procID(fromdi);
      const Box& fromBox = level[fromdi];
      if ((fromBox.bigEnd(0) < ghost.smallEnd(0)) && isSorted )
      {
        //can skip rest cuz we haven't gotten to something interesting
        continue;
      }

      if (ghost.intersectsNotEmpty(fromBox))
      {
        Box srcBox(ghost); // ??
        srcBox &= fromBox; // ??

        Box destBox = srcBox + a_shift;

        if (fromProcID == m_procID)
        { // local move
          AsyncRegion region(todi, destBox);
          m_regions.insert(region);
        }
        else
        {
          AsyncRegion region(todi, destBox);
          m_regions.insert(region);
        }
      }
      if ((fromBox.smallEnd(0) > ghost.bigEnd(0)) && isSorted)
      {
        // Can break out of loop, since we know that the smallEnd
        // of all the remaining boxes are lexigraphically beyond this ghosted box.
        break;
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////

  // loop over all dest/to DI's
  for (vector<DataIndex>::iterator vdi = vectorDestDI.begin();
      vdi != vectorDestDI.end(); ++vdi)
  {

    const DataIndex todi(*vdi);

    Box ghost(dest[todi]);
    ghost -= a_shift;

    ghost.grow(a_ghost);

    const unsigned int toProcID = dest.procID(todi);

    // then for each level/from DI on this processor, see if they intersect
    for (vector<DataIndex>::iterator vli = vectorLevelOnProcDI.begin();
        vli != vectorLevelOnProcDI.end(); ++vli)
    {

      // at this point, i know m_procID == fromProcID

      const DataIndex fromdi(*vli);
      const Box& fromBox = level[fromdi];

      if ((fromBox.bigEnd(0) < ghost.smallEnd(0)) && isSorted)
      {
        //can skip rest cuz we haven't gotten to something interesting
        continue;
      }

      if (ghost.intersectsNotEmpty(fromBox))
      {
        Box srcBox(ghost); // ??
        srcBox &= fromBox; // ??

        Box destBox = srcBox + a_shift;

        if (toProcID == m_procID)
        { // local move
          AsyncSendInstructions item(fromdi, todi, srcBox, destBox);
          m_senderMotionPlan.insert(item);
        }
        else
        {
          AsyncSendInstructions item(fromdi, todi, srcBox, destBox);
          m_senderMotionPlan.insert(item);
        }
      }
      if ((fromBox.smallEnd(0) > ghost.bigEnd(0)) && isSorted)
      {
        //can break out of loop, since we know that the smallEnd
        // of all the remaining boxes are lexigraphically beyond this ghosted box.
        break;
      }
    }
  }

  if (isPeriodic && a_shift != IntVect::Zero)
  {
    MayDay::Error("AsyncCopier::define - domain periodic and a non-zero shift in the copy is not implemented");
  }

  /////////////////////////////////////////////////////////////////////////////

  // put periodic intersection checking in here for "to" boxes
  if (isPeriodic)
  {
    for (LayoutIterator to(a_dest.layoutIterator()); to.ok(); ++to)
    {
      Box ghost(dest[to()]);
      ghost.grow(a_ghost);
      //unsigned int toProcID = dest.procID(to());  // unused variable

      // only do this if ghost box hangs over domain edge
      if (!domainBox.contains(ghost))
      {
        // add the dataIndex for this box to the list
        // of boxes which we need to come back to
        periodicallyFilledToVect.push_back(DataIndex(to()));
        // now check to see if we need to grow the
        // periodic check radius
        if (!grownDomainCheckBox.contains(ghost))
        {
          // grow the domainCheckBox until it contains ghost
          while (!grownDomainCheckBox.contains(ghost))
          {
            grownDomainCheckBox.grow(1);
            periodicCheckRadius++;
          }
        } // end if we need to grow radius around domain

      } //end if ghost box is not contained in domain
    } // end if periodic
  }

  // Here ends the so-called N-squared optimizations.  the rest is unchanged. (ndk)

  // now do periodic checking, if necessary
  if (isPeriodic)
    {

      // the only "from" boxes we will need to check
      // will be those within periodicCheckRadius of the
      // domain boundary. so, create a box to screen out
      // those which we will need to check.
      Box shrunkDomainBox = a_domain.domainBox();
      shrunkDomainBox.grow(-periodicCheckRadius);

      ShiftIterator shiftIt = a_domain.shiftIterator();
      IntVect shiftMult(domainBox.size());

      // now loop over "from" boxes
      for (LayoutIterator from(a_src.layoutIterator()); from.ok(); ++from)
        {
          // first check to see whether we need to look at this box
          const Box& fromBox = level[from()];

          if (!shrunkDomainBox.contains(fromBox))
            {
              unsigned int fromProcID = level.procID(from());

              // check to see if fromBox is contained in domain,
              // if not, add it to the list of fromBoxes we need to
              // go back and check separately to see if it will
              // fill one of the "to" boxes
              if (!domainBox.contains(fromBox))
                {
                  periodicFromVect.push_back(DataIndex(from()));

                  if (!grownFromDomainCheckBox.contains(fromBox))
                    {
                      while (!grownFromDomainCheckBox.contains(fromBox))
                        {
                          grownFromDomainCheckBox.grow(1);
                          periodicFromCheckRadius++;
                        }
                    } // end if we need to grow domain check box
                } // end if fromBox is outside domain

              // now loop over those "to" boxes which were not contained
              // in the domain
              for (int toRef = 0; toRef < periodicallyFilledToVect.size(); toRef++)
                {
                  DataIndex toIndex = periodicallyFilledToVect[toRef];
                  unsigned int toProcID = dest.procID(toIndex);

                  // don't worry about anything that doesn't involve this proc
                  if (toProcID != m_procID && fromProcID != m_procID)
                    {
                      // do nothing
                    }
                  else
                    {
                      Box ghost(dest[toIndex]);
                      ghost.grow(a_ghost);
                      // now need to loop over shift vectors and look at images
                      for (shiftIt.begin(); shiftIt.ok(); ++shiftIt)
                        {
                          IntVect shiftVect(shiftIt()*shiftMult);
                          ghost.shift(shiftVect);
                          if (ghost.intersectsNotEmpty(fromBox)) // rarely happens
                            {
                              Box intersectBox(ghost);
                              intersectBox &= fromBox;
                              Box toBox(intersectBox);
                              toBox.shift(-shiftVect);
                              if (toProcID == fromProcID) // local move
                                {
                                  AsyncSendInstructions item( DataIndex(from())
                                                            , DataIndex(toIndex)
                                                            , intersectBox
                                                            , toBox);
                                  m_senderMotionPlan.insert(item);

                                  AsyncRegion region( DataIndex(toIndex)
                                                    , toBox); 
                                  m_regions.insert(region);
                                }
                              else if (fromProcID == m_procID)
                                {
                                  AsyncSendInstructions item( DataIndex(from())
                                                            , DataIndex(toIndex)
                                                            , intersectBox
                                                            , toBox);
                                  m_senderMotionPlan.insert(item);
                                }
                              else
                                {
                                  AsyncRegion region( DataIndex(toIndex)
                                                    , toBox); 
                                  m_regions.insert(region);
                                }

                            } // end if shifted box intersects

                          ghost.shift(-shiftVect);
                        } // end loop over shift vectors
                    } // end if either from box or to box are on this proc
                } // end loop over destination boxes
            } // end if source box is close to domain boundary
        } // end loop over destination boxes

      // now go back through the "from" boxes which were outside
      // the domain and see if they intersect any toBoxes
      if (periodicFromVect.size() != 0)
        {
          // the only "to" boxes we will need to check
          // will be those within periodicCheckRadius of the
          // domain boundary. so, create a box to screen out
          // those which we will need to check.
          shrunkDomainBox = a_domain.domainBox();
          shrunkDomainBox.grow(-periodicFromCheckRadius);

          // now loop over the "to" boxes
          for (LayoutIterator to(a_dest.layoutIterator()); to.ok(); ++to)
            {
              // first check to see whether we need to look at this box
              Box ghost(dest[to()]);
              ghost.grow(a_ghost);

              if (!shrunkDomainBox.contains(ghost))
                {
                  unsigned int toProcID = a_dest.procID(to());

                  // now loop over those "from" boxes which are not
                  // contained by the domain
                  for (int fromRef = 0; fromRef < periodicFromVect.size(); fromRef++)
                    {
                      DataIndex fromIndex = periodicFromVect[fromRef];
                      const Box& fromBox = level[fromIndex];
                      unsigned int fromProcID = level.procID(fromIndex);

                      // don't worry about anything which doesn't involve
                      // this proc
                      if (toProcID != m_procID && fromProcID != m_procID)
                        {
                          // do nothing
                        }
                      else
                        {
                          // now need to loop over shift vectors and look at images
                          for (shiftIt.begin(); shiftIt.ok(); ++shiftIt)
                          {
                            IntVect shiftVect(shiftIt()*shiftMult);
                            ghost.shift(shiftVect);
                            if (ghost.intersectsNotEmpty(fromBox))
                            {
                              Box intersectBox(ghost);
                              intersectBox &= fromBox;
                              Box toBox(intersectBox);
                              toBox.shift(-shiftVect);
                              if (toProcID == fromProcID) // local move
                              {
                                AsyncSendInstructions item( DataIndex(fromIndex)
                                                          , DataIndex(to())
                                                          , intersectBox
                                                          , toBox);
                                m_senderMotionPlan.insert(item);

                                AsyncRegion region( DataIndex(to())
                                                  , toBox);
                                m_regions.insert(region);
                              }
                              else if (fromProcID == m_procID)
                              {
                                AsyncSendInstructions item( DataIndex(fromIndex)
                                                          , DataIndex(to())
                                                          , intersectBox
                                                          , toBox);
                                m_senderMotionPlan.insert(item);
                              }
                              else
                              {
                                AsyncRegion region( DataIndex(to())
                                                  , toBox);
                                m_regions.insert(region);
                              }

                            } // end if shifted box intersects

                            ghost.shift(-shiftVect);
                          } // end loop over shift vectors
                        } // end if either from box or to box are on this proc
                    } // end loop over "from" boxes
                } // end if destination box is close to domain boundary
            } // end loop over destination boxes
        } // end if any of the "From" boxes were outside the domain

    } // end if we need to do anything for periodicity
}

void AsyncCopier::exchangeDefine(const DisjointBoxLayout& a_grids,
                                 const IntVect& a_ghost,
                                 int a_procID)
{
  clear();

  m_src  = a_grids;
  m_dest = a_grids;

  m_procID = a_procID;

  DataIterator dit = a_grids.dataIterator();
  NeighborIterator nit(a_grids);
  for (dit.begin(); dit.ok(); ++dit)
  {
    const Box& b = a_grids[dit];
    int toProcID = a_grids.procID(dit());

    if (toProcID != m_procID)
        continue;

    Box bghost(b);
    bghost.grow(a_ghost);
    for (nit.begin(dit()); nit.ok(); ++nit)
    {
      Box neighbor = nit.box();
      int fromProcID = a_grids.procID(nit());
      if (neighbor.intersectsNotEmpty(bghost))
      {
        Box box(neighbor & bghost);

        if (fromProcID == m_procID)
        { // local move
          AsyncSendInstructions item( DataIndex(nit())
                                    , dit() 
                                    , nit.unshift(box) 
                                    , box);
          m_senderMotionPlan.insert(item);

          AsyncRegion region( dit()
                            , box);
          m_regions.insert(region);
        }
        else
        {
          AsyncRegion region( dit()
                            , box);
          m_regions.insert(region);
        }
      }
      neighbor.grow(a_ghost);
      if (neighbor.intersectsNotEmpty(b) && fromProcID != m_procID)
      {
        Box box(neighbor & b);
        AsyncSendInstructions item( dit() 
                                  , DataIndex(nit()) 
                                  , box
                                  , nit.unshift(box)); 
        m_senderMotionPlan.insert(item);
      }
    }

  }
}

#include "NamespaceFooter.H"
