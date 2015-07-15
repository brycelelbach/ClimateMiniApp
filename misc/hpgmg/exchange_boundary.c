//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
// perform a (intra-level) ghost zone exchange
//  NOTE exchange_boundary() only exchanges the boundary.  
//  It will not enforce any boundary conditions
//  BC's are either the responsibility of a separate function or should be fused into the stencil
void exchange_boundary(level_type * level, int id, int shape){
  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;

  if(shape>=STENCIL_MAX_SHAPES)shape=STENCIL_SHAPE_BOX;  // shape must be < STENCIL_MAX_SHAPES in order to safely index into exchange_ghosts[]
  int my_tag = (level->tag<<4) | shape;
  int buffer=0;
  int n;

  #ifdef USE_MPI
  int nMessages = level->exchange_ghosts[shape].num_recvs + level->exchange_ghosts[shape].num_sends;
  MPI_Request *recv_requests = level->exchange_ghosts[shape].requests;
  MPI_Request *send_requests = level->exchange_ghosts[shape].requests + level->exchange_ghosts[shape].num_recvs;

  // loop through packed list of MPI receives and prepost Irecv's...
  if(level->exchange_ghosts[shape].num_recvs>0){
    _timeStart = CycleTime();
    #ifdef USE_MPI_THREAD_MULTIPLE
    #pragma omp parallel for schedule(dynamic,1)
    #endif
    for(n=0;n<level->exchange_ghosts[shape].num_recvs;n++){
      MPI_Irecv(level->exchange_ghosts[shape].recv_buffers[n],
                level->exchange_ghosts[shape].recv_sizes[n],
                MPI_DOUBLE,
                level->exchange_ghosts[shape].recv_ranks[n],
                my_tag,
                MPI_COMM_WORLD,
                &recv_requests[n]
      );
    }
    _timeEnd = CycleTime();
    level->cycles.ghostZone_recv += (_timeEnd-_timeStart);
  }


  // pack MPI send buffers...
  if(level->exchange_ghosts[shape].num_blocks[0]){
    _timeStart = CycleTime();
    PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[shape].num_blocks[0])
    for(buffer=0;buffer<level->exchange_ghosts[shape].num_blocks[0];buffer++){
      CopyBlock(level,id,&level->exchange_ghosts[shape].blocks[0][buffer]);
    }
    _timeEnd = CycleTime();
    level->cycles.ghostZone_pack += (_timeEnd-_timeStart);
  }

 
  // loop through MPI send buffers and post Isend's...
  if(level->exchange_ghosts[shape].num_sends>0){
    _timeStart = CycleTime();
    #ifdef USE_MPI_THREAD_MULTIPLE
    #pragma omp parallel for schedule(dynamic,1)
    #endif
    for(n=0;n<level->exchange_ghosts[shape].num_sends;n++){
      MPI_Isend(level->exchange_ghosts[shape].send_buffers[n],
                level->exchange_ghosts[shape].send_sizes[n],
                MPI_DOUBLE,
                level->exchange_ghosts[shape].send_ranks[n],
                my_tag,
                MPI_COMM_WORLD,
                &send_requests[n]
      ); 
    }
    _timeEnd = CycleTime();
    level->cycles.ghostZone_send += (_timeEnd-_timeStart);
  }
  #endif


  // exchange locally... try and hide within Isend latency... 
  if(level->exchange_ghosts[shape].num_blocks[1]){
    _timeStart = CycleTime();
    PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[shape].num_blocks[1])
    for(buffer=0;buffer<level->exchange_ghosts[shape].num_blocks[1];buffer++){
      CopyBlock(level,id,&level->exchange_ghosts[shape].blocks[1][buffer]);
    }
    _timeEnd = CycleTime();
    level->cycles.ghostZone_local += (_timeEnd-_timeStart);
  }


  // wait for MPI to finish...
  #ifdef USE_MPI 
  if(nMessages){
    _timeStart = CycleTime();
    MPI_Waitall(nMessages,level->exchange_ghosts[shape].requests,level->exchange_ghosts[shape].status);
    _timeEnd = CycleTime();
    level->cycles.ghostZone_wait += (_timeEnd-_timeStart);
  }


  // unpack MPI receive buffers 
  if(level->exchange_ghosts[shape].num_blocks[2]){
    _timeStart = CycleTime();
    PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[shape].num_blocks[2])
    for(buffer=0;buffer<level->exchange_ghosts[shape].num_blocks[2];buffer++){
      CopyBlock(level,id,&level->exchange_ghosts[shape].blocks[2][buffer]);
    }
    _timeEnd = CycleTime();
    level->cycles.ghostZone_unpack += (_timeEnd-_timeStart);
  }
  #endif

 
  level->cycles.ghostZone_total += (uint64_t)(CycleTime()-_timeCommunicationStart);
}