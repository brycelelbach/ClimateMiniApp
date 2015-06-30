//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <immintrin.h>
//------------------------------------------------------------------------------------------------------------------------------
#define apply_op_betai_ij(x_n,idx) (                                                                                                                       \
  minus_b_h2inv_STENCIL_TWELFTH*(                                                                                                                          \
     + beta_i[idx        ]*( 15.0*(x_n[idx-1      ]-x_n[idx]) - (x_n[idx-2       ]-x_n[idx+1      ]) )                                                     \
     + beta_i[idx+1      ]*( 15.0*(x_n[idx+1      ]-x_n[idx]) - (x_n[idx+2       ]-x_n[idx-1      ]) )                                                     \
     + 0.25*(                                                                                                                                              \
       + (beta_i[idx        +jStride]-beta_i[idx        -jStride]) * (x_n[idx-1      +jStride]-x_n[idx+jStride]-x_n[idx-1      -jStride]+x_n[idx-jStride]) \
       + (beta_i[idx        +kStride]-beta_i[idx        -kStride]) * (x_n[idx-1      +kStride]-x_n[idx+kStride]-x_n[idx-1      -kStride]+x_n[idx-kStride]) \
       + (beta_i[idx+1      +jStride]-beta_i[idx+1      -jStride]) * (x_n[idx+1      +jStride]-x_n[idx+jStride]-x_n[idx+1      -jStride]+x_n[idx-jStride]) \
       + (beta_i[idx+1      +kStride]-beta_i[idx+1      -kStride]) * (x_n[idx+1      +kStride]-x_n[idx+kStride]-x_n[idx+1      -kStride]+x_n[idx-kStride]) \
     )                                                                                                                                                     \
   )                                                                                                                                                       \
)
#define apply_op_betaj_ij(x_n,idx) (                                                                                                                        \
  minus_b_h2inv_STENCIL_TWELFTH*(                                                                                                                          \
     + beta_j[idx        ]*( 15.0*(x_n[idx-jStride]-x_n[idx]) - (x_n[idx-jStride2]-x_n[idx+jStride]) )                                                     \
     + beta_j[idx+jStride]*( 15.0*(x_n[idx+jStride]-x_n[idx]) - (x_n[idx+jStride2]-x_n[idx-jStride]) )                                                     \
     + 0.25*(                                                                                                                                              \
       + (beta_j[idx        +1      ]-beta_j[idx        -1      ]) * (x_n[idx-jStride+1      ]-x_n[idx+1      ]-x_n[idx-jStride-1      ]+x_n[idx-1      ]) \
       + (beta_j[idx        +kStride]-beta_j[idx        -kStride]) * (x_n[idx-jStride+kStride]-x_n[idx+kStride]-x_n[idx-jStride-kStride]+x_n[idx-kStride]) \
       + (beta_j[idx+jStride+1      ]-beta_j[idx+jStride-1      ]) * (x_n[idx+jStride+1      ]-x_n[idx+1      ]-x_n[idx+jStride-1      ]+x_n[idx-1      ]) \
       + (beta_j[idx+jStride+kStride]-beta_j[idx+jStride-kStride]) * (x_n[idx+jStride+kStride]-x_n[idx+kStride]-x_n[idx+jStride-kStride]+x_n[idx-kStride]) \
     )                                                                                                                                                     \
   )                                                                                                                                                       \
)
#define apply_op_betak_ij(x_n,idx) (                                                                                                                        \
  minus_b_h2inv_STENCIL_TWELFTH*(                                                                                                                          \
     + beta_k[idx        ]*( 15.0*(x_n[idx-kStride]-x_n[idx]) - (x_n[idx-kStride2]-x_n[idx+kStride]) )                                                     \
     + beta_k[idx+kStride]*( 15.0*(x_n[idx+kStride]-x_n[idx]) - (x_n[idx+kStride2]-x_n[idx-kStride]) )                                                     \
     + 0.25*(                                                                                                                                              \
       + (beta_k[idx        +1      ]-beta_k[idx        -1      ]) * (x_n[idx-kStride+1      ]-x_n[idx+1      ]-x_n[idx-kStride-1      ]+x_n[idx-1      ]) \
       + (beta_k[idx        +jStride]-beta_k[idx        -jStride]) * (x_n[idx-kStride+jStride]-x_n[idx+jStride]-x_n[idx-kStride-jStride]+x_n[idx-jStride]) \
       + (beta_k[idx+kStride+1      ]-beta_k[idx+kStride-1      ]) * (x_n[idx+kStride+1      ]-x_n[idx+1      ]-x_n[idx+kStride-1      ]+x_n[idx-1      ]) \
       + (beta_k[idx+kStride+jStride]-beta_k[idx+kStride-jStride]) * (x_n[idx+kStride+jStride]-x_n[idx+jStride]-x_n[idx+kStride-jStride]+x_n[idx-jStride]) \
     )                                                                                                                                                     \
   )                                                                                                                                                       \
)
//------------------------------------------------------------------------------------------------------------------------------
void smooth(level_type * level, int x_id, int rhs_id, double a, double b){
  int s;
  for(s=0;s<2*NUM_SMOOTHS;s++){ // there are two sweeps per GSRB smooth

    // exchange the ghost zone...
    #ifdef GSRB_OOP // out-of-place GSRB ping pongs between x and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}
    #else // in-place GSRB only operates on x
                 exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,        x_id,stencil_get_shape());
    #endif

    // apply the smoother...
    uint64_t _timeStart = CycleTime();

    #pragma omp parallel
    {
    const int  ghosts  = level->box_ghosts;
    const int jStride  = level->box_jStride;
    const int kStride  = level->box_kStride;
    const int jStride2 = jStride*2;
    const int kStride2 = kStride*2;
    const int     dim  = level->box_dim;
    const double h2inv = 1.0/(level->h*level->h);
    const double minus_b_h2inv_STENCIL_TWELFTH = -b*h2inv*STENCIL_TWELFTH;
    const double quarter = 0.25;
    const double fifteen = 15.0;
    __m512d STENCIL_TWELFTH_splat = _mm512_extload_pd(&minus_b_h2inv_STENCIL_TWELFTH,_MM_UPCONV_PD_NONE,_MM_BROADCAST_1X8,_MM_HINT_NONE);
    __m512d         quarter_splat = _mm512_extload_pd(&quarter                      ,_MM_UPCONV_PD_NONE,_MM_BROADCAST_1X8,_MM_HINT_NONE);
    __m512d         fifteen_splat = _mm512_extload_pd(&fifteen                      ,_MM_UPCONV_PD_NONE,_MM_BROADCAST_1X8,_MM_HINT_NONE);

    #if 0
    const int numThreads = omp_get_num_threads();
    const int threadID = omp_get_thread_num();
    int ijmax = dim*jStride; // box relative
    int ijlo,ijhi; // thread relative
    ijlo = (ijmax*(threadID  )) / numThreads;
    ijhi = (ijmax*(threadID+1)) / numThreads;

                              ijlo = (ijlo & ~0x7); // round down
    if(threadID!=numThreads-1)ijhi = (ijhi & ~0x7); // round down (except the last thread which must finish the plane)
    if(ijhi > ijmax)ijhi=ijmax;
    int VL = ijhi-ijlo;
    #else
    const int numThreads = omp_get_num_threads();
    const int threadID = omp_get_thread_num();
    int ijmax = dim*jStride; // box relative
    int ijlo,ijhi; // thread relative
    int klo = threadID & 0x1;
    ijlo = (ijmax*((threadID>>1)  )) / (numThreads>>1);
    ijhi = (ijmax*((threadID>>1)+1)) / (numThreads>>1);

                              ijlo = (ijlo & ~0x7); // round down
    if(threadID< numThreads-2)ijhi = (ijhi & ~0x7); // round down (except the last thread which must finish the plane)
    if(ijhi > ijmax)ijhi=ijmax;
    int VL = ijhi-ijlo;
    #endif

    int box;
    for(box=0;box<level->num_my_boxes;box++){
      int ij,k;
      const int color000 = (level->my_boxes[box].low.i^level->my_boxes[box].low.j^level->my_boxes[box].low.k^s)&1;  // is element 000 red or black on *THIS* sweep

      #warning GSRB using pre-computed 1.0/0.0 FP array for Red-Black to facilitate vectorization...
      #if 0
      for(k=0;k<dim;k++){
      #else 
      for(k=klo;k<dim;k+=2){
      #endif
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        const double * __restrict__ RedBlack = level->RedBlack_FP                          + ghosts*(1+jStride        ) + ijlo + kStride*((k^color000)&0x1);
        const double * __restrict__ rhs      = level->my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride) + ijlo + k*kStride;
        const double * __restrict__ alpha    = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride) + ijlo + k*kStride;
        const double * __restrict__ beta_i   = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride) + ijlo + k*kStride;
        const double * __restrict__ beta_j   = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride) + ijlo + k*kStride;
        const double * __restrict__ beta_k   = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride) + ijlo + k*kStride;
        const double * __restrict__ Dinv     = level->my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride) + ijlo + k*kStride;
        #ifdef GSRB_OOP
        const double * __restrict__ x_n;
              double * __restrict__ x_np1;
                       if((s&1)==0){x_n      = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + ijlo + k*kStride;
                                    x_np1    = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride) + ijlo + k*kStride;}
                               else{x_n      = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride) + ijlo + k*kStride;
                                    x_np1    = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + ijlo + k*kStride;}
        #else
        const double * __restrict__ x_n      = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + ijlo + k*kStride; // i.e. [0] = first non ghost zone point
              double * __restrict__ x_np1    = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + ijlo + k*kStride; // i.e. [0] = first non ghost zone point
        #endif
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        __assume_aligned(rhs   ,64);
        __assume_aligned(alpha ,64);
        __assume_aligned(beta_i,64);
        __assume_aligned(beta_j,64);
        __assume_aligned(beta_k,64);
        __assume_aligned(Dinv  ,64);
        __assume_aligned(x_n   ,64);
        __assume_aligned(x_np1 ,64);
        #if (BOX_ALIGN_JSTRIDE == 8)
        __assume((+jStride ) % 8 == 0);
        __assume((-jStride ) % 8 == 0);
        __assume((+jStride2) %16 == 0);
        __assume((-jStride2) %16 == 0);
        #endif
        #if (BOX_ALIGN_KSTRIDE == 8)
        __assume((+kStride ) % 8 == 0);
        __assume((-kStride ) % 8 == 0);
        __assume((+kStride2) %16 == 0);
        __assume((-kStride2) %16 == 0);
        #endif


        #if 0
        for(ij=0;ij<VL;ij+=8){
          __m512d xminus8 = _mm512_load_pd(x_n+ij-8);
          __m512d x222    = _mm512_load_pd(x_n+ij  );
          __m512d xplus8  = _mm512_load_pd(x_n+ij+8);
          __m512d x022    = _mm512_castsi512_pd(_mm512_alignr_epi32(_mm512_castpd_si512(x222  ),_mm512_castpd_si512(xminus8),12));
          __m512d x122    = _mm512_castsi512_pd(_mm512_alignr_epi32(_mm512_castpd_si512(x222  ),_mm512_castpd_si512(xminus8),14));
          __m512d x322    = _mm512_castsi512_pd(_mm512_alignr_epi32(_mm512_castpd_si512(xplus8),_mm512_castpd_si512(x222   ), 2));
          __m512d x422    = _mm512_castsi512_pd(_mm512_alignr_epi32(_mm512_castpd_si512(xplus8),_mm512_castpd_si512(x222   ), 4));
          
          __m512d b222    = _mm512_load_pd(beta_i+ij  );
          __m512d bplus8  = _mm512_load_pd(beta_i+ij+8);
          __m512d b322    = _mm512_castsi512_pd(_mm512_alignr_epi32(_mm512_castpd_si512(bplus8),_mm512_castpd_si512(b222   ), 2));

          __m512d highorder = _mm512_fmadd_pd(b222, _mm512_fmsub_pd(fifteen_splat,_mm512_sub_pd(x122,x222), _mm512_sub_pd(x022,x322) ),
                                _mm512_mul_pd(b322, _mm512_fmsub_pd(fifteen_splat,_mm512_sub_pd(x322,x222), _mm512_sub_pd(x422,x122) )) );

          __m512d b212;b212 = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(b212,beta_i+ij  -jStride),beta_i+ij  -jStride+8);
          __m512d b232;b232 = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(b232,beta_i+ij  +jStride),beta_i+ij  +jStride+8);
          __m512d b221;b221 =                                     _mm512_load_pd(beta_i+ij  -kStride);
          __m512d b223;b223 =                                     _mm512_load_pd(beta_i+ij  +kStride);
          __m512d b332;b332 = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(b322,beta_i+ij+1+jStride),beta_i+ij+1+jStride+8);
          __m512d b312;b312 = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(b322,beta_i+ij+1-jStride),beta_i+ij+1-jStride+8);
          __m512d b323;b323 = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(b322,beta_i+ij+1+kStride),beta_i+ij+1+kStride+8);
          __m512d b321;b321 = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(b322,beta_i+ij+1-kStride),beta_i+ij+1-kStride+8);

          __m512d mixedpartial;
          __m512d x0a;x0a = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(x0a,x_n+ij-1+jStride),x_n+ij-1+jStride+8);
          __m512d x0b;x0b = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(x0b,x_n+ij  +jStride),x_n+ij  +jStride+8);
          __m512d x0c;x0c = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(x0c,x_n+ij-1-jStride),x_n+ij-1-jStride+8);
          __m512d x0d;x0d = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(x0d,x_n+ij  -jStride),x_n+ij  -jStride+8);
          mixedpartial =   _mm512_mul_pd( _mm512_sub_pd(b232,b212), _mm512_add_pd(_mm512_sub_pd(_mm512_sub_pd(x0a,x0b),x0c),x0d)             );

          __m512d x1a;x1a = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(x1a,x_n+ij-1+kStride),x_n+ij-1+kStride+8);
          __m512d x1b;x1b =                                    _mm512_load_pd(x_n+ij  +kStride);
          __m512d x1c;x1c = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(x1c,x_n+ij-1-kStride),x_n+ij-1-kStride+8);
          __m512d x1d;x1d =                                    _mm512_load_pd(x_n+ij  -kStride);
          mixedpartial = _mm512_fmadd_pd( _mm512_sub_pd(b223,b221), _mm512_add_pd(_mm512_sub_pd(_mm512_sub_pd(x1a,x1b),x1c),x1d),mixedpartial);

          __m512d x2a;x2a = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(x2a,x_n+ij+1+jStride),x_n+ij+1+jStride+8);
          __m512d x2c;x2c = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(x2c,x_n+ij+1-jStride),x_n+ij+1-jStride+8);
          mixedpartial = _mm512_fmadd_pd( _mm512_sub_pd(b332,b312), _mm512_add_pd(_mm512_sub_pd(_mm512_sub_pd(x2a,x0b),x2c),x0d),mixedpartial);

          __m512d x3a;x3a = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(x3a,x_n+ij+1+kStride),x_n+ij+1+kStride+8);
          __m512d x3c;x3c = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(x3c,x_n+ij+1-kStride),x_n+ij+1-kStride+8);
          mixedpartial = _mm512_fmadd_pd( _mm512_sub_pd(b323,b321), _mm512_add_pd(_mm512_sub_pd(_mm512_sub_pd(x3a,x1b),x3c),x1d),mixedpartial);

          __m512d result = _mm512_mul_pd(STENCIL_TWELFTH_splat,_mm512_fmadd_pd(quarter_splat,mixedpartial,highorder));

          _mm512_store_pd(x_np1+ij,result);
  
        }
        #else
        #pragma omp simd aligned(rhs,alpha,beta_j,beta_k,Dinv,x_n,x_np1:64)
        for(ij=0;ij<VL;ij++){x_np1[ij] = apply_op_betai_ij(x_n,ij);}
        #endif
        #pragma omp simd aligned(rhs,alpha,beta_j,beta_k,Dinv,x_n,x_np1:64)
        for(ij=0;ij<VL;ij++){x_np1[ij] += apply_op_betaj_ij(x_n,ij);}
        #pragma omp simd aligned(rhs,alpha,beta_j,beta_k,Dinv,x_n,x_np1:64)
        for(ij=0;ij<VL;ij++){x_np1[ij] += apply_op_betak_ij(x_n,ij);}
        #ifdef USE_HELMHOLTZ
        #pragma omp simd aligned(rhs,alpha,beta_j,beta_k,Dinv,x_n,x_np1:64)
        for(ij=0;ij<VL;ij++){x_np1[ij] += a*alpha[ij]*x_n[ij];}
        #endif
        #if 1
        #pragma omp simd aligned(rhs,alpha,beta_j,beta_k,Dinv,x_n,x_np1:64)
        for(ij=0;ij<VL;ij++){x_np1[ij] = x_n[ij] + RedBlack[ij]*Dinv[ij]*(rhs[ij]-x_np1[ij]);}
        #else
        for(ij=0;ij<VL;ij+=8){
          __m512d rb8 = _mm512_loadunpackhi_pd(_mm512_loadunpacklo_pd(rb8,RedBlack+ij),RedBlack+ij+8);
          _mm512_store_pd(x_np1+ij,_mm512_fmadd_pd(rb8, _mm512_mul_pd(_mm512_load_pd(Dinv+ij), _mm512_sub_pd(_mm512_load_pd(rhs+ij),_mm512_load_pd(x_np1+ij))), _mm512_load_pd(x_n+ij) ) );
        } 
        #endif
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      } // k loop
    } // boxes
  }
    level->cycles.smooth += (uint64_t)(CycleTime()-_timeStart);
  } // s-loop
}


//------------------------------------------------------------------------------------------------------------------------------
