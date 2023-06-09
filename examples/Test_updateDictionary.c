/*
 * Copyright (c) Yann Collet, Meta Platforms, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <malloc.h>
#include <unistd.h> /* access check file exisit. */
#include <pthread.h>
#include <getopt.h>

#include "../lib/zstd.h"
#include "../lib/common.h"
#include "../lib/common/debug.h"
#include "../lib/zstd_errors.h"
#include "../lib/zdict.h"
#include "../programs/util.h"
#include "../programs/dibio.h"          /* DiB_trainFromFiles */
#include "../programs/timefn.h"         /* UTIL_time_t, UTIL_clockSpanMicro, UTIL_getTime */
#include "../lib/common/error_private.h"
#include "../lib/compress/zstd_compress_internal.h"

// #define KB *(1<<10)
// #define MB *(1<<20)
#define DICT_BUFFER_SIZE 32768
#define MAX_DICTSIZE 112640      
#define DEFAULT_ACCEL_TEST 1
#define SAMPLESIZE_MAX (128 KB)
// #define MIN(a,b)    ((a) < (b) ? (a) : (b))
#define DiB_rotl32_1(x,r) ((x << r) | (x >> (32 - r)))
#define DISPLAYLEVEL 2
#define CLEVEL  3
#define BLOCKSIZE 4096
#define COMPCHUNKSIZE (5 MB)
#define TRAINCHUNKSIZE (1 MB)
#define DSDROPMIN   0.5
#define DSMIN       0.1
static const unsigned kDefaultRegression = 1;

/* Parameter for train and compress.*/
typedef struct pthread_train_compress
{
    void *srcBuffer;
    void *dictBuffer;
    void *dictBuffer_old;
    pthread_mutex_t lock;
    pthread_cond_t updateDict;
    pthread_cond_t dictComp;
    size_t MaxDictSize;
    size_t totalConsumedSize;
    size_t dict_Exist;                  /* 0:Dictionary doesn't exit,train by@train().
                                         1: Dictionary exit,consume by@dictCompress().*/
    size_t trainDict_Break;             /* 0: Continue train.
                                        -1:Fininsh training.*/
    size_t srcSize;                     /* Source data size. */
    size_t dictSize;
    size_t dictSize_old;
    size_t blockSize;                   /* Compress and Train blocksize. */
    size_t tempcSize;                   /* Compress chunksize.*/
    size_t temptSize;                   /*Train chunksize.*/
    // size_t trainNewDict;             /* 1: Train new dictionary.
    //                                  0: Use the old dictionary.*/
    size_t trainConsumed;
    size_t exitThread;
   /* Match Hit. */
    size_t dictHit_singelChunk;         /* Singel Chunk dictionary match hit. */
    size_t srcHit_singelChunk;          /* Singel Chunk source match hit. */
    size_t dictHit_total;               /* Total dictionary match hit. */
    size_t srcHit_total;                /* Total source match hit. */
    size_t check_blockNumb_train;
    size_t check_blockNumb_comp;
    // double hitRatio_singelChunk;    
    // double hitRatio_total;
    /* Match Length. */
    // double mLengthRatio_singelChunk;    
    // double mLengthRatio_total;
    size_t dictmLength_singelChunk;     /* Singel Chunk dictionary  mLength. */
    size_t srcmLength_singelChunk;      /* Singel Chunk source mLength. */
    size_t dictmLength_total;           /* Total dictionary mLength. */
    size_t srcmLength_total;            /* Total source  mLength. */
    int trainDictOnce;                  /* Only use the first dictionary. */
    void* dictBuffer_first;             /* The first dictionary.  */
    size_t dictSize_first;
    double ratio ;                      /* Compress Ratio. */
    // double *saveHit_ratio;      
    double *saveComp_ratio;             /* Compress Ratio each chunk. */
    // double *saveMatch_ratio;         /* mLength dict each singel chunk / srcLength each chunk*/
    double *saveMS_ratio;               /* mLength each chunk / chunkSize (srcSize). */
    double *saveDS_ratio;               /* dict mLength each chunk / chunkSize (srcSize). */
    double *saveMD_ratio;               /* Src match reatio change singel chunk. */
    double *saveMSDiff_ratio;           /* Difference = (the second MSRatio - the first MSRatio)*/
    double *saveDSDiff_ratio;
    double *savetrainTime_singelChunk;      /* Train time singel chunk. */
    double *savecompTime_singelChunk;       /* COmpress time singel chunk. */
    double MSRatio_total;                   /* MS: (Match Length in src) / (src Size) */
    double DSRatio_total;                   /* DS: (Match Length in dict) / (src Size)*/
    double trainTime;
    double compTime;
    double cctime;                          /* each chunk compress time. */
    double dsDrop;                          /* Drop to the value,will train a new dictionary. */
    int count;
    int display;
    int update_not;
    int count_unnormal;
    size_t tTimes;
    size_t cTimes;
}TC_params;
TC_params tc_params;    /* Galobal paramter for Train and Compress. */
void TC_params_free(TC_params *tc_p);
static ZDICT_fastCover_params_t defaultFastCoverParams(void);
/* Set samplessizes. */
static int SetsamplesSizes(size_t srcSize,size_t blockSize,size_t* samplesSizes);

/* Use the srcbuffer training to get a dictionary。 */
static size_t  DictionaryTrain_Stream(TC_params *tc_parameters,ZDICT_fastCover_params_t *parameters);

/* Use the given dictionary to compress secBuffer. */
static size_t DC_Stream(TC_params *tc_parameters);

/* Thread function :Regular Compress data.*/
// static size_t RegularComp_Stream(REG_compress_params regParams);

/* Thread function :Dictionary Compress data.*/
static void* multiple_DictionaryCompress_stream(void  *tc_parameters);

/* Thread function :Dictionary Training.*/
static void *multiple_DictionaryTrain_stream(void *tc_parameters);

/* Dictionary Compress function. */
static size_t DictionaryComp_Stream(void* srcBuffer,size_t srcSize,void* OutBuffer,ZSTD_CCtx* cctx);
// static size_t DictionaryComp_Stream(void* srcBuffer,size_t srcSize,void* dictBuffer,size_t dictSize,void* OutBuffer);
/* Initalize TC_params */
static int initalize_TC_params(TC_params *tc_p,void *srcBuffer,size_t srcSize,size_t MaxDictSize,
                                size_t blockSize,size_t compressChunkSize,size_t trainChunkSize,
                                int trainDictOnce,int display,int update);

/* Get size parameters. */
static int readU32FromCharChecked_(const char* stringPtr, unsigned* value);
/* Help function. */
void help_function();
/* Save compress ratio data. */
int saveDataToFile(char* file,double data);
int saveSingelChunk_data_ToFile(char *file,int compOrHit);
void saveData( char* ftotalComp_Ratio, char* fsingelComp_Ratio, char* ftotalHit_Ratio, char* fsingelHit_Ratio, 
                char* ftotalMatch_Ratio,char* fsingelMatch_Ratio,char* ftotalMS_Ratio,char* fsingelMS_Ratio,
                char* ftotalDS_Ratio,char* fsingelDS_Ratio,char* fsingelDSDiff,char *fsingelDS,
                char* ftotalTrain_Time,char* fsingelTrain_Time,char* ftotalComp_Time,char* fsingelComp_Time);

static U32 DiB_rand_(U32* src);
static int Random_LoadTrainBuffer(size_t trainSize,size_t blockSize,size_t endPosition,size_t *samplePosition);

void help_function(){
    printf("example: ./Test_updateDictionary file -B 4096 -T 1MB -C 5MB -M 110KB --trainDictOnce 0 -D 2\n");
    printf("\n");
    printf("-O# --trainDictOnce# \t: 1 Only use the first dictionary\n\t\t\t0 default:The first chunk use a empty dictionary,the second chunk use the dictionary trained from the first chunk,and so on.\n\t\t\t3 Each chunk use a new dictioary trained from current chunk.\n");
    printf("-B# --blockSize# \t: cut file into independent blocks of size # (default: 4096bytes.)\n");
    printf("-T# --trainChunkSize# \t: cut file into independent chunk of size when training dictionary # (default: 1 MB.)\n");
    printf("-M# --maxDictSize# \t: limit dictionary to specified size (default: 112640bytes.)\n");
    printf("-C# --compressChunkSize#: cut file into independent chunk of size when compress with dictionary # (default: 5 MB.)\n");
}

static int readU32FromCharChecked_(const char* stringPtr, unsigned* value)
{
    unsigned result = 0;
    while ((*stringPtr >='0') && (*stringPtr <='9')) {
        unsigned const max = ((unsigned)(-1)) / 10;
        unsigned last = result;
        if (result > max) return 1; /* overflow error */
        result *= 10;
        result += (unsigned)(*stringPtr - '0');
        if (result < last) return 1; /* overflow error */
        (stringPtr)++ ;
    }
    if ((*stringPtr=='K') || (*stringPtr=='M')) {
        unsigned const maxK = ((unsigned)(-1)) >> 10;
        if (result > maxK) return 1; /* overflow error */
        result <<= 10;
        if (*stringPtr=='M') {
            if (result > maxK) return 1; /* overflow error */
            result <<= 10;
        }
        (stringPtr)++;  /* skip `K` or `M` */
        if (*stringPtr=='i') (stringPtr)++;
        if (*stringPtr=='B') (stringPtr)++;
    }
    *value = result;
    return 0;
}

static int initalize_TC_params(TC_params *tc_p,void *srcBuffer,size_t srcSize,size_t MaxDictSize,
                                size_t blockSize,size_t compressChunkSize,size_t trainChunkSize,int trainDictOnce,
                                int display,int update)
{
    int check = 0;
    if (pthread_mutex_init(&tc_p->lock,NULL) != 0 )     {printf("pthread_mutex_init Fail!!!\n");check +=1;}
    if (pthread_cond_init(&tc_p->updateDict,NULL) != 0) {printf("pthread_cond_init Fail!!!\n");check +=1;}
    if (pthread_cond_init(&tc_p->dictComp,NULL) != 0)   {printf("pthread_cond_init Fail!!!\n");check +=1;}
    tc_p->srcSize = srcSize;
    tc_p->blockSize = blockSize;
    tc_p->dictSize = MaxDictSize;
    tc_p->dictSize_old = MaxDictSize;
    tc_p->tempcSize = compressChunkSize;    /* Compress chunksize. */
    tc_p->temptSize = trainChunkSize;       /* Train chunksize. */
    tc_p->MaxDictSize = MaxDictSize;        /* Max Dictionary size. */
    tc_p->totalConsumedSize = 0;            /* Consumed data size. */
    tc_p->dict_Exist = 0;                    /* Does the  dictionary exist. */
    tc_p->trainDict_Break = 0;              /* Whether */
    tc_p->trainDictOnce = trainDictOnce;
    // tc_p->trainNewDict = 0;
    tc_p->dictBuffer = calloc(MaxDictSize,sizeof(char));
    tc_p->dictBuffer_old = calloc(MaxDictSize,sizeof(char));
    tc_p->dictBuffer_first = calloc(MaxDictSize,sizeof(char));
    tc_p->dictSize_first = MaxDictSize;
    // memset(tc_p->dictBuffer,0xff,MAX_DICTSIZE);
    // memset(tc_p->dictBuffer_old,0xff,MAX_DICTSIZE);
    tc_p->srcBuffer = calloc(srcSize,sizeof(char));
    tc_p->exitThread = 0;
    tc_p->ratio = 0.0;
    /* Match Hit. */
    tc_p->dictHit_singelChunk = 0;
    tc_p->srcHit_singelChunk = 0;
    // tc_p->hitRatio_singelChunk = 0.0;
    tc_p->dictHit_total = 0;
    tc_p->srcHit_total = 0;
    // tc_p->hitRatio_total = 0.0;
    /* Match Length. */
    tc_p->dictmLength_singelChunk = 0;
    tc_p->srcmLength_singelChunk = 0;
    // tc_p->mLengthRatio_singelChunk = 0.0;
    tc_p->dictmLength_total = 0;
    tc_p->srcmLength_total = 0;
    // tc_p->mLengthRatio_total  = 0.0;

    tc_p->check_blockNumb_train = 0;
    tc_p->check_blockNumb_comp = 0;
    if (memmove(tc_p->srcBuffer,srcBuffer,srcSize) == NULL )    {printf("Initalize srcBuffer Fail!\n");}
    int nb = ((compressChunkSize + blockSize -1)/blockSize);
    tc_p->saveComp_ratio = (double*)calloc(nb,sizeof(double));
    // tc_p->saveHit_ratio = (double*)calloc(nb,sizeof(double));
    // tc_p->saveMatch_ratio = (double*)calloc(nb,sizeof(double));
    tc_p->saveMS_ratio = (double*)calloc(nb,sizeof(double));
    tc_p->saveMSDiff_ratio = (double*)calloc(nb,sizeof(double));
    tc_p->saveDS_ratio = (double*)calloc(nb,sizeof(double));
    tc_p->saveDSDiff_ratio = (double*)calloc(nb,sizeof(double));
    tc_p->savetrainTime_singelChunk = (double*)calloc(nb,sizeof(double));
    tc_p->savecompTime_singelChunk = (double*)calloc(nb,sizeof(double));

    tc_p->compTime = 0.0;
    tc_p->trainTime = 0.0;
    tc_p->DSRatio_total = 0.0;
    tc_p->MSRatio_total = 0.0;
    tc_p->count = 0;
    tc_p->display = display;
    tc_p->cctime = 0.0;
    tc_p->dsDrop = DSDROPMIN;
    tc_p->count_unnormal = 0;
    tc_p->tTimes = 0;
    if (update != 0)    {tc_p->dsDrop = (double)((update*10.0)/100.0);}
    return check;
}

// static int SvaeFile(char* FileName,size_t Size,void* srcBuff){
//     const char* f = FileName;
//     size_t size = Size;
//     void* buff = srcBuff;
//     int check = 0;
//     FILE* fp = fopen(f,"wb");
//     if (fp != NULL){
//         fwrite(buff,size,1,fp);
//     }
//     else{
//         check = 1;
//     }
//     fclose(fp);
//     return check;

// }
/* Regular Compression */
// static void compress_orDie(const char* fname, const char* oname)
// {
//     size_t fSize;
//     void* const fBuff = mallocAndLoadFile_orDie(fname, &fSize);
//     size_t const cBuffSize = ZSTD_compressBound(fSize);
//     void* const cBuff = malloc_orDie(cBuffSize);

//     size_t const cSize = ZSTD_compress(cBuff, cBuffSize, fBuff, fSize, 1);
//     CHECK_ZSTD(cSize);

//     saveFile_orDie(oname, cBuff, cSize);

//     /* success */
//     printf("%25s : %6u -> %7u - %s \n", fname, (unsigned)fSize, (unsigned)cSize, oname);

//     free(fBuff);
//     free(cBuff);
// }
/* Create a ouputFile like "xxx.zst" */
// static char* createOutFilename_orDie(const char* filename)
// {
//     size_t const inL = strlen(filename);
//     size_t const outL = inL + 5;
//     void* const outSpace = malloc_orDie(outL);
//     memset(outSpace, 0, outL);
//     strcat(outSpace, filename);
//     strcat(outSpace, ".zst");
//     return (char*)outSpace;
// }

/* Parameters for train dictionary. */
static ZDICT_fastCover_params_t defaultFastCoverParams(void)
{
    ZDICT_fastCover_params_t params;
    ZDICT_params_t zParams;
    zParams.compressionLevel = CLEVEL;
    zParams.notificationLevel = (unsigned) DISPLAYLEVEL;
    zParams.dictID = 0;
    memset(&params, 0, sizeof(params));
    params.d = 8;
    params.f = 20;
    params.steps = 4;
    params.splitPoint = 0.75; /* different from default splitPoint of cover */
    params.accel = DEFAULT_ACCEL_TEST;
    params.shrinkDict = 0;
    params.shrinkDictMaxRegression = kDefaultRegression;
    params.nbThreads = 1;
    params.zParams = zParams;
    return params;
}

 /* Create contex for dictcompress or dictdecompress. */
// static ZSTD_DDict* createDict_orDie(const char* dictFileName)
// {
//     size_t dictSize;
//     printf("loading dictionary %s \n", dictFileName);
//     void* const dictBuffer = mallocAndLoadFile_orDie(dictFileName, &dictSize);
//     ZSTD_DDict* const ddict = ZSTD_createDDict(dictBuffer, dictSize);
//     CHECK(ddict != NULL, "ZSTD_createDDict() failed!");
//     free(dictBuffer);
//     return ddict;
// }

/* Load file into buff. */
static void* loadFiletoBuff(const char * FileName,size_t *FileSize){
    size_t size;
    void *fBuff = 
        (unsigned char *) mallocAndLoadFile_orDie (FileName,
                    &size);
    *FileSize = size;
    return fBuff;
}
/* Load dictbuff into cctx. */
static void *loadDictToCCtx(ZSTD_CCtx* cctx,const void *dict,size_t dictSize){
    // CHECK( ZSTD_CCtx_loadDictionary(cctx, dict, dictSize) );
    ZSTD_CCtx_loadDictionary(cctx, dict, dictSize) ;
}

static int SetsamplesSizes(size_t srcSize,size_t blockSize,size_t* samplesSizes){

    size_t const sSize = srcSize;
    size_t const bSize = blockSize;
    unsigned nbSamples = (int)((srcSize+bSize-1)/bSize);
    int nbSamplesLoaded = 0;
    size_t srcSizeDataLoaded = 0;
    int check = 0;
    // {
    //     printf("srcSize = %ld\n",sSize);
    //     printf("blockSize = %ld\n",blockSize);
    // }
    if ( (sSize <= 0) || (bSize <= 0) ){    /* Check srcsize and block size .*/
            if(sSize <= 0){printf("Error:Sampledata size is zero or negative.\n");}
            if(bSize <=0){printf("Error:BlockSize is zero or negative.\n");}
            check = 1;
        } 
    else{
        while ( (nbSamplesLoaded < nbSamples) && (srcSizeDataLoaded < sSize) ){
                srcSizeDataLoaded = bSize > 0 ?
                                (size_t)MIN(sSize, (S64)bSize) :
                                (size_t)MIN(sSize, SAMPLESIZE_MAX );
                samplesSizes[nbSamplesLoaded++]  = srcSizeDataLoaded; 
                if(srcSizeDataLoaded >= sSize)
                    break;
                    /* If src-chuhking is enabled,continute  setting samplesSizes */
                while (srcSizeDataLoaded < sSize && nbSamplesLoaded < nbSamples){
                    size_t chunksize = MIN((size_t)(sSize-srcSizeDataLoaded),bSize);
                    if( (srcSizeDataLoaded+chunksize) > sSize)
                        break;
                    samplesSizes[nbSamplesLoaded++] += chunksize;
                    srcSizeDataLoaded += chunksize;
                }
            }
    }
    /* Check if samplesSize has been set correctly.*/
    size_t d = (int)((srcSize+blockSize-1)/blockSize);
    size_t count = 0;
    for (int i =0;(i < nbSamples) && (i < d);i++){
        size_t tempSize = samplesSizes[i];
        if(( tempSize != blockSize ) && ( i != nbSamples-1 )){
            printf("samplesLoaded [%d]  = %ld\n",i,tempSize);
            count++;
        }
    }
    if (count){
        printf("Total Nbsamples = %u\n",nbSamples);
        printf("Error SamplesSize Counts  = %ld\n",count);
        check = 1;
    }
    
    return check;
}

static U32 DiB_rand_(U32* src)
{
    static const U32 prime1 = 2654435761U;
    static const U32 prime2 = 2246822519U;
    U32 rand32 = *src;
    rand32 *= prime1;
    rand32 ^= prime2;
    rand32  = DiB_rotl32_1(rand32, 13);
    *src = rand32;
    return rand32 >> 5;
}
static int Random_LoadTrainBuffer(size_t trainSize,size_t blockSize,size_t endPosition,size_t *samplePosition)
{
    int nbSamples = (int)((trainSize+blockSize-1)/blockSize);
    int i = 0;
    int chunkSize = endPosition/nbSamples;
    U32 seed = 0xFD2FB528;
    if ( endPosition < trainSize ){
        printf("Error : Reaming data is not enough for generating a new dictionary!!!\n");
        return -1;
    }
    for(i=0;i < nbSamples;i++){
        samplePosition[i] = DiB_rand_(&seed)%(chunkSize)+chunkSize*i;
        if ( (endPosition-samplePosition[i]) < blockSize) {
            samplePosition[i] = samplePosition[i] - blockSize;
        }
        if ( samplePosition[i] > endPosition ) {
            printf("Random Loading data into train buffer fail.!");
            return -1;
        }
    }
    if (tc_params.display == 1){
        printf("Train Loop: Random Load into TrainBuffer.\n");
    }
    return 0;
}
/* Get a Dictionary by training the sample data stream.*/
static size_t  DictionaryTrain_Stream(TC_params *tc_parameters,ZDICT_fastCover_params_t *parameters)
{    
    size_t result = 0;
    pthread_mutex_lock(&tc_parameters->lock);
    while ( tc_parameters->dict_Exist == 1 )
    {
        if (tc_parameters->trainDict_Break == -1) break;
        if (tc_parameters->exitThread == 1) {
            pthread_mutex_unlock(&tc_parameters->lock);
            return -1;
        }
        if (tc_parameters->display == 1){
            printf("Waitting dictionary has been consumed.\n");
        }
        pthread_cond_wait(&tc_parameters->updateDict,&tc_parameters->lock);
        if (tc_parameters->display == 1){
           printf("\nBack to train!!!\n");
        }
        
        if (tc_parameters->trainDictOnce == 2){
            if (((tc_parameters->srcSize)-(tc_parameters->totalConsumedSize)) <= (tc_parameters->temptSize)){
                tc_parameters->trainDict_Break = -1;
                pthread_cond_signal(&tc_parameters->dictComp);
                pthread_mutex_unlock(&tc_parameters->lock);
                return -1;
            }
            if (tc_parameters->display == 1){
                printf("\nDon't train new dictionary.\n");
            }
            tc_parameters->dict_Exist = 1;
            pthread_cond_signal(&tc_parameters->dictComp);
            pthread_mutex_unlock(&tc_parameters->lock);
            return -2;
        }
    } 
    /* Check if the data is exhausted. */
    if (((tc_parameters->srcSize)-(tc_parameters->totalConsumedSize)) <= (tc_parameters->temptSize)){
        tc_parameters->trainDict_Break = -1;
        result = -1;
    }
    /* If the remaining data is enough to generate a dictionary, continue training. */
    else{
        // int check_1 =0;
        
        
        size_t MaxDictSize = tc_parameters->MaxDictSize;
        void*  dictBuffer = malloc(MaxDictSize);  
        size_t dictSize = 0;
        size_t trainChunkSize = tc_parameters->temptSize;
        size_t blockSize = tc_parameters->blockSize;
        void* sampleBuffer = malloc(tc_parameters->temptSize+32);
        
        /* Set samplesSizes .*/
        int nbSamples = (int)((trainChunkSize+blockSize-1)/blockSize);
        if (tc_parameters->display == 1 ){
            printf("Train dictSize = %ld\n",MaxDictSize);
            printf("Train source size = %ld\n",trainChunkSize);
            printf("Train blocksize = %ld\n",blockSize);
            printf("\n");
        }
        
        size_t *samplePosition;
        samplePosition = (size_t*)calloc(nbSamples,sizeof(size_t));
        size_t endPosition = MIN((tc_parameters->srcSize - tc_parameters->totalConsumedSize),tc_parameters->tempcSize);
        int check_randLoad = Random_LoadTrainBuffer(trainChunkSize,blockSize,endPosition,samplePosition);
        size_t* sampleSizes;
        sampleSizes = (size_t*)calloc(nbSamples,sizeof(size_t));
        int check = SetsamplesSizes(trainChunkSize,blockSize,sampleSizes);
        if (check != 0){
            printf("Error: Set samplesSize Fail!!!\n");
            result = -1;
        }
        int i = 0;
        size_t samplebufferPosition = 0;
        size_t position = tc_parameters->totalConsumedSize;
        if ( (tc_parameters->count >= 1 ) && ( (tc_parameters->trainDictOnce == 4)||(tc_parameters->trainDictOnce == 5)) ){
            position = tc_parameters->totalConsumedSize - tc_parameters->tempcSize;
        }
        for(i=0;i<nbSamples;i++){
            if (memmove(sampleBuffer+samplebufferPosition,tc_parameters->srcBuffer+position+samplePosition[i],sampleSizes[i]) == NULL){
                    printf("Error: Copy train buffer fail!!!\n");
            }
            samplebufferPosition += sampleSizes [i];
        }
        if (tc_parameters->display == 1){
            printf("Train Loop: Start training.\n");
        }
        clock_t start_time,end_time;
        start_time = clock();
        dictSize = ZDICT_optimizeTrainFromBuffer_fastCover(dictBuffer, MaxDictSize,
                                                                sampleBuffer, sampleSizes, nbSamples,
                                                                parameters);

        end_time = clock();
        double time_singelChunk = 0.0;
        time_singelChunk = (double)(end_time - start_time)/CLOCKS_PER_SEC;
        if ( (dictSize > 0) && (dictSize <= MAX_DICTSIZE) ){
             if (tc_parameters->display == 1 ){
                printf("Train Loop: Dictionary Size = %ld.\n",dictSize);
            }
        
            tc_parameters->dictSize_old = tc_parameters->dictSize;
            if ( memmove(tc_parameters->dictBuffer_old,tc_parameters->dictBuffer,tc_parameters->dictSize) == NULL ){
                printf("Error Train:Copy tc_parameters->dictBuffer to tc_parameters->dictBuffer_old Fail!!!\n");
                result = -1;
            }
            result = tc_parameters->dictSize_old;
            /* If you change the copy order,it will improve the compress ratio.
                Because the training data is closer to the compressed data. trainDictOnce = 0 equal to trainDictOnce = 3.*/
            tc_parameters->dictSize = dictSize;
            if ( memmove(tc_parameters->dictBuffer,dictBuffer,dictSize) == NULL ){
                printf("Error Train:Copy dictBuffer to tc_parameters Fail!!!\n");
                result = -1;
            }
            if (tc_parameters->trainDictOnce == 1){
                tc_parameters->dictSize_first = dictSize;
                result = tc_parameters->dictSize_first;
                if ( memmove(tc_parameters->dictBuffer_first,dictBuffer,dictSize) == NULL ){
                    printf("Error Train:Copy dictBuffer to dictBuffer_first Fail!!!\n");
                    result = -1;
                }
                else{
                    tc_parameters->trainDictOnce = 2;
                    printf("Only tran dictionary once.\n");
                }
            }
        }
        else{
            printf("Error Train: Train Dictionary Fail!!!\nDictionary size = %ld\n",dictSize);
            result = -1;
        }
        
        tc_parameters->savetrainTime_singelChunk[tc_parameters->count] = time_singelChunk;
        tc_parameters->trainTime += time_singelChunk;
        free(dictBuffer);
        free(sampleBuffer);
        free(samplePosition);
        free(sampleSizes);
         if (tc_parameters->display == 1 ){
                printf("Train Loop: ConsumedSize = %ld\n",tc_parameters->totalConsumedSize);
                printf("Train Loop Remain = %ld\n",(tc_parameters->srcSize)-(tc_parameters->totalConsumedSize));
            }
        tc_parameters->dict_Exist = 1;
    }
    if (tc_parameters->trainDict_Break == -1){
        if (tc_parameters->display == 1 ){
                printf("Remaining data size isn't enough for training a new dictionary.\n");
                printf("Remaining data size = %ld\n",tc_parameters->srcSize-tc_parameters->totalConsumedSize);
            }
    }
    if (result == -1){
        tc_parameters->trainDict_Break = -1;
    }

    pthread_cond_signal(&tc_parameters->dictComp);
    pthread_mutex_unlock(&tc_parameters->lock);
    if (tc_parameters->display == 1 ){
               printf("Train trainDict_Break = %ld\n",tc_parameters->trainDict_Break);
        }
    return result;
}
/*  Comer case:
                if the src buffer which can't be compressed , outBuff.pos will be assignment an wrong value which is very samll and not true.
                to avoid this case must initalized cctx->streamStage :  
                cctx->streamStage = zcss_init.

                ZSTD doesn't have an stream compress mode api that you can use it to compress with chunk size small like 4096 bytes.
                In multiple compress mode ,you can use -B# to set block size.
                (minimum value is 512KB in default compress level,the minimum value will change if the compress level upper or lower.)*/
static size_t DictionaryComp_Stream(void* srcBuffer,size_t srcSize,void* OutBuffer,ZSTD_CCtx* cctx)
// static size_t DictionaryComp_Stream(void* srcBuffer,size_t srcSize,void* dictBuffer,size_t dictSize,void* OutBuffer)
{

    void *srcBuff = srcBuffer;
    size_t outSize = srcSize;
    void  *outBuffer = OutBuffer;

    /*Compress srcBuff*/
    ZSTD_inBuffer inBuff = { srcBuff, srcSize, 0 };
    ZSTD_outBuffer outBuff= { outBuffer, outSize, 0 };
    clock_t start_time,end_time;
    start_time = clock();
    ZSTD_compressStream2(cctx, &outBuff, &inBuff, ZSTD_e_end);
    cctx->streamStage = zcss_init;
    end_time = clock();
    double time_block = 0.0;
    time_block = (double)(end_time - start_time)/CLOCKS_PER_SEC;
    tc_params.cctime += time_block;
    tc_params.dictHit_singelChunk += dict_Hit;
    tc_params.srcHit_singelChunk += src_Hit;
    tc_params.dictmLength_singelChunk += dict_ml;
    tc_params.srcmLength_singelChunk += src_ml;
    outSize = outBuff.pos;
    // ZSTD_freeCCtx(cctx);
    return outSize;
} 
static size_t DC_Stream(TC_params *tc_parameters){

    int result = 0;
    pthread_mutex_lock(&tc_parameters->lock);
    /* Check whether dictionary exit.
        If dictionary doen't exit,call @train training a dictionary. */
    while ( tc_parameters->dict_Exist == 0 )
    {   
        if ( tc_parameters->trainDict_Break == -1){
            if (tc_parameters->display == 1){
                printf("Reuse the old dictionary!!!\n(If reamining data szie >0 ).\n");
            }
            
            break;
        }
        if (tc_parameters->display == 1){
            printf("Dictionary hasn't finished.\n");
        }
        pthread_cond_wait(&tc_parameters->dictComp,&tc_parameters->lock);
        if (tc_parameters->display == 1){
           printf("\nBack to Compress!!!\n");
        }
    }
    /* Compress with dictionary. */
    if ( tc_parameters->totalConsumedSize < tc_parameters->srcSize ){
        size_t dictSize = 0;
        void *dictBuffer;
        
        /* Copy dictionary  to dictBuffer. */
        // if ( tc_parameters->trainNewDict == 1 ){
        if ( tc_parameters->trainDictOnce == 2 ){
            dictSize = tc_parameters->dictSize_first;
            dictBuffer = calloc(dictSize,sizeof(char));
            // printf("\ntrainDIctOnce = %d\n",tc_parameters->trainDictOnce);
            if ( memmove(dictBuffer,tc_parameters->dictBuffer_first,dictSize) == NULL ){
                printf("Error DictCompress: Copy dictBuffer Fail!\n");
                tc_parameters->exitThread = 1;
                free(dictBuffer);  
                pthread_cond_signal(&tc_parameters->updateDict);
                pthread_mutex_unlock(&tc_parameters->lock);          
                return -1;
            }
            if (tc_parameters->display == 1){
                printf("Use the first dictionary\n\n");
            }
                    
        }
        else {
            if (tc_parameters->trainDictOnce == 3 || tc_parameters->trainDictOnce == 4
                || tc_parameters->trainDictOnce == 5){
                dictSize = tc_parameters->dictSize;
                dictBuffer = calloc(dictSize,sizeof(char));
                if ( memmove(dictBuffer,tc_parameters->dictBuffer,dictSize) == NULL ){
                    printf("Error DictCompress: Copy dictBuffer Fail!\n");
                    tc_parameters->exitThread = 1;
                    free(dictBuffer);  
                    pthread_cond_signal(&tc_parameters->updateDict);
                    pthread_mutex_unlock(&tc_parameters->lock);          
                    return -1;
                }
            }
            if (tc_parameters->trainDictOnce == 0){
                dictSize = tc_parameters->dictSize_old;
                dictBuffer = calloc(dictSize,sizeof(char));
                if ( memmove(dictBuffer,tc_parameters->dictBuffer_old,dictSize) == NULL ){
                    printf("Error DictCompress: Copy dictBuffer Fail!\n");
                    tc_parameters->exitThread = 1;
                    free(dictBuffer);  
                    pthread_cond_signal(&tc_parameters->updateDict);
                    pthread_mutex_unlock(&tc_parameters->lock);          
                    return -1;
                }
            }
            
        }
        /* Copy data to cBuffer waitting compressing. */
        size_t cSize = MIN((tc_parameters->srcSize - tc_parameters->totalConsumedSize),
                                tc_parameters->tempcSize );
        void *cBuffer = calloc(cSize,sizeof(char));
        size_t position = tc_parameters->totalConsumedSize;

        if (tc_parameters->display == 1){
                printf("DictCompress position start =%ld\n",position);
        }
        
        if ( memmove(cBuffer,tc_parameters->srcBuffer+position,cSize) == NULL ){
            printf("Error DictCompress: Copy data to cBuffer Fail!\n");
            tc_parameters->exitThread = 1;
            free(dictBuffer);
            free(cBuffer);
            pthread_cond_signal(&tc_parameters->updateDict);
            pthread_mutex_unlock(&tc_parameters->lock);
            return -1;
        }
        /* Set Chunks. */
        size_t blocksize = tc_parameters->blockSize;
        size_t nb = ((cSize+blocksize-1)/blocksize);
        size_t *SamplesSize = calloc(nb,sizeof(size_t));
        int check = SetsamplesSizes(cSize,blocksize,SamplesSize);
        if (check){
            free (cBuffer);
            free(SamplesSize);
            free(dictBuffer);
            tc_parameters->exitThread = 1;
            pthread_cond_signal(&tc_parameters->updateDict);
            pthread_mutex_unlock(&tc_parameters->lock);
            return -1;
        } 
        size_t tempConsumeSize = 0;
        size_t dictCompressSize = 0;
        ZSTD_CCtx* cctx = ZSTD_createCCtx();
        loadDictToCCtx(cctx,dictBuffer,dictSize);
        // double time_singelChunk = 0.0;
        /* Dictionary Compress each chunk. */
        for (int i = 0;i < nb;i++){
            if ((tempConsumeSize >= cSize) || (tempConsumeSize < 0)){
                free(SamplesSize);
                free(dictBuffer);
                free (cBuffer);
                ZSTD_freeCCtx(cctx);
                tc_parameters->exitThread = 1;
                pthread_cond_signal(&tc_parameters->updateDict);
                pthread_mutex_unlock(&tc_parameters->lock);
                return -1;
            } 
            void *tempCbuffer = malloc(SamplesSize[i]);
            memcpy(tempCbuffer,cBuffer+tempConsumeSize,SamplesSize[i]);
            
            void *oBuffer = malloc(blocksize);
            // clock_t start_time,end_time;
            // start_time = clock();
            // cctx->streamStage = zcss_load;
            dictCompressSize += DictionaryComp_Stream(tempCbuffer,SamplesSize[i],oBuffer,cctx);
            // end_time = clock();
            // time_singelChunk += (double)(end_time - start_time )/CLOCKS_PER_SEC;
            // dictCompressSize += DictionaryComp_Stream(tempCbuffer,SamplesSize[i],dictBuffer,dictSize,oBuffer);
            tempConsumeSize +=SamplesSize[i];
            free(tempCbuffer);
            free(oBuffer);
        }
        tc_parameters->savecompTime_singelChunk[tc_parameters->count] = tc_parameters->cctime;
        tc_parameters->compTime += tc_parameters->cctime;
        tc_parameters->cctime = 0.0;
        ZSTD_freeCCtx(cctx);
        tc_parameters->check_blockNumb_comp += nb;
        result = dictCompressSize;
        tc_parameters->totalConsumedSize += cSize;
        /* Match Hit. */
        // tc_parameters->hitRatio_singelChunk = (tc_parameters->dictHit_singelChunk*1.0/tc_parameters->srcHit_singelChunk);
        tc_parameters->dictHit_total += tc_parameters->dictHit_singelChunk;
        tc_parameters->srcHit_total += tc_parameters->srcHit_singelChunk;

        /* Match Length. */
        // tc_parameters->mLengthRatio_singelChunk = (tc_parameters->dictmLength_singelChunk*1.0/tc_parameters->srcmLength_singelChunk);
        tc_parameters->dictmLength_total += tc_parameters->dictmLength_singelChunk;
        tc_parameters->srcmLength_total += tc_parameters->srcmLength_singelChunk;
        // tc_parameters->saveHit_ratio[tc_parameters->count] = tc_parameters->hitRatio_singelChunk;
        // tc_parameters->saveMatch_ratio[tc_parameters->count] = tc_parameters->mLengthRatio_singelChunk;
        tc_parameters->saveMS_ratio[tc_parameters->count] = (tc_parameters->srcmLength_singelChunk*1.0/cSize);
        // double DSRatio = (1 MB)*((1.0/tc_parameters->srcmLength_singelChunk)-(1.0/tc_parameters->dictmLength_singelChunk));
        // tc_parameters->saveDS_ratio[tc_parameters->count] = DSRatio;
        tc_parameters->saveDS_ratio[tc_parameters->count] = (tc_parameters->dictmLength_singelChunk*1.0/cSize);
        if (tc_parameters->count >= 1){
            int n = tc_parameters->count;
            tc_parameters->saveMSDiff_ratio[n] = tc_parameters->saveMS_ratio[n] - tc_parameters->saveMS_ratio[n-1]; 
            tc_parameters->saveDSDiff_ratio[n] = tc_parameters->saveDS_ratio[n-1] - tc_parameters->saveDS_ratio[n];
        }
        if (tc_parameters->display == 1){
            printf("DictCompress Loop: This time DS Ratio = %f\n",tc_parameters->saveDS_ratio[tc_parameters->count]);
            printf("DictCompress Loop: This time MS Ratio = %f\n",tc_parameters->saveMS_ratio[tc_parameters->count]);
            // printf("DictCompress Loop: This time Hit Ratio = %f\n",tc_parameters->hitRatio_singelChunk);
            printf("DictCompress Loop: This time Dict Hit = %ld \t Src Hit = %ld\n",tc_parameters->dictHit_singelChunk,tc_parameters->srcHit_singelChunk);
            // printf("DictCompress Loop: This time ML Ratio = %f\n",tc_parameters->mLengthRatio_singelChunk);
            printf("DictCompress Loop: This time Dict ML = %ld \t Src ML = %ld\n",tc_parameters->dictmLength_singelChunk,tc_parameters->srcmLength_singelChunk);
        }
        /* Reset singelChunk hit. */
        tc_parameters->dictHit_singelChunk = 0; 
        tc_parameters->srcHit_singelChunk = 0;
        tc_parameters->dictmLength_singelChunk = 0; 
        tc_parameters->srcmLength_singelChunk= 0;
        double tempCompRatio = (cSize*1.0/dictCompressSize);
        tc_parameters->saveComp_ratio[tc_parameters->count] = tempCompRatio;
        
        if (tc_parameters->display == 1){
            printf("DictCompress Loop: This time consumed data size = %ld\n",cSize);
            printf("DictCompress Loop: Compressed size = %ld\n",dictCompressSize);
            printf("DictCompress Loop: Compressed Ratio = %f\n",tempCompRatio);
            printf("DictCompress Loop: Total ConsumedSize = %ld\n",tc_parameters->totalConsumedSize);
        }
        tc_parameters->dict_Exist = 0;
        /* Check if the dictionary is valid.*/
        if ( ((tc_parameters->count) >= 1) && ( (tc_parameters->trainDictOnce == 4) || (tc_parameters->trainDictOnce == 5 ))){
            int n = tc_parameters->count;
            double dsBefore = tc_parameters->saveDS_ratio[n-1];
            double dsdrop = tc_parameters->saveDSDiff_ratio[n]/dsBefore;
            if ( dsdrop <= tc_parameters->dsDrop ){
                tc_parameters->dict_Exist = 1;
                if ( (tc_parameters->saveDS_ratio[n] < DSMIN) && (tc_parameters->trainDictOnce == 5 )){
                    tc_parameters->dict_Exist = 0;
                    tc_parameters->count_unnormal += 1;
                }    
            }
            if (tc_parameters->totalConsumedSize >= tc_parameters->srcSize){
                printf("D\n");
                tc_parameters->dict_Exist = 0;
            }     
        }
        tc_parameters->count += 1;
        free(SamplesSize);
        free(dictBuffer);
        free(cBuffer);
    }
    else{
        if (tc_parameters->display == 1){
            printf("Dictionary Compress Finish.\n");
        }
        result = -1;
    }
    pthread_cond_signal(&tc_parameters->updateDict);
    pthread_mutex_unlock(&tc_parameters->lock);    
    if (tc_parameters->display == 1){
            printf("Compress trainDict_Break = %ld\n",tc_parameters->trainDict_Break);
            printf("Remaining data size = %ld\n",((tc_parameters->srcSize)-(tc_parameters->totalConsumedSize)));
    }
    return result;
} 
/* Regualr Compress the srcBuffer given,and return the compressed size.
    blockSize = 0 ,compress data at one time without chunking. */
// static size_t RegularComp_Stream(void* srcBuffer,size_t srcSize,void* outBuffer,size_t blockSize)
// {   
//     void *buffIn = srcBuffer;
//     // void *outBuffer = outBuffer;
//     size_t outSize = 0;
//     if (blockSize > 0 ){
//         int nbSamples =  (int)((srcSize+blockSize-1)/blockSize);
//         size_t *samplesSize = calloc(nbSamples,sizeof(size_t));
//         int check = SetsamplesSizes(srcSize,blockSize,samplesSize);
//         if (check){
//             printf("Error: Regular ,Set samplesSize fail!!!\n");
//             free(samplesSize);
//             return -1;
//         }
//         int i = 0;
//         size_t position = 0;
//         for (i;i < nbSamples; i++){
//             ZSTD_inBuffer inBuff = { buffIn+position, samplesSize[i], 0 };
//             ZSTD_outBuffer outBuff= { outBuffer+outSize, blockSize, 0 };
//             ZSTD_CCtx* const cctx = ZSTD_createCCtx();
//             ZSTD_EndDirective const mode = ZSTD_e_end;
//             /* If all the input data is been consumed ,return 0.*/
//             size_t remaining = ZSTD_compressStream2(cctx,&outBuff,&inBuff,mode);
//             position += samplesSize[i];
//             if (remaining){
//                 printf("Data is not fully compressed!\n");
//                 printf("Remaining = %ld\n",remaining);
//             }
//             outSize += outBuff.pos;
//         }
//         free(samplesSize);
//     }
//     if (blockSize == 0 ){
//             ZSTD_inBuffer inBuff = { buffIn, srcSize, 0 };
//             ZSTD_outBuffer outBuff= { outBuffer, srcSize, 0 };
//             ZSTD_CCtx* const cctx = ZSTD_createCCtx();
//             ZSTD_EndDirective const mode = ZSTD_e_end;
//             /* If all the input data is been consumed ,return 0.*/
//             size_t remaining = ZSTD_compressStream2(cctx,&outBuff,&inBuff,mode);
//             if (remaining){
//                 printf("Data is not fully compressed!\n");
//                 printf("Remaining = %ld\n",remaining);
//             }
//             outSize = outBuff.pos;
//     }
//     if (blockSize < 0 )
//         printf("BlockSize is negative,please set a blockSize > 0!!!\n");
//     return outSize;
// }

static void* multiple_DictionaryTrain_stream(void* tc_p){

    TC_params *tc_parameters = (TC_params *)(tc_p);
    size_t srcSize = tc_parameters->srcSize;
    size_t totalConsumedSize = tc_parameters->totalConsumedSize;
    size_t blockSize = tc_parameters->blockSize;
    size_t trainSize = tc_parameters->temptSize;
    size_t dictSize = 0;
    
    if (tc_parameters->display == 1){
        printf("Train: Source size = %ld\n",srcSize);
        printf("Train: blockSize = %ld\n",blockSize);
        printf("Train: trainSize = %ld\n",trainSize);
        printf("Train: totalConsumedSize = %ld\n",totalConsumedSize);
        printf("\n");
    }
    int i = 0;
    ZDICT_fastCover_params_t fastCoverParams = defaultFastCoverParams();
    while ( 1 )
    {           
        dictSize = DictionaryTrain_Stream(tc_parameters,&fastCoverParams);
        if ( dictSize > 0 ){
            if ( tc_parameters->display == 1){
                printf("Train dictionary %d times\n",i);
            }  
            i++;            
        }
        if (dictSize == -1) break;
    }     
    if (tc_parameters->display == 1){
        printf("Finish training program!\nExit.\n\n");  
    }
    tc_parameters->tTimes = i;
                                      
    return 0;
}
static void* multiple_DictionaryCompress_stream(void *tc_p)
{   
    TC_params *tc_parameters = (TC_params *)(tc_p);
    size_t srcSize = tc_parameters->srcSize;
    // size_t totalConsumeSize = 0;
    // size_t compchunkSize = tc_parameters->tempcSize;
    // size_t trainSize = tc_parameters->temptSize;
    // size_t blocksize = tc_parameters->blockSize;
    size_t dictCompressSize = 0;
    if (tc_parameters->display <= 2){
        printf("Compress: srcSize = %ld\n",tc_parameters->srcSize);
        printf("Compress: blockSize = %ld\n",tc_parameters->blockSize);
    }
    int i = 0;
    size_t check = 0;
    // size_t nbchunk = srcSize/(compchunkSize);
    while ( 1 )
    {
        if (tc_parameters->display == 1 ){
            printf("\nHers is dictcompress loop.\n");
        }
        check = DC_Stream(tc_parameters);
        i++;
        if (check  == -1) {
            break;
        }
        else{
            if (tc_parameters->display == 1){
                printf("Compress %d times.\n",i);
            }
            dictCompressSize += check;
        }
    }
    tc_parameters->cTimes = i;
    double ratio = (srcSize*1.0/dictCompressSize);
    tc_parameters->ratio = ratio;
    // double hitRatio = (tc_parameters->dictHit_total*1.0/tc_parameters->srcHit_total);
    // tc_parameters->hitRatio_total = hitRatio;
    if (tc_parameters->display <= 2){
        printf("Source size  = %ld\n",srcSize);
        printf("Dictionary Compress Size = %ld\n",dictCompressSize);
        printf("Compress Ratio: %f\n",ratio);
    }
    printf("Finish Compress!\nExit.\n\n");
    return 0;
}
void TC_params_free(TC_params *tc_p){
    
    free(tc_p->srcBuffer);
    /* free dictBuffer. */
    free(tc_p->dictBuffer);
    free(tc_p->dictBuffer_first);
    free(tc_p->dictBuffer_old);
    /*free save_ratio. */
    free(tc_p->saveComp_ratio);
    // free(tc_p->saveHit_ratio);
    // free(tc_p->saveMatch_ratio);
    free(tc_p->saveMS_ratio);
    free(tc_p->saveDS_ratio);
    free(tc_p->savecompTime_singelChunk);
    free(tc_p->savetrainTime_singelChunk);
    free(tc_p->saveDSDiff_ratio);
    free((tc_p->saveMSDiff_ratio));
    pthread_mutex_destroy(&tc_p->lock);
    pthread_cond_destroy(&tc_p->dictComp);
    pthread_cond_destroy(&tc_p->updateDict);

}
int saveDataToFile(char* file,double data){
    FILE* fp = fopen(file,"a+");
    if (fp != NULL){
        fprintf(fp,"%f\n",data);
    }  
    else {
        printf("Save data fail, file open error!!!\n");
        return -1;
    }
    fclose(fp);
    return 0;
}
int saveSingelChunk_data_ToFile(char *file,int compOrHit){
    FILE* fp = fopen(file,"a+");
    if (fp != NULL)
    {
        for (int i = 0;i < tc_params.count;i++)
        {
            if (compOrHit == 0){
                fprintf(fp,"%f\n",tc_params.saveComp_ratio[i]);
            }
            // if (compOrHit == 1){
            //     // fprintf(fp,"%f\n",tc_params.saveHit_ratio[i]);
            // }
            if (compOrHit == 2){
                // fprintf(fp,"%f\n",tc_params.saveMatch_ratio[i]);
            }
            if (compOrHit == 3){
                 fprintf(fp,"%f\n",tc_params.saveMS_ratio[i]);
            }
            // if (compOrHit == 4){
            //     //  fprintf(fp,"%f\n",tc_params.saveDS_ratio[i]);
            // }
            if (compOrHit == 5){
                 fprintf(fp,"%f\n",tc_params.saveMSDiff_ratio[i]);
            }
            if (compOrHit == 6){
                 fprintf(fp,"%f\n",tc_params.saveDS_ratio[i]);
            }
            if (compOrHit == 7){
                 fprintf(fp,"%f\n",tc_params.savetrainTime_singelChunk[i]);
            }
            if (compOrHit == 8){
                 fprintf(fp,"%f\n",tc_params.savecompTime_singelChunk[i]);
            }
        }
    }
    else {
        printf("Save data fail, file open error!!!\n");
        return -1;
    }
        
    fclose(fp);
    return 0;
}


void saveData( char* ftotalComp_Ratio, char* fsingelComp_Ratio, char* ftotalHit_Ratio,char* fsingelHit_Ratio, 
                char* ftotalMatch_Ratio,char* fsingelMatch_Ratio,char* ftotalMS_Ratio,char* fsingelMS_Ratio,
                char* ftotalDS_Ratio,char* fsingelDS_Ratio,char* fsingelMSDiff,char* fsingelDS,
                char* ftotalTrain_Time,char* fsingelTrain_Time,char* ftotalComp_Time,char* fsingelComp_Time)
{
    int check_t = 0;
    int check_s = 0;
    // if ( ftotalHit_Ratio != NULL ){
    //     // check_t = saveDataToFile(ftotalHit_Ratio,tc_params.hitRatio_total);
    // }

    // if ( ftotalMatch_Ratio != NULL ){
    //     // check_t = saveDataToFile(ftotalMatch_Ratio,tc_params.mLengthRatio_total);
    // }
    if ( ftotalDS_Ratio != NULL ){
       check_t = saveDataToFile(ftotalDS_Ratio,tc_params.DSRatio_total);
    }
    // check_t = saveDataToFile(ftotalHit_Ratio,tc_params.hitRatio_total);
    // check_t = saveDataToFile(ftotalMatch_Ratio,tc_params.mLengthRatio_total);
    // check_t = saveDataToFile(ftotalDS_Ratio,tc_params.DSRatio_total);
    check_t = saveDataToFile(ftotalMS_Ratio,tc_params.MSRatio_total);
    check_t = saveDataToFile(ftotalComp_Ratio,tc_params.ratio);
    check_t = saveDataToFile(ftotalComp_Time,tc_params.compTime);
    check_t = saveDataToFile(ftotalTrain_Time,tc_params.trainTime);

    
    // if ( fsingelHit_Ratio != NULL ){
    //     check_s = saveSingelChunk_data_ToFile(fsingelHit_Ratio,1);
    // }
    // if ( fsingelMatch_Ratio != NULL ){
    //     check_s = saveSingelChunk_data_ToFile(fsingelMatch_Ratio,2);
    // }
    check_s = saveSingelChunk_data_ToFile(fsingelComp_Ratio,0);
    // check_s = saveSingelChunk_data_ToFile(fsingelHit_Ratio,1);
    // check_s = saveSingelChunk_data_ToFile(fsingelMatch_Ratio,2);
    check_s = saveSingelChunk_data_ToFile(fsingelMS_Ratio,3);
    // check_s = saveSingelChunk_data_ToFile(fsingelDS_Ratio,4);
    check_s = saveSingelChunk_data_ToFile(fsingelMSDiff,5);
    check_s = saveSingelChunk_data_ToFile(fsingelDS,6);
    check_s = saveSingelChunk_data_ToFile(fsingelTrain_Time,7);
    check_s = saveSingelChunk_data_ToFile(fsingelComp_Time,8);

}

int main(int argc,char* argv[]) {
    // int check = 1;
     if (argc < 6) {
        printf("Parameters at least 6.\n");
        help_function();
        return 0;
    }
    char *file_in = argv[1];
    char *saveCompressRatio = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/cRatio.txt";
    // char *saveHitRatio = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/hRatio.txt";
    // char *saveMLength = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/mLRatio.txt";
    char *saveMSRatio = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/MSRatio.txt";
    char *saveDSRatio = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/DSRatio.txt";
    char *saveTrainTime = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/TrainTime.txt";
    char *saveCompTime = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/CompTime.txt";


    char *saveCompressRatio_singel = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/cRatio_singel.txt";
    // char *saveHitRatio_singel = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/hRatio_singel.txt";
    // char *saveMLength_singel = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/mLRatio_singel.txt";
    char *saveMSRatio_singel = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/MSRatio_singel.txt";
    // char *saveDSRAtio_singel = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/DSRatio_singel.txt";
    char *saveMSDiff_singel = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/MSDiff_singel.txt";
    char *saveDS_singel = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/DS_singel.txt";
    char *saveTrainTime_singel = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/TrainTime_singel.txt";
    char *saveCompTime_singel = "/home/yonghui/dictSize/test_mydictComp/Test_python/TestData/CompTime_singel.txt";


    unsigned blockSize = BLOCKSIZE;
    unsigned compressChunkSize = COMPCHUNKSIZE;
    unsigned trainChunkSize = TRAINCHUNKSIZE;
    unsigned maxDictSize = MAX_DICTSIZE;
    int trainDictOnce = 0;
    int display = 0;
    size_t srcSize;
    void*  srcBuffer = loadFiletoBuff(file_in,&srcSize);
    if (srcSize < trainChunkSize)
    {
        printf("Source size is too samll,at least > 2 MB");
        free(srcBuffer);
        return 0;
    }
    int opt;
    int option_index = 0;
    char *optstring = "B:T:C:M:D:O:U:";
    int update = 0;
    static struct option long_options[] = {
        {"blockSize", required_argument, NULL, 'B'},
        {"trainChunkSize",  required_argument, NULL, 'T'},
        {"compressChunkSize", required_argument, NULL, 'C'},
        {"maxDictSize", required_argument, NULL, 'M'},
        {"trainDictOnce", required_argument, NULL, 'O'},
        {"display", required_argument, NULL, 'D'},
        {"updateLimit", required_argument, NULL, 'U'},
        {0, 0, 0, 0}
    };
    while ( (opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1)
    {
            switch(opt)
            {   
                case 'B':   readU32FromCharChecked_(optarg,&blockSize);
                            break;
                case 'C':   readU32FromCharChecked_(optarg,&compressChunkSize);
                            break;
                case 'T':   readU32FromCharChecked_(optarg,&trainChunkSize);
                            break;
                case 'M':   readU32FromCharChecked_(optarg,&maxDictSize);
                            break;
                case 'O':   trainDictOnce = (int)(*optarg - '0');
                            break;
                case 'D':   display = (int)(*optarg - '0');
                            break;
                case 'U':   update = (int)(*optarg - '0');
                            break;
                default:    printf("Error: Failed to parse parameter.\n");
                            break;                   
            }
    }
    
    if (initalize_TC_params(&tc_params,srcBuffer,srcSize,maxDictSize,blockSize,compressChunkSize,trainChunkSize,trainDictOnce,display,update) != 0){
        printf("Initalize TC_paramters Fail!!!\n");
        free(srcBuffer);
        TC_params_free(&tc_params);
        return 0;
    }
    if (tc_params.display <= 2){
        printf("trainDictOnce = %d\n",trainDictOnce);
        printf("dropLimit = %f\n",tc_params.dsDrop);
    }
    pthread_t th_train,th_dictcomp;
    /* Dictionary Train. */
    pthread_create(&th_train,NULL,multiple_DictionaryTrain_stream,(void*)(&tc_params));
    pthread_create(&th_dictcomp,NULL,multiple_DictionaryCompress_stream,(void*)(&tc_params));
        
    pthread_join(th_train,NULL);
    pthread_join(th_dictcomp,NULL);
    
    double hitratio = (tc_params.dictHit_total*1.0/tc_params.srcHit_total);
    double mLratio = (tc_params.dictmLength_total*1.0/tc_params.srcmLength_total);
    // tc_params.mLengthRatio_total = mLratio;
    double MSRatio = (tc_params.srcmLength_total*1.0/tc_params.totalConsumedSize);
    tc_params.MSRatio_total = MSRatio;
    // double DSRatio = (1 MB)*((1.0/tc_params.srcmLength_total) - (1.0/tc_params.dictmLength_total));
    double DSRatio = ((1.0*tc_params.dictmLength_total)/tc_params.srcSize);
    tc_params.DSRatio_total = DSRatio;
    if (tc_params.display <= 2){
        printf("Total:\tDSratio: %f\n",DSRatio);
        printf("Total:\tMSratio: %f\n",MSRatio);
        printf("Total:\tHit Ratio: %f\n",hitratio);
        printf("Total:\tDict Hit = %ld \tSrc Hit = %ld\n",tc_params.dictHit_total,tc_params.srcHit_total);
        printf("Total:\tML Ratio: %f\n",mLratio);
        printf("Total:\tDict ML = %ld \tSrc ML = %ld\n",tc_params.dictmLength_total,tc_params.srcmLength_total);
        printf("Total:\tComp Times  = %ld\n",tc_params.cTimes);
        printf("Total:\tTrain Times  = %ld\n",tc_params.tTimes);
        printf("Unnormal count  = %d\n",tc_params.count_unnormal);
        
    }
    // saveData(saveCompressRatio,saveCompressRatio_singel,saveHitRatio,saveHitRatio_singel,saveMLength,saveMLength_singel,saveMSRatio,saveMSRatio_singel,saveDSRatio,saveDSRAtio_singel,saveMSDiff_singel,saveDS_singel);
    saveData(saveCompressRatio,saveCompressRatio_singel,NULL,NULL,NULL,NULL,saveMSRatio,saveMSRatio_singel,saveDSRatio,NULL,saveMSDiff_singel,saveDS_singel,saveCompTime,saveCompTime_singel,saveTrainTime,saveTrainTime_singel);
    TC_params_free(&tc_params);
    free(srcBuffer);

    return 0;
}
 
