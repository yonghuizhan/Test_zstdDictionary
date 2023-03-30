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

#include "../lib/zstd.h"
#include "../lib/common.h"
#include "../lib/zstd_errors.h"
#include "../lib/zdict.h"
#include "../programs/util.h"
#include "../programs/dibio.h"          /* DiB_trainFromFiles */
#include "../programs/timefn.h"         /* UTIL_time_t, UTIL_clockSpanMicro, UTIL_getTime */


#define KB *(1<<10)
#define MB *(1<<20)
#define DICT_BUFFER_SIZE 32768
#define MAX_DICTSIZE 112640      
#define DEFAULT_ACCEL_TEST 1
#define SAMPLESIZE_MAX (128 KB)
#define MIN(a,b)    ((a) < (b) ? (a) : (b))
#define DISPLAYLEVEL 2
#define CLEVEL  3
static const unsigned kDefaultRegression = 1;
/* Parameters for Dictionary Train. */
typedef struct 
{
    void *dictBuffer;
    size_t blockSize;
    void *srcBuffer;
    size_t srcSize;
    size_t totalConsumedSize;
    size_t *dictSize;
}DICT_train_params;
/* Parameters for Dictionary Compress. */
typedef struct 
{
    void *dictBuffer;
    size_t *dictSize;
    void *srcBuffer;
    size_t srcSize;
    size_t *totalConsumedSize;
    size_t compSize;
    size_t blockSize;
    size_t trainSize;   /*The data size used for training dictionary.*/
    void *outBuffer;
}DICT_compress_params;

/* Parameters for Regular Compress. */
typedef struct {
    void *srcBuffer;
    size_t srcSize;
    void *outBuffer;
    size_t *dictsize;
    size_t chunkSize;
    size_t blockSize;
}REG_compress_params;
struct producons
{
    size_t totalConsumedSize;    /*数据*/
    pthread_mutex_t lock;       //互斥锁
    int readpos,writepos;       //读写位置
    pthread_cond_t nottempty;   //条件变量  非空
    pthread_cond_t notfull;     //条件变量  非满
};
static ZDICT_fastCover_params_t defaultFastCoverParams(void);
/* Set samplessizes. */
static int SetsamplesSizes(size_t srcSize,size_t blockSize,size_t* samplesSizes);

/* Use the srcbuffer training to get a dictionary。 */
static size_t  DictionaryTrain_Stream(void* const outBuffer,size_t MaxOutSize,
                                        void* srcBuffer,size_t srcSize,
                                        size_t blockSize,ZDICT_fastCover_params_t *parameters);

/* Use the given dictionary to compress secBuffer. */
static size_t DictionaryComp_Stream(void* srcBuffer,size_t srcSize,
                                    void* dictBuffer,size_t dictSize,void* OutBuffer);

/* Thread function :Regular Compress data.*/
static size_t RegularComp_Stream(REG_compress_params regParams);

/* Thread function :Dictionary Compress data.*/
static int multiple_DictionaryCompress_stream(DICT_compress_params dictcParams);

/* Thread function :Dictionary Training.*/
static size_t multiple_DictionaryTrain_stream(void* tParams);
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void init(struct producons *prod)
{
    pthread_mutex_init(&prod->lock,NULL);       //初始化互斥锁
    pthread_cond_init(&prod->nottempty,NULL);    //初始化条件变量
    pthread_cond_init(&prod->notfull,NULL);     //初始化条件变量
    prod->readpos = 0;
    prod->writepos = 0;
    prod->totalConsumedSize = 0;
}
static int SvaeFile(char* FileName,size_t Size,void* srcBuff){
    const char* f = FileName;
    size_t size = Size;
    void* buff = srcBuff;
    int check = 0;
    FILE* fp = fopen(f,"wb");
    if (fp != NULL){
        fwrite(buff,size,1,fp);
    }
    else{
        check = 1;
    }
    fclose(fp);
    return check;

}
/* Regular Compression */
static void compress_orDie(const char* fname, const char* oname)
{
    size_t fSize;
    void* const fBuff = mallocAndLoadFile_orDie(fname, &fSize);
    size_t const cBuffSize = ZSTD_compressBound(fSize);
    void* const cBuff = malloc_orDie(cBuffSize);

    size_t const cSize = ZSTD_compress(cBuff, cBuffSize, fBuff, fSize, 1);
    CHECK_ZSTD(cSize);

    saveFile_orDie(oname, cBuff, cSize);

    /* success */
    printf("%25s : %6u -> %7u - %s \n", fname, (unsigned)fSize, (unsigned)cSize, oname);

    free(fBuff);
    free(cBuff);
}
/* Create a ouputFile like "xxx.zst" */
static char* createOutFilename_orDie(const char* filename)
{
    size_t const inL = strlen(filename);
    size_t const outL = inL + 5;
    void* const outSpace = malloc_orDie(outL);
    memset(outSpace, 0, outL);
    strcat(outSpace, filename);
    strcat(outSpace, ".zst");
    return (char*)outSpace;
}

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
static ZSTD_DDict* createDict_orDie(const char* dictFileName)
{
    size_t dictSize;
    printf("loading dictionary %s \n", dictFileName);
    void* const dictBuffer = mallocAndLoadFile_orDie(dictFileName, &dictSize);
    ZSTD_DDict* const ddict = ZSTD_createDDict(dictBuffer, dictSize);
    CHECK(ddict != NULL, "ZSTD_createDDict() failed!");
    free(dictBuffer);
    return ddict;
}

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
            if(sSize <= 0){printf("error:Sampledata size is zero or negative.\n");}
            if(bSize <=0){printf("error:BlockSize is zero or negative.\n");}
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
    /* Check whether samplesSize has been set correctly.*/
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
/* Get a Dictionary by training the sample data stream.*/
static size_t  DictionaryTrain_Stream(void* const outBuffer,size_t MaxOutSize,
                            void* srcBuffer,size_t srcSize,
                            size_t blockSize,ZDICT_fastCover_params_t *parameters)
{   int check_1 =1;
    void*  dictBuffer = outBuffer;  
    size_t dictSize = 0;
    size_t maxDictSize = MaxOutSize;
    void* sampleBuffer = malloc(srcSize+32);
    /* Set samplesSizes .*/
    int nbSamples = (int)((srcSize+blockSize-1)/blockSize);
    if (check_1){
        printf("Train dictSize = %ld\n",dictSize);
        printf("Train source size = %ld\n",srcSize);
        printf("Train blocksize = %ld\n",blockSize);
        printf("\n");
    }
    
    size_t* sampleSizes;
    sampleSizes = (size_t*)calloc(nbSamples,sizeof(size_t));
    int check = SetsamplesSizes(srcSize,blockSize,sampleSizes);
    
    /* Copy SamplesData to sampleBuffer. */
    memmove(sampleBuffer,srcBuffer,srcSize);
    if(strncmp(sampleBuffer,srcBuffer,srcSize)){
        printf("Copy SamplesData to SampleBuffer Fail!\n");
        check = 1;
        }
    if (!check){   
    dictSize = ZDICT_optimizeTrainFromBuffer_fastCover(dictBuffer, maxDictSize,
                                                              sampleBuffer, sampleSizes, nbSamples,
                                                              parameters);
    }
   
    free(sampleBuffer);
    free(sampleSizes);
    return dictSize;
}
static size_t DictionaryComp_Stream(void* srcBuffer,size_t srcSize,void* dictBuffer,size_t dictSize,void* OutBuffer){

    void *srcBuff = srcBuffer;
    void *dictBuff = dictBuffer;
    size_t outSize = srcSize;
    void  *outBuffer = OutBuffer;
    size_t dictsize = dictSize;
    // outBuffer = malloc_orDie(outSize);
    /*HHHH*/
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    /* Load dict into cctx localDict */
    loadDictToCCtx(cctx,dictBuff,dictsize);

    /*Compress srcBuff*/
    ZSTD_inBuffer inBuff = { srcBuff, srcSize, 0 };
    ZSTD_outBuffer outBuff= { outBuffer, outSize, 0 };
    ZSTD_compressStream2(cctx, &outBuff, &inBuff, ZSTD_e_end);
    outSize =outBuff.pos;
    
    ZSTD_freeCCtx(cctx);
    return outSize;
} 
/* Regualr Compress the srcBuffer given,and return the compressed size.*/
// static size_t RegularComp_Stream(void* srcBuffer,size_t srcSize,void* outBuffer)
static size_t RegularComp_Stream(REG_compress_params regParams)
{   
    void *buffIn = regParams.srcBuffer;
    size_t sSize = regParams.srcSize;
    void *outBuffer = regParams.outBuffer;
    size_t *dictSize = regParams.dictsize;
    size_t chunksize = regParams.chunkSize;
    size_t nbchunk = sSize/regParams.chunkSize;
    size_t blocksize = regParams.blockSize;
    size_t totalConsumedSize = 0;
    int nb = 0;
    while ( nb <= nbchunk){
        if (*dictSize == 0){
            size_t tempCompSize = MIN((sSize - totalConsumedSize),chunksize);
            size_t nbsamples = 0;
            size_t *sampleSizes = calloc(0,0);
            ZSTD_inBuffer inBuff = { buffIn, sSize, 0 };
            ZSTD_outBuffer outBuff= { outBuffer, sSize, 0 };
            ZSTD_CCtx* const cctx = ZSTD_createCCtx();
            ZSTD_EndDirective const mode = ZSTD_e_end;
            /* If all the input data is been consumed ,return 0.*/
            size_t remaining = ZSTD_compressStream2(cctx,&outBuff,&inBuff,mode);
            if (remaining){
                printf("Don't compress the data stream completely!\n");
                printf("Remaining = %ld\n",remaining);
            }
            size_t outSize = outBuff.pos;
            return outSize;
        }
    }
    
    
}

static size_t multiple_DictionaryTrain_stream(void* tParams_){
    DICT_train_params *tParams = (DICT_train_params*)tParams_;
    void *dictBuffer = tParams->dictBuffer;
    size_t srcSize = tParams->srcSize;
    void *srcBuffer = tParams->srcBuffer;
    // pthread_mutex_lock(&mutex);
    size_t totalConsumedSize = 0;
    size_t blockSize = tParams->blockSize;
    size_t trainSize = 2 MB;
    void *trainBuffer = malloc(trainSize);
    size_t dictSize = 0;
    size_t nb = srcSize/(trainSize * 5);
    printf("\nnb = %ld\n",nb);
    int check = 1;
    if (check){
        printf("MSource size = %ld\n",srcSize);
        printf("MblockSize = %ld\n",blockSize);
        printf("MdictSize = %ld\n",dictSize);
        printf("\n");
    }
    int i = 0;
    while ( i < nb )
    {   
        // pthread_mutex_lock(&mutex);
        if ( (srcSize - totalConsumedSize) >= (2*trainSize) ){
            memcpy(trainBuffer,srcBuffer+totalConsumedSize,trainSize);
            // totalConsumedSize += trainSize;
            ZDICT_fastCover_params_t fastCoverParams = defaultFastCoverParams();
            dictSize = DictionaryTrain_Stream(dictBuffer,MAX_DICTSIZE,
                                                trainBuffer,trainSize,blockSize,
                                                &fastCoverParams);
            tParams->dictSize = dictSize;
            printf("tParams->dictSize = %ld\n",tParams->dictSize);
            printf("Train size  = %ld\n",trainSize);
            printf("Dictionary Size = %ld\n",dictSize);
            printf("Train data position is %ld\n",totalConsumedSize);
            if ( dictSize == 0 ){
                printf("Training Dictionary Fail!\n");
            } 
        }
        printf("\n");
        totalConsumedSize += trainSize*5;
        // pthread_mutex_unlock(&mutex);
        i++;
    }
    free(trainBuffer);                                           
    return dictSize;
}
static int multiple_DictionaryCompress_stream(DICT_compress_params dictcParams)
{   
    size_t dictSize = dictcParams.dictSize;
    void *dictBuffer = dictcParams.dictBuffer;
    void *srcBuffer = dictcParams.srcBuffer;
    size_t srcSize = dictcParams.srcSize;
    size_t totalConsumeSize = 0;
    size_t compSize = dictcParams.compSize;
    size_t trainSize = dictcParams.trainSize;
    size_t blocksize = dictcParams.blockSize;
    // void *outBuffer = dictcParams.outBuffer;
    size_t dictCompressSize = 0;
    size_t nbchunk = srcSize/(compSize);
    int n = 0;
    while (n <= nbchunk){
        // pthread_mutex_lock(&mutex);
        if ( (dictSize > 0) && (dictSize <= MAX_DICTSIZE) 
                            && (totalConsumeSize < srcSize)){
            size_t cSize = MIN((srcSize - totalConsumeSize),(compSize));
            void *cBuffer = malloc(cSize);
            size_t tLoad = totalConsumeSize - trainSize; /* The size of data that will be compressed. */
            memcpy(cBuffer,srcBuffer+tLoad,cSize);
            totalConsumeSize += cSize;
            /* Set Chunks. */
            size_t nb = ((cSize+blocksize-1)/blocksize);
            size_t *SamplesSize = calloc(nb,sizeof(size_t));
            int check = SetsamplesSizes(cSize,blocksize,SamplesSize);
            if (check){
                free (cBuffer);
                return 0;
            } 
            size_t tempConsumeSize = 0;
            /* Dictionary Compress each chunk. */
            for (int i =0;i < nb;i++){
                if ((tempConsumeSize >= cSize) || (tempConsumeSize < 0)){
                    free(SamplesSize);
                    // free(dictBuffer);
                    free (cBuffer);
                    return 0;
                } 
                void *tempCbuffer = malloc(SamplesSize[i]);
                memcpy(tempCbuffer,cBuffer+tempConsumeSize,SamplesSize[i]);
                
                void *oBuffer = malloc(blocksize);
                dictCompressSize += DictionaryComp_Stream(tempCbuffer,SamplesSize[i],dictBuffer,dictSize,oBuffer);
                
                tempConsumeSize +=SamplesSize[i];
                free(tempCbuffer);
                free(oBuffer);
            }
            printf("Temp Dictionary Compressed Size = %ld\n",dictCompressSize);
            free(SamplesSize);
            free(cBuffer);
        }
        // pthread_mutex_unlock(&mutex);
    }
    // if ( (dictSize < 0) || (dictSize > MAX_DICTSIZE)){
    //     printf("Error:A failure occurred during dictionary training. \n ");
    //     printf("Dictionary Size  = %ld \n ",dictSize);
    //     printf("The data consumed is %ld \n",totalConsumeSize);
    //     // free(TrainBuffer);
    //     // free(dictBuffer);
    //     return 0;
    // }
    if (srcSize <= totalConsumeSize){
        printf("\nSource data has been consumed!\n");
    }
    return dictCompressSize;
}
static DICT_train_params*
initalDictTrainParams(void *dictBuffer,void *srcBuffer,
                            size_t srcSize,size_t blockSize,
                            size_t totalConsumedSize)
{
    DICT_train_params *trainParams = (DICT_train_params*)malloc(sizeof(DICT_train_params));
    trainParams->blockSize = blockSize;
    trainParams->totalConsumedSize = totalConsumedSize;
    trainParams->dictBuffer = dictBuffer;
    trainParams->srcBuffer = srcBuffer;
    trainParams->srcSize = srcSize;
    trainParams->dictSize = 0;
    return trainParams;
}
static DICT_compress_params 
initalDictCompressParams(void *dictBuffer,size_t *dictSize,
                            void *srcBuffer,size_t srcSize,
                            size_t *totalConsumedSize,size_t compSize,
                            size_t blockSize,size_t trainSize,void *outBuffer)
{
    DICT_compress_params dictcParams ;
    dictcParams.blockSize = blockSize;
    dictcParams.dictBuffer = dictBuffer;
    dictcParams.dictSize = dictSize;
    dictcParams.totalConsumedSize = totalConsumedSize;
    dictcParams.srcBuffer = srcBuffer;
    dictcParams.srcSize = srcSize;
    dictcParams.trainSize = trainSize;
    dictcParams.outBuffer = outBuffer;
    dictcParams.compSize = compSize;
    return dictcParams;
}
static REG_compress_params
initalRegCompressParams(void *srcBuffer,size_t srcSize,
                        size_t *dictSize,size_t chunkSize,
                        size_t blocksize,void *outBuffer)
{   
    REG_compress_params regParams;
    regParams.srcBuffer = srcBuffer;
    regParams.srcSize = srcSize;
    regParams.outBuffer = outBuffer;
    regParams.dictsize = dictSize;
    regParams.chunkSize = chunkSize;
    regParams.blockSize = blocksize;
    return regParams;
}
struct producons buffer;
int main(int argc,char* argv[]) {
    int check = 1;
    char *file_in = argv[1];
    size_t MaxDictSize = MAX_DICTSIZE;
    size_t blockSize = 4096;
    size_t srcSize;
    
    void*  srcBuffer = loadFiletoBuff(file_in,&srcSize);
    size_t compSize = 10 MB;
    size_t trainSize = 2 MB;
    size_t totalConsumeSize = 0;    /* Data size has been consumed. */
    size_t dictCompressSize = 0;    /* Data size compressed with dictionary. */
    size_t regCOmpressSize = 0;
    pthread_t th_train;//th_dictcomp,th_regcomp;
    init(&buffer);
   
    printf("srcSize = %ld\n",srcSize);
        size_t dictSize = 0;
        void *dictBuffer = malloc(MAX_DICTSIZE);
        printf("Total cunsumed data is %ld\n",totalConsumeSize);
        printf("\n");

        /* Dictionary Train. */
        // DICT_train_params tParams = (DICT_train_params*)malloc(sizeof(DICT_train_params));
        // printf("src address :%ld\n",srcBuffer);
        DICT_train_params* tParams = initalDictTrainParams(dictBuffer,srcBuffer,
                                                            srcSize,blockSize,totalConsumeSize);
        // printf("tParams address :%ld",tParams->trainBuffer);
        if(check){
            printf("TsrcSize = %ld\n",tParams->srcSize);
            printf("tblockSize = %ld\n",tParams->blockSize);
            printf("TdictSize = %ld\n",tParams->dictSize);
            printf("\n");
        }
        // pthread_create(&th_train,NULL,multiple_DictionaryTrain_stream,(void *)tParams);
        multiple_DictionaryTrain_stream(tParams);
        /* Dictionary Compress. */
        // void *outBuffer = malloc(compSize);
        // DICT_compress_params dictcParams = initalDictCompressParams(dictBuffer,&dictSize,srcBuffer,
        //                                                                 srcSize,&totalConsumeSize,compSize,
        //                                                                 blockSize,trainSize,outBuffer);
        // pthread_create(&th_dictcomp,NULL,multiple_DictionaryCompress_stream,&dictcParams);

        // /* Regular Compress. */
        // void *regOutBuffer = malloc(compSize);
        // REG_compress_params regParams = initalRegCompressParams(srcBuffer,srcSize,&dictSize,compSize,blockSize,regOutBuffer);
        // pthread_create(&th_regcomp,NULL,RegularComp_Stream,&regParams);
        // // printf("Source size = %ld\n",srcSize);
        // if ( (dictSize < 0) || (dictSize > MaxDictSize)){
        //     printf("Error:A failure occurred during dictionary training. \n ");
        //     printf("Dictionary Size  = %ld \n ",dictSize);
        //     printf("The data consumed is %ld \n",totalConsumeSize);
        // }
        free(dictBuffer);
        free(tParams);
        // free(outBuffer);
        // free(regOutBuffer);
        printf("\n");
    
    // printf("Dictionary Compressed data size = %ld\n",dictCompressSize);
    pthread_join(th_train,NULL);
    // pthread_join(th_dictcomp,NULL);
    // pthread_join(th_regcomp,NULL);
    free(srcBuffer);
    
    return 0;
}
 
