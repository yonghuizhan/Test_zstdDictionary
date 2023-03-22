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
#include <unistd.h> /*access check file exisit.*/

#include "../lib/zstd.h"
#include "../lib/common.h"
#include "../lib/zstd_errors.h"
#include "../lib/zdict.h"
#include "../programs/util.h"
#include "../programs/dibio.h"          /* DiB_trainFromFiles */
#include "../programs/timefn.h"         /* UTIL_time_t, UTIL_clockSpanMicro, UTIL_getTime */


#define KB *(1<<10)
#define DICT_BUFFER_SIZE 32768
#define MAX_DICTSIZE 112640      
#define DEFAULT_ACCEL_TEST 1
#define SAMPLESIZE_MAX (128 KB)
#define MIN(a,b)    ((a) < (b) ? (a) : (b))
#define DISPLAYLEVEL 2
#define CLEVEL  3
static const unsigned kDefaultRegression = 1;


static int readU32FromCharChecked(const char** stringPtr, unsigned* value)
{
    unsigned result = 0;
    while ((**stringPtr >='0') && (**stringPtr <='9')) {
        unsigned const max = ((unsigned)(-1)) / 10;
        unsigned last = result;
        if (result > max) return 1; /* overflow error */
        result *= 10;
        result += (unsigned)(**stringPtr - '0');
        if (result < last) return 1; /* overflow error */
        (*stringPtr)++ ;
    }
    if ((**stringPtr=='K') || (**stringPtr=='M')) {
        unsigned const maxK = ((unsigned)(-1)) >> 10;
        if (result > maxK) return 1; /* overflow error */
        result <<= 10;
        if (**stringPtr=='M') {
            if (result > maxK) return 1; /* overflow error */
            result <<= 10;
        }
        (*stringPtr)++;  /* skip `K` or `M` */
        if (**stringPtr=='i') (*stringPtr)++;
        if (**stringPtr=='B') (*stringPtr)++;
    }
    *value = result;
    return 0;
}
static unsigned readU32FromChar(const char** stringPtr) {
    static const char errorMsg[] = "error: numeric value overflows 32-bit unsigned int";
    unsigned result;
    if (readU32FromCharChecked(stringPtr, &result)) 
    return result;
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

// static const char*
// FIO_determineCompressedName(const char* srcFileName, const char* suffix){
//     static size_t dfnbCapacity = 0;
//     static char* dstFileNameBuffer = NULL;   /* using static allocation : this function cannot be multi-threaded */
//     size_t sfnSize = strlen(srcFileName);
//     size_t const srcSuffixLen = strlen(suffix);

//     if (dfnbCapacity <= sfnSize+srcSuffixLen+1) {
//         /* resize buffer for dstName */
//         free(dstFileNameBuffer);
//         dfnbCapacity = sfnSize + srcSuffixLen + 30;
//         dstFileNameBuffer = (char*)malloc(dfnbCapacity);
//         if (!dstFileNameBuffer) {
//             printf("Create outputName fail.\n");
//         }
//     }

//     memcpy(dstFileNameBuffer, srcFileName, sfnSize);
//     memcpy(dstFileNameBuffer+sfnSize, suffix, srcSuffixLen+1 /* Include terminating null */);
//     return dstFileNameBuffer;
// }

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

size_t TEST_zstd_dictComp(   char* srcFile,size_t *sSize,   char* dictFile,size_t *dSize){
    
    
    size_t srcSize;
    size_t dictSize;
    char *input_file = srcFile; 
    char *dict_file = dictFile;
    void *srcBuff = loadFiletoBuff(input_file,&srcSize);
    void *dictBuff = loadFiletoBuff(dict_file,&dictSize);
 
    *sSize = srcSize;
    *dSize = dictSize; 
    size_t outSize = srcSize;
    void  *outFile = malloc_orDie(outSize);

    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    /* Load dict into cctx localDict */
    loadDictToCCtx(cctx,dictBuff,dictSize);

    /*Compress srcBuff*/
    ZSTD_inBuffer inBuff = { (char*)srcBuff, srcSize, 0 };
    ZSTD_outBuffer outBuff= { outFile, outSize, 0 };
    ZSTD_compressStream2(cctx, &outBuff, &inBuff, ZSTD_e_end);
    outSize =outBuff.pos;
    char* output_file = NULL;
    //    char* outDirName = NULL;
    // char* suffix = ".zst";
    // output_file  = FIO_determineCompressedName(srcFile, suffix);  /* cannot fail */
    output_file = createOutFilename_orDie(srcFile);
    FILE *fp_out=fopen(output_file,"wb");
    if (fp_out == NULL){
        printf("Fail to save compressed file!\n");
    }
    else{
        fwrite(outBuff.dst,outSize,1,fp_out);
    }
    fclose(fp_out);
    ZSTD_freeCCtx(cctx);
    free(srcBuff);
    free(dictBuff);
    free(outFile);
    free(output_file);
    return outSize;
}
/* Train a dictionary with a srcFile. */
void TEST_zstd_train_dict(char *srcFileNmae, char *outFile,size_t bSize){
    const int optimize = 1;         /* 1 To find the best parameters 0 or not*/
    unsigned int memLimit = 0;
    size_t blockSize = bSize;
    char * outFileName = outFile;
    unsigned int maxDictSize = MAX_DICTSIZE;
    int operationResult = 0;
    size_t argCount = 6;                        /* It is a optional parameters , which is bigger than filenames number.*/
    FileNamesTable* filenames = UTIL_allocateFileNamesTable((size_t)argCount);  /* argCount >= 1 */
    
    filenames->fileNames[filenames->tableSize] = srcFileNmae;
    filenames->tableSize++;
    ZDICT_fastCover_params_t fastCoverParams = defaultFastCoverParams();
    /* Train dictionary from srcFile.*/
    operationResult = DiB_trainFromFiles(outFileName, maxDictSize, filenames->fileNames, 
                                            (int)filenames->tableSize, blockSize, 
                                            NULL, NULL, &fastCoverParams, 
                                            optimize, memLimit);
    UTIL_freeFileNamesTable(filenames);
}

void TEST_zstd_dictDecomp(   char* dictFileName,   char* srcName,size_t *srcSize,   char* outputFileNmae,size_t* outputSize ){
    
    void* dictBuffer;
       char* dictName = dictFileName;
    size_t dictSize;

       char* outName;
    void* outBuffer;
       char* inputNmae = srcName;
    size_t inputSize;
    void* inputBuffer;

    /*Load dictionary into dictBuffer and create outBuff.*/
    // dictBuffer = loadFiletoBuff(dictName,&dictSize);
    ZSTD_DDict* const dictPtr = createDict_orDie(dictName);
    inputBuffer = loadFiletoBuff(inputNmae,&inputSize);

    unsigned long long  const  outSize = ZSTD_getFrameContentSize(inputBuffer,inputSize);   /*Get the size of file that before compress from freame.*/
    CHECK(outSize != ZSTD_CONTENTSIZE_ERROR, "%s: not compressed by zstd!", inputNmae);
    CHECK(outSize != ZSTD_CONTENTSIZE_UNKNOWN, "%s: original size unknown!", inputNmae);
    outBuffer = malloc_orDie((size_t) outSize);

    ZSTD_DCtx* const dctx = ZSTD_createDCtx();
    CHECK(dctx != NULL, "ZSTD_createDCtx() failed!");
    size_t dSize = ZSTD_decompress_usingDDict(dctx, outBuffer, outSize, inputBuffer, inputSize, dictPtr);  /* Use dictionary decompress.*/
    CHECK_ZSTD(dSize);
    /* When zstd knows the content size, it will error if it doesn't match. */
    CHECK(dSize == outSize, "Impossible because zstd will check this condition!");

    *srcSize = inputSize;
    *outputSize = outSize;
    
    char* dfile = outputFileNmae;
    FILE *fp_decomp = fopen(dfile,"wb");
    if (fp_decomp == NULL){
        printf("Save decompress file fail!\n");
    }
    else{
        fwrite(outBuffer,dSize,1,fp_decomp);
        printf("%25s : %6u -> %7u \n", inputNmae, (unsigned)inputSize, (unsigned)outSize);
    }
    fclose(fp_decomp);

    ZSTD_freeDCtx(dctx);
    ZSTD_freeDDict(dictPtr);
    // ZSTD_DStream* ddctx = ZSTD_createDStream();
    // ZSTD_DCtx_loadDictionary(ddctx,dictBuffer,dictSize);
    // free(dictBuffer);
    free(inputBuffer);
    free(outBuffer);
}

void TEST_zstd_regComp(const char *FileName,const char *outputFile){
    const char *srcFileName = FileName;
    char *outFileName = outputFile;
    if (outFileName == NULL){
        outFileName = createOutFilename_orDie(srcFileName);
    }
    compress_orDie(srcFileName,outFileName);
}
/* Load srcBuffer to sampleBuffer. */
// static int loadintoSamples(void* buffer, size_t bufferSize,void* srcBuffer,
//                             size_t* sampleSizes, int nbSamples,
//                             size_t targetBlockSize, 
//                             int displayLevel )
// {
//     char* const buff = (char*)buffer;
//     char* src = srcBuffer;
//     size_t totalDataLoaded = 0;
//     size_t fileDataLoaded = 0;
//     int nbSamplesLoaded = 0;
//     S64 const fileSize = bufferSize;
    
//     while ( nbSamplesLoaded < nbSamples ) {        
//         /* Load the first chunk of data from the srcBUffer. */
//         fileDataLoaded = targetBlockSize > 0 ?
//                             (size_t)MIN(fileSize, (S64)targetBlockSize) :
//                             (size_t)MIN(fileSize, SAMPLESIZE_MAX );
        
//         if (totalDataLoaded + fileDataLoaded > bufferSize)
//             break;
//         if ( memmove(buff+totalDataLoaded,srcBuffer,fileDataLoaded) == NULL) {
//             printf("Copy data to buffer fail from sample.\n ");
//             return 0;
//         }
//         sampleSizes[nbSamplesLoaded++] = fileDataLoaded;
//         totalDataLoaded += fileDataLoaded;

//         /* If srcbuffer-chunking is enabled, load the rest of the srcbuffer as more samples */
//         if (targetBlockSize > 0) {
//             while( (S64)fileDataLoaded < fileSize && nbSamplesLoaded < nbSamples ) {
//                 size_t const chunkSize = MIN((size_t)(fileSize-fileDataLoaded), targetBlockSize);
//                 if (totalDataLoaded + chunkSize > *) /* buffer is full */
//                     break;

//                 if (fread( buff+totalDataLoaded, 1, chunkSize, f ) != chunkSize)
//                     EXM_THROW(11, "Pb reading %s", fileNamesTable[fileIndex]);
//                 sampleSizes[nbSamplesLoaded++] = chunkSize;
//                 totalDataLoaded += chunkSize;
//                 fileDataLoaded += chunkSize;
//             }
//         }    
//     }
//     DISPLAYLEVEL(2, "\r%79s\r", "");
//     DISPLAYLEVEL(4, "Loaded %d KB total training data, %d nb samples \n",
//         (int)(totalDataLoaded / (1 KB)), nbSamplesLoaded );
//     *bufferSizePtr = totalDataLoaded;
//     return nbSamplesLoaded;
// }
/* Set samplesSizes.
    @return 1:error
            0 complete set samplesizes.*/
int SetsamplesSizes(size_t srcSize,size_t blockSize,size_t* samplesSizes){
    size_t sSize = srcSize;
    size_t bSize = blockSize;
    unsigned nbSamples = (int)((srcSize+bSize-1)/bSize);
    int nbSamplesLoaded = 0;
    size_t srcSizeDataLoaded = 0;
    if ( (sSize <= 0) || (bSize <= 0) ){    /* Check srcsize and block size .*/
            if(sSize <= 0){printf("error:Sampledata size is zero or negative.\n");}
            if(bSize <=0){printf("error:BlockSize is zero or negative.\n");}
            return 1;
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
                    size_t chunksize = MIN((sSize-srcSizeDataLoaded),bSize);
                    if( (srcSizeDataLoaded+chunksize) > sSize)
                        break;
                    samplesSizes[nbSamplesLoaded++] += chunksize;
                    srcSizeDataLoaded += chunksize;
                }
            }
    }
    size_t d = (int)((sSize+bSize-1)/bSize);
    for (int i =0;(i < nbSamplesLoaded) && (i < d);i++){
        printf("samplesLoaded [%d]  = %ld\n",i,samplesSizes[i]);
    }
    return 0;
}
/* Get a Dictionary by training the sample data stream.*/
static size_t  DictionaryTrain_stream(void* const outBuffer,size_t MaxOutSize,
                            void* srcBuffer,size_t srcSize,
                            size_t blockSize,ZDICT_fastCover_params_t *parameters)
{
    void* const dictBuffer = outBuffer;
    size_t dictSize = 0;
    size_t maxDictSize = MaxOutSize;
    void* sampleBuffer = malloc(srcSize+32);
    /* Set samplesSizes .*/
    int nbSamples = (int)((srcSize+blockSize-1)/blockSize);
    size_t* sampleSizes;
    sampleSizes = (size_t*)malloc(nbSamples*sizeof(size_t));
    int check = SetsamplesSizes(srcSize,blockSize,sampleSizes);
    if(check){return 0;}
    /* Copy SamplesData to sampleBuffer. */
    memmove(sampleBuffer,srcBuffer,srcSize);
    if(strncmp(sampleBuffer,srcBuffer,srcSize)){
        printf("Copy SamplesData to SampleBuffer Fail!\n");return 0;
        }
    dictSize = ZDICT_optimizeTrainFromBuffer_fastCover(dictBuffer, maxDictSize,
                                                              sampleBuffer, sampleSizes, nbSamples,
                                                              parameters);
    free(sampleBuffer);
    free(sampleSizes);
    return dictSize;
}
int main(int argc,char* argv[]) {
    int check = 1;
    // char* input_file = "/home/yonghui/dictSize/test_mydictComp/nci";
    char* input_file = "/home/yonghui/dictSize/test_mydictComp/CHANGELOG";
    // char* output_file = "/home/yonghui/dictSize/test_mydictComp/nci.zst";
    // char* output_file_reg = "/home/yonghui/dictSize/test_mydictComp/reg_nci.zst";

    char* train_dictFile = "/home/yonghui/dictSize/test_mydictComp/dict_nci_test.bin";
    // char* cfile = output_file;
    // char* dfile = "/home/yonghui/dictSize/test_mydictComp/nci_test";
    // char *dfile = "../../Dictionary/dict.bin";
    char *dcompfile = "../../DictComp/temp";
    char *ftemp = "../../Temp/temp";
    /* Load src file and dict file*/
    size_t srcSize;
    size_t dictSize;
    size_t outSize;     /*Output file size*/
    // size_t blockSize = 4096;
    size_t decompSize;
    size_t cSize;
    size_t dSize;
    // int mode = 0;
    
    // if(check) {printf("A\n");}
    // TEST_zstd_train_dict(input_file,train_dictFile,blockSize);
    {   
        size_t MaxDictSize = MAX_DICTSIZE;
        size_t blockSize = 4096;
        size_t srcSize;
        size_t dictSize;
        void* const dictBuffer = malloc(MaxDictSize);
        // void* const dictBuffer;
        void*  srcBuffer = loadFiletoBuff(input_file,&srcSize);

        ZDICT_fastCover_params_t fastCoverParams = defaultFastCoverParams();
        dictSize = DictionaryTrain_stream(dictBuffer,MaxDictSize,srcBuffer,srcSize,blockSize,&fastCoverParams);
        if (check){
            printf("Src size  = %ld\n",srcSize);
        }
        if (dictSize == 0){
            printf("Training Dictionary Fail!\n");
        } 
        else{
                FILE* fp_dict = fopen(train_dictFile,"wb");
                if(fp_dict != NULL){
                    fwrite(dictBuffer,dictSize,1,fp_dict);
                    printf("Dictionaey size = %ld\n",dictSize);
                }
                else{
                    printf("Error:Save Dictionary File Fail!\n");
                }
                fclose(fp_dict);
        }
        free(srcBuffer);
        free(dictBuffer);
    }
        

    // if(check) {printf("B\n");}
    // TEST_zstd_regComp(input_file,output_file_reg);

    // if(check) {printf("C\n");}
    // TEST_zstd_dictComp(input_file,&srcSize,train_dictFile,&dictSize);

    // if (mode){
    //     if(check) {printf("D\n");}
    //     TEST_zstd_dictDecomp(train_dictFile,cfile,&cSize,dfile,&dSize);
    //     if(check) {printf("E\n");}
    // }
    return 0;
}

