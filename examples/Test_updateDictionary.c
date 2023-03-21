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



#define DICT_BUFFER_SIZE 32768
#define MAX_DICTSIZE 112640      
#define DEFAULT_ACCEL_TEST 1
static const unsigned kDefaultRegression = 1;
/*Regular Compression*/
static void compress_orDie(const char* fname, const char* oname)
{
    size_t fSize;
    void* const fBuff = mallocAndLoadFile_orDie(fname, &fSize);
    size_t const cBuffSize = ZSTD_compressBound(fSize);
    void* const cBuff = malloc_orDie(cBuffSize);

    /* Compress.
     * If you are doing many compressions, you may want to reuse the context.
     * See the multiple_simple_compression.c example.
     */
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
    memset(&params, 0, sizeof(params));
    params.d = 8;
    params.f = 20;
    params.steps = 4;
    params.splitPoint = 0.75; /* different from default splitPoint of cover */
    params.accel = DEFAULT_ACCEL_TEST;
    params.shrinkDict = 0;
    params.shrinkDictMaxRegression = kDefaultRegression;
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

/* Load file into buff.*/
static void* loadFiletoBuff(const char * FileName,size_t *FileSize){
    size_t size;
    void *fBuff = 
        (unsigned char *) mallocAndLoadFile_orDie (FileName,
                    &size);
    *FileSize = size;
    return fBuff;
}
/*Load dictbuff into cctx.*/
static void *loadDictToCCtx(ZSTD_CCtx* cctx,const void *dict,size_t dictSize){
    // CHECK( ZSTD_CCtx_loadDictionary(cctx, dict, dictSize) );
    ZSTD_CCtx_loadDictionary(cctx, dict, dictSize) ;
}

/* Train a dictionary with a srcFile.*/
void TEST_zstd_dictComp(   char* srcFile,size_t *sSize,   char* dictFile,size_t *dSize){
    
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
        fwrite(outBuff.dst,outBuff.pos,1,fp_out);
    }
    fclose(fp_out);
    ZSTD_freeCCtx(cctx);
    free(srcBuff);
    free(dictBuff);
    free(outFile);
    free(output_file);
}

void TEST_zstd_train_dict(   char *srcFileNmae,   char *outFile,size_t bSize){
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
    size_t    dSize = ZSTD_decompress_usingDDict(dctx, outBuffer, outSize, inputBuffer, inputSize, dictPtr);  /* Use dictionary decompress.*/
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
int main(int argc,char* argv[]) {

    // int i = 0;
    // char *inputFile;
    // char *outputFile;
    // char *dictFile;

    // dictFile = getFileName(argv[1]);
    // outputFile = getFileName(argv[2]);
    // inputFile = getFileName(argv[3]);
    
    // int check = 0;
    // check += CheckFileExisit(dictFile);
    // check += CheckFileExisit(inputFile);
    // check += CheckFileExisit(outputFile);
    
    int check = 1;
    char* input_file = "/home/yonghui/dictSize/test_mydictComp/nci";
//   char* input_file = "/home/yonghui/dictSize/test_mydictComp/mr";
    char* output_file = "/home/yonghui/dictSize/test_mydictComp/nci.zst";
    char* output_file_reg = "/home/yonghui/dictSize/test_mydictComp/reg_nci.zst";

    char* train_dictFile = "/home/yonghui/dictSize/test_mydictComp/dict_nci_test.bin";
    char* cfile = output_file;
    char* dfile = "/home/yonghui/dictSize/test_mydictComp/nci_test";
    /* Load src file and dict file*/
    size_t srcSize;
    size_t dictSize;
    size_t outSize;     /*Output file size*/
    size_t blockSize = 4096;
    size_t decompSize;
    size_t cSize;
    size_t dSize;
    // outSize = srcSize;
    if(check) printf("A\n");
    TEST_zstd_train_dict(input_file,train_dictFile,blockSize);
    if(check) printf("B\n");
    TEST_zstd_regComp(input_file,output_file_reg);
    if(check) printf("C\n");
    TEST_zstd_dictComp(input_file,&srcSize,train_dictFile,&dictSize);
    if(check) printf("D\n");
    TEST_zstd_dictDecomp(train_dictFile,cfile,&cSize,dfile,&dSize);
    if(check) printf("E\n");
    return 0;
}

