#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#define KB *(1<<10)
#define MB *(1<<20)
#define MIN(a,b)    ((a) < (b) ? (a) : (b))
#define COMP_SIZE 
#define MAX_SIZE 31 MB 
typedef struct  pthread_test
{
    pthread_mutex_t lock;
    pthread_cond_t updateDict;
    pthread_cond_t dictComp;
    size_t totalConsumedSize;
    size_t dict_Exit;           /* 0:Dictionary doesn't exit,train by@train().
                                    1: Dictionary exit,consume by@dictCompress().*/
    size_t trainDict_Break;     /* 0: Continue train.
                                    -1:Fininsh training.*/
    size_t srcSize;             /* Source data size. */
    size_t blockSize;           /* Compress and Train blocksize. */
    size_t tempcSize;           /* Compress chunksize.*/
    size_t temptSize;           /*Train chunksize.*/
    
}Updatedict_Compress;

struct pthread_test train_dictComp;

int train(Updatedict_Compress *up_c);
void *train_or_not(void*arg);
int  dictCompress(Updatedict_Compress *up_c);
void *compress_or_not(void*arg);

/* Initalize struct Updatedict_Compress.*/
void inital_Updatedict_Compress(Updatedict_Compress *up_c){
    pthread_mutex_init(&up_c->lock,NULL);
    pthread_cond_init(&up_c->updateDict,NULL);
    pthread_cond_init(&up_c->dictComp,NULL);
    up_c->totalConsumedSize = 0;
    up_c->dict_Exit= 0;
    up_c->trainDict_Break = 0;
    up_c->blockSize = 4 KB;
    up_c->srcSize = MAX_SIZE;
    up_c->tempcSize = 10 MB;
    up_c->temptSize = 2 MB;
}
/* Train dictionary or not. */
void *train_or_not(void*arg){
    size_t ssize = train_dictComp.srcSize;
    printf("Train: ssize = %ld\n",ssize);
    printf("Train: blockSize = %ld\n",train_dictComp.blockSize);
    printf("Train: totalconsumedsize = %ld\n",train_dictComp.totalConsumedSize);
    // printf("Train: ssize-train_dictComp.totalConsumedSize = %ld\n",ssize-train_dictComp.totalConsumedSize);
    // printf("Train: 2*trainsize = %d\n",2*trainsize);
    int check = 0;
    int i = 0;
    size_t train_or_not = train_dictComp.srcSize-train_dictComp.temptSize;
    if ( train_or_not <=0 ){
        printf("Source data size is too samll,it does'n enough to train a dictionary\n");
        printf("At least bigger than %ld MB\n",train_dictComp.temptSize/ (1 MB));
    }
    while (1)
    {   
        printf("\nHere is train loop.\n");
        printf("Train: consumedsize = %ld\n",train_dictComp.totalConsumedSize);
        check = train(&train_dictComp);
        i++;
        if (check == -1){
            break;
        } 
        else{
            printf("Train dictionary %d times\n",i);
        }
        
    }
    printf("Finish training program!\nExit.\n\n");
}
/* Train dictionary. */
int train(Updatedict_Compress *up_c){
    int result = 0;
    pthread_mutex_lock(&up_c->lock);
    while (up_c->dict_Exit == 1)
    {
        if (up_c->trainDict_Break == -1) break; 
        printf("Waitting dictionary has been consumed.\n");
        pthread_cond_wait(&up_c->updateDict,&up_c->lock);
        printf("\nBack to train!!!\n");
    }    
    /* Check whether finish training. */
    if (((up_c->srcSize)-(up_c->totalConsumedSize)) <= (2*up_c->temptSize)){
        up_c->trainDict_Break = -1;
        result = -1;
    }
    else{
        up_c->totalConsumedSize += 2 MB;
        printf("Train Loop: ConsumedSize = %ld\n",up_c->totalConsumedSize);
        printf("Remaining = %ld\n",((up_c->srcSize)-(up_c->totalConsumedSize)));
        up_c->dict_Exit = 1;
    }
    if (up_c->trainDict_Break == -1){
        printf("Remaining data size isn't enough for training a new dictionary.\n");
        printf("Remaining data size = %ld\n",up_c->srcSize-up_c->totalConsumedSize);
    }
    pthread_cond_signal(&up_c->dictComp);
    pthread_mutex_unlock(&up_c->lock);
    printf("Train trainDict_Break = %ld\n",up_c->trainDict_Break);
    return result;
}
/* Compress or not.*/
void *compress_or_not(void*arg){
    size_t ssize = train_dictComp.srcSize;
    printf("Compress: srcSize = %ld\n",train_dictComp.srcSize);
    printf("Compress: blockSize = %ld\n",train_dictComp.blockSize);
    printf("Compress: totalconsumedsize = %ld\n",train_dictComp.totalConsumedSize);
    int i = 0;
    int check = 0;
    while (1)
    {
        printf("\nHers is dictcompress loop.\n");
        check = dictCompress(&train_dictComp);
        i++;
        if (check  == -1) {
            break;
        }
        else{
            printf("Compress %d times.\n",i);
        }
        
    }
    printf("Finish Compress!\nExit.\n\n");
}
int  dictCompress(Updatedict_Compress *up_c){
    int result = 0;
    pthread_mutex_lock(&up_c->lock);
    /* Check whether dictionary exit.
        If dictionary doen't exit,call @train training a dictionary. */
    while ( up_c->dict_Exit == 0 )
    {   
        if ( up_c->trainDict_Break == -1){
            printf("Reuse the old dictionary!!!\n(If reamining data szie >0 ).\n");
            result = -1;
            break;
        }
        printf("Dictionary hasn't finished.\n");
        pthread_cond_wait(&up_c->dictComp,&up_c->lock);
        printf("\nBack to Compress!!!\n");
    }
    /* Compress with dictionary. */
    if (up_c->totalConsumedSize < up_c->srcSize){
        size_t tempLoad = MIN((up_c->srcSize-up_c->totalConsumedSize),(up_c->tempcSize-up_c->temptSize));
        up_c->totalConsumedSize += tempLoad;
        printf("DictCompress Loop: ConsumedSize = %ld\n",up_c->totalConsumedSize);
        up_c->dict_Exit = 0;
    }
    pthread_cond_signal(&up_c->updateDict);
    pthread_mutex_unlock(&up_c->lock);    
    printf("Compress trainDict_Break = %ld\n",up_c->trainDict_Break);
    printf("Remaining data size = %ld\n",((up_c->srcSize)-(up_c->totalConsumedSize)));
    return result;
}
int main(int argc,char* argv[]){
    pthread_t th_train,th_dictComp;
    inital_Updatedict_Compress(&train_dictComp);
    
    pthread_create(&th_train,NULL,train_or_not,0);
    pthread_create(&th_dictComp,NULL,compress_or_not,0);

    pthread_join(th_train,NULL);
    pthread_join(th_dictComp,NULL);
    return 0;
}