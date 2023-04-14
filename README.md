###  These test code base on zstd compression .



The Dictionary Compression -UpdateDIctionary test code file is 

~~~
Test_updateDictionary.c
~~~

which path is:

~~~ 
examples/Test_updateDictionary.c.
~~~

#### First,you should compile zstd 

Before yout build zstd,you should set an environment variables: **LD_LIBRARY_PATH **

~~~
LD_LIBRARY_PATH = /home/user name/.../Test_zstdDictionary/lib"
~~~

You should **replace the path as yourself.**

Must set the environment variables: **LD_LIBRARY_PATH**, otherwise the program will not run correctly!!!

~~~
 make clean && make- j8
~~~

#### Then 

~~~
/examples make clean && make -j8
~~~

~~~
./Test_updateDictionary File -B# -C# -T# -M#
~~~


