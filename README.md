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

~~~
 make clean && make- j8
~~~

#### Then 

~~~
/examples make clean && make -j8
~~~

~~~
./Test_updateDictionary File
~~~

The program will print :

Source size:

Dictionary Compress size:

Dictionary Compress Ration: