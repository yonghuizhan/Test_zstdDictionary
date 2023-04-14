# ################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under both the BSD-style license (found in the
# LICENSE file in the root directory of this source tree) and the GPLv2 (found
# in the COPYING file in the root directory of this source tree).
# You may select, at your option, one of the above-listed licenses.
# ################################################################

# LIBDIR =../lib
# CPPFLAGS += -I$(LIBDIR)
# LIB = $(LIBDIR)/libzstd.a


# .PHONY: default
# default: all

# .PHONY: all
# all: simple_compression simple_decompression \
# 	multiple_simple_compression\
# 	dictionary_compression dictionary_decompression \
# 	streaming_compression streaming_decompression \
# 	multiple_streaming_compression streaming_memory_usage

# $(LIB) :
# 	$(MAKE) -C $(LIBDIR) libzstd.a

# simple_compression.o: common.h
# simple_compression : $(LIB)

# simple_decompression.o: common.h
# simple_decompression : $(LIB)

# multiple_simple_compression.o: common.h
# multiple_simple_compression : $(LIB)

# dictionary_compression.o: common.h
# dictionary_compression : $(LIB)

# dictionary_decompression.o: common.h
# dictionary_decompression : $(LIB)

# streaming_compression.o: common.h
# streaming_compression : $(LIB)

# multiple_streaming_compression.o: common.h
# multiple_streaming_compression : $(LIB)

# streaming_decompression.o: common.h
# streaming_decompression : $(LIB)

# streaming_memory_usage.o: common.h
# streaming_memory_usage : $(LIB)


# .PHONY:clean
# clean:
# 	@$(RM) core *.o tmp* result* *.zst \
#         simple_compression simple_decompression \
#         multiple_simple_compression \
#         dictionary_compression dictionary_decompression \
#         streaming_compression streaming_decompression \
#         multiple_streaming_compression streaming_memory_usage
# 	@echo Cleaning completed

# .PHONY:test
# test: all
# 	cp README.md tmp
# 	cp Makefile tmp2
# 	@echo -- Simple compression tests
# 	./simple_compression tmp
# 	./simple_decompression tmp.zst
# 	./multiple_simple_compression *.c
# 	./streaming_decompression tmp.zst > /dev/null
# 	@echo -- Streaming memory usage
# 	./streaming_memory_usage
# 	@echo -- Streaming compression tests
# 	./streaming_compression tmp
# 	./streaming_decompression tmp.zst > /dev/null
# 	@echo -- Edge cases detection
# 	! ./streaming_decompression tmp    # invalid input, must fail
# 	! ./simple_decompression tmp       # invalid input, must fail
# 	touch tmpNull                      # create 0-size file
# 	./simple_compression tmpNull
# 	./simple_decompression tmpNull.zst # 0-size frame : must work
# 	@echo -- Multiple streaming tests
# 	./multiple_streaming_compression *.c
# 	@echo -- Dictionary compression tests
# 	./dictionary_compression tmp2 tmp README.md
# 	./dictionary_decompression tmp2.zst tmp.zst README.md
# 	$(RM) tmp* *.zst
# 	@echo tests completed
CC = gcc
TARGET = Test_updateDictionary
SRC = Test_updateDictionary.c
OBJ = ../programs/obj/conf_aec26574c336b900571a5deec4052692/util.o \
../programs/obj/conf_aec26574c336b900571a5deec4052692/dibio.o \
../programs/obj/conf_aec26574c336b900571a5deec4052692/timefn.o \
../programs/obj/conf_aec26574c336b900571a5deec4052692/error_private.o \
Test_updateDictionary.o
LDFLAGS = -L../lib/ -lzstd -lpthread
CFLAGS = -c -g -O0 -Wall
$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)
Test_updateDictionary.o:Test_updateDictionary.c 
	$(CC) $(CFLAGS) $< -o $@
	@echo "done"
.PHONY: clean 
clean:
	rm -f Test_updateDictionary.o $(TARGET)

# ../programs/obj/conf_aec26574c336b900571a5deec4052692/zstd_double_fast.o 
# ../programs/obj/conf_aec26574c336b900571a5deec4052692/zstd_double_fast.o 

# CC = gcc
# TARGET = withoutThread
# SRC = withoutThread.c
# OBJ =  ../programs/obj/conf_aec26574c336b900571a5deec4052692/util.o \
# ../programs/obj/conf_aec26574c336b900571a5deec4052692/dibio.o \
# ../programs/obj/conf_aec26574c336b900571a5deec4052692/timefn.o \
# ../programs/obj/conf_aec26574c336b900571a5deec4052692/error_private.o \
# withoutThread.o
# # ../programs/obj/conf_aec26574c336b900571a5deec4052692/zstdmt_compress.o \


# LDFLAGS = -L../lib/ -lzstd 
# #LDFLAGS = -lzstd 
# # CFLAGS = -c -g -O0 -Wall -Wl,-rpath,/home/yonghui/dictSize/test_mydictComp/BackUp/Test_zstdDictionary/lib 
# CFLAGS = -c -g -O0 -Wall
# $(TARGET): $(OBJ)
# #$(CC) -Wall -Wl,-rpath,/home/yonghui/dictSize/test_mydictComp/BackUp/Test_zstdDictionary/lib  -o $@ $^ $(LDFLAGS)
# 	$(CC)  -o $@ $^ $(LDFLAGS)
# withoutThread.o:withoutThread.c
# 	$(CC) $(CFLAGS) $< -o $@
# 	@echo "done"
# .PHONY: clean 
# clean:
# 	rm -f withoutThread.o $(TARGET)
