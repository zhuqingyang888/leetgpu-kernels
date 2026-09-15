NVCC := nvcc
NVCCFLAGS := -O3
INCLUDES := -I../common

# 当前算子目录名，例如 1_Vector_Addition
DIR := $(CURDIR)
NAME := $(notdir $(DIR))

# 所有编译结果统一放到 ../build/<算子名>/
BUILD_DIR := ../build/$(NAME)
TARGET := $(BUILD_DIR)/$(NAME)

SRCS := $(wildcard *.cu)
OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(SRCS))
HEADERS := $(wildcard *.cuh)

.PHONY: all run sass clean

all: $(TARGET)

# 创建 build 目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 链接
$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

# 编译
$(BUILD_DIR)/%.o: %.cu $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# 运行
run: $(TARGET)
	$(TARGET)

# 导出 SASS
sass: $(TARGET)
	cuobjdump --dump-sass $(TARGET) > $(BUILD_DIR)/sass.txt
	@echo "SASS saved to $(BUILD_DIR)/sass.txt"

# 只删除当前算子的 build
clean:
	rm -rf $(BUILD_DIR)