#!/bin/bash

# 步骤 1：清理现有的 temp_repo 目录并重新创建
rm -rf temp_repo  # 删除已有的 temp_repo 目录
mkdir temp_repo   # 重新创建 temp_repo 目录

# 步骤 2：初始化 Git 仓库
cd temp_repo
git init

# 步骤 3：删除现有的 remote 配置并重新添加远程仓库
git remote remove origin  # 删除现有的 origin 配置
git remote add -f origin https://github.com/AGI-Edgerunners/LLM-Adapters.git

# 步骤 4：启用 Sparse Checkout
git config core.sparseCheckout true

# 步骤 5：指定要下载的文件路径
echo "dataset/" >> .git/info/sparse-checkout  # 下载 commonsense datasets 文件夹
echo "ft-training_set/commonsense_170k.json" >> .git/info/sparse-checkout  # 下载 commonsense_170k.json 文件

# 步骤 6：拉取指定内容
git pull origin main  # 如果主分支是 main
# git pull origin master  # 如果主分支是 master

# 步骤 7：返回上一级目录并查看下载的内容
cd ..
ls temp_repo/dataset  # 查看 commonsense datasets 文件夹
ls temp_repo/ft-training_set  # 查看 ft-training_set 文件夹
mv -f temp_repo/dataset ./dataset  # 移动 dataset 文件夹到当前目录
mv -f temp_repo/ft-training_set/commonsense_170k.json ./commonsense_170k.json  # 移动 commonsense_170k.json 文件到当前目录
